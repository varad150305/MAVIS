"""Unit tests for prototype/llm_stage.py with a scripted fake runtime and a fake Ollama HTTP server.
No model is needed. Real-model validation is in selector-independent script llm_validation/."""
import copy
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))

from llm_stage import (  # noqa: E402
    DocumentAssistant, LLMConfig, LLMError, NO_CONTEXT_MESSAGE, OllamaRuntime, format_result, numbers_in,
    pages_from_document, quote_match, quote_supported, verify_recovery,
)


def doc(*texts):
    return {"document_id": "d", "pages": [{"page_number": i + 1, "raw_text": t, "corrected_text": t,
                                           "confidence_label": "HIGH"} for i, t in enumerate(texts)]}


class FakeRuntime:
    """Returns scripted JSON by stage (recovery prompts contain 'repair OCR')."""

    def __init__(self, recover=None, answer=None, fail_on=None):
        self.recover, self.answer, self.fail_on = recover, answer, fail_on
        self.calls = []

    def chat(self, system, user, schema, max_tokens):
        stage = "recover" if "repair OCR" in system else "answer"
        self.calls.append((stage, user))
        if self.fail_on == stage:
            raise LLMError("runtime_unavailable", "down")
        payload = self.recover(user) if stage == "recover" else self.answer(user)
        return json.dumps(payload), {"prompt_tokens": 10, "output_tokens": 5, "output_s": 1.0, "prompt_s": 0.1,
                                     "total_s": 1.1, "load_s": 0.0, "done_reason": "stop"}


def identity_recover(_user):
    return {"corrected_text": None, "uncertain_spans": []}


def make_recover(mapping):
    def f(user):
        raw = user.split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0]
        return mapping.get(raw, {"corrected_text": raw, "uncertain_spans": []})
    return f


# ----------------------------------------------------------------- helpers
def test_pages_from_document_skips_empty_pages_and_keeps_numbers():
    d = {"pages": [{"page_number": 1, "raw_text": "  "}, {"page_number": 2, "raw_text": "Hello"}, {"page_number": 3}]}
    assert pages_from_document(d) == [{"page": 2, "text": "Hello", "confidence_label": None, "raw_text_missing": False}]
    assert pages_from_document(None) == [] and pages_from_document({}) == []


def test_numbers_in():
    assert numbers_in("Total 129.50 on 14/03/2026 at 10:30, id 4821-7") == ["129.50", "14/03/2026", "10:30", "4821-7"]


def test_verify_recovery_accepts_small_letter_fixes_and_rejects_rewrites():
    raw = "The quick brovvn fox jumps ouer the Iazy dog"
    ok, issues, _ = verify_recovery(raw, "The quick brown fox jumps over the lazy dog", [], 0.8)
    assert ok and not issues
    ok, issues, _ = verify_recovery(raw, "A fast animal leaps across a sleepy canine in the garden today", [], 0.8)
    assert not ok and "changed too much" in issues[0]


def test_verify_recovery_rejects_invented_or_dropped_numbers_and_flags_lookalike_fixes():
    raw = "Invoice 4821 total 129.50 paid"
    ok, issues, _ = verify_recovery(raw, "Invoice 4821 total 139.50 paid", [], 0.8)
    assert not ok and any("dropped or altered '129.50'" in i for i in issues)      # 2->3 is not an OCR look-alike
    ok, issues, _ = verify_recovery(raw, "Invoice 4821 total 7777 paid", [], 0.8)
    assert not ok and any("introduced '7777'" in i for i in issues)
    ok, issues, _ = verify_recovery(raw, "Invoice total paid", [], 0.5)
    assert not ok and any("dropped or altered" in i for i in issues)
    # a genuine look-alike repair (l -> 1) is accepted but always flagged as unverified
    ok, issues, extra = verify_recovery("Invoice 4821 total l29.50 paid", "Invoice 4821 total 129.50 paid", [], 0.8)
    assert ok and extra and extra[0]["text"] == "129.50" and extra[0]["possible_readings"] == ["l29.50"]
    _, _, extra = verify_recovery("Invoice 4821 total l29.50 paid", "Invoice 4821 total 129.50 paid",
                                  [{"text": "129.50"}], 0.8)
    assert extra == []           # already flagged by the model


def test_verify_recovery_handles_attached_punctuation_and_letter_words_with_digit_lookalikes():
    # regression from the first real-model run: 'l.5%' -> '1.5%' and 'Subtota1' -> 'Subtotal' were wrongly rejected
    ok, issues, extra = verify_recovery("accrue l.5% interest. Subtota1: 388.50", "accrue 1.5% interest. Subtotal: 388.50", [], 0.8)
    assert ok, issues
    assert [e["text"] for e in extra] == ["1.5"]          # the numeric repair is flagged; the word repair is not
    # a year that is NOT derivable from the OCR reading is still rejected: 202S -> 2025 is fine, 2023 is invented
    assert verify_recovery("Date 07 November 202S", "Date 07 November 2025", [], 0.8)[0]
    ok, issues, _ = verify_recovery("Date 07 November 202S", "Date 07 November 2023", [], 0.8)
    assert not ok and any("202S" in i for i in issues)


def test_quote_supported_is_tolerant_of_case_and_punctuation_but_not_invention():
    texts = ["Invoice 4821 total 129.50 paid on 14 March."]
    assert quote_supported("total 129.50 paid", texts)
    assert quote_supported("INVOICE 4821 TOTAL 129.50", texts)
    assert not quote_supported("total 999.00 paid", texts)
    assert not quote_supported("", texts)
    # a quote that differs from the OCR text by OCR-level noise is only an approximate match, never "exact"
    ratio, kind = quote_match("Indice No 4821", ["Inudice No 4821 Date 14 Manch"])
    assert kind == "approx" and 0.85 <= ratio < 1.0
    assert quote_match("total 129.50", texts)[1] == "exact"
    assert quote_match("shipping address is Pune", texts)[1] == "none"


# ----------------------------------------------------------------- ask() flow
def test_ask_answers_with_verified_evidence_and_never_modifies_the_ocr_document():
    d = doc("Invoice 4821 total l29.50 paid on 14 March")
    before = copy.deepcopy(d)
    rt = FakeRuntime(
        recover=make_recover({d["pages"][0]["raw_text"]: {
            "corrected_text": "Invoice 4821 total 129.50 paid on 14 March",
            "uncertain_spans": [{"text": "129.50", "reason": "l vs 1", "possible_readings": ["129.50"]}]}}),
        answer=lambda u: {"answerable_from_document": True, "answer": "The total is 129.50 (reading uncertain).",
                          "evidence": [{"page": 1, "quote": "total 129.50 paid"}], "general_knowledge": ""})
    r = DocumentAssistant(LLMConfig(), rt).ask(d, "What is the total?")
    assert r["error"] is None and r["source"] == "document"
    assert r["evidence"][0]["verified_in_ocr_text"] is True
    assert r["pages"][0]["correction_accepted"] and r["pages"][0]["ocr_text"].endswith("l29.50 paid on 14 March")
    assert r["uncertain_spans"][0]["page"] == 1
    assert "[?129.50?]" in rt.calls[-1][1]          # uncertain span marked for the answering stage
    assert d == before                                # the OCR result is untouched


def test_hallucinated_correction_is_rejected_and_raw_ocr_is_used():
    d = doc("Invoice 4821 total 129.50 paid on 14 March")
    rt = FakeRuntime(recover=lambda u: {"corrected_text": "Invoice 4821 total 5000.00 paid on 14 March", "uncertain_spans": []},
                     answer=lambda u: {"answerable_from_document": True, "answer": "129.50",
                                       "evidence": [{"page": 1, "quote": "total 129.50"}], "general_knowledge": ""})
    r = DocumentAssistant(LLMConfig(), rt).ask(d, "total?")
    assert r["pages"][0]["correction_accepted"] is False
    assert r["pages"][0]["corrected_text"] == d["pages"][0]["raw_text"]
    assert any("correction rejected" in w for w in r["warnings"])
    assert "5000.00" not in rt.calls[-1][1]


def test_answer_absent_from_document_is_marked_not_in_document():
    d = doc("Quarterly report. Revenue grew in the north region.")
    rt = FakeRuntime(recover=make_recover({}),
                     answer=lambda u: {"answerable_from_document": False, "answer": "The captured document does not say.",
                                       "evidence": [], "general_knowledge": ""})
    r = DocumentAssistant(LLMConfig(), rt).ask(d, "Who is the CEO?")
    assert r["source"] == "not_in_document" and r["answerable_from_document"] is False
    assert "does not contain" in format_result(r)


def test_general_knowledge_is_kept_separate_from_the_document_answer():
    d = doc("Solve for x: 2x + 3 = 11")
    rt = FakeRuntime(recover=make_recover({}), answer=lambda u: {
        "answerable_from_document": False, "answer": "The document asks the question but gives no solution.",
        "evidence": [], "general_knowledge": "From general knowledge, not the document: x = 4."})
    r = DocumentAssistant(LLMConfig(), rt).ask(d, "What is x?")
    assert r["source"] == "general_knowledge"
    assert r["general_knowledge"].startswith("From general knowledge, not the document:")
    assert "x = 4" not in r["answer"]


def test_unverified_evidence_and_missing_evidence_are_flagged():
    d = doc("Invoice 4821 total 129.50")
    rt = FakeRuntime(recover=make_recover({}), answer=lambda u: {
        "answerable_from_document": True, "answer": "It is 999.", "general_knowledge": "",
        "evidence": [{"page": 1, "quote": "total 999.00 paid in full"}]})
    r = DocumentAssistant(LLMConfig(), rt).ask(d, "total?")
    assert r["source"] == "document_unverified" and r["evidence"][0]["verified_in_ocr_text"] is False
    assert "NOT FOUND IN OCR TEXT" in format_result(r)
    rt2 = FakeRuntime(recover=make_recover({}), answer=lambda u: {
        "answerable_from_document": True, "answer": "x", "evidence": [], "general_knowledge": ""})
    r2 = DocumentAssistant(LLMConfig(), rt2).ask(d, "total?")
    assert r2["source"] == "document_unverified" and any("without supporting evidence" in w for w in r2["warnings"])


def test_empty_or_missing_ocr_context_gives_explicit_message_without_calling_the_model():
    rt = FakeRuntime()
    for d in (None, {"pages": []}, doc(""), doc("   ", "")):
        r = DocumentAssistant(LLMConfig(), rt).ask(d, "anything?")
        assert r["error"]["code"] == "no_context" and r["error"]["message"] == NO_CONTEXT_MESSAGE
        assert format_result(r).startswith("[no_context]")
    assert rt.calls == []
    assert DocumentAssistant(LLMConfig(), rt).ask(doc("text"), "  ")["error"]["code"] == "empty_request"


def test_runtime_failure_returns_error_not_exception_and_keeps_finished_stages():
    d = doc("Some text here")
    rt = FakeRuntime(recover=make_recover({}), answer=lambda u: {}, fail_on="answer")
    r = DocumentAssistant(LLMConfig(), rt).ask(d, "q?")
    assert r["error"]["code"] == "runtime_unavailable"
    assert r["pages"] and r["pages"][0]["ocr_text"] == "Some text here"      # recovery result survives
    rt2 = FakeRuntime(fail_on="recover")
    assert DocumentAssistant(LLMConfig(), rt2).ask(d, "q?")["error"]["code"] == "runtime_unavailable"


def test_bad_json_from_the_model_is_reported():
    class Bad(FakeRuntime):
        def chat(self, *a, **k):
            return "not json", {"prompt_tokens": 1, "output_tokens": 1, "output_s": 1, "prompt_s": 1, "total_s": 1,
                                "load_s": 0, "done_reason": "stop"}
    r = DocumentAssistant(LLMConfig(), Bad()).ask(doc("text"), "q?", recover=False)
    assert r["error"]["code"] == "bad_output"


def test_recovery_is_cached_per_page_text_so_follow_up_questions_skip_stage_one():
    d = doc("Alpha beta gamma")
    rt = FakeRuntime(recover=make_recover({}), answer=lambda u: {
        "answerable_from_document": True, "answer": "a", "general_knowledge": "",
        "evidence": [{"page": 1, "quote": "Alpha beta"}]})
    a = DocumentAssistant(LLMConfig(), rt)
    a.ask(d, "q1?")
    a.ask(d, "q2?")
    assert [c[0] for c in rt.calls] == ["recover", "answer", "answer"]


def test_summarize_and_recover_modes_and_recover_disabled():
    d = doc("Alpha beta gamma", "Delta epsilon")
    rt = FakeRuntime(recover=make_recover({}), answer=lambda u: {
        "answerable_from_document": True, "answer": "- summary", "general_knowledge": "",
        "evidence": [{"page": 2, "quote": "Delta epsilon"}]})
    a = DocumentAssistant(LLMConfig(), rt)
    r = a.ask(d, "", mode="summarize")
    assert r["error"] is None and r["answer"] == "- summary" and "Summarise the document" in rt.calls[-1][1]
    r = a.ask(d, "", mode="recover")
    assert r["error"] is None and r["answer"] == "" and "[Page 2]" in r["corrected_text"]
    n = len(rt.calls)
    a.ask(d, "q?", recover=False)
    assert [c[0] for c in rt.calls[n:]] == ["answer"]


def test_every_page_reaches_the_model_in_order_nothing_is_truncated():
    d = doc(*[f"Page body number {i} " + "word " * 80 for i in range(1, 21)])
    rt = FakeRuntime(recover=make_recover({}), answer=lambda u: {
        "answerable_from_document": False, "answer": "no", "evidence": [], "general_knowledge": ""})
    cfg = LLMConfig(max_recover_pages=2, num_ctx=32768)
    r = DocumentAssistant(cfg, rt).ask(d, "q?")
    prompt = [c for c in rt.calls if c[0] == "answer"][0][1]
    positions = [prompt.index(f"=== PAGE {i} ===") for i in range(1, 21)]
    assert positions == sorted(positions) and len(positions) == 20           # all 20 pages, in order
    assert all(f"Page body number {i} " in prompt for i in range(1, 21))     # full text, also beyond the recovery cap
    assert any("Only the first 2 pages" in w for w in r["warnings"])         # the recovery cap is reported, not hidden
    assert not any("truncated" in w.lower() for w in r["warnings"])


def test_a_document_that_does_not_fit_the_context_is_an_error_not_a_silent_cut():
    d = doc(*["lorem ipsum " * 400] * 6)
    rt = FakeRuntime(recover=make_recover({}), answer=lambda u: {})
    r = DocumentAssistant(LLMConfig(num_ctx=4096), rt).ask(d, "q?", recover=False)
    assert r["error"]["code"] == "context_too_large" and "Nothing was cut" in r["error"]["message"]
    assert "MAVIS_LLM_NUM_CTX" in r["error"]["message"]
    assert rt.calls == []                                                     # never sent a partial document


def test_not_finalized_document_is_refused_and_failed_pages_are_reported():
    base = doc("Alpha beta", "Gamma delta")
    rt = FakeRuntime(recover=make_recover({}), answer=lambda u: {
        "answerable_from_document": True, "answer": "a", "general_knowledge": "",
        "evidence": [{"page": 1, "quote": "Alpha beta"}]})
    bad = {**base, "processing": {"status": "incomplete", "pages_ok": 1, "pages_failed": []}}
    r = DocumentAssistant(LLMConfig(), rt).ask(bad, "q?")
    assert r["error"]["code"] == "not_finalized" and rt.calls == []
    partial = {**base, "processing": {"status": "complete_with_errors", "pages_ok": 2, "pages_failed": [3]}}
    r = DocumentAssistant(LLMConfig(), rt).ask(partial, "q?")
    assert r["error"] is None and any("page(s) [3]" in w for w in r["warnings"])


def test_result_carries_the_hash_of_the_exact_document_it_was_given_and_does_not_modify_it():
    import hashlib
    d = {**doc("Alpha beta"), "processing": {"status": "complete", "pages_ok": 1, "pages_failed": []}}
    before = copy.deepcopy(d)
    rt = FakeRuntime(recover=make_recover({}), answer=lambda u: {
        "answerable_from_document": True, "answer": "a", "general_knowledge": "",
        "evidence": [{"page": 1, "quote": "Alpha beta"}]})
    r = DocumentAssistant(LLMConfig(), rt).ask(d, "q?")
    assert r["document_sha256"] == hashlib.sha256(json.dumps(before, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert d == before


# ----------------------------------------------------------------- Ollama HTTP client
class _Handler(BaseHTTPRequestHandler):
    mode = "ok"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"models": [{"name": "gemma3:4b"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.mode == "missing":
            self._send(404, {"error": "model 'gemma3:4b' not found, try pulling it first"})
        elif self.mode == "slow":
            time.sleep(1.5)
            self._send(200, {})
        elif self.mode == "full":
            self._send(200, {"message": {"content": "{}"}, "done_reason": "stop", "prompt_eval_count": 8192})
        elif self.mode == "length":
            self._send(200, {"message": {"content": "{"}, "done_reason": "length"})
        else:
            self._send(200, {"message": {"content": "{\"ok\": true}"}, "done_reason": "stop", "eval_count": 7,
                             "eval_duration": 1_000_000_000, "prompt_eval_count": 3, "total_duration": 2_000_000_000})


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ollama_runtime_success_and_stats(server):
    srv, url = server
    _Handler.mode = "ok"
    rt = OllamaRuntime(LLMConfig(base_url=url))
    assert rt.list_models() == ["gemma3:4b"]
    content, stats = rt.chat("s", "u", {"type": "object"}, 50)
    assert json.loads(content) == {"ok": True} and stats["output_tokens"] == 7 and stats["total_s"] == 2.0


def test_ollama_runtime_model_missing_gives_exact_pull_command_and_no_substitution(server):
    srv, url = server
    _Handler.mode = "missing"
    with pytest.raises(LLMError) as e:
        OllamaRuntime(LLMConfig(base_url=url, model="gemma3:4b")).chat("s", "u", {}, 10)
    assert e.value.code == "model_missing" and "ollama pull gemma3:4b" in e.value.message


def test_ollama_runtime_timeout_and_truncation(server):
    srv, url = server
    _Handler.mode = "slow"
    with pytest.raises(LLMError) as e:
        OllamaRuntime(LLMConfig(base_url=url, timeout_s=0.3)).chat("s", "u", {}, 10)
    assert e.value.code == "timeout"
    _Handler.mode = "length"
    with pytest.raises(LLMError) as e:
        OllamaRuntime(LLMConfig(base_url=url)).chat("s", "u", {}, 10)
    assert e.value.code == "output_truncated"


def test_prompt_that_fills_the_whole_context_is_reported_as_possible_truncation(server):
    srv, url = server
    _Handler.mode = "full"
    with pytest.raises(LLMError) as e:
        OllamaRuntime(LLMConfig(base_url=url, num_ctx=8192)).chat("s", "u", {}, 10)
    assert e.value.code == "context_too_large" and "cut off" in e.value.message


def test_runtime_unavailable_when_nothing_listens():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    cfg = LLMConfig(base_url=f"http://127.0.0.1:{port}")
    r = DocumentAssistant(cfg).ask(doc("Some text"), "q?")
    assert r["error"]["code"] == "runtime_unavailable" and "Ollama" in r["error"]["message"]
    with pytest.raises(LLMError) as e:
        OllamaRuntime(cfg).list_models()
    assert e.value.code == "runtime_unavailable"


def test_default_config_targets_gemma3_4b_and_env_overrides(monkeypatch):
    monkeypatch.delenv("MAVIS_LLM_MODEL", raising=False)
    assert LLMConfig().model == "gemma3:4b"
    monkeypatch.setenv("MAVIS_LLM_MODEL", "gemma3:1b")
    monkeypatch.setenv("MAVIS_LLM_RECOVER", "0")
    c = LLMConfig()
    assert c.model == "gemma3:1b" and c.recover is False


def test_low_ocr_legibility_is_warned_and_answers_are_not_called_verified():
    pytest.importorskip("wordfreq")
    garbled = "ACuie indiatial Siapplies Lat Inudice No 4821 Cuesberer Pryas Raghunithan Hyelrade pamp guanty"
    d = doc(garbled)
    rt = FakeRuntime(recover=make_recover({}), answer=lambda u: {
        "answerable_from_document": True, "answer": "The invoice is 4821.", "general_knowledge": "",
        "evidence": [{"page": 1, "quote": "Inudice No 4821"}]})
    r = DocumentAssistant(LLMConfig(), rt).ask(d, "invoice number?")
    assert r["ocr_legibility"][1] < 0.6
    assert any("heavily garbled" in w for w in r["warnings"])
    assert r["source"] == "document_unverified"


def test_numbers_in_the_answer_that_are_not_in_the_ocr_text_are_flagged():
    d = doc("Meeting on 07 November 202S about the budget")
    rt = FakeRuntime(recover=make_recover({}), answer=lambda u: {
        "answerable_from_document": True, "answer": "The meeting was on 7 November 2025.", "general_knowledge": "",
        "evidence": [{"page": 1, "quote": "07 November 202S"}]})
    r = DocumentAssistant(LLMConfig(), rt).ask(d, "when?")
    assert "2025" in r["unsupported_numbers"]
    assert r["source"] == "document_unverified" and any("do not appear in the OCR text" in w for w in r["warnings"])


def _mixed_doc():
    clean = "Invoice 4821 total due 419.58 paid by the customer within thirty days"
    garbled = "Secson Quesbon wps apmmp tuyom mod pe soys fiad hte sumof roots"
    return doc(clean, garbled)


def _answer_rt(quote, page):
    return FakeRuntime(recover=make_recover({}), answer=lambda u: {
        "answerable_from_document": True, "answer": "It is 4821.", "general_knowledge": "",
        "evidence": [{"page": page, "quote": quote}]})


def test_a_garbled_page_does_not_downgrade_an_answer_that_cites_only_a_clean_page():
    pytest.importorskip("wordfreq")
    r = DocumentAssistant(LLMConfig(), _answer_rt("Invoice 4821 total due", 1)).ask(_mixed_doc(), "invoice number?")
    assert r["ocr_legibility"][2] < 0.6 < r["ocr_legibility"][1]
    assert r["source"] == "document" and r["evidence"][0]["page"] == 1
    assert any("Page 2: only" in w for w in r["warnings"])              # the garbled page is still announced
    assert not any("relies on page" in w for w in r["warnings"])


def test_an_answer_citing_the_garbled_page_is_unverified():
    pytest.importorskip("wordfreq")
    r = DocumentAssistant(LLMConfig(), _answer_rt("Quesbon wps apmmp", 2)).ask(_mixed_doc(), "what?")
    assert r["source"] == "document_unverified" and any("relies on page(s) [2]" in w for w in r["warnings"])


def test_evidence_is_attributed_to_the_page_where_the_quote_is_found():
    pytest.importorskip("wordfreq")
    # the model claims page 2 but the quote is on page 1 -> reported page is corrected and the claim is kept
    r = DocumentAssistant(LLMConfig(), _answer_rt("Invoice 4821 total due", 2)).ask(_mixed_doc(), "invoice number?")
    ev = r["evidence"][0]
    assert ev["page"] == 1 and ev["page_reported_by_model"] == 2 and ev["match"] == "exact"
    assert r["source"] == "document"                                    # judged by the real page (clean), not the claimed one
