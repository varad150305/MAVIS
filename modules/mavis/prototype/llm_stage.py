"""Local-LLM document assistant for MAVIS: question answering, summarisation and
conservative OCR-text recovery on top of the existing OCR result.

Input is the existing `app.processed_document` (PostProcessingPipeline.process_document):
{"document_id", "pages": [{"page_number", "raw_text", "corrected_text", "confidence_label", ...}]}
(or an exported raw_output.json / corrected_output.json, which has the same page keys).
Nothing here modifies that structure or any OCR text: results are returned as a NEW dict.

Flow, one model (default Gemma 3 4B via a local Ollama server), two logically separate stages:
  1. recover  - per page: fix clear OCR recognition errors only, list `uncertain_spans`.
                A programmatic check rejects a page's correction (keeping the raw OCR text)
                if it changes too much or introduces a number that cannot be derived from
                the OCR text. Numeric corrections are always flagged as uncertain.
  2. answer   - answer / summarise from the recovered text. The model must separate what the
                document says (`answer`, `evidence`) from its own general knowledge
                (`general_knowledge`). Each evidence quote is checked against the OCR text.
No internet access: the model only sees the captured document. A retrieval/search component
can later be added by passing extra context to `ask(..., extra_context=...)`.

The module uses only the standard library (urllib) and never raises from `ask()`:
failures come back as result["error"] = {"code", "message"}.
Config (env, all optional): MAVIS_LLM_MODEL (default gemma3:4b), MAVIS_LLM_URL
(default http://127.0.0.1:11434), MAVIS_LLM_NUM_CTX, MAVIS_LLM_TIMEOUT_S, MAVIS_LLM_RECOVER (1/0),
MAVIS_LLM_KEEP_ALIVE. The model is never substituted: if it is missing you get a
`model_missing` error with the exact pull command.

CLI:  python prototype/llm_stage.py --doc exports/corrected_output.json --ask "What is the total?"
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_MODEL = "gemma3:4b"
DEFAULT_URL = "http://127.0.0.1:11434"

NO_CONTEXT_MESSAGE = "No OCR text has been captured yet, so there is nothing to ask about."


class LLMError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


def _env_bool(name, default):
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class LLMConfig:
    model: str = field(default_factory=lambda: os.environ.get("MAVIS_LLM_MODEL", DEFAULT_MODEL))
    base_url: str = field(default_factory=lambda: os.environ.get("MAVIS_LLM_URL", DEFAULT_URL).rstrip("/"))
    num_ctx: int = field(default_factory=lambda: int(os.environ.get("MAVIS_LLM_NUM_CTX", "16384")))
    timeout_s: float = field(default_factory=lambda: float(os.environ.get("MAVIS_LLM_TIMEOUT_S", "600")))
    keep_alive: str = field(default_factory=lambda: os.environ.get("MAVIS_LLM_KEEP_ALIVE", "10m"))
    recover: bool = field(default_factory=lambda: _env_bool("MAVIS_LLM_RECOVER", True))
    max_recover_pages: int = field(default_factory=lambda: int(os.environ.get("MAVIS_LLM_MAX_RECOVER_PAGES", "6")))
    # pages beyond max_recover_pages are NOT dropped: they reach the model as raw OCR text (and a warning says so)
    chars_per_token: float = 2.5        # deliberately pessimistic for OCR text (measured: see docs)
    prompt_overhead_tokens: int = 700   # system prompt + schema + request wrapper
    max_answer_tokens: int = 700
    max_recover_tokens: int = 900
    min_recovery_similarity: float = 0.80   # corrected text must stay this close to the OCR text


# ---------------------------------------------------------------- runtime
class OllamaRuntime:
    """Minimal Ollama HTTP client (/api/tags, /api/chat). Other runtimes can implement the same two methods."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _post(self, path, payload, timeout):
        req = urllib.request.Request(self.cfg.base_url + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if e.code == 404 and "not found" in body.lower():
                raise LLMError("model_missing",
                               f"Model '{self.cfg.model}' is not installed in Ollama. Install it with: "
                               f"ollama pull {self.cfg.model}   (or set MAVIS_LLM_MODEL to a model you have).")
            raise LLMError("runtime_error", f"Ollama returned HTTP {e.code}: {body[:300]}")
        except (urllib.error.URLError, ConnectionError) as e:
            raise LLMError("runtime_unavailable",
                           f"Cannot reach the local LLM runtime at {self.cfg.base_url} ({getattr(e, 'reason', e)}). "
                           "Install and start Ollama (https://ollama.com) or set MAVIS_LLM_URL.")
        except TimeoutError:
            raise LLMError("timeout", f"The model did not answer within {timeout:.0f} s (MAVIS_LLM_TIMEOUT_S).")
        except OSError as e:           # socket.timeout is an OSError subclass on 3.10+ TimeoutError; keep a safety net
            if "timed out" in str(e):
                raise LLMError("timeout", f"The model did not answer within {timeout:.0f} s (MAVIS_LLM_TIMEOUT_S).")
            raise LLMError("runtime_unavailable", f"Cannot reach the local LLM runtime: {e}")

    def list_models(self):
        try:
            with urllib.request.urlopen(self.cfg.base_url + "/api/tags", timeout=5) as r:
                return [m["name"] for m in json.loads(r.read().decode("utf-8")).get("models", [])]
        except (urllib.error.URLError, OSError) as e:
            raise LLMError("runtime_unavailable",
                           f"Cannot reach the local LLM runtime at {self.cfg.base_url} ({getattr(e, 'reason', e)}). "
                           "Install and start Ollama (https://ollama.com) or set MAVIS_LLM_URL.")

    def chat(self, system, user, schema, max_tokens):
        payload = {"model": self.cfg.model, "stream": False, "format": schema, "keep_alive": self.cfg.keep_alive,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "options": {"temperature": 0, "num_ctx": self.cfg.num_ctx, "num_predict": max_tokens}}
        r = self._post("/api/chat", payload, self.cfg.timeout_s)
        content = (r.get("message") or {}).get("content", "")
        stats = {"total_s": r.get("total_duration", 0) / 1e9, "load_s": r.get("load_duration", 0) / 1e9,
                 "prompt_tokens": r.get("prompt_eval_count", 0), "prompt_s": r.get("prompt_eval_duration", 0) / 1e9,
                 "output_tokens": r.get("eval_count", 0), "output_s": r.get("eval_duration", 0) / 1e9,
                 "done_reason": r.get("done_reason")}
        if stats["prompt_tokens"] >= self.cfg.num_ctx - 16:
            raise LLMError("context_too_large",
                           f"The prompt filled the whole model context ({stats['prompt_tokens']} of {self.cfg.num_ctx} "
                           "tokens), so the runtime may have cut off part of the document. Raise MAVIS_LLM_NUM_CTX "
                           "(needs more RAM) or use a smaller document.")
        if stats["done_reason"] == "length":
            raise LLMError("output_truncated", "The model ran out of output tokens before finishing its answer.")
        return content, stats


# ---------------------------------------------------------------- schemas and prompts
RECOVER_SCHEMA = {
    "type": "object",
    "properties": {
        "corrected_text": {"type": "string"},
        "uncertain_spans": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "reason": {"type": "string"},
            "possible_readings": {"type": "array", "items": {"type": "string"}}},
            "required": ["text", "reason", "possible_readings"]}}},
    "required": ["corrected_text", "uncertain_spans"]}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable_from_document": {"type": "boolean"},
        "answer": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "page": {"type": "integer"}, "quote": {"type": "string"}}, "required": ["page", "quote"]}},
        "general_knowledge": {"type": "string"}},
    "required": ["answerable_from_document", "answer", "evidence", "general_knowledge"]}

RECOVER_SYSTEM = """You repair OCR output. The input is raw OCR text from a camera photo of a paper document. Reply with JSON only.

Rules:
1. corrected_text = the same text with ONLY clear recognition errors fixed: look-alike characters (0/O, 1/l/I, 5/S, rn/m, vv/w), words broken or merged by wrong spacing, stray symbols inside words. Keep the original line breaks, order and wording. Never add, remove, rephrase, summarise, translate or reorder anything.
2. Never guess names, numbers, dates, currency amounts, units, identifiers or mathematical expressions. If one looks wrong, incomplete or unreadable, leave it EXACTLY as in the OCR text and list it in uncertain_spans.
3. uncertain_spans lists every passage you could not read or repair with confidence: text (copied exactly from the OCR), reason, possible_readings (may be empty).
4. If the text is empty or meaningless, return it unchanged with no uncertain_spans."""

ANSWER_SYSTEM = """You help a user understand a scanned paper document. You only see OCR text of it, which can contain recognition errors. Reply with JSON only.

Rules:
1. Use ONLY the document text below to fill "answer". Do not use outside knowledge there.
2. If the document does not contain the information needed, set answerable_from_document=false and set answer to a short statement that the captured document does not say. Do not guess.
3. "evidence": short quotes copied VERBATIM from the document text that support the answer, each with its page number. No evidence is allowed only when answerable_from_document is false.
4. Passages marked [?...?] are uncertain OCR readings. If your answer depends on one, say the reading is uncertain. Never state a number, name, date or amount that is not in the text.
5. "general_knowledge": leave empty unless the user's request needs facts that are not in the document; then write brief help starting with "From general knowledge, not the document:". Never mix it into "answer"."""


# ---------------------------------------------------------------- helpers
def _norm_ws(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _norm_quote(text):
    return re.sub(r"[^\w]+", " ", (text or "").lower()).strip()


_NUM = re.compile(r"\d+(?:[.,:/-]\d+)*")


def numbers_in(text):
    return _NUM.findall(text or "")


try:
    from wordfreq import zipf_frequency
except ImportError:                      # optional: the legibility warning is skipped without it
    zipf_frequency = None

LOW_LEGIBILITY = 0.60


def legibility(text):
    """Share of 3+ letter words that are ordinary English words (wordfreq zipf >= 2). None if unknown.
    Clean or mildly damaged OCR scored 0.72-1.0 and heavily garbled OCR 0.31-0.53 on the validation set
    (in-sample: the threshold 0.60 was read off those cases)."""
    if zipf_frequency is None:
        return None
    words = [w for w in re.findall(r"[A-Za-z]+", text or "") if len(w) >= 3]
    if not words:
        return None
    return sum(zipf_frequency(w.lower(), "en") >= 2.0 for w in words) / len(words)


def pages_from_document(doc):
    """[{page, text}] from processed_document (or an export file). Only pages with OCR text."""
    pages = []
    for p in (doc or {}).get("pages", []) or []:
        text = (p.get("raw_text") or "").strip()
        if text:
            pages.append({"page": int(p.get("page_number", len(pages) + 1)), "text": text,
                          "confidence_label": p.get("confidence_label"),
                          "raw_text_missing": bool(p.get("raw_text_missing"))})
    return pages


_CANON = str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8", "Z": "2",
                        "b": "6", "g": "9", "q": "9"})


def _canon(token):
    """Map OCR look-alike characters onto digits so 'l29.50' and '129.50' compare equal."""
    return token.translate(_CANON)


def _tokens(text):
    out = []
    for t in re.findall(r"\S+", text or ""):
        t = t.strip("%$,.;:()[]'\"")
        if t:
            out.append(t)
    return out


def _has_digit(token):
    return any(c.isdigit() for c in token)


def verify_recovery(raw, corrected, spans, min_similarity):
    """Check a proposed correction against the OCR text.

    Returns (accepted, issues, extra_spans). Rejects when the text changed too much, or when a
    digit-bearing token (number, date, amount, identifier) is dropped, invented, or changed by more than
    OCR look-alike characters (l/1, O/0, S/5 ...). Numeric look-alike repairs that pass are returned as
    extra uncertain spans so they are never presented as verified.
    """
    issues, extra = [], []
    nraw, ncor = _norm_ws(raw), _norm_ws(corrected)
    if not ncor:
        return False, ["empty correction"], extra
    sim = difflib.SequenceMatcher(None, nraw, ncor, autojunk=False).ratio()
    if sim < min_similarity:
        issues.append(f"correction changed too much (similarity {sim:.2f} < {min_similarity})")
    raw_tok, cor_tok = _tokens(nraw), _tokens(ncor)
    raw_set, cor_set = set(raw_tok), set(cor_tok)
    raw_canon, cor_canon = {_canon(t): t for t in raw_tok}, {_canon(t) for t in cor_tok}
    flagged = " ".join(s.get("text", "") for s in spans)
    for r in sorted(t for t in raw_set if _has_digit(t) and t not in cor_set):
        if _canon(r) not in cor_canon:
            issues.append(f"dropped or altered '{r}' from the OCR text")
    for c in sorted(t for t in cor_set if _has_digit(t) and t not in raw_set):
        origin = raw_canon.get(_canon(c))
        if origin is None:
            issues.append(f"introduced '{c}' that is not derivable from the OCR text")
        elif c not in flagged:
            extra.append({"text": c, "reason": f"changed from the OCR reading {origin!r} (look-alike characters); "
                          "not verified", "possible_readings": [origin]})
    return not issues, issues, extra


APPROX_RATIO = 0.85


def quote_match(quote, texts):
    """Best match of `quote` in the page texts, as (ratio, kind). Case/punctuation-insensitive; kind is
    'exact' (substring), 'approx' (aligned word window with ratio >= 0.85, i.e. small OCR-level differences)
    or 'none'. A quote that is only approximately present is NOT presented as verbatim source text."""
    q = _norm_quote(quote)
    if len(q) < 3:
        return 0.0, "none"
    qwords = q.split()
    best = 0.0
    for t in texts:
        nt = _norm_quote(t)
        if q in nt:
            return 1.0, "exact"
        words = nt.split()
        for k in {max(1, len(qwords) - 1), len(qwords), len(qwords) + 1}:
            for i in range(0, max(1, len(words) - k + 1)):
                matcher = difflib.SequenceMatcher(None, q, " ".join(words[i:i + k]), autojunk=False)
                # ratio() can never exceed quick_ratio() (a cheap character-count bound), so a window whose bound is
                # below the approx threshold cannot change the verdict and its exact ratio is not computed. This took
                # a 5-page answer from ~15 s to well under a second; exact / approx results and their ratios are
                # unchanged, and a 'none' verdict reports the best ratio among the windows that were actually scored.
                if matcher.quick_ratio() < APPROX_RATIO:
                    continue
                best = max(best, matcher.ratio())
    return best, ("approx" if best >= APPROX_RATIO else "none")


def quote_supported(quote, texts):
    return quote_match(quote, texts)[1] != "none"


def _parse_json(content, what):
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise LLMError("bad_output", f"The model returned invalid JSON for the {what} stage: {e}")
    if not isinstance(data, dict):
        raise LLMError("bad_output", f"The model returned a non-object for the {what} stage.")
    return data


# ---------------------------------------------------------------- the assistant
class DocumentAssistant:
    def __init__(self, cfg=None, runtime=None):
        self.cfg = cfg or LLMConfig()
        self.runtime = runtime or OllamaRuntime(self.cfg)
        self._recovery_cache = {}

    # -- stage 1 -----------------------------------------------------------------
    def recover_page(self, page):
        key = hashlib.sha1(f"{self.cfg.model}\n{page['text']}".encode("utf-8")).hexdigest()
        if key in self._recovery_cache:
            return {**self._recovery_cache[key], "cached": True}
        self._check_fits(page["text"], self.cfg.max_recover_tokens, f"text of page {page['page']}")
        content, stats = self.runtime.chat(RECOVER_SYSTEM, "OCR text:\n<<<\n" + page["text"] + "\n>>>",
                                           RECOVER_SCHEMA, self.cfg.max_recover_tokens)
        data = _parse_json(content, "recovery")
        corrected = data.get("corrected_text", "")
        spans = [s for s in data.get("uncertain_spans", []) if isinstance(s, dict) and s.get("text")]
        ok, issues, extra = verify_recovery(page["text"], corrected, spans, self.cfg.min_recovery_similarity)
        out = {"page": page["page"], "ocr_text": page["text"],
               "corrected_text": corrected if ok else page["text"],
               "correction_accepted": ok, "rejection_reasons": issues,
               "uncertain_spans": (spans + extra) if ok else [],
               "changed": ok and _norm_ws(corrected) != _norm_ws(page["text"]), "stats": stats, "cached": False}
        self._recovery_cache[key] = out
        return out

    # -- context -----------------------------------------------------------------
    @staticmethod
    def _context(pages, recovered):
        """All pages, in page order, nothing dropped."""
        blocks = []
        for p in pages:
            rec = recovered.get(p["page"])
            text = rec["corrected_text"] if rec else p["text"]
            for sp in (rec["uncertain_spans"] if rec else []):
                if sp["text"] in text:
                    text = text.replace(sp["text"], f"[?{sp['text']}?]", 1)
            blocks.append(f"=== PAGE {p['page']} ===\n{text}")
        return "\n\n".join(blocks)

    def _check_fits(self, document_text, output_tokens, what):
        """Refuse (never truncate) when the text cannot fit the configured model context."""
        est = int(len(document_text) / self.cfg.chars_per_token) + self.cfg.prompt_overhead_tokens + output_tokens
        if est > self.cfg.num_ctx:
            raise LLMError(
                "context_too_large",
                f"The {what} needs about {est} tokens but the model context is {self.cfg.num_ctx} tokens. Nothing was "
                "cut. Raise MAVIS_LLM_NUM_CTX (more RAM; Gemma 3 supports up to 128K), or work on fewer pages.")

    # -- both stages ---------------------------------------------------------------
    def ask(self, doc, request, mode="answer", recover=None, extra_context=None):
        """mode: 'answer' | 'summarize' | 'recover'. Returns a result dict; never raises."""
        t0 = time.perf_counter()
        warnings = []
        result = {"mode": mode, "request": request, "model": self.cfg.model, "error": None,
                  "pages": [], "corrected_text": "", "uncertain_spans": [], "answer": "",
                  "answerable_from_document": None, "evidence": [], "general_knowledge": "",
                  "source": None, "unsupported_numbers": [], "document_sha256": None, "warnings": warnings, "timing": {}, "ocr_text_unmodified": True}
        processing = (doc or {}).get("processing")
        if processing is not None:
            result["document_sha256"] = hashlib.sha256(
                json.dumps(doc, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            if processing.get("status") not in ("complete", "complete_with_errors"):
                result["error"] = {"code": "not_finalized", "message":
                                   "The document is not finalized (OCR has not finished); nothing was sent to the model."}
                return result
            if processing.get("pages_failed"):
                warnings.append(f"OCR produced no text for page(s) {processing['pages_failed']}; "
                                "they are not part of this document.")
        pages = pages_from_document(doc)
        for p in pages:
            if p["raw_text_missing"]:
                warnings.append(f"Page {p['page']}: the original OCR text is not in this export; the corrected text "
                                "is used instead.")
        if not pages:
            result["error"] = {"code": "no_context", "message": NO_CONTEXT_MESSAGE}
            return result
        if mode == "answer" and not (request or "").strip():
            result["error"] = {"code": "empty_request", "message": "Type a question first."}
            return result
        low_pages = []
        result["ocr_legibility"] = {}
        for p in pages:
            lg = legibility(p["text"])
            if lg is not None:
                result["ocr_legibility"][p["page"]] = round(lg, 2)
                if lg < LOW_LEGIBILITY:
                    low_pages.append(p["page"])
                    warnings.append(f"Page {p['page']}: only {lg:.0%} of the OCR words are recognisable English words, "
                                    "so the text is heavily garbled; answers may repeat OCR errors.")
        do_recover = self.cfg.recover if recover is None else recover
        recovered = {}
        try:
            if do_recover or mode == "recover":
                t1 = time.perf_counter()
                for p in pages[: self.cfg.max_recover_pages]:
                    recovered[p["page"]] = self.recover_page(p)
                if len(pages) > self.cfg.max_recover_pages:
                    warnings.append(f"Only the first {self.cfg.max_recover_pages} pages were checked for OCR errors.")
                result["timing"]["recover_s"] = round(time.perf_counter() - t1, 2)
                for rec in recovered.values():
                    if not rec["correction_accepted"]:
                        warnings.append(f"Page {rec['page']}: correction rejected, raw OCR kept "
                                        f"({'; '.join(rec['rejection_reasons'])}).")
                result["pages"] = [{k: v for k, v in rec.items() if k != "stats"} for rec in recovered.values()]
                result["corrected_text"] = "\n\n".join(
                    f"[Page {r['page']}]\n{r['corrected_text']}" for r in recovered.values())
                result["uncertain_spans"] = [{"page": r["page"], **s} for r in recovered.values()
                                             for s in r["uncertain_spans"]]
            if mode == "recover":
                result["source"] = "ocr_recovery"
                return self._finish(result, t0, recovered, None)
            context = self._context(pages, recovered)
            if extra_context:
                context += "\n\n=== EXTRA CONTEXT (not the captured document) ===\n" + extra_context
            self._check_fits(context, self.cfg.max_answer_tokens, "document")
            task = ("Summarise the document in at most 5 short bullet points." if mode == "summarize"
                    else request.strip())
            t2 = time.perf_counter()
            content, stats = self.runtime.chat(ANSWER_SYSTEM, f"DOCUMENT:\n{context}\n\nUSER REQUEST: {task}",
                                               ANSWER_SCHEMA, self.cfg.max_answer_tokens)
            result["timing"]["answer_s"] = round(time.perf_counter() - t2, 2)
            data = _parse_json(content, "answer")
            result["answer"] = str(data.get("answer", "")).strip()
            result["answerable_from_document"] = bool(data.get("answerable_from_document"))
            result["general_knowledge"] = str(data.get("general_knowledge", "")).strip()
            source_texts = [p["text"] for p in pages] + [r["corrected_text"] for r in recovered.values()]
            page_texts = {p["page"]: [p["text"]] + ([recovered[p["page"]]["corrected_text"]] if p["page"] in recovered else [])
                          for p in pages}
            for ev in data.get("evidence", []):
                if isinstance(ev, dict) and ev.get("quote"):
                    # attribute the quote to the page where it is actually found (the model's page number is only
                    # a hint: it is tried first, so it wins ties)
                    order = ([ev.get("page")] if ev.get("page") in page_texts else []) +                             [pg for pg in page_texts if pg != ev.get("page")]
                    ratio, kind, found = 0.0, "none", None
                    for pg in order:
                        r, k = quote_match(ev["quote"], page_texts[pg])
                        if r > ratio:
                            ratio, kind, found = r, k, pg
                    entry = {"page": found if kind != "none" else ev.get("page"), "quote": ev["quote"],
                             "verified_in_ocr_text": kind != "none", "match": kind, "match_ratio": round(ratio, 2)}
                    if kind != "none" and ev.get("page") != found:
                        entry["page_reported_by_model"] = ev.get("page")
                    result["evidence"].append(entry)
            ev_ok = bool(result["evidence"]) and all(e["verified_in_ocr_text"] for e in result["evidence"])
            if result["answerable_from_document"] and not result["evidence"]:
                warnings.append("The model gave an answer without supporting evidence; treat it as unverified.")
            elif result["answerable_from_document"] and not ev_ok:
                warnings.append("Some evidence quotes could not be found in the OCR text; treat them as unverified.")
            context_numbers = set(numbers_in(chr(10).join(source_texts + ([extra_context] if extra_context else []))))
            result["unsupported_numbers"] = [n for n in numbers_in(result["answer"]) if n not in context_numbers]
            if result["answerable_from_document"] and result["unsupported_numbers"]:
                warnings.append("The answer contains number(s) " + ", ".join(sorted(set(result["unsupported_numbers"])))
                                + " that do not appear in the OCR text (computed or guessed); treat as unverified.")
            cited_low = sorted({e["page"] for e in result["evidence"] if e["page"] in low_pages})
            if result["answerable_from_document"] and cited_low:
                warnings.append(f"The answer relies on page(s) {cited_low} whose OCR text is heavily garbled; "
                                "treat it as unverified.")
            clean = ev_ok and not result["unsupported_numbers"] and not cited_low
            result["source"] = ("document" if result["answerable_from_document"] and clean else
                                "document_unverified" if result["answerable_from_document"] else
                                "general_knowledge" if result["general_knowledge"] else "not_in_document")
            return self._finish(result, t0, recovered, stats)
        except LLMError as e:
            result["error"] = {"code": e.code, "message": e.message}
            return self._finish(result, t0, recovered, None)

    @staticmethod
    def _finish(result, t0, recovered, answer_stats):
        result["timing"]["total_s"] = round(time.perf_counter() - t0, 2)
        toks = sum(r["stats"]["output_tokens"] for r in recovered.values() if not r.get("cached"))
        ptoks = sum(r["stats"]["prompt_tokens"] for r in recovered.values() if not r.get("cached"))
        if answer_stats:
            toks += answer_stats["output_tokens"]
            ptoks += answer_stats["prompt_tokens"]
            result["timing"]["answer_output_tokens_per_s"] = round(
                answer_stats["output_tokens"] / answer_stats["output_s"], 2) if answer_stats["output_s"] else None
        result["timing"]["prompt_tokens"], result["timing"]["output_tokens"] = ptoks, toks
        return result


def format_result(r):
    """Plain-text rendering for the UI / CLI."""
    if r.get("error"):
        return f"[{r['error']['code']}] {r['error']['message']}"
    if r["mode"] == "recover":
        out = [r["corrected_text"] or "(nothing to recover)"]
        for s in r["uncertain_spans"]:
            out.append(f"? p{s['page']}: {s['text']!r} - {s['reason']}")
    else:
        out = [r["answer"] or "(no answer)"]
        if r["evidence"]:
            out.append("\nEvidence:")
            out += [f"- p{e['page']}: \"{e['quote']}\"" + ("" if e["match"] == "exact" else "  [approximate match to the OCR text]" if e["match"] == "approx"
                    else "  [NOT FOUND IN OCR TEXT]")
                    for e in r["evidence"]]
        if r["general_knowledge"]:
            out.append("\n" + r["general_knowledge"])
        if r["source"] == "not_in_document":
            out.append("\n(The captured document does not contain this.)")
    for w in r["warnings"]:
        out.append(f"! {w}")
    out.append(f"\n[{r['model']} - {r['timing'].get('total_s')} s]")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ask the local LLM about an OCR export")
    ap.add_argument("--doc", required=True, help="an exports/ directory (or one of its files)")
    ap.add_argument("--ask", default="")
    ap.add_argument("--mode", choices=["answer", "summarize", "recover"])
    ap.add_argument("--no-recover", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from postprocess.formatter import OCRFormatter     # merges raw_output.json + corrected_output.json
    doc = OCRFormatter.load_document(a.doc)
    mode = a.mode or ("answer" if a.ask else "summarize")
    r = DocumentAssistant().ask(doc, a.ask, mode=mode, recover=False if a.no_recover else None)
    print(json.dumps(r, indent=1, ensure_ascii=False) if a.json else format_result(r))
    return 1 if r["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
