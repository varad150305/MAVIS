"""Tests for the web UI layer (webapp/): headless scan session, document store, gating rules and HTTP surface.

Real threads and the real selector/post-processing code; only the camera, the OCR reader and the model are scripted.
Run: py -3.11 -m pytest tests/test_webapp.py
"""
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "prototype"))
sys.path.insert(0, str(REPO))

import webapp.session as session_mod  # noqa: E402
from llm_stage import DocumentAssistant, LLMConfig  # noqa: E402
from ocr_engine import PrototypeOCREngine  # noqa: E402
from webapp.core import Busy, Refused, WebApp  # noqa: E402
from webapp.session import ScanSession  # noqa: E402


PRODUCTION_COOLDOWN_S = session_mod.OCR_FOCUS_COOLDOWN_SEC


# ---------------------------------------------------------------- scripted camera / OCR / model
def blocky(seed):
    """A frame whose 32x32 fingerprint differs strongly from any other seed."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (18, 32), dtype=np.uint8)
    return cv2.cvtColor(cv2.resize(small, (640, 360), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)


class FakeStream:
    url = "fake://camera"

    def __init__(self, pages=3, fps=50.0, page_s=1.0, connect_ok=True, connect_delay=0.0, stall_after=None):
        self.pages, self.fps, self.page_s = pages, fps, page_s
        self.connect_ok, self.connect_delay, self.stall_after = connect_ok, connect_delay, stall_after
        self.idx, self.t0, self.released = 0, None, False
        self._gate = threading.Event()
        self.rng = np.random.default_rng(1)

    def connect(self):
        time.sleep(self.connect_delay)
        return self.connect_ok

    def read(self):
        if self.stall_after is not None and self.idx >= self.stall_after:
            self._gate.wait()                       # a camera that stops answering
            return False, None
        now = time.perf_counter()
        self.t0 = self.t0 or now
        due = self.t0 + self.idx / self.fps
        if due > now:
            time.sleep(due - now)
        page = min(int(self.idx / (self.fps * self.page_s)), self.pages - 1)
        frame = np.clip(blocky(100 + page).astype(np.int16) + self.rng.integers(-2, 3, (360, 640, 3)), 0, 255).astype(np.uint8)
        self.idx += 1
        return True, frame

    def release(self):
        self.released = True
        self._gate.set()


class FakeEngine(PrototypeOCREngine):
    fail_calls = ()                 # 1-based OCR call numbers that raise
    calls = 0

    def ocr(self, crop):
        type(self).calls += 1
        if type(self).calls in type(self).fail_calls:
            raise RuntimeError("injected OCR failure")
        time.sleep(0.05)
        text = f"Quarterly report page total {int(crop.mean())} units"
        return {"raw_text": text, "detections": [{"bbox": [0, 0, 1, 1], "text": text, "confidence": 0.9}]}, 50.0, 1.0


@pytest.fixture(autouse=True)
def fast_cooldown(monkeypatch):
    monkeypatch.setattr(session_mod, "OCR_FOCUS_COOLDOWN_SEC", 0.2)
    FakeEngine.calls, FakeEngine.fail_calls = 0, ()
    yield


def wait_for(cond, timeout=20.0, what="condition"):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def make_session(tmp_path, stream, **kw):
    finished = []
    s = ScanSession("scan-test", stream, {"kind": "demo", "label": "test"}, tmp_path / "scan-test",
                    engine_factory=FakeEngine, on_finished=finished.append, **kw)
    s.finished_calls = finished
    return s


def run_scan(tmp_path, pages=3, **stream_kw):
    s = make_session(tmp_path, FakeStream(pages=pages, **stream_kw))
    s.start()
    wait_for(lambda: s.snapshot()["capture"]["pages_captured"] >= pages, what=f"{pages} captured pages")
    s.stop_scanning()
    wait_for(lambda: s.phase in ("finished", "failed"), what="finalization")
    return s


# ---------------------------------------------------------------- session
def test_capture_rules_are_the_production_values():
    # The B0 rule is the frozen production rule: 3 s cooldown after a page, then any frame whose 32x32 fingerprint differs by
    # more than 5.0. (The workspace version of this test compares these with the legacy Tk app, which is not part of this
    # repository.) PRODUCTION_COOLDOWN_S is read before the fixture shortens the live value.
    assert session_mod.DUPLICATE_DISTANCE_THRESHOLD == 5.0
    assert PRODUCTION_COOLDOWN_S == 3.0


def test_scan_runs_to_a_finalized_document(tmp_path):
    s = run_scan(tmp_path)
    assert s.phase == "finished" and s.finalized and not s.empty
    doc = json.loads(s.snapshot_json())
    assert [p["page_number"] for p in doc["pages"]] == [1, 2, 3]
    assert doc["processing"] == {"status": "complete", "pages_ok": 3, "pages_failed": []}
    export = tmp_path / "scan-test" / "export"
    assert {p.name for p in export.glob("*.json")} == {"raw_output.json", "corrected_output.json", "metadata.json"}
    assert json.loads((export / "metadata.json").read_text())["schema_version"] == 2
    assert (tmp_path / "scan-test" / "pages" / "page_1.jpg").stat().st_size > 0
    meta = json.loads((tmp_path / "scan-test" / "meta.json").read_text())
    assert meta["page_count"] == 3 and meta["mean_confidence"] == 0.9
    assert s.finished_calls == [s]
    assert s.llm_blocker() is None
    snap = s.snapshot()
    assert [st["state"] for st in snap["stages"]] == ["done"] * 4
    assert snap["ocr"]["completed"] == 3 and snap["ocr"]["queue_depth"] == 0 and snap["ocr"]["mean_confidence"] == 0.9


def test_the_model_is_blocked_until_the_document_is_finalized(tmp_path):
    s = make_session(tmp_path, FakeStream())
    s.start()
    wait_for(lambda: s.phase == "scanning", what="scanning")
    assert "not finished" in s.llm_blocker()
    s.stop_scanning()
    wait_for(lambda: s.phase == "finished", what="finalization")
    assert s.llm_blocker() is None


def test_stop_does_not_discard_queued_pages(tmp_path):
    """Pages still queued when Stop is pressed are all read (slow OCR, stop right after the last capture)."""
    class SlowEngine(FakeEngine):
        def ocr(self, crop):
            time.sleep(0.4)
            return super().ocr(crop)
    s = ScanSession("scan-slow", FakeStream(pages=3, page_s=0.6), {"kind": "demo", "label": "t"}, tmp_path / "slow",
                    engine_factory=SlowEngine)
    s.start()
    wait_for(lambda: s.snapshot()["capture"]["pages_captured"] >= 3, what="captures")
    s.stop_scanning()
    mid = s.snapshot()
    assert mid["phase"] == "finalizing" and mid["capture"]["accepting"] is False
    wait_for(lambda: s.phase == "finished", what="finalization")
    assert [p["page_number"] for p in json.loads(s.snapshot_json())["pages"]] == [1, 2, 3]


def test_a_failed_page_is_reported_and_numbering_follows_capture_order(tmp_path):
    FakeEngine.fail_calls = (2,)
    s = run_scan(tmp_path)
    doc = json.loads(s.snapshot_json())
    assert [p["page_number"] for p in doc["pages"]] == [1, 3]
    assert doc["processing"]["status"] == "complete_with_errors" and doc["processing"]["pages_failed"] == [2]
    snap = s.snapshot()
    assert [p["state"] for p in snap["pages"]] == ["done", "error", "done"]
    assert next(st for st in snap["stages"] if st["key"] == "read_pages")["state"] == "warn"


def test_two_scans_in_one_process_use_separate_engines(tmp_path):
    a = run_scan(tmp_path / "a")
    b = run_scan(tmp_path / "b")
    assert a.engine is not b.engine
    assert a.finalized and b.finalized
    assert len(json.loads(b.snapshot_json())["pages"]) == 3


def test_a_stalled_camera_cannot_freeze_finalization(tmp_path, monkeypatch):
    monkeypatch.setattr(ScanSession, "CAMERA_STOP_TIMEOUT_S", 0.6)
    s = make_session(tmp_path, FakeStream(pages=2, page_s=0.5, stall_after=30))
    s.start()
    wait_for(lambda: s.snapshot()["capture"]["pages_captured"] >= 2, what="captures")
    time.sleep(0.4)                                     # the camera is now silent
    s.stop_scanning()
    wait_for(lambda: s.phase == "finished", timeout=15, what="finalization despite the stalled camera")
    stop = next(st for st in s.snapshot()["stages"] if st["key"] == "stop_capture")
    assert stop["state"] == "warn" and "did not respond" in stop["detail"]
    assert len(json.loads(s.snapshot_json())["pages"]) == 2


def test_connect_failure_and_cancel(tmp_path):
    s = make_session(tmp_path, FakeStream(connect_ok=False))
    s.start()
    wait_for(lambda: s.phase == "failed", what="failure")
    assert "Could not connect" in s.error and not s.active()
    c = make_session(tmp_path / "c", FakeStream(connect_delay=0.5))
    c.start()
    assert c.cancel() and c.phase == "cancelled"
    time.sleep(0.8)
    assert c.phase == "cancelled" and c.stream.released


def test_an_empty_scan_creates_no_document(tmp_path):
    s = make_session(tmp_path, FakeStream(pages=1))
    s._accepting = False                                # the user stops before any page is captured
    s.start()
    wait_for(lambda: s.phase == "scanning", what="scanning")
    s.stop_scanning()
    wait_for(lambda: s.phase == "finished", what="finalization")
    assert s.empty and s.snapshot_json() is None and s.snapshot()["document_id"] is None


def test_c3_is_still_selectable_and_b0_is_the_default(tmp_path):
    assert ScanSession("x", FakeStream(), {"kind": "demo", "label": "t"}, tmp_path).selector is None   # B0
    assert ScanSession("x", FakeStream(), {"kind": "demo", "label": "t"}, tmp_path, selector_mode="c3").selector is not None


# ---------------------------------------------------------------- DroidCam watermark, blank views, timed capture
def test_the_droidcam_watermark_is_recognised_in_its_garbled_forms_and_ordinary_text_is_kept():
    from webapp.watermark import is_watermark
    for seen in ("using droidcam app", "using droiccam.app", "ueingdroidcam.app", "droidcam.app", "usEringdraidcamap",
                 "usig roicamoth.ir", "using droidcam.app"):
        assert is_watermark(seen), seen
    for text in ("Dr. Narayan G. Hegde*", "Controlling Our", "Super-Ego", "Quarterly Sales Report (units)", "using the app",
                 "Invoice 4821 total 129.50", "Bhavan's Journal, 1-15 September, 2026 37", "application", "campaign",
                 "a long sentence that only mentions droidcam in passing and is real text of the page"):
        assert not is_watermark(text), text


def test_a_watermark_fused_onto_real_text_loses_only_the_watermark():
    from webapp.watermark import split_detections, strip_edges
    # real RapidOCR output for a frame with the watermark in the same band as the first text line
    assert strip_edges("using droidcaControlling Our Super-Ego by Dr. Narayan G. Hegde") == \
        "Controlling Our Super-Ego by Dr. Narayan G. Hegde"
    assert strip_edges("Controlling Our Super-Ego using droidcam.app") == "Controlling Our Super-Ego"
    assert strip_edges("usEringdraidcamap shared and resonant journey where heard") == "shared and resonant journey where heard"
    for text in ("Using the app is easy", "Dr. Narayan G. Hegde", "Quarterly total 4821 units paid on 14 March"):
        assert strip_edges(text) == text
    kept, removed = split_detections([{"text": "using droidcam app", "bbox": [0, 0, 1, 1], "confidence": .9},
                                      {"text": "using droidcaHello world again", "bbox": [0, 5, 1, 6], "confidence": .8},
                                      {"text": "plain line", "bbox": [0, 9, 1, 10], "confidence": .9}])
    assert [d["text"] for d in kept] == ["Hello world again", "plain line"] and len(removed) == 2


class WatermarkEngine(FakeEngine):
    """Every page carries the watermark box; the 2nd page carries nothing else."""
    def ocr(self, crop):
        type(self).calls += 1
        n = type(self).calls
        wm = {"bbox": [0, 0, 160, 20], "text": "using droidcam app", "confidence": 0.9}
        body = {"bbox": [0, 60, 400, 80], "text": f"Quarterly report page {n}", "confidence": 0.9}
        dets = [wm] if n == 2 else [wm, body]
        return {"raw_text": " ".join(d["text"] for d in dets), "detections": dets}, 50.0, 1.0


def test_watermark_is_dropped_before_layout_and_blank_views_are_left_out(tmp_path):
    s = ScanSession("scan-wm", FakeStream(pages=3), {"kind": "demo", "label": "t"}, tmp_path / "wm",
                    engine_factory=WatermarkEngine)
    s.start()
    wait_for(lambda: s.snapshot()["capture"]["pages_captured"] >= 3, what="captures")
    s.stop_scanning()
    wait_for(lambda: s.phase == "finished", what="finalization")
    doc = json.loads(s.snapshot_json())
    assert [p["page_number"] for p in doc["pages"]] == [1, 3]                    # the watermark-only view is not a page
    assert all("droidcam" not in p["raw_text"].lower() for p in doc["pages"])
    assert doc["pages"][0]["raw_text"] == "Quarterly report page 1"
    assert doc["processing"] == {"status": "complete", "pages_ok": 2, "pages_failed": []}
    snap = s.snapshot()
    assert [p["blank"] for p in snap["pages"]] == [False, True, False] and snap["capture"]["watermark_ignored"] == 3
    meta = json.loads((tmp_path / "wm" / "meta.json").read_text())
    assert meta["blank_pages"] == [2] and meta["page_count"] == 2
    assert "droidcam" not in json.dumps(json.loads((tmp_path / "wm" / "export" / "raw_output.json").read_text())).lower()


def test_timed_capture_takes_a_page_every_interval_without_change_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "MIN_INTERVAL_S", 0.3)
    s = ScanSession("scan-timer", FakeStream(pages=1, page_s=60), {"kind": "demo", "label": "t"}, tmp_path / "t",
                    selector_mode="interval", interval_s=0.4, engine_factory=FakeEngine)
    s.start()
    wait_for(lambda: s.phase == "scanning", what="scanning")
    assert s.snapshot()["capture"]["next_capture_at"] is not None
    wait_for(lambda: s.snapshot()["capture"]["pages_captured"] >= 4, what="four timed captures of one unchanged view")
    s.stop_scanning()
    wait_for(lambda: s.phase == "finished", what="finalization")
    snap = s.snapshot()
    assert {p["reason"] for p in snap["pages"]} == {"timer"} and s.selector is None and snap["capture"]["mode"] == "interval"
    assert len(json.loads(s.snapshot_json())["pages"]) >= 4                      # no deduplication in timed mode


def test_timed_capture_sends_the_sharpest_frame_of_the_last_half_second(tmp_path):
    s = ScanSession("x", FakeStream(), {"kind": "demo", "label": "t"}, tmp_path, selector_mode="interval", interval_s=5)
    sent = []
    s._submit = lambda frame, fp, score, reason, ms: sent.append((frame, score, reason)) or True
    sharp = blocky(1)
    blurry = cv2.GaussianBlur(sharp, (0, 0), 6)
    s._next_capture_at = time.perf_counter() + 0.3                              # inside the pick window
    s._interval_step(blurry)
    s._interval_step(sharp)
    s._interval_step(blurry)
    assert sent == []                                                           # not due yet
    time.sleep(0.35)
    s._interval_step(blurry)                                                    # the tick arrives with a blurry frame
    assert len(sent) == 1 and sent[0][0] is sharp and sent[0][2] == "timer"


def test_unknown_capture_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        ScanSession("x", FakeStream(), {"kind": "demo", "label": "t"}, tmp_path, selector_mode="nope")


# ---------------------------------------------------------------- app core: gating, preparation, questions
class ScriptedRuntime:
    def __init__(self):
        self.model_calls, self.gate, self.aborts = [], None, 0

    def list_models(self):
        return ["gemma3:4b"]

    def abort_all(self):                                   # what CancellableOllamaRuntime does to a request in flight
        self.aborts += 1
        if self.in_flight:                                 # like the real runtime: only requests already running
            self._aborted = True
            if self.gate is not None:
                self.gate.set()

    _aborted = False
    in_flight = False

    def chat(self, system, user, schema, max_tokens):
        from llm_stage import LLMError
        stage = "recover" if "repair OCR" in system else "answer"
        self.model_calls.append(stage)
        if self.gate is not None and stage == "answer":
            self.in_flight = True
            self.gate.wait(5)
            self.in_flight = False
            if self._aborted:
                self._aborted = False
                raise LLMError("cancelled", "The assistant was stopped.")
        if stage == "recover":
            body = {"corrected_text": user.split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0], "uncertain_spans": []}
        else:
            body = {"answerable_from_document": True, "answer": "It is a quarterly report.", "general_knowledge": "",
                    "evidence": [{"page": 1, "quote": "Quarterly report page total"}]}
        return json.dumps(body), {"prompt_tokens": 1, "output_tokens": 1, "output_s": 1, "prompt_s": 1, "total_s": 1,
                                  "load_s": 0, "done_reason": "stop"}


@pytest.fixture
def web(tmp_path):
    runtime = ScriptedRuntime()
    assistant = DocumentAssistant(LLMConfig(), runtime)
    return WebApp(library_root=tmp_path / "lib", assistant=assistant, engine_factory=FakeEngine, warm_ocr=False,
                  health_thread=False), runtime


def test_the_model_never_runs_during_a_scan_and_questions_open_after_preparation(web):
    app, runtime = web
    app.start_scan(FakeStream(pages=2), {"kind": "demo", "label": "t"})
    with pytest.raises(Refused):
        app.start_scan(FakeStream(), {"kind": "demo", "label": "t"})           # one scan at a time
    wait_for(lambda: app.scan.snapshot()["capture"]["pages_captured"] >= 2, what="captures")
    with pytest.raises(Refused):
        app.ask(app.scan.id, "answer", "anything?")                          # scan active -> refused
    assert runtime.model_calls == []                                           # no model traffic while capturing
    app.stop_scan()
    sid = app.scan.id
    wait_for(lambda: (app.prepare_state(sid) or {}).get("status") == "ready", what="context preparation")
    assert runtime.model_calls and set(runtime.model_calls) == {"recover"}      # prepared only after finalization
    assert app.scan.finalized and app.scan.llm_blocker() is None
    turn = app.ask(sid, "answer", "What is this?")
    assert turn["result"]["answer"] == "It is a quarterly report." and turn["result"]["source"] == "document"
    assert runtime.model_calls.count("recover") == 2                           # the answer used the cached recoveries
    chat = json.loads((app.library.root / sid / "chat.json").read_text())
    assert chat["turns"][0]["question"] == "What is this?"
    assert app.library.list()[0]["turn_count"] == 1


def test_only_one_model_request_runs_at_a_time(web):
    app, runtime = web
    app.start_scan(FakeStream(pages=1), {"kind": "demo", "label": "t"})
    wait_for(lambda: app.scan.snapshot()["capture"]["pages_captured"] >= 1, what="capture")
    app.stop_scan()
    sid = app.scan.id
    wait_for(lambda: (app.prepare_state(sid) or {}).get("status") == "ready", what="preparation")
    runtime.gate = threading.Event()
    result = {}
    t = threading.Thread(target=lambda: result.update(turn=app.ask(sid, "answer", "q1")))
    t.start()
    wait_for(lambda: app.snapshot()["busy"] is not None, what="request in flight")
    with pytest.raises(Busy):
        app.ask(sid, "answer", "q2")
    runtime.gate.set()
    t.join(10)
    assert result["turn"]["question"] == "q1" and app.snapshot()["busy"] is None
    assert app.start_scan(FakeStream(pages=1), {"kind": "demo", "label": "t"}) is app.scan   # allowed once idle


def test_a_new_scan_hard_stops_a_running_question(web):
    """A scan always wins: the question in flight is aborted, nothing is saved for it, and the scan starts."""
    app, runtime = web
    app.start_scan(FakeStream(pages=1), {"kind": "demo", "label": "t"})
    wait_for(lambda: app.scan.snapshot()["capture"]["pages_captured"] >= 1, what="capture")
    app.stop_scan()
    sid = app.scan.id
    wait_for(lambda: (app.prepare_state(sid) or {}).get("status") == "ready", what="preparation")
    runtime.gate = threading.Event()
    outcome = {}

    def ask():
        try:
            outcome["turn"] = app.ask(sid, "answer", "a slow question")
        except Refused as exc:
            outcome["refused"] = str(exc)
    t = threading.Thread(target=ask)
    t.start()
    wait_for(lambda: app.snapshot()["busy"] is not None, what="request in flight")
    app.dismiss_scan()
    started = time.time()
    new_scan = app.start_scan(FakeStream(pages=1), {"kind": "demo", "label": "t"})
    t.join(10)
    assert time.time() - started < 5 and runtime.aborts >= 1
    assert "stopped because a new scan" in outcome["refused"] and "turn" not in outcome
    assert app.snapshot()["busy"] is None and app.scan is new_scan and new_scan.active()
    assert json.loads((app.library.root / sid / "chat.json").read_text())["turns"] == [] if (app.library.root / sid / "chat.json").exists() else True


def test_a_new_scan_cancels_the_context_preparation_and_it_can_be_restarted(web):
    app, runtime = web
    runtime.gate = threading.Event()
    app.start_scan(FakeStream(pages=1), {"kind": "demo", "label": "t"})
    wait_for(lambda: app.scan.snapshot()["capture"]["pages_captured"] >= 1, what="capture")
    app.stop_scan()
    sid = app.scan.id
    wait_for(lambda: (app.prepare_state(sid) or {}).get("status") in ("ready", "running"), what="preparation started")
    app.dismiss_scan()
    app.start_scan(FakeStream(pages=1), {"kind": "demo", "label": "t"})
    assert (app.prepare_state(sid) or {}).get("status") in ("ready", "cancelled")   # never left running
    assert not app._llm_lock.locked()


def test_without_recovery_the_preparation_reads_the_whole_document_into_the_model_once(tmp_path):
    runtime = ScriptedRuntime()
    cfg = LLMConfig()
    cfg.recover = False
    app = WebApp(library_root=tmp_path / "lib", assistant=DocumentAssistant(cfg, runtime), engine_factory=FakeEngine,
                 warm_ocr=False, health_thread=False)
    prompts = []
    real_chat = runtime.chat
    runtime.chat = lambda system, user, schema, max_tokens: (prompts.append((user, max_tokens)), real_chat(system, user, schema, max_tokens))[1]
    s = app.start_scan(FakeStream(pages=2), {"kind": "demo", "label": "t"})
    wait_for(lambda: s.snapshot()["capture"]["pages_captured"] >= 2, what="captures")
    app.stop_scan()
    wait_for(lambda: (app.prepare_state(s.id) or {}).get("status") == "ready", what="preparation")
    st = app.prepare_state(s.id)
    assert st["done"] == st["total"] == 2
    assert len(prompts) == 1 and prompts[0][1] == 1                              # one single-token request, no recovery
    assert prompts[0][0].startswith("DOCUMENT:\n=== PAGE 1 ===") and prompts[0][0].endswith("USER REQUEST: ")
    turn = app.ask(s.id, "answer", "What is this?")
    assert prompts[1][0].startswith(prompts[0][0])                               # the question shares the primed prefix
    assert turn["result"]["answer"]


# ---------------------------------------------------------------- deleting scans
def test_deleting_a_scan_removes_only_that_scan(web):
    app, _runtime = web
    ids = []
    for _ in range(2):
        s = app.start_scan(FakeStream(pages=1), {"kind": "demo", "label": "t"})
        wait_for(lambda: s.snapshot()["capture"]["pages_captured"] >= 1, what="capture")
        app.stop_scan()
        wait_for(lambda: s.phase == "finished" and (app.prepare_state(s.id) or {}).get("status") == "ready", what="ready")
        ids.append(s.id)
        app.dismiss_scan()
        time.sleep(1.1)                                   # scan ids are timestamped to the second
    app.ask(ids[0], "answer", "q")
    app.delete_document(ids[0])
    assert not (app.library.root / ids[0]).exists() and (app.library.root / ids[1] / "snapshot.json").exists()
    assert [d["id"] for d in app.library.list()] == [ids[1]] and app.prepare_state(ids[0]) is None
    with pytest.raises(Refused):
        app.delete_document(ids[0])                       # already gone
    with pytest.raises(Refused):
        app.delete_document("..")                         # never anything outside the library
    other = app.library.root.parent / "keep.txt"
    other.write_text("x")
    with pytest.raises(Refused):
        app.delete_document("../keep.txt")
    assert other.exists()
    s = app.start_scan(FakeStream(pages=1), {"kind": "demo", "label": "t"})
    with pytest.raises(Refused):
        app.delete_document(ids[1])                       # not while a scan is running
    assert (app.library.root / ids[1]).exists()


# ---------------------------------------------------------------- abortable Ollama transport
def test_abort_all_stops_a_hanging_request_at_once_and_normal_requests_still_work():
    import http.server
    from llm_stage import LLMConfig as Cfg, LLMError
    from webapp.llm_runtime import CancellableOllamaRuntime
    release = threading.Event()

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/api/hang":
                release.wait(20)
                return
            body = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = Cfg(base_url=f"http://127.0.0.1:{server.server_address[1]}")
        rt = CancellableOllamaRuntime(cfg)
        assert rt._post("/api/ok", {}, 5) == {"ok": True}
        err = {}

        def call():
            try:
                rt._post("/api/hang", {}, 30)
            except LLMError as exc:
                err["code"] = exc.code
        t = threading.Thread(target=call)
        t.start()
        time.sleep(0.4)
        started = time.time()
        assert rt.abort_all() == 1
        t.join(5)
        assert err == {"code": "cancelled"} and time.time() - started < 3
        assert rt._post("/api/ok", {}, 5) == {"ok": True}                         # not poisoned for later requests
    finally:
        release.set()
        server.shutdown()


def test_unavailable_model_is_reported_without_losing_the_document(tmp_path):
    class Down(ScriptedRuntime):
        def list_models(self):
            from llm_stage import LLMError
            raise LLMError("runtime_unavailable", "Cannot reach the local LLM runtime.")
    app = WebApp(library_root=tmp_path / "lib", assistant=DocumentAssistant(LLMConfig(), Down()),
                 engine_factory=FakeEngine, warm_ocr=False, health_thread=False)
    assert app.llm["status"] == "unavailable"
    app.start_scan(FakeStream(pages=1), {"kind": "demo", "label": "t"})
    wait_for(lambda: app.scan.snapshot()["capture"]["pages_captured"] >= 1, what="capture")
    app.stop_scan()
    sid = app.scan.id
    wait_for(lambda: (app.prepare_state(sid) or {}).get("status") == "unavailable", what="unavailable status")
    doc = app.library.get(sid)
    assert doc is not None and doc.detail()["pages"][0]["raw_text"].startswith("Quarterly report")


# ---------------------------------------------------------------- HTTP surface
def test_http_surface(web):
    from aiohttp.test_utils import TestClient, TestServer
    from webapp.server import make_app
    app, _runtime = web

    async def go():
        async with TestClient(TestServer(make_app(app))) as client:
            r = await client.get("/api/state")
            assert r.status == 200 and (await r.json())["scan"] is None
            r = await client.get("/")
            assert r.status == 200 and "MAVIS" in await r.text()
            r = await client.get("/api/state", headers={"Host": "evil.example"})
            assert r.status == 403                                              # DNS-rebinding guard
            r = await client.post("/api/scan/stop", json={}, headers={"Origin": "https://evil.example"})
            assert r.status == 403                                              # cross-site POST
            r = await client.post("/api/scan/stop", json={})
            assert r.status == 409 and (await r.json())["error"]["code"] == "refused"
            r = await client.post("/api/scan/start", json={"kind": "droidcam", "ip": "", "port": "4747"})
            assert r.status == 400
            r = await client.post("/api/scan/start", json={"kind": "droidcam", "ip": "1.2.3.4", "port": "99999"})
            assert r.status == 400
            r = await client.post("/api/scan/start", json={"kind": "demo", "selector": "interval", "interval_s": 1})
            assert r.status == 400 and "between 2 and 30" in (await r.json())["error"]["message"]
            r = await client.post("/api/scan/start", json={"kind": "demo", "selector": "interval", "interval_s": "x"})
            assert r.status == 400
            r = await client.post("/api/scan/start", json={"kind": "demo", "selector": "bogus"})
            assert r.status == 400
            r = await client.post("/api/documents/nope/ask", json={"mode": "answer", "question": "x"})
            assert r.status == 409
            r = await client.get("/api/documents/..%5Csecret")
            assert r.status in (404, 400)
            r = await client.get("/api/events")
            assert r.status == 200 and r.headers["Content-Type"].startswith("text/event-stream")
            first = await asyncio.wait_for(r.content.readline(), 5)
            assert first.startswith(b"data: ")
            r.close()
    asyncio.run(go())


def test_the_local_model_is_capped_to_a_thread_count_on_every_request():
    """So the model cannot use every core and freeze the browser; the same value everywhere avoids model reloads."""
    import http.server
    from llm_stage import LLMConfig as Cfg
    from webapp.llm_runtime import DEFAULT_THREADS, CancellableOllamaRuntime
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            seen.append((self.path, json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))))
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = Cfg(base_url=f"http://127.0.0.1:{server.server_address[1]}")
        rt = CancellableOllamaRuntime(cfg)
        assert rt.threads == DEFAULT_THREADS == 8
        rt._post("/api/chat", {"model": "m", "options": {"temperature": 0}}, 5)
        rt._post("/api/generate", {"model": "m", "keep_alive": "10m"}, 5)
        rt._post("/api/other", {"model": "m"}, 5)
        assert seen[0][1]["options"] == {"temperature": 0, "num_thread": 8}
        assert seen[1][1]["options"] == {"num_thread": 8} and "options" not in seen[2][1]
        off = CancellableOllamaRuntime(cfg, threads=0)
        off._post("/api/chat", {"model": "m"}, 5)
        assert "options" not in seen[3][1]                                    # 0 = Ollama's default
    finally:
        server.shutdown()
