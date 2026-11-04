"""Gemini as the page reader, local OCR only when offline. Tested against a local mock of the API; no request leaves this machine."""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "prototype"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import webapp.gemini_reader as gr  # noqa: E402
from test_webapp import FakeEngine, FakeStream, fast_cooldown, wait_for  # noqa: E402,F401  (fast_cooldown: autouse)
from webapp.core import Refused, WebApp  # noqa: E402
from webapp.gemini_reader import (GeminiReader, ReaderSettings, ReaderUnavailable, RemoteFirstEngine,  # noqa: E402
                                  make_engine_factory)
from webapp.session import ScanSession  # noqa: E402

KEY = "TEST-SECRET-KEY-12345"
CLOSED = "http://127.0.0.1:9"            # nothing listens here: connecting fails at once (the computer is "offline")


class Mock:
    """Fake Gemini endpoint. `behaviour` is a function(request_index) -> (status, body, delay_s, headers)."""

    def __init__(self, behaviour):
        self.requests, self.behaviour, self.raw = [], behaviour, []
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _answer(self, body):
                mock.requests.append({"method": self.command, "path": self.path, "headers": dict(self.headers), "body": body,
                                      "t": time.time()})
                status, payload, delay, headers = mock.behaviour(len(mock.requests))
                time.sleep(delay)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                mock.raw.append({"path": self.path, "headers": dict(self.headers), "raw_body": raw})
                try:
                    body = json.loads(raw)
                except ValueError:                                  # e.g. OCR.space's multipart/form-data
                    body = {"_non_json": True}
                self._answer(body)

            def do_GET(self):
                self._answer(None)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()


def ok(text):
    return 200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}, 0, None


def settings(url, **extra):
    env = {"GEMINI_API_KEY": KEY, "MAVIS_GEMINI_URL": url, "MAVIS_GEMINI_RPM": "600", **extra}
    return ReaderSettings(env)


FRAME = np.full((360, 640, 3), 200, dtype=np.uint8)


@pytest.fixture(autouse=True)
def quick_backoff(monkeypatch):
    monkeypatch.setattr(gr, "SERVER_ERROR_RETRY_S", 0.05)
    monkeypatch.setattr(gr, "OFFLINE_REPROBE_S", 0.3)
    FakeEngine.calls = 0                                   # counts pages read by the (fake) local engine


@pytest.fixture
def mock_ok():
    m = Mock(lambda i: ok("Hello world\nsecond line"))
    yield m
    m.close()


# ---------------------------------------------------------------- the reader itself
def test_the_reader_sends_the_image_with_the_key_in_a_header_only(mock_ok):
    text = GeminiReader(settings(mock_ok.url)).read(FRAME)
    assert text == "Hello world\nsecond line"
    req = mock_ok.requests[0]
    assert req["headers"]["x-goog-api-key"] == KEY and KEY not in req["path"]
    assert req["path"] == "/v1beta/models/gemini-2.5-flash:generateContent"
    parts = req["body"]["contents"][0]["parts"]
    assert "Transcribe ALL the text" in parts[0]["text"] and parts[1]["inline_data"]["mime_type"] == "image/jpeg"
    assert req["body"]["generationConfig"]["temperature"] == 0


@pytest.mark.parametrize("status,payload,headers,reason,retry,retryable", [
    (429, {}, {"Retry-After": "7"}, "rate limit", 7.0, True),
    (429, {}, None, "rate limit", 30.0, True),
    (429, {"error": {"details": [{"retryDelay": "34s"}]}}, None, "rate limit", 34.0, True),
    (429, {"error": {"message": "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}, None, "daily quota", None, False),
    (403, {}, None, "key was rejected", None, False),
    (400, {}, None, "HTTP 400", None, False),
    (500, {}, None, "HTTP 500", 0.05, True),
    (200, {"promptFeedback": {"blockReason": "SAFETY"}}, None, "blocked", None, False),
    (200, {"candidates": []}, None, "no answer", None, True),
])
def test_every_failure_is_classified_and_never_leaks_the_key(status, payload, headers, reason, retry, retryable):
    m = Mock(lambda i: (status, payload, 0, headers))
    try:
        with pytest.raises(ReaderUnavailable) as err:
            GeminiReader(settings(m.url)).read(FRAME)
    finally:
        m.close()
    e = err.value
    assert reason in e.reason and e.retry_after == retry and e.retryable is retryable and not e.network and KEY not in e.reason


def test_a_connection_failure_is_a_network_error_without_the_key():
    with pytest.raises(ReaderUnavailable) as err:
        GeminiReader(settings(CLOSED)).read(FRAME)
    assert err.value.network and err.value.retryable and "network error" in err.value.reason and KEY not in err.value.reason


def test_the_key_and_model_check_sends_no_page_and_reports_each_situation(mock_ok):
    assert GeminiReader(settings(mock_ok.url)).check() == ("ok", "")
    assert mock_ok.requests[0]["method"] == "GET" and mock_ok.requests[0]["path"] == "/v1beta/models/gemini-2.5-flash"
    assert mock_ok.requests[0]["body"] is None and mock_ok.requests[0]["headers"]["x-goog-api-key"] == KEY
    for status, expected in ((403, "rejected"), (400, "rejected"), (404, "model_missing"), (503, "unreachable")):
        m = Mock(lambda i, s=status: (s, {}, 0, None))
        try:
            assert GeminiReader(settings(m.url)).check()[0] == expected
        finally:
            m.close()
    assert GeminiReader(settings(CLOSED)).check()[0] == "offline"
    assert GeminiReader(ReaderSettings({})).check()[0] == "missing_key"


def test_settings_never_expose_the_key_and_gemini_is_the_default_provider():
    s = ReaderSettings({"GEMINI_API_KEY": KEY})
    assert s.has_key and not s.local_only and KEY not in json.dumps(s.public())
    pub = s.public()
    assert (pub["provider"], pub["model"], pub["has_key"], pub["remote"], pub["label"]) ==         ("gemini", "gemini-2.5-flash", True, True, "Google (Gemini)")
    assert ReaderSettings({"GEMINI_API_KEY": KEY, "MAVIS_READER": "local"}).local_only        # development override
    local = object()
    assert make_engine_factory(ReaderSettings({"MAVIS_READER": "local"}), local) is local
    assert make_engine_factory(s, local) is not local


# ---------------------------------------------------------------- scans: Gemini first, local only when offline
def run_session(tmp_path, url, pages=3, parallel=3, mode="b0", **env):
    s = settings(url, MAVIS_GEMINI_PARALLEL=str(parallel), **env)
    sess = ScanSession("scan-gem", FakeStream(pages=pages), {"kind": "demo", "label": "t"}, tmp_path / "g",
                       engine_factory=make_engine_factory(s, FakeEngine), selector_mode=mode)
    sess.start()
    wait_for(lambda: sess.snapshot()["capture"]["pages_captured"] >= pages, what="captures")
    sess.stop_scanning()
    wait_for(lambda: sess.phase == "finished", timeout=40, what="finalization")
    return sess


def test_pages_are_read_by_gemini_and_the_local_engine_is_never_touched_while_online(tmp_path):
    m = Mock(lambda i: ok("using droidcam.app\nInvoice 4821\nusing droidcam app\nTotal 129.50"))
    try:
        sess = run_session(tmp_path, m.url)
    finally:
        m.close()
    doc = json.loads(sess.snapshot_json())
    assert [p["page_number"] for p in doc["pages"]] == [1, 2, 3]
    assert all(p["raw_text"] == "Invoice 4821\nTotal 129.50" for p in doc["pages"])          # watermark cleaned
    snap = sess.snapshot()
    assert {p["reader"] for p in snap["pages"]} == {"gemini"} and FakeEngine.calls == 0     # no local OCR at all
    assert snap["capture"]["watermark_ignored"] == 6 and snap["ocr"]["mean_confidence"] is None
    assert json.loads((tmp_path / "g" / "meta.json").read_text())["readers"] == ["gemini"]
    assert KEY not in json.dumps(snap) and KEY not in (tmp_path / "g" / "meta.json").read_text()
    assert len(m.requests) == 3 and all(r["headers"]["x-goog-api-key"] == KEY for r in m.requests)


def test_offline_pages_are_read_locally_and_gemini_is_not_hammered(tmp_path):
    sess = run_session(tmp_path, CLOSED)
    doc = json.loads(sess.snapshot_json())
    assert [p["page_number"] for p in doc["pages"]] == [1, 2, 3] and FakeEngine.calls == 3
    snap = sess.snapshot()
    assert {p["reader"] for p in snap["pages"]} == {"local"}
    warns = [e["msg"] for e in snap["activity"] if "was not used" in e["msg"]]
    assert len(warns) == 3 and all("no internet connection" in w and "local reader" in w for w in warns)
    assert json.loads((tmp_path / "g" / "meta.json").read_text())["readers"] == ["local"]


def test_gemini_is_used_again_as_soon_as_the_connection_is_back():
    m = Mock(lambda i: ok("from gemini"))
    online = {"up": False}
    try:
        engine = RemoteFirstEngine(GeminiReader(settings(m.url)), FakeEngine, parallel=1, rpm=600,
                                   probe=lambda: online["up"])
        engine._offline, engine._next_probe_at = True, 0.0                      # it had been offline
        results = []

        def read(n):
            r = engine._read_page(n, FRAME, {"queue_entered_at": time.perf_counter()})
            results.append((r[1], r[2]["reader"]))
        read(1)
        assert results[-1][1] == "local" and len(m.requests) == 0               # still offline: no request to Gemini
        online["up"] = True
        time.sleep(0.35)                                                         # past the re-probe interval
        read(2)
        assert results[-1] == ("from gemini", "gemini") and not engine.is_offline()
    finally:
        m.close()


def test_a_rate_limit_makes_pages_wait_and_then_read_them_instead_of_failing(tmp_path):
    """The free tier allows only a few requests a minute: pages wait for the quota (and the pace drops), nothing is lost."""
    m = Mock(lambda i: (429, {}, 0, {"Retry-After": "1"}) if i in (2, 3) else ok("Page text"))
    try:
        sess = run_session(tmp_path, m.url, pages=3, parallel=1)
    finally:
        m.close()
    snap = sess.snapshot()
    doc = json.loads(sess.snapshot_json())
    assert [p["page_number"] for p in doc["pages"]] == [1, 2, 3] and FakeEngine.calls == 0      # all read, none local
    assert all(p["state"] == "done" for p in snap["pages"])
    assert any("waited" in e["msg"] and "rate limit" in e["msg"] for e in snap["activity"])


def test_a_rate_limit_that_will_not_clear_in_time_fails_the_page_visibly_not_locally(tmp_path, monkeypatch):
    monkeypatch.setattr(gr, "MAX_RATE_WAIT_S", 2.0)
    m = Mock(lambda i: (429, {}, 0, {"Retry-After": "120"}))
    try:
        sess = run_session(tmp_path, m.url, pages=3, parallel=1)
    finally:
        m.close()
    snap = sess.snapshot()
    assert [p["state"] for p in snap["pages"]] == ["error"] * 3 and FakeEngine.calls == 0   # failed visibly, no local OCR
    assert "longer than MAVIS waits" in snap["pages"][2]["error"]
    assert len(m.requests) == 1                                                  # then it stopped asking for 120 s
    assert sess.empty and sess.snapshot_json() is None                           # nothing readable: no document is made


def test_the_pace_slows_down_after_a_rate_limit_and_the_pause_is_reported():
    m = Mock(lambda i: (429, {}, 0, {"Retry-After": "1"}) if i == 1 else ok("text"))
    try:
        engine = RemoteFirstEngine(GeminiReader(settings(m.url)), FakeEngine, parallel=1, rpm=600)
        before = engine._interval
        metrics = {"queue_entered_at": time.perf_counter()}
        r = engine._read_page(1, FRAME, metrics)
        assert r[1] == "text" and engine._interval == pytest.approx(before * 1.5) and metrics["rate_limit_wait_s"] > 0.5
        assert engine.rate_limited_until() is None                               # the pause has passed
    finally:
        m.close()


def test_thinking_is_switched_off_and_a_model_that_refuses_that_is_asked_without_it():
    m = Mock(lambda i: ok("text"))
    try:
        GeminiReader(settings(m.url)).read(FRAME)
        assert m.requests[0]["body"]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
        GeminiReader(settings(m.url, MAVIS_GEMINI_THINKING_BUDGET="-1")).read(FRAME)
        assert "thinkingConfig" not in m.requests[1]["body"]["generationConfig"]
    finally:
        m.close()
    m = Mock(lambda i: (400, {"error": {"message": "This model only works in thinking mode."}}, 0, None) if i == 1 else ok("text"))
    try:
        reader = GeminiReader(settings(m.url))
        assert reader.read(FRAME) == "text" and len(m.requests) == 2
        assert "thinkingConfig" not in m.requests[1]["body"]["generationConfig"]
        reader.read(FRAME)
        assert "thinkingConfig" not in m.requests[2]["body"]["generationConfig"]   # remembered: not sent again
    finally:
        m.close()


def test_a_page_that_gemini_fails_to_read_is_reported_failed_and_the_rest_are_kept(tmp_path):
    m = Mock(lambda i: (400, {}, 0, None) if i == 2 else ok("Page text"))
    try:
        sess = run_session(tmp_path, m.url, pages=3, parallel=1)
    finally:
        m.close()
    doc = json.loads(sess.snapshot_json())
    assert [p["page_number"] for p in doc["pages"]] == [1, 3]
    assert doc["processing"] == {"status": "complete_with_errors", "pages_ok": 2, "pages_failed": [2]}
    assert "HTTP 400" in sess.snapshot()["pages"][1]["error"] and FakeEngine.calls == 0


def test_transient_errors_are_retried_then_succeed_and_a_bad_key_is_not_retried():
    m = Mock(lambda i: (500, {}, 0, None) if i < 3 else ok("finally"))
    try:
        engine = RemoteFirstEngine(GeminiReader(settings(m.url)), FakeEngine, parallel=1, rpm=600, retries=2)
        r = engine._read_page(1, FRAME, {"queue_entered_at": time.perf_counter()})
        assert r[1] == "finally" and r[2]["attempts"] == 3 and len(m.requests) == 3 and FakeEngine.calls == 0
    finally:
        m.close()
    m = Mock(lambda i: (403, {}, 0, None))
    try:
        engine = RemoteFirstEngine(GeminiReader(settings(m.url)), FakeEngine, parallel=1, rpm=600, retries=2)
        r = engine._read_page(1, FRAME, {"queue_entered_at": time.perf_counter()})
        assert r[4] is not None and "rejected" in str(r[4]) and len(m.requests) == 1 and FakeEngine.calls == 0
    finally:
        m.close()
    m = Mock(lambda i: (500, {}, 0, None))
    try:
        engine = RemoteFirstEngine(GeminiReader(settings(m.url)), FakeEngine, parallel=1, rpm=600, retries=1)
        r = engine._read_page(1, FRAME, {"queue_entered_at": time.perf_counter()})
        assert r[4] is not None and "HTTP 500" in str(r[4]) and len(m.requests) == 2    # one retry, then a visible failure
    finally:
        m.close()


def test_several_pages_are_read_at_the_same_time():
    m = Mock(lambda i: (200, ok("text")[1], 0.6, None))
    try:
        engine = RemoteFirstEngine(GeminiReader(settings(m.url)), FakeEngine, parallel=3, rpm=600)
        worker = threading.Thread(target=engine.worker_loop, daemon=True)
        worker.start()
        t0 = time.time()
        for n in (1, 2, 3):
            assert engine.submit(n, FRAME, {})
        engine.stop()
        engine.wait()
        worker.join(5)
        elapsed = time.time() - t0
        results = []
        while (r := engine.poll_result()) is not None:
            results.append(r)
    finally:
        m.close()
    assert sorted(r[0] for r in results) == [1, 2, 3] and all(r[4] is None and r[1] == "text" for r in results)
    assert elapsed < 1.5                                                              # sequential would take >= 1.8 s


def test_requests_are_spaced_to_the_configured_rate():
    m = Mock(lambda i: ok("text"))
    try:
        engine = RemoteFirstEngine(GeminiReader(settings(m.url)), FakeEngine, parallel=3, rpm=120)   # one per 0.5 s
        worker = threading.Thread(target=engine.worker_loop, daemon=True)
        worker.start()
        for n in (1, 2, 3):
            engine.submit(n, FRAME, {})
        engine.stop()
        engine.wait()
        worker.join(5)
    finally:
        m.close()
    starts = sorted(r["t"] for r in m.requests)
    assert starts[1] - starts[0] >= 0.4 and starts[2] - starts[1] >= 0.4


# ---------------------------------------------------------------- app level
def test_scanning_is_refused_without_a_key_and_the_local_reader_is_never_built(tmp_path):
    app = WebApp(library_root=tmp_path / "lib", reader_settings=ReaderSettings({}), health_thread=False)
    wait_for(lambda: app.snapshot()["reader"]["status"] == "missing_key", what="reader check")
    state = app.snapshot()["reader"]
    assert state["provider"] == "gemini" and state["has_key"] is False and "GEMINI_API_KEY" in state["message"]
    with pytest.raises(Refused) as err:
        app.start_scan(FakeStream(), {"kind": "demo", "label": "t"})
    assert "API key" in str(err.value) and "GEMINI_API_KEY" in str(err.value) and app.scan is None
    assert "ocr" not in app.snapshot()                                                 # no local engine to warm up


def test_scanning_is_refused_when_google_rejects_the_key_but_allowed_when_merely_offline(tmp_path):
    app = WebApp(library_root=tmp_path / "lib", reader_settings=settings(CLOSED), health_thread=False)
    wait_for(lambda: app.snapshot()["reader"]["status"] == "offline", what="offline status")
    assert app._reader_blocker() is None                                               # offline: the local reader takes over
    m = Mock(lambda i: (403, {}, 0, None))
    try:
        app2 = WebApp(library_root=tmp_path / "lib2", reader_settings=settings(m.url), health_thread=False)
        wait_for(lambda: app2.snapshot()["reader"]["status"] == "rejected", what="rejected status")
        with pytest.raises(Refused) as err:
            app2.start_scan(FakeStream(), {"kind": "demo", "label": "t"})
        assert "rejected" in str(err.value)
    finally:
        m.close()


def test_no_local_ocr_library_is_loaded_by_the_web_app_while_online():
    """Importing and running the web app must not import RapidOCR or onnxruntime (the local reader is built lazily, offline only)."""
    import subprocess
    code = ("import sys; sys.path.insert(0, r'%s'); import webapp.server, webapp.core; "
            "bad = [m for m in ('rapidocr', 'onnxruntime') if m in sys.modules]; print('LOADED', bad)" % REPO)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120).stdout
    assert "LOADED []" in out, out


# ---------------------------------------------------------------- failed pages stay visible and can be read again
def test_a_failed_page_is_listed_with_its_photo_and_reason_and_can_be_read_again(tmp_path):
    from llm_stage import DocumentAssistant, LLMConfig
    from test_webapp import ScriptedRuntime
    posts = {"n": 0, "fail_second": True}

    def behaviour(i):
        req = m.requests[-1]
        if req["body"] is None:                                    # the key / model check
            return 200, {}, 0, None
        posts["n"] += 1
        if posts["fail_second"] and posts["n"] == 2:
            return 400, {}, 0, None
        return ok(f"Quarterly report page {posts['n']}")
    m = Mock(behaviour)
    try:
        app = WebApp(library_root=tmp_path / "lib", reader_settings=settings(m.url, MAVIS_GEMINI_PARALLEL="1"),
                     assistant=DocumentAssistant(LLMConfig(), ScriptedRuntime()), health_thread=False)
        wait_for(lambda: app.snapshot()["reader"]["status"] == "ok", what="reader check")
        s = app.start_scan(FakeStream(pages=3), {"kind": "demo", "label": "t"})
        wait_for(lambda: s.snapshot()["capture"]["pages_captured"] >= 3, what="captures")
        app.stop_scan()
        wait_for(lambda: s.phase == "finished", timeout=30, what="finalization")
        wait_for(lambda: (app.prepare_state(s.id) or {}).get("status") == "ready", what="preparation")
        doc = app.library.get(s.id)
        detail = doc.detail()
        assert [(p["n"], p["status"], p["has_image"]) for p in detail["pages"]] == [(1, "ok", True), (2, "failed", True), (3, "ok", True)]
        assert "HTTP 400" in detail["pages"][1]["error"] and detail["page_count"] == 2 and detail["captured_count"] == 3
        document_id, sha = doc.meta["document_id"], doc.meta["snapshot_sha256"]

        with pytest.raises(Refused):
            app.reread_page(s.id, 1)                               # already read
        with pytest.raises(Refused):
            app.reread_page(s.id, 9)                               # not a page of this scan
        app.dismiss_scan()
        s2 = app.start_scan(FakeStream(pages=1), {"kind": "demo", "label": "t"})                # a new scan is running
        with pytest.raises(Refused) as err:
            app.reread_page(s.id, 2)
        assert "scan is running" in str(err.value)
        wait_for(lambda: s2.phase == "scanning", what="second scan")
        app.stop_scan()
        wait_for(lambda: s2.phase == "finished", timeout=30, what="second scan finished")
        posts["fail_second"] = False
        result = app.reread_page(s.id, 2)
        assert result == {"status": "ok", "page": 2}
        detail = doc.detail()
        assert [p["status"] for p in detail["pages"]] == ["ok", "ok", "ok"] and detail["page_count"] == 3
        assert detail["pages"][1]["raw_text"].startswith("Quarterly report page") and not detail["pages_failed"]
        assert doc.meta["document_id"] == document_id and doc.meta["snapshot_sha256"] != sha        # same document, new text
        saved = json.loads((doc.dir / "snapshot.json").read_text())
        assert [p["page_number"] for p in saved["pages"]] == [1, 2, 3] and saved["processing"]["pages_failed"] == []
        raw = json.loads((doc.dir / "export" / "raw_output.json").read_text())
        assert [p["page_number"] for p in raw["pages"]] == [1, 2, 3]
        wait_for(lambda: (app.prepare_state(s.id) or {}).get("status") == "ready", what="the model re-reads the document")
    finally:
        m.close()
