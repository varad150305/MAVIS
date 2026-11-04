"""OCR.space as the page reader (MAVIS_OCR_PROVIDER=ocrspace), tested against a local mock of its API. No request leaves this machine."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "prototype"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import webapp.gemini_reader as gr  # noqa: E402
from test_gemini_reader import CLOSED, FRAME, Mock, fast_cooldown, quick_backoff, run_session  # noqa: E402,F401
from test_webapp import FakeEngine, wait_for  # noqa: E402,F401
from webapp.core import WebApp  # noqa: E402
from webapp.gemini_reader import ReaderSettings, ReaderUnavailable, make_reader  # noqa: E402
from webapp.ocrspace_reader import MAX_UPLOAD_BYTES, OcrSpaceReader, encode_under_limit  # noqa: E402

KEY = "OCRSPACE-TEST-KEY-9876"


def ocr_ok(text):
    return 200, {"ParsedResults": [{"ParsedText": text, "FileParseExitCode": 1, "ErrorMessage": ""}], "OCRExitCode": 1,
                 "IsErroredOnProcessing": False}, 0, None


def errored(message, status=200):
    return status, {"IsErroredOnProcessing": True, "OCRExitCode": 99, "ErrorMessage": [message], "ParsedResults": None}, 0, None


def os_settings(url, **extra):
    return ReaderSettings({"MAVIS_OCR_PROVIDER": "ocrspace", "OCRSPACE_API_KEY": KEY, "MAVIS_OCRSPACE_URL": url,
                           "MAVIS_OCRSPACE_RPM": "600", **extra})


@pytest.fixture
def mock_ok():
    m = Mock(lambda i: ocr_ok("Hello world\r\nsecond line\r\n"))
    yield m
    m.close()


def test_the_request_follows_the_documented_api_and_keeps_the_key_in_the_header(mock_ok):
    text = OcrSpaceReader(os_settings(mock_ok.url)).read(FRAME)
    assert text == "Hello world\nsecond line"                              # CRLF lines normalised, edges trimmed
    sent = mock_ok.raw[0]
    assert sent["path"] == "/parse/image" and sent["headers"]["apikey"] == KEY
    assert sent["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")
    body = sent["raw_body"]
    assert KEY.encode() not in body and KEY not in sent["path"]
    assert b'name="OCREngine"\r\n\r\n2' in body and b'name="language"\r\n\r\neng' in body
    assert b'name="scale"\r\n\r\ntrue' in body and b'name="detectOrientation"\r\n\r\ntrue' in body
    assert b'name="isTable"\r\n\r\nfalse' in body and b'name="isOverlayRequired"\r\n\r\nfalse' in body
    assert b'name="file"; filename="page.jpg"' in body and b"\xff\xd8" in body            # a JPEG file part


def test_engine_3_language_auto_and_table_mode_are_configurable(mock_ok):
    OcrSpaceReader(os_settings(mock_ok.url, MAVIS_OCRSPACE_ENGINE="3", MAVIS_OCRSPACE_LANGUAGE="auto",
                               MAVIS_OCRSPACE_TABLE="1")).read(FRAME)
    body = mock_ok.raw[0]["raw_body"]
    assert b'name="OCREngine"\r\n\r\n3' in body and b'name="language"' not in body and b'name="isTable"\r\n\r\ntrue' in body


def test_a_page_with_no_text_is_an_empty_transcription_not_an_error():
    m = Mock(lambda i: ocr_ok(""))
    try:
        assert OcrSpaceReader(os_settings(m.url)).read(FRAME) == ""
    finally:
        m.close()


def test_a_big_photo_is_compressed_under_the_free_plans_1_mb_limit():
    rng = np.random.default_rng(1)
    noisy = rng.integers(0, 255, (3000, 4000, 3), dtype=np.uint8)                       # worst case: incompressible
    data = encode_under_limit(noisy)
    assert len(data) <= MAX_UPLOAD_BYTES < 1_000_000
    assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is not None
    small = encode_under_limit(FRAME)
    assert len(small) < 100_000


@pytest.mark.parametrize("status,reply,reason,retry,retryable,rate", [
    (200, errored("You may only perform this action upto maximum 10 number of times within 600 seconds")[1], "rate limit", 600.0, True, True),
    (429, {}, "rate limit", 60.0, True, True),
    (200, errored("Daily request limit reached for this key")[1], "daily request limit", None, False, False),
    (200, errored("Invalid API key provided")[1], "rejected the API key", None, False, False),
    (403, {}, "rejected the API key", None, False, False),
    (200, errored("File size exceeds the maximum limit of 1024 KB")[1], "larger than OCR.space allows", None, False, False),
    (200, {"ParsedResults": [{"FileParseExitCode": -20, "ErrorMessage": "Timed out waiting for results"}], "OCRExitCode": 2},
     "could not finish", 3.0, True, False),
    (500, {}, "HTTP 500", 5.0, True, False),
    (200, {"ParsedResults": []}, "no result", None, True, False),
])
def test_errors_are_classified_from_the_reply_and_never_leak_the_key(status, reply, reason, retry, retryable, rate):
    m = Mock(lambda i: (status, reply, 0, None))
    try:
        with pytest.raises(ReaderUnavailable) as err:
            OcrSpaceReader(os_settings(m.url)).read(FRAME)
    finally:
        m.close()
    e = err.value
    assert reason in e.reason and e.retry_after == retry and e.retryable is retryable and e.rate_limited is rate
    assert KEY not in e.reason and not e.network


def test_a_connection_failure_is_a_network_error():
    with pytest.raises(ReaderUnavailable) as err:
        OcrSpaceReader(os_settings(CLOSED)).read(FRAME)
    assert err.value.network and "network error" in err.value.reason and KEY not in err.value.reason


def test_the_check_never_spends_a_request_of_the_daily_quota(mock_ok):
    assert OcrSpaceReader(os_settings(mock_ok.url)).check()[0] == "ok"
    assert mock_ok.raw == []                                                              # nothing was sent
    assert OcrSpaceReader(os_settings(CLOSED)).check()[0] == "offline"
    assert OcrSpaceReader(ReaderSettings({"MAVIS_OCR_PROVIDER": "ocrspace"})).check() == ("missing_key", "OCRSPACE_API_KEY is not set.")


def test_provider_settings_and_selection():
    s = ReaderSettings({"MAVIS_OCR_PROVIDER": "ocrspace", "OCRSPACE_API_KEY": KEY, "GEMINI_API_KEY": "other"})
    assert (s.provider, s.label, s.key_env, s.ocr_engine, s.parallel, s.rpm) == ("ocrspace", "OCR.space", "OCRSPACE_API_KEY", "2", 2, 20.0)
    pub = s.public()
    assert KEY not in json.dumps(pub) and "other" not in json.dumps(pub)
    assert (pub["provider"], pub["label"], pub["model"], pub["remote"], pub["key_env"]) == ("ocrspace", "OCR.space", "engine 2", True, "OCRSPACE_API_KEY")
    assert "OCRSPACE_API_KEY" in pub["setup"] and "freekey" in pub["setup"]
    assert isinstance(make_reader(s), OcrSpaceReader)
    assert ReaderSettings({"GEMINI_API_KEY": "k"}).provider == "gemini"                  # Gemini stays the default
    with pytest.raises(ValueError):
        ReaderSettings({"MAVIS_OCR_PROVIDER": "nope"})
    assert not ReaderSettings({"MAVIS_OCR_PROVIDER": "ocrspace", "GEMINI_API_KEY": "k"}).has_key   # each provider uses its own key


def test_scans_are_read_by_ocrspace_with_the_same_cleanup_and_the_local_engine_untouched(tmp_path):
    m = Mock(lambda i: ocr_ok("using droidcam.app\r\nInvoice 4821\r\nTotal 129.50\r\n"))
    try:
        s = os_settings(m.url, MAVIS_OCRSPACE_PARALLEL="2")
        sess = run_session_with(tmp_path, s)
    finally:
        m.close()
    doc = json.loads(sess.snapshot_json())
    assert [p["page_number"] for p in doc["pages"]] == [1, 2, 3]
    assert all(p["raw_text"] == "Invoice 4821\nTotal 129.50" for p in doc["pages"])      # watermark removed
    snap = sess.snapshot()
    assert {p["reader"] for p in snap["pages"]} == {"ocrspace"} and FakeEngine.calls == 0
    assert any("read by OCR.space" in e["msg"] for e in snap["activity"])
    assert json.loads((tmp_path / "g" / "meta.json").read_text())["readers"] == ["ocrspace"]
    assert KEY not in json.dumps(snap) and KEY not in (tmp_path / "g" / "meta.json").read_text()
    assert len(m.raw) == 3 and all(r["headers"]["apikey"] == KEY for r in m.raw)


def run_session_with(tmp_path, settings, pages=3):
    from test_webapp import FakeStream
    from webapp.session import ScanSession
    sess = ScanSession("scan-os", FakeStream(pages=pages), {"kind": "demo", "label": "t"}, tmp_path / "g",
                       engine_factory=gr.make_engine_factory(settings, FakeEngine))
    sess.start()
    wait_for(lambda: sess.snapshot()["capture"]["pages_captured"] >= pages, what="captures")
    sess.stop_scanning()
    wait_for(lambda: sess.phase == "finished", timeout=40, what="finalization")
    return sess


def test_ocrspaces_rate_limit_message_makes_pages_wait_then_read(tmp_path):
    m = Mock(lambda i: errored("You may only perform this action upto maximum 10 number of times within 1 seconds")
             if i == 2 else ocr_ok("Page text"))
    try:
        sess = run_session_with(tmp_path, os_settings(m.url, MAVIS_OCRSPACE_PARALLEL="1"))
    finally:
        m.close()
    snap = sess.snapshot()
    assert [p["state"] for p in snap["pages"]] == ["done"] * 3 and FakeEngine.calls == 0
    assert any("waited" in e["msg"] and "OCR.space" in e["msg"] for e in snap["activity"])


def test_offline_pages_fall_back_to_the_local_reader_with_ocrspace_too(tmp_path):
    sess = run_session_with(tmp_path, os_settings(CLOSED))
    assert {p["reader"] for p in sess.snapshot()["pages"]} == {"local"} and FakeEngine.calls == 3
    assert any("OCR.space was not used" in e["msg"] and "no internet connection" in e["msg"] for e in sess.snapshot()["activity"])


def test_the_app_reports_the_selected_provider_and_asks_for_its_own_key(tmp_path):
    app = WebApp(library_root=tmp_path / "lib", reader_settings=ReaderSettings({"MAVIS_OCR_PROVIDER": "ocrspace"}), health_thread=False)
    wait_for(lambda: app.snapshot()["reader"]["status"] == "missing_key", what="reader check")
    state = app.snapshot()["reader"]
    assert state["provider"] == "ocrspace" and state["label"] == "OCR.space" and "OCRSPACE_API_KEY" in state["message"]
    from webapp.core import Refused
    from test_webapp import FakeStream
    with pytest.raises(Refused) as err:
        app.start_scan(FakeStream(), {"kind": "demo", "label": "t"})
    assert "OCR.space API key" in str(err.value) and "OCRSPACE_API_KEY" in str(err.value)
