"""Camera reader that stays at the live edge, and the ONNX thread cap for the local OCR reader."""
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "prototype"))
sys.path.insert(0, str(REPO))

from webapp.live_stream import LiveStream  # noqa: E402

T0 = time.perf_counter()
BASE = cv2.resize(np.random.default_rng(1).integers(60, 200, (36, 64, 3), dtype=np.uint8), (640, 360))


def frame_jpeg():
    """A frame that carries the moment it was captured (ms since T0, 24 binary blocks)."""
    ms = int((time.perf_counter() - T0) * 1000)
    f = BASE.copy()
    for b in range(24):
        f[0:16, b * 16:(b + 1) * 16] = 255 if (ms >> b) & 1 else 0
    return cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def age_of(frame):
    g = cv2.cvtColor(frame[0:16, 0:384], cv2.COLOR_BGR2GRAY)
    ms = sum((1 << b) for b in range(24) if g[8, b * 16 + 8] > 128)
    return (time.perf_counter() - T0) - ms / 1000.0


def start_camera(fps=30.0):
    stop = threading.Event()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--boundary")
            self.end_headers()
            nxt = time.perf_counter()
            try:
                while not stop.is_set():
                    time.sleep(max(0, nxt - time.perf_counter()))
                    jpg = frame_jpeg()                         # a live camera sends what it sees now
                    self.wfile.write(b"--boundary\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                                     + b"\r\n\r\n" + jpg + b"\r\n")
                    nxt = max(nxt + 1 / fps, time.perf_counter())
            except Exception:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    server.daemon_threads = True
    server.stop_event = stop
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/video"


def test_the_reader_keeps_full_frame_rate_and_catches_up_after_being_starved():
    server, url = start_camera()
    stream = LiveStream(url)
    try:
        assert stream.connect()
        started, steady = time.time(), 0
        while time.time() - started < 2.0:                    # keeping up: every frame, fresh
            ok, frame = stream.read()
            assert ok
            steady += 1
        assert steady >= 50                                    # ~30 fps camera (a plain reader gets the same here)
        assert stream.stale_dropped < 10                       # nothing needed skipping while it kept up
        time.sleep(2.0)                                        # the CPU is busy: nobody reads for 2 s
        ok, frame = stream.read()                              # the first read afterwards is already the live frame
        assert ok and age_of(frame) < 0.35, age_of(frame)
        assert stream.stale_dropped >= 1                       # it skipped buffered frames to get there
        ages = [age_of(stream.read()[1]) for _ in range(30)]
        assert max(ages) < 0.35
    finally:
        stream.release()
        server.shutdown()


def test_a_dead_stream_reports_failure_instead_of_hanging_or_raising():
    server, url = start_camera()
    stream = LiveStream(url)
    assert stream.connect()
    server.stop_event.set()                                    # the camera goes away: its connection closes
    server.shutdown()
    results = []
    for _ in range(300):                                       # the connection is gone: reads must fail, not hang
        ok, _ = stream.read()
        results.append(ok)
        if not ok:
            break
    assert results[-1] is False and stream.read() == (False, None)
    stream.release()
    assert LiveStream("http://127.0.0.1:9/video").connect() is False


def test_the_thread_cap_builds_the_same_reader_with_only_the_thread_settings_added(monkeypatch):
    import rapidocr
    from ocr_engine import PrototypeOCREngine
    from webapp.ocr_threads import install_reader
    seen = []

    class Recorder:
        def __init__(self, params=None):
            seen.append(dict(params))

    monkeypatch.setattr(rapidocr, "RapidOCR", Recorder)
    monkeypatch.setattr(PrototypeOCREngine, "_reader", None)
    PrototypeOCREngine._init_reader()                          # what the frozen engine builds
    monkeypatch.setattr(PrototypeOCREngine, "_reader", None)
    assert install_reader(4) is True
    original, capped = seen
    extra = {k: v for k, v in capped.items() if k not in original}
    assert extra == {"EngineConfig.onnxruntime.intra_op_num_threads": 4, "EngineConfig.onnxruntime.inter_op_num_threads": 1}
    assert {k: v for k, v in capped.items() if k in original} == original
    assert install_reader(4) is False                          # a reader already exists: left alone
    monkeypatch.setattr(PrototypeOCREngine, "_reader", None)
    assert install_reader(0) is False and PrototypeOCREngine._reader is None   # 0 = library default
