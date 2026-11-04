"""One live scan: camera -> frame selection -> OCR queue -> drain -> post-processing -> frozen document.

This is the headless counterpart of prototype/main.py's PrototypeApp (same capture rules, same shutdown order, same
finalization snapshot), without any Tk dependency, so a browser can observe it. All numbers exposed through
`snapshot()` are real counters/timestamps of this session; nothing is estimated or animated on the server.

Shutdown order (unchanged from main.py): stop accepting captures -> camera/capture threads finish -> selector flush ->
OCR queue drains completely -> post-processing -> export -> frozen snapshot (finalized=True is set last). The LLM is
never started from here.
"""
import copy
import hashlib
import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "prototype", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from frame_selector import build_selector  # noqa: E402
from ocr_engine import PrototypeOCREngine  # noqa: E402
from postprocess.pipeline import PostProcessingPipeline  # noqa: E402

from webapp.gemini_reader import REMOTE_ENGINES  # noqa: E402
from webapp.watermark import clean_text_lines, split_detections  # noqa: E402

# Same values as main.PrototypeApp (tests/test_webapp_session.py asserts they stay identical).
DUPLICATE_DISTANCE_THRESHOLD = 5.0
OCR_FOCUS_COOLDOWN_SEC = 3.0
LOW_CONTENT_CHARS = 40        # a page with fewer characters than this is flagged "little text" in the UI
CAPTURE_MODES = ("b0", "c3", "c5", "interval")
DEFAULT_INTERVAL_S = 5.0
MIN_INTERVAL_S, MAX_INTERVAL_S = 2.0, 30.0
INTERVAL_PICK_WINDOW_S = 0.5  # timed capture takes the sharpest frame of the last 0.5 s before the tick

STAGES = [
    ("stop_capture", "Stopping capture"),
    ("read_pages", "Reading your pages"),
    ("clean_text", "Cleaning and structuring text"),
    ("save", "Saving the document"),
]


class PageRecord:
    def __init__(self, number, metrics, jpeg, reason):
        self.number = number
        self.metrics = metrics
        self.jpeg = jpeg
        self.reason = reason
        self.captured_at = time.time()
        self.finished_at = None
        self.text = ""
        self.error = None
        self.mean_confidence = None
        self.detections = 0
        self.ocr_s = None
        self.finished = False
        self.watermark_boxes = 0
        self.reader = "local"                      # who read this page: "local" (PP-OCR) or "gemini"

    @property
    def blank(self):
        return self.finished and self.error is None and not self.text.strip()

    @property
    def state(self):
        if self.error is not None:
            return "error"
        if self.finished:
            return "done"
        return "reading" if self.metrics.get("ocr_started_at") else "queued"


def _encode_jpeg(frame, max_width, quality):
    height, width = frame.shape[:2]
    if width > max_width:
        frame = cv2.resize(frame, (max_width, int(height * max_width / width)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


class ScanSession:
    OCR_QUEUE_MAX = PrototypeOCREngine.OCR_QUEUE_MAXSIZE
    CAMERA_STOP_TIMEOUT_S = 8.0
    VIDEO_MAX_WIDTH = 800

    def __init__(self, session_id, stream, source, out_dir, selector_mode="b0", engine_factory=PrototypeOCREngine,
                 pipeline_factory=None, on_finished=None, page_image_width=1100, interval_s=DEFAULT_INTERVAL_S):
        if selector_mode not in CAPTURE_MODES:
            raise ValueError(f"unknown capture mode {selector_mode!r}")
        self.id = session_id
        self.stream = stream
        self.source = source                       # {"kind": "droidcam"|"demo", "label": str}
        self.out_dir = Path(out_dir)
        self.selector_mode = selector_mode
        # None = no frame selector: B0 (cooldown + fingerprint dedup) or the plain timer ("interval")
        self.selector = build_selector(selector_mode) if selector_mode in ("c3", "c5") else None
        self.interval_s = min(MAX_INTERVAL_S, max(MIN_INTERVAL_S, float(interval_s)))
        self._next_capture_at = None               # perf_counter time of the next timed capture
        self._window_best = None                   # (sharpness, frame) among the frames just before the tick
        self._engine_factory = engine_factory
        self._pipeline_factory = pipeline_factory or (lambda export_dir: PostProcessingPipeline(export_dir=export_dir))
        self._on_finished = on_finished
        self._page_image_width = page_image_width

        self._lock = threading.Lock()
        self.phase = "connecting"       # connecting | scanning | finalizing | finished | failed | cancelled
        self.error = ""
        self.created_at = time.time()
        self.scan_started_at = None
        self.finalize_started_at = None
        self.finished_at = None
        self._cancelled = False

        self.engine = None
        self.pipeline = None
        self._ocr_thread = None
        self._camera_thread = None
        self._capture_thread = None
        self._collector_thread = None
        self._shutdown_thread = None
        self._camera_stop = threading.Event()
        self._camera_done = threading.Event()
        self._capture_done = threading.Event()
        self._collector_stop = threading.Event()
        self._accepting = True

        self._latest_frame = None
        self._frame_seq = 0
        self._frame_times = deque(maxlen=90)
        self._frame_size = None
        self._frames_read = 0
        self.connection = "connecting"  # connecting | live | lost | closed
        self.connection_error = ""
        self._video_cache = (-1, b"", 0.0)
        self._sent_times = deque(maxlen=60)
        self._stale_logged, self._stale_log_at = 0, 0.0

        self._pages = {}
        self._last_page = 0
        self._last_fingerprint = None
        self._cooldown_until = 0.0
        self._capture_status = "waiting"   # waiting | cooldown | duplicate | watching | submitted | queue_full
        self._dropped = 0
        self._last_capture_at = None
        self._watermark_total = 0

        self._events = deque(maxlen=120)
        self._event_id = 0
        self.stages = [{"key": k, "label": l, "state": "pending", "detail": "", "started_at": None, "ended_at": None}
                       for k, l in STAGES]

        self.finalized = False
        self.finalize_error = ""
        self.empty = False
        self.document_id = None            # doc_xxxx assigned by the post-processing pipeline
        self._snapshot_json = None
        self._snapshot_sha256 = None

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        threading.Thread(target=self._connect_and_run, name="web-connect", daemon=True).start()

    def cancel(self):
        """Abort while connecting (nothing has been captured yet)."""
        with self._lock:
            if self.phase != "connecting":
                return False
            self._cancelled = True
            self.phase = "cancelled"
        self._log("info", "Connection cancelled")
        return True

    def _connect_and_run(self):
        self._log("info", f"Connecting to {self.source['label']}...")
        try:
            ok = bool(self.stream.connect())
            reason = ""
        except Exception as exc:  # noqa: BLE001
            ok, reason = False, f"{type(exc).__name__}: {exc}"
        if self._cancelled:
            self._safe_release()
            return
        if not ok:
            with self._lock:
                self.phase = "failed"
                self.connection = "closed"
                self.error = (f"Could not connect to {self.source['label']}. Check that the DroidCam app is running, "
                              "that the phone and this computer are on the same network, and that the IP address "
                              "and port are right." + (f" ({reason})" if reason else ""))
            self._log("error", "Could not connect to the camera")
            self._safe_release()
            return
        try:
            self.engine = self._engine_factory()
            self.pipeline = self._pipeline_factory(self.out_dir / "export")
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.phase = "failed"
                self.error = f"Could not start the scanner: {type(exc).__name__}: {exc}"
            self._safe_release()
            return
        if self.selector is not None:
            cv2.setNumThreads(1)           # as in main.py: keep OpenCV off the OCR worker's cores
        self._ocr_thread = threading.Thread(target=self.engine.worker_loop, name="web-ocr", daemon=True)
        self._camera_thread = threading.Thread(target=self._camera_loop, name="web-camera", daemon=True)
        self._capture_thread = threading.Thread(target=self._capture_loop, name="web-capture", daemon=True)
        self._collector_thread = threading.Thread(target=self._collector_loop, name="web-collector", daemon=True)
        with self._lock:
            self.phase = "scanning"
            self.connection = "live"
            self.scan_started_at = time.time()
        if self.selector_mode == "interval":
            self._next_capture_at = time.perf_counter() + self.interval_s
        self._ocr_thread.start()
        self._camera_thread.start()
        self._capture_thread.start()
        self._collector_thread.start()
        self._log("info", "Camera connected - scanning")

    def _safe_release(self):
        try:
            self.stream.release()
        except Exception:  # noqa: BLE001
            pass

    def stop_scanning(self):
        """Stop accepting pages and finish everything already captured. Idempotent."""
        with self._lock:
            if self.phase != "scanning" or self._shutdown_thread is not None:
                return False
            self.phase = "finalizing"
            self.finalize_started_at = time.time()
            self._accepting = False
            self._shutdown_thread = threading.Thread(target=self._shutdown_worker, name="web-shutdown", daemon=True)
        self._log("info", "Scanning stopped - finishing the queued pages")
        self._shutdown_thread.start()
        return True

    # ------------------------------------------------------------------ camera / capture
    def _camera_loop(self):
        try:
            while not self._camera_stop.is_set():
                success, frame = self.stream.read()
                if not success:
                    with self._lock:
                        if self.connection == "live":
                            self.connection = "lost"
                            self.connection_error = "The camera stream stopped sending frames."
                    self._log("error", "Camera stream stopped")
                    break
                now = time.time()
                stale = getattr(self.stream, "stale_dropped", 0)
                if stale - self._stale_logged >= 10 and now - self._stale_log_at > 10:
                    self._log("warn", f"The picture had fallen {stale - self._stale_logged} frames behind the camera "
                                      "(the computer was busy); skipped ahead to the live frame")
                    self._stale_logged, self._stale_log_at = stale, now
                with self._lock:
                    self._latest_frame = frame
                    self._frame_seq += 1
                    self._frames_read += 1
                    self._frame_times.append(now)
                    self._frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        finally:
            self._safe_release()
            self._camera_done.set()

    def _capture_loop(self):
        last_seq = -1
        try:
            while not self._camera_stop.is_set():
                with self._lock:
                    frame, seq, accepting = self._latest_frame, self._frame_seq, self._accepting
                if frame is None or seq == last_seq:
                    time.sleep(0.005)
                    continue
                last_seq = seq
                if not accepting:
                    continue
                if self.selector is not None:
                    self._select_and_submit(frame)
                    continue
                if self.selector_mode == "interval":
                    self._interval_step(frame)
                    continue
                self._b0_step(frame)
        finally:
            self._capture_done.set()

    @staticmethod
    def _sharpness(frame):
        small = cv2.resize(frame, (640, max(1, int(frame.shape[0] * 640 / frame.shape[1]))), interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

    def _interval_step(self, frame):
        """Timed capture: one page every `interval_s` seconds, no change detection.

        The only choice made is which frame of the last INTERVAL_PICK_WINDOW_S seconds goes to OCR: the sharpest, so a
        moment of hand shake at the tick does not decide the page.
        """
        now = time.perf_counter()
        if self._next_capture_at is None:
            self._next_capture_at = now + self.interval_s
        due_in = self._next_capture_at - now
        self._capture_status = "timer_wait"
        if due_in <= INTERVAL_PICK_WINDOW_S:
            score = self._sharpness(frame)
            if self._window_best is None or score > self._window_best[0]:
                self._window_best = (score, frame)
        if due_in > 0:
            return
        score, best = self._window_best or (self._sharpness(frame), frame)
        self._window_best = None
        self._submit(best, self._fingerprint(best), score, "timer", 0.0)
        self._next_capture_at += self.interval_s
        if self._next_capture_at <= now:
            self._next_capture_at = now + self.interval_s

    @staticmethod
    def _fingerprint(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _b0_step(self, frame):
        """The original production rule: 3 s cooldown after a page, then any frame that is not a near-duplicate."""
        now = time.perf_counter()
        if now < self._cooldown_until:
            self._capture_status = "cooldown"
            return
        fingerprint = self._fingerprint(frame)
        if self._last_fingerprint is not None:
            distance = float(np.mean(np.abs(fingerprint - self._last_fingerprint)))
            if distance <= DUPLICATE_DISTANCE_THRESHOLD:
                self._capture_status = "duplicate"
                return
        if self._submit(frame, fingerprint, 0.0, "new_view", 0.0):
            self._last_fingerprint = fingerprint.copy()
            self._cooldown_until = time.perf_counter() + OCR_FOCUS_COOLDOWN_SEC

    def _select_and_submit(self, frame):
        start = time.perf_counter()
        commits = self.selector.process(frame, start)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not commits and self._capture_status in ("waiting", "submitted"):
            self._capture_status = "watching"
        for commit in commits:
            self._submit_commit(commit, elapsed_ms)

    def _submit_commit(self, commit, selection_ms=0.0):
        return self._submit(commit["frame"], commit["fingerprint"], commit["sharpness"], commit["reason"], selection_ms)

    def _submit(self, frame, fingerprint, sharpness, reason, selection_ms):
        page_number = self._last_page + 1
        metrics = {
            "localization_ms": 0.0, "selection_ms": selection_ms, "sharpness_ms": 0.0, "stability_ms": 0.0,
            "fingerprint": fingerprint, "sharpness": sharpness, "accepted_at": time.perf_counter(),
            "crop_width": int(frame.shape[1]), "crop_height": int(frame.shape[0]), "selector_reason": reason,
        }
        record = PageRecord(page_number, metrics, _encode_jpeg(frame, self._page_image_width, 78), reason)
        with self._lock:
            self._pages[page_number] = record      # registered first so the collector can always find it
        if self.engine.submit(page_number, frame, metrics):
            self._last_page = page_number
            self._capture_status = "submitted"
            self._last_capture_at = time.time()
            self._log("capture", f"Page {page_number} captured", page=page_number)
            return True
        with self._lock:
            self._pages.pop(page_number, None)
        self._dropped += 1
        self._capture_status = "queue_full"
        self._log("warn", f"The reading queue is full - a page view was skipped ({self._dropped} so far)")
        return False

    # ------------------------------------------------------------------ OCR results
    def _collector_loop(self):
        while True:
            result = self.engine.poll_result()
            if result is None:
                if self._collector_stop.is_set():
                    return
                time.sleep(0.04)
                continue
            self._record_result(result)

    def _record_result(self, result):
        page_number, text, metrics, ocr_result, error = result
        with self._lock:
            record = self._pages.get(page_number)
        if record is None:
            return
        threshold = getattr(self.engine, "CONFIDENCE_THRESHOLD", 0.5)
        detections, removed = [], []
        reader = metrics.get("reader", "local")
        if error is None and (ocr_result or {}).get("engine") in REMOTE_ENGINES:
            text, n_removed = clean_text_lines(text)           # a transcription has no boxes: clean it line by line
            removed = [None] * n_removed
        elif error is None:
            # The DroidCam watermark is dropped box by box, then the engine's own line layout is rebuilt from the rest.
            detections, removed = split_detections((ocr_result or {}).get("detections", []))
            if removed:
                text = self.engine._reconstruct_layout(detections) if detections else ""
        with self._lock:
            record.finished_at = time.time()
            if error is not None:
                record.error = str(error)
            else:
                record.text = text
                record.reader = reader
                record.watermark_boxes = len(removed)
                self._watermark_total += len(removed)
                record.ocr_s = round(float(metrics.get("ocr_inference_ms", 0.0)) / 1000.0, 2)
                used = [d["confidence"] for d in detections if d.get("text") and d["confidence"] >= threshold]
                record.detections = len(used)
                record.mean_confidence = round(sum(used) / len(used), 3) if used else None
            record.finished = True
        if error is not None:
            self._log("error", f"Page {page_number}: reading failed ({error})", page=page_number)
        elif not text.strip():
            self._log("warn", f"Page {page_number}: no readable text" + (" (only the DroidCam watermark)" if removed else "")
                      + " - it will be left out", page=page_number)
        else:
            conf = "" if record.mean_confidence is None else f", confidence {record.mean_confidence:.0%}"
            who = f" by {getattr(self.engine, 'label', reader)}" if reader != "local" else ""
            self._log("ocr", f"Page {page_number} read{who}: {len(text):,} characters{conf}", page=page_number)
        if metrics.get("rate_limit_wait_s"):
            self._log("warn", f"Page {page_number}: waited {metrics['rate_limit_wait_s']:.0f}s for "
                              f"{getattr(self.engine, 'label', 'the page reader')}'s rate limit", page=page_number)
        if metrics.get("fallback_reason"):
            self._log("warn", f"Page {page_number}: {getattr(self.engine, 'label', 'the online reader')} was not used "
                              f"({metrics['fallback_reason']}); read with the local reader", page=page_number)

    # ------------------------------------------------------------------ shutdown + finalization
    def _stage(self, key):
        return next(s for s in self.stages if s["key"] == key)

    def _begin_stage(self, key, detail=""):
        with self._lock:
            stage = self._stage(key)
            stage.update(state="active", detail=detail, started_at=time.time())

    def _end_stage(self, key, detail=None, warn=False):
        with self._lock:
            stage = self._stage(key)
            stage.update(state="warn" if warn else "done", ended_at=time.time())
            if detail is not None:
                stage["detail"] = detail

    def _pending_pages(self):
        with self._lock:
            return [p for p in self._pages.values() if not p.finished]

    def _shutdown_worker(self):
        try:
            self._do_shutdown()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.finalize_error = f"{type(exc).__name__}: {exc}"
                self.error = f"Finalizing failed: {self.finalize_error}"
                self.phase = "failed"
            self._log("error", f"Finalizing failed: {self.finalize_error}")
            return
        if self._on_finished is not None:
            try:
                self._on_finished(self)
            except Exception as exc:  # noqa: BLE001
                self._log("error", f"Could not open the document: {type(exc).__name__}: {exc}")

    def _do_shutdown(self):
        self._begin_stage("stop_capture", "Closing the camera")
        self._camera_stop.set()
        camera_ok = self._camera_done.wait(self.CAMERA_STOP_TIMEOUT_S)
        capture_ok = self._capture_done.wait(self.CAMERA_STOP_TIMEOUT_S)
        with self._lock:
            if self.connection == "live":
                self.connection = "closed"
        if self.selector is not None and capture_ok:
            for commit in self.selector.flush():       # the steady view the user was on when they pressed stop
                self._submit_commit(commit)
        if camera_ok and capture_ok:
            self._end_stage("stop_capture", "Camera closed")
        else:
            self._end_stage("stop_capture", "The camera did not respond in time; continuing with the pages already "
                                            "captured", warn=True)
            self._log("warn", "The camera did not close in time; continuing with the captured pages")

        total = len(self._pages)
        self._begin_stage("read_pages", f"0 of {total} pages")
        self.engine.stop()
        dead_since = None
        while True:
            pending = self._pending_pages()
            worker_alive = self._ocr_thread is not None and self._ocr_thread.is_alive()
            if not pending and self.engine.queue_depth() == 0 and not self.engine.is_processing():
                break
            if pending and not worker_alive and self.engine.queue_depth() == 0:
                # The worker has exited. Its last results may still be on their way to the collector, so only
                # give up on a page after a grace period in which nothing arrived.
                dead_since = dead_since or time.time()
                if time.time() - dead_since > 1.5:
                    for p in self._pending_pages():    # the reader died without reporting these pages
                        with self._lock:
                            p.error, p.finished = "The OCR worker stopped before finishing this page.", True
                    break
            else:
                dead_since = None
            with self._lock:
                self._stage("read_pages")["detail"] = f"{total - len(pending)} of {total} pages"
            time.sleep(0.1)
        self.engine.wait()
        self._collector_stop.set()
        self._collector_thread.join(timeout=5)
        if self._ocr_thread is not None:
            self._ocr_thread.join(timeout=3)
        failed = sorted(n for n, p in self._pages.items() if p.error is not None)
        blank = sorted(n for n, p in self._pages.items() if p.blank)
        self._end_stage("read_pages", f"{total - len(failed)} of {total} pages read"
                        + (f" - {len(failed)} could not be read" if failed else "")
                        + (f" - {len(blank)} blank view(s) left out" if blank else ""), warn=bool(failed))

        self._begin_stage("clean_text", "Fixing common OCR mistakes and structuring text")
        # A view with no readable text (for example only the DroidCam watermark) is not a page of the document.
        ok_numbers = sorted(n for n, p in self._pages.items() if p.error is None and not p.blank)
        processed, processing = None, {"status": "complete" if not failed else "complete_with_errors",
                                       "pages_ok": len(ok_numbers), "pages_failed": failed}
        if ok_numbers:
            processed = self.pipeline.process_document([self._pages[n].text for n in ok_numbers])
            for number, page in zip(ok_numbers, processed["pages"]):
                page["page_number"] = number           # keep capture numbering when a page failed (defect D2)
            self.document_id = processed["document_id"]
        self._end_stage("clean_text", f"{len(ok_numbers)} pages processed")

        self._begin_stage("save", "Writing the export files")
        if processed is not None:
            self.pipeline.export_results(processed, processing=processing)
            document = copy.deepcopy(processed)
            document["processing"] = processing
            text = json.dumps(document, sort_keys=True, ensure_ascii=False)
            self._snapshot_json = text
            self._snapshot_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            self._write_session_files(processed)
        else:
            self.empty = True
        self._end_stage("save", "Document saved" if processed is not None else "Nothing to save")
        with self._lock:
            self.finalized = True                      # last: whoever sees True also sees the snapshot
            self.finished_at = time.time()
            self.phase = "finished"
        self._log("info", "Document finalized" if processed is not None else "No readable pages were captured")

    def _write_session_files(self, processed):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        pages_dir = self.out_dir / "pages"
        pages_dir.mkdir(exist_ok=True)
        pages_meta = []
        for n in sorted(self._pages):
            p = self._pages[n]
            if p.jpeg:
                (pages_dir / f"page_{n}.jpg").write_bytes(p.jpeg)
            pages_meta.append({"page": n, "state": p.state, "chars": len(p.text), "mean_confidence": p.mean_confidence,
                               "ocr_s": p.ocr_s, "reason": p.reason, "error": p.error, "reader": p.reader})
        created = datetime.fromtimestamp(self.created_at)
        confs = [p.mean_confidence for p in self._pages.values() if p.mean_confidence is not None and not p.blank]
        meta = {
            "id": self.id, "document_id": self.document_id, "title": f"Scan {created:%d %b %Y, %H:%M}",
            "created_at": created.isoformat(timespec="seconds"), "source": self.source, "selector": self.selector_mode,
            "page_count": len(processed["pages"]), "pages_failed": [p["page"] for p in pages_meta if p["error"]],
            "blank_pages": sorted(n for n, p in self._pages.items() if p.blank),
            "readers": sorted({p.reader for p in self._pages.values() if p.error is None}),
            "watermark_boxes_ignored": self._watermark_total, "interval_s": self.interval_s if self.selector_mode == "interval" else None,
            "scan_seconds": round((self.finalize_started_at or time.time()) - (self.scan_started_at or time.time()), 1),
            "mean_confidence": round(sum(confs) / len(confs), 3) if confs else None,
            "snapshot_sha256": self._snapshot_sha256, "pages": pages_meta,
        }
        (self.out_dir / "snapshot.json").write_text(self._snapshot_json, encoding="utf-8")
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ queries
    def llm_blocker(self):
        """Why the LLM must not run now (None = allowed). Checks the real state, like main.PrototypeApp._llm_blocker."""
        if self.finalize_error:
            return f"Finalization failed ({self.finalize_error})."
        if not self.finalized:
            return "Scanning is not finished yet. Questions open once every page has been read."
        if (not (self._camera_done.is_set() and self._capture_done.is_set()) or self.engine.is_processing()
                or self.engine.queue_depth() > 0):
            return "Reading is still active; wait until it has finished."
        return None

    def active(self):
        return self.phase in ("connecting", "scanning", "finalizing")

    def snapshot_json(self):
        return self._snapshot_json

    def page_image(self, number):
        with self._lock:
            record = self._pages.get(number)
            return record.jpeg if record else None

    def note_video_frame(self):
        """Called by the HTTP layer for every feed frame it writes to a viewer (for the 'shown' fps readout)."""
        with self._lock:
            self._sent_times.append(time.time())

    def _sent_fps(self, now):
        with self._lock:
            times = [t for t in self._sent_times if now - t <= 3.0]
        if len(times) < 2:
            return 0.0
        return round((len(times) - 1) / max(times[-1] - times[0], 1e-6), 1)

    def video_jpeg(self, last_seq, min_interval=0.1, quality=65):
        """(seq, jpeg) of the newest frame if it is newer than `last_seq` (rate-limited), else (last_seq, None)."""
        with self._lock:
            frame, seq = self._latest_frame, self._frame_seq
            cached_seq, cached, cached_at = self._video_cache
        if frame is None or seq == last_seq:
            return last_seq, None
        if cached_seq == seq:
            return seq, cached
        if time.time() - cached_at < min_interval:
            return last_seq, None
        jpeg = _encode_jpeg(frame, self.VIDEO_MAX_WIDTH, quality)
        with self._lock:
            self._video_cache = (seq, jpeg, time.time())
        return seq, jpeg

    def _fps(self, now):
        with self._lock:
            times = [t for t in self._frame_times if now - t <= 3.0]
        if len(times) < 2:
            return 0.0
        return round((len(times) - 1) / max(times[-1] - times[0], 1e-6), 1)

    def _log(self, kind, message, page=None):
        with self._lock:
            self._event_id += 1
            self._events.append({"id": self._event_id, "t": time.time(), "kind": kind, "msg": message, "page": page})

    def snapshot(self):
        now = time.time()
        with self._lock:
            pages = [self._page_view(self._pages[n]) for n in sorted(self._pages)]
            events = list(self._events)[-40:]
            stages = [dict(s) for s in self.stages]
            frame_size, frames, last_frame_at = self._frame_size, self._frames_read, (
                self._frame_times[-1] if self._frame_times else None)
            phase, connection, conn_error = self.phase, self.connection, self.connection_error
        reading = next((p for p in pages if p["state"] == "reading"), None)
        done = [p for p in pages if p["state"] == "done"]
        failed = [p for p in pages if p["state"] == "error"]
        confs = [p["confidence"] for p in done if p["confidence"] is not None]
        ocr_times = [p["ocr_s"] for p in done if p["ocr_s"]]
        engine = self.engine
        cooldown = max(0.0, self._cooldown_until - time.perf_counter()) if self.selector is None else 0.0
        return {
            "id": self.id, "phase": phase, "error": self.error, "source": self.source, "selector": self.selector_mode,
            "started_at": self.scan_started_at, "finalize_started_at": self.finalize_started_at,
            "connection": {"state": connection, "error": conn_error, "fps": self._fps(now), "frames": frames,
                           "size": list(frame_size) if frame_size else None,
                           "last_frame_age_s": round(now - last_frame_at, 1) if last_frame_at else None,
                           "shown_fps": self._sent_fps(now), "stale_dropped": getattr(self.stream, "stale_dropped", 0)},
            "capture": {"accepting": self._accepting, "status": self._capture_status, "cooldown_s": round(cooldown, 1),
                        "pages_captured": len(pages), "skipped_queue_full": self._dropped,
                        "last_capture_at": self._last_capture_at, "mode": self.selector_mode,
                        "interval_s": self.interval_s if self.selector_mode == "interval" else None,
                        "next_capture_at": round(now + (self._next_capture_at - time.perf_counter()), 2)
                        if self.selector_mode == "interval" and self._next_capture_at and phase == "scanning" else None,
                        "watermark_ignored": self._watermark_total},
            "ocr": {"queue_depth": engine.queue_depth() if engine else 0, "queue_max": self.OCR_QUEUE_MAX,
                    "busy": bool(engine and engine.is_processing()), "current_page": reading["n"] if reading else None,
                    "current_started_at": reading["started_at"] if reading else None, "completed": len(done),
                    "failed": len(failed),
                    "rate_limited_until": engine.rate_limited_until() if engine is not None and hasattr(engine, "rate_limited_until") else None,
                    "mean_confidence": round(sum(confs) / len(confs), 3) if confs else None,
                    "avg_page_s": round(sum(ocr_times) / len(ocr_times), 1) if ocr_times else None},
            "pages": pages, "activity": events, "stages": stages,
            "finalized": self.finalized, "empty": self.empty, "document_id": self.id if self.finalized and not self.empty else None,
        }

    def _page_view(self, p):
        started = p.metrics.get("ocr_started_at")
        started_wall = None
        if started and not p.finished:
            started_wall = round(time.time() - (time.perf_counter() - started), 2)
        preview = " ".join(p.text.split())[:150]
        return {"n": p.number, "state": p.state, "chars": len(p.text), "confidence": p.mean_confidence,
                "ocr_s": p.ocr_s, "reason": p.reason, "preview": preview, "error": p.error,
                "captured_at": p.captured_at, "started_at": started_wall,
                "blank": p.blank, "reader": p.reader,
                "low_content": bool(p.finished and p.error is None and not p.blank
                                    and len(p.text.strip()) < LOW_CONTENT_CHARS)}
