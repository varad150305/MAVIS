"""Application state shared by the HTTP handlers: the one active scan, the document library and the local model.

Rules enforced here (not only in the browser):
  * one scan at a time;
  * the model is never started while a scan is connecting, capturing or finalizing, and a document created by a scan
    only becomes askable once that scan reports finalized with its OCR idle (ScanSession.llm_blocker);
  * one model request at a time (questions and the context-preparation step share one lock).
"""
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "prototype", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import os  # noqa: E402

import cv2  # noqa: E402

from llm_stage import (ANSWER_SCHEMA, ANSWER_SYSTEM, DocumentAssistant, LLMConfig, LLMError,  # noqa: E402
                       pages_from_document)
from ocr_engine import PrototypeOCREngine  # noqa: E402
from postprocess.pipeline import PostProcessingPipeline  # noqa: E402

from webapp.watermark import clean_text_lines, split_detections  # noqa: E402
from webapp.gemini_reader import REMOTE_ENGINES, ReaderSettings, make_engine_factory, make_reader  # noqa: E402
from webapp.llm_runtime import CancellableOllamaRuntime  # noqa: E402
from webapp.session import ScanSession  # noqa: E402
from webapp.workspace import Library  # noqa: E402

HEALTH_INTERVAL_S = 15.0
CANCEL_WAIT_S = 15.0
STOPPED_BY_SCAN = "The assistant was stopped because a new scan was started."


def default_assistant():
    """The web app's model setup. Per-page OCR recovery is OFF unless MAVIS_LLM_RECOVER=1: it measured no accuracy
    benefit (8/18 either way, see MAVIS_FREEZE_REPORT known issue 4) and costs minutes of CPU per document."""
    cfg = LLMConfig()
    if "MAVIS_LLM_RECOVER" not in os.environ:
        cfg.recover = False
    return DocumentAssistant(cfg, CancellableOllamaRuntime(cfg))


class Busy(Exception):
    pass


class Refused(Exception):
    """A request that must not run now (message is shown to the user)."""


class WebApp:
    def __init__(self, library_root=None, assistant=None, engine_factory=None, pipeline_factory=None,
                 warm_ocr=True, health_thread=True, reader_settings=None):
        self.library = Library(library_root or ROOT / "exports" / "web")
        self.assistant = assistant or default_assistant()
        # Pages are read by Gemini; the local PP-OCR engine is built only if the computer is offline (gemini_reader.py).
        self.reader = reader_settings or ReaderSettings()
        self._api = make_reader(self.reader)                         # the chosen cloud OCR API (Gemini or OCR.space)
        self._engine_injected = engine_factory is not None          # tests inject their own page reader
        self._engine_factory = engine_factory or make_engine_factory(self.reader, PrototypeOCREngine)
        self.reader_state = {"status": "checking", "message": f"Checking the {self.reader.label} connection..."}
        if not self._engine_injected and not self.reader.local_only:
            threading.Thread(target=self.check_reader, name="web-reader-check", daemon=True).start()
        else:
            self.reader_state = {"status": "ok", "message": ""}
        self._pipeline_factory = pipeline_factory
        self.scan = None
        self._scan_lock = threading.Lock()
        self._llm_lock = threading.Lock()
        self._busy = None                      # {"doc_id", "question", "mode", "started_at"} while a question runs
        self._prepare = {}                     # doc_id -> status dict
        self._docs_version = 0
        self._cancel_epoch = 0                 # bumped by cancel_llm(); work started before the bump is discarded
        self.llm = {"status": "checking", "model": self.assistant.cfg.model, "message": "Checking the local model...",
                    "checked_at": None}
        self.check_llm()
        if health_thread:
            threading.Thread(target=self._health_loop, name="web-llm-health", daemon=True).start()

    # ------------------------------------------------------------------ page reader (Gemini)
    def check_reader(self):
        """Is the API key set/accepted and the service reachable? Sends no page data."""
        status, message = self._api.check()
        self.reader_state = {"status": status, "message": message, "checked_at": time.time()}
        return self.reader_state

    def _reader_blocker(self):
        """Why a scan cannot read pages at all (None = fine). Offline is fine: the local reader takes over."""
        if self._engine_injected or self.reader.local_only:
            return None
        if not self.reader.has_key:
            return f"Page reading needs a {self.reader.label} API key. {self.reader.setup_hint}"
        if self.reader_state.get("status") in ("rejected", "model_missing"):
            return self.reader_state["message"]
        return None

    def check_llm(self):
        model = self.assistant.cfg.model
        try:
            names = self.assistant.runtime.list_models()
        except LLMError as exc:
            self.llm.update(status="unavailable", message=exc.message, checked_at=time.time())
            return self.llm
        except Exception as exc:  # noqa: BLE001
            self.llm.update(status="unavailable", message=f"{type(exc).__name__}: {exc}", checked_at=time.time())
            return self.llm
        if any(n == model or n.split(":")[0] == model.split(":")[0] and ":" not in model for n in names):
            self.llm.update(status="ok", message="", checked_at=time.time())
        else:
            self.llm.update(status="model_missing", checked_at=time.time(),
                            message=f"The model '{model}' is not installed. Install it with: ollama pull {model}")
        return self.llm

    def _health_loop(self):
        while True:
            time.sleep(HEALTH_INTERVAL_S)
            if self.scan is not None and self.scan.active():
                continue                       # no model traffic at all while a scan is running
            self.check_llm()

    # ------------------------------------------------------------------ scanning
    def cancel_llm(self):
        """Hard-stop all model work: in-flight request aborted, queued preparation/questions discarded.

        Returns once the model lock is free (or CANCEL_WAIT_S has passed)."""
        self._cancel_epoch += 1
        abort = getattr(self.assistant.runtime, "abort_all", None)
        deadline = time.time() + CANCEL_WAIT_S
        while True:
            # Abort again on every round: a request that had acquired the lock but not yet reached the model when the
            # first abort ran (the question is still being prepared) would otherwise start afterwards and run to the end.
            if abort is not None:
                abort()
            if self._llm_lock.acquire(timeout=0.25):
                self._llm_lock.release()
                return
            if time.time() > deadline:
                return

    def _unload_model(self):
        """Free the model's memory while scanning (a loaded 4B model holds ~5 GB). Not an inference: keep_alive 0."""
        try:
            cfg = self.assistant.cfg
            self.assistant.runtime._post("/api/generate", {"model": cfg.model, "keep_alive": 0}, 10)
        except Exception:  # noqa: BLE001 - best effort; Ollama may be off or the runtime may be a test double
            pass

    def start_scan(self, stream, source, selector_mode="b0", interval_s=5.0):
        with self._scan_lock:
            if self.scan is not None and self.scan.active():
                raise Refused("A scan is already running. Finish it first.")
            blocker = self._reader_blocker()
            if blocker:
                raise Refused(blocker)
            # A new scan always wins: whatever the model is doing (a question, the context preparation) is stopped
            # first, because reading and answering never run at the same time.
            self.cancel_llm()
            session_id = "scan-" + datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir = self.library.root / session_id
            self.scan = ScanSession(session_id, stream, source, out_dir, selector_mode=selector_mode,
                                    engine_factory=self._engine_factory, pipeline_factory=self._pipeline_factory,
                                    on_finished=self._scan_finished, interval_s=interval_s)
            self.scan.start()
            threading.Thread(target=self._unload_model, name="web-unload-model", daemon=True).start()
            return self.scan

    def _scan_finished(self, session):
        """Called on the scan's shutdown thread once the document is frozen and saved."""
        if session.empty or not session.finalized:
            return
        self.library.forget(session.id)
        self._docs_version += 1
        self.prepare(session.id)

    def stop_scan(self):
        if self.scan is None or not self.scan.stop_scanning():
            raise Refused("There is no scan to stop.")

    def cancel_scan(self):
        if self.scan is None or not self.scan.cancel():
            raise Refused("There is nothing to cancel.")

    def dismiss_scan(self):
        """Leave the finished/failed scan screen. Refused while the scan is still doing work."""
        scan = self.scan
        if scan is None:
            return
        if scan.active():
            raise Refused("The scan is still running.")
        self.scan = None

    # ------------------------------------------------------------------ model gating
    def _llm_blocker(self, doc_id):
        scan = self.scan
        if scan is not None:
            if scan.active():
                return "A scan is in progress. Questions open once scanning has finished and every page has been read."
            if scan.id == doc_id:
                blocker = scan.llm_blocker()
                if blocker:
                    return blocker
        return None

    # ------------------------------------------------------------------ context preparation
    def prepare_state(self, doc_id):
        return self._prepare.get(doc_id)

    def prepare(self, doc_id, force=False):
        """Warm the model and review the first pages (the assistant caches the result, so the first question is fast)."""
        doc = self.library.get(doc_id)
        if doc is None:
            raise Refused("Unknown document.")
        blocker = self._llm_blocker(doc_id)
        if blocker:
            raise Refused(blocker)
        current = self._prepare.get(doc_id)
        if current and (current["status"] in ("running", "queued") or (current["status"] == "ready" and not force)):
            return current
        pages = pages_from_document(json.loads(doc.snapshot_json))
        cfg = self.assistant.cfg
        targets = pages[: cfg.max_recover_pages] if cfg.recover else []
        # Without per-page recovery the preparation is two steps: load the model, then read the whole document into it.
        status = {"status": "queued", "done": 0, "total": len(targets) or 2, "message": "Waiting for the model...",
                  "started_at": time.time(), "warnings": [], "ended_at": None}
        self._prepare[doc_id] = status
        threading.Thread(target=self._prepare_worker, args=(doc_id, targets, status, self._cancel_epoch),
                         name="web-prepare", daemon=True).start()
        return status

    def _prime_document(self, doc_id):
        """Read the whole document into the model once (prompt processing only, one output token).

        ask() sends 'DOCUMENT:<all pages>' followed by the question; Ollama reuses the processed prefix, so measured on
        a 12-page scan the first question drops from ~90 s to ~15 s. The model is not asked anything."""
        doc = self.library.get(doc_id)
        if doc is None:
            return
        pages = pages_from_document(json.loads(doc.snapshot_json))
        context = DocumentAssistant._context(pages, {})
        self.assistant._check_fits(context, self.assistant.cfg.max_answer_tokens, "document")
        try:
            self.assistant.runtime.chat(ANSWER_SYSTEM, f"DOCUMENT:\n{context}\n\nUSER REQUEST: ", ANSWER_SCHEMA, 1)
        except LLMError as exc:
            if exc.code != "output_truncated":         # one token is all that was asked for
                raise

    def _warm_model(self):
        """Load the model into memory without asking it anything (empty generate request)."""
        cfg = self.assistant.cfg
        post = getattr(self.assistant.runtime, "_post", None)
        if post is not None:
            post("/api/generate", {"model": cfg.model, "keep_alive": cfg.keep_alive}, cfg.timeout_s)

    def _prepare_worker(self, doc_id, targets, status, epoch):
        with self._llm_lock:
            if epoch != self._cancel_epoch:
                status.update(status="cancelled", message="Stopped because a new scan was started.", ended_at=time.time())
                return
            try:
                status.update(status="running", message="Loading the local AI model into memory...")
                health = self.check_llm()
                if health["status"] != "ok":
                    status.update(status="unavailable", message=health["message"], ended_at=time.time())
                    return
                try:
                    self._warm_model()
                    if not targets:
                        status.update(done=1, message="Reading the whole document into the model (about a minute for "
                                                      "a long document)...")
                        self._prime_document(doc_id)
                        status["done"] = 2
                except LLMError as exc:
                    if exc.code == "cancelled":
                        status.update(status="cancelled", message="Stopped because a new scan was started.",
                                      ended_at=time.time())
                        return
                    if exc.code == "context_too_large":
                        status["warnings"].append(exc.message)         # questions will report this themselves
                    elif exc.code in ("runtime_unavailable", "model_missing"):
                        status.update(status="unavailable", message=exc.message, ended_at=time.time())
                        return
                    else:
                        raise
                for page in targets:
                    status["message"] = f"Reviewing page {page['page']} for reading mistakes"
                    try:
                        self.assistant.recover_page(page)
                    except LLMError as exc:
                        if exc.code == "cancelled":
                            status.update(status="cancelled", message="Stopped because a new scan was started.",
                                          ended_at=time.time())
                            return
                        if exc.code in ("runtime_unavailable", "model_missing"):
                            self.llm.update(status="unavailable" if exc.code == "runtime_unavailable" else "model_missing",
                                            message=exc.message, checked_at=time.time())
                            status.update(status="unavailable", message=exc.message, ended_at=time.time())
                            return
                        status["warnings"].append(f"Page {page['page']}: {exc.message}")
                    status["done"] += 1
                status.update(status="ready", message="Ready", ended_at=time.time())
            except Exception as exc:  # noqa: BLE001
                status.update(status="failed", message=f"{type(exc).__name__}: {exc}", ended_at=time.time())

    # ------------------------------------------------------------------ questions
    def ask(self, doc_id, mode, question):
        doc = self.library.get(doc_id)
        if doc is None:
            raise Refused("Unknown document.")
        if mode not in ("answer", "summarize"):
            raise Refused("Unknown request type.")
        question = (question or "").strip()
        if mode == "answer" and not question:
            raise Refused("Type a question first.")
        blocker = self._llm_blocker(doc_id)
        if blocker:
            raise Refused(blocker)
        if self.llm["status"] in ("unavailable", "model_missing"):
            self.check_llm()                   # the user may have started Ollama since the last check
        epoch = self._cancel_epoch
        if not self._llm_lock.acquire(blocking=False):
            preparing = any(s["status"] in ("running", "queued") for s in self._prepare.values())
            raise Busy("The assistant is still preparing the document." if preparing
                       else "The assistant is still working on the previous request.")
        try:
            if epoch != self._cancel_epoch:
                raise Refused(STOPPED_BY_SCAN)
            self._busy = {"doc_id": doc_id, "mode": mode, "question": question, "started_at": time.time()}
            snapshot = json.loads(doc.snapshot_json)           # private copy; the stored text is never mutated
            result = self.assistant.ask(snapshot, question, mode=mode)
            if (result.get("error") or {}).get("code") == "cancelled" or epoch != self._cancel_epoch:
                raise Refused(STOPPED_BY_SCAN)                 # nothing is saved for a request that was cut off
            label = question if mode == "answer" else "Summarize this document"
            turn = doc.add_turn(mode, label, result)
            self._docs_version += 1
            return turn
        finally:
            self._busy = None
            self._llm_lock.release()

    # ------------------------------------------------------------------ reading a page again
    def reread_page(self, doc_id, page_number):
        """Read a page that failed (or came out blank) again from its saved photo, and put it into the document.

        Reading and answering never overlap, so this takes the model lock and is refused while a scan runs."""
        doc = self.library.get(doc_id)
        if doc is None:
            raise Refused("Unknown document.")
        state = doc.page_state(page_number)
        if state is None:
            raise Refused("That page does not exist in this document.")
        if state == "ok":
            raise Refused("That page has already been read.")
        if self.scan is not None and self.scan.active():
            raise Refused("A scan is running. Pages can be read again when it has finished.")
        blocker = self._reader_blocker()
        if blocker:
            raise Refused(blocker)
        photo = doc.page_image_path(page_number)
        frame = cv2.imread(str(photo)) if photo is not None else None
        if frame is None:
            raise Refused("The photo of that page is not available.")
        if not self._llm_lock.acquire(blocking=False):
            raise Busy("The assistant is busy. Try again when it has finished.")
        try:
            engine = self._engine_factory()
            if not hasattr(engine, "_read_page"):
                raise Refused("Reading a page again is not available with this page reader.")
            _n, text, _metrics, ocr_result, error = engine._read_page(page_number, frame,
                                                                       {"queue_entered_at": time.perf_counter()})
            if error is not None:
                raise Refused(f"Page {page_number} could not be read: {error}")
            engine_name = (ocr_result or {}).get("engine")
            reader = engine_name if engine_name in REMOTE_ENGINES else "local"
            if reader != "local":
                text, _removed = clean_text_lines(text)
            else:
                kept, removed = split_detections((ocr_result or {}).get("detections", []))
                if removed:
                    text = engine._reconstruct_layout(kept) if kept else ""
            result = doc.add_page_text(page_number, text,
                                       self._pipeline_factory or (lambda d: PostProcessingPipeline(export_dir=d)), reader)
            self._prepare.pop(doc_id, None)                    # the document changed: the model must read it again
            self._docs_version += 1
        finally:
            self._llm_lock.release()
        if result == "ok":
            self.prepare(doc_id)
        return {"status": result, "page": page_number}

    # ------------------------------------------------------------------ deleting scans
    def delete_document(self, doc_id):
        """Remove a scan (document, page photos and conversation) from disk. Asked for by the user in the UI."""
        if self.library.get(doc_id) is None:
            raise Refused("Unknown document.")
        if self.scan is not None and self.scan.active():
            raise Refused("A scan is running. Stop it before deleting scans.")
        if self._llm_lock.locked():
            self.cancel_llm()                  # the model may be working on this very document
        try:
            self.library.delete(doc_id)
        except (ValueError, OSError) as exc:
            raise Refused(f"Could not delete that scan: {exc}")
        self._prepare.pop(doc_id, None)
        self._docs_version += 1

    # ------------------------------------------------------------------ state for the browser
    def snapshot(self):
        scan = self.scan
        scan_view = scan.snapshot() if scan is not None else None
        prepare = {k: {key: v[key] for key in ("status", "done", "total", "message", "started_at", "ended_at", "warnings")}
                   for k, v in self._prepare.items()}
        busy = dict(self._busy) if self._busy else None
        reader = {**self.reader.public(), **self.reader_state, "offline_fallback": "local reader (this computer)"}
        return {"now": time.time(), "reader": reader, "llm": dict(self.llm),
                "busy": busy, "prepare": prepare,
                "docs_version": self._docs_version, "scan": scan_view,
                "limits": {"max_recover_pages": self.assistant.cfg.max_recover_pages,
                           "recover": self.assistant.cfg.recover}}
