import re
import queue
import threading
import time
from pathlib import Path

import cv2

from profiling import profiler

import warnings
# Narrowly suppress known-harmless third-party noise (CPU-only builds warn
# about missing ccache and CPU pinned-memory on every import/init). Genuine
# runtime errors and warnings are never touched here - each filter matches
# one specific known message, not a whole category or module.
warnings.filterwarnings("ignore", message="No ccache found.*")
warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator.*")

DEBUG_OCR = False

class PrototypeOCREngine:
    """CPU-only RapidOCR (PP-OCRv5 mobile) adapter for the clean prototype."""

    OCR_QUEUE_MAXSIZE = 8
    RAPIDOCR_CONFIDENCE_THRESHOLD = 0.50
    OCR_PADDING_PX = 12
    CONFIDENCE_THRESHOLD = RAPIDOCR_CONFIDENCE_THRESHOLD
    # Config that produced the benchmarked numbers in
    # ocr_benchmark/docs/OCR_MODEL_DECISION.md (native resolution, not the
    # downscaled variant). Model cache is a production-owned copy of
    # ocr_benchmark/models/rapidocr/cache_5 - the benchmark artifacts under
    # ocr_benchmark/ are never read at runtime.
    MODEL_CACHE_DIR = Path(__file__).resolve().parent / "models" / "rapidocr" / "cache_5"
    DET_LIMIT_SIDE_LEN = 736
    DET_LIMIT_TYPE = "min"
    _reader = None
    _init_time_ms = None
    _init_lock = threading.Lock()

    def __init__(self):
        self._jobs = queue.Queue(maxsize=self.OCR_QUEUE_MAXSIZE)
        self._results = queue.Queue()
        self._stop_event = threading.Event()
        self._busy = threading.Event()

    def submit(self, page_number, crop, metrics):
        metrics["queue_entered_at"] = time.perf_counter()
        try:
            self._jobs.put_nowait((page_number, crop, metrics))
            metrics["ocr_queue_depth"] = self._jobs.qsize()
            profiler.sample("ocr_queue_depth", metrics["ocr_queue_depth"])
            return True
        except queue.Full:
            profiler.count("ocr_queue_full")
            print("[OCR QUEUE FULL]")
            return False

    def has_capacity(self):
        return not self._jobs.full()

    def queue_depth(self):
        return self._jobs.qsize()

    def is_processing(self):
        return self._busy.is_set()

    def worker_loop(self):
        while not self._stop_event.is_set() or not self._jobs.empty():
            try:
                page_number, crop, metrics = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue

            self._busy.set()
            try:
                preprocessing_start = time.perf_counter()
                prepared_crop, crop_metrics = self._prepare_crop(crop)
                crop_metrics["ocr_preprocessing_ms"] = (
                    time.perf_counter() - preprocessing_start
                ) * 1000.0
                metrics.update(crop_metrics)
                print(
                    f"[CAPTURE] page={page_number} "
                    f"crop={crop_metrics['crop_width']}x{crop_metrics['crop_height']} "
                    f"padded={crop_metrics['padded_width']}x{crop_metrics['padded_height']} "
                    f"scaled={crop_metrics['scaled_width']}x{crop_metrics['scaled_height']} "
                    f"queue={metrics['ocr_queue_depth']}/{self.OCR_QUEUE_MAXSIZE}"
                )
                metrics["ocr_started_at"] = time.perf_counter()
                metrics["queue_wait_ms"] = (
                    metrics["ocr_started_at"] - metrics["queue_entered_at"]
                ) * 1000.0
                ocr_result, inference_ms, layout_ms = self.ocr(prepared_crop)
                metrics["ocr_inference_ms"] = inference_ms
                metrics["layout_reconstruction_ms"] = layout_ms
                metrics["ocr_finished_at"] = time.perf_counter()
                # ocr() is page-agnostic (it only sees a crop), so the
                # page_number - the one piece of metadata that makes this
                # schema self-contained for a future postprocessing/NLP
                # consumer - is attached here, where it's actually known.
                ocr_result["page_number"] = page_number
                text = ocr_result["raw_text"]
                self._results.put((page_number, text, metrics, ocr_result, None))
            except Exception as exc:
                print(
                    f"[OCR ERROR] page={page_number} "
                    f"type={type(exc).__name__} "
                    f"error={exc}"
                )
                metrics["error"] = str(exc)
                metrics["ocr_finished_at"] = time.perf_counter()
                self._results.put((page_number, "", metrics, None, exc))
            finally:
                self._busy.clear()
                self._jobs.task_done()

    def poll_result(self):
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop_event.set()

    def wait(self):
        self._jobs.join()

    @classmethod
    def _init_reader(cls):
        if cls._reader is not None:
            return cls._reader
        with cls._init_lock:
            if cls._reader is not None:
                return cls._reader
            from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR
            params = {
                # PP-OCRv5 only ships a "ch" (language-agnostic) detector;
                # recognition is "en" since MAVIS targets English text.
                "Det.lang_type": LangDet.CH,
                "Det.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Det.limit_side_len": cls.DET_LIMIT_SIDE_LEN,
                "Det.limit_type": cls.DET_LIMIT_TYPE,
                "Rec.lang_type": LangRec.EN,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Global.model_root_dir": str(cls.MODEL_CACHE_DIR),
            }
            start = time.perf_counter()
            print("[OCR] Initializing RapidOCR (PP-OCRv5 mobile)...")
            cls._reader = RapidOCR(params=params)
            cls._init_time_ms = (time.perf_counter() - start) * 1000.0
            print("[OCR] RapidOCR ready")
            return cls._reader

    @staticmethod
    def _reconstruct_layout(ocr_data):
        blocks = []
        for result in ocr_data:
            confidence = result["confidence"]
            if confidence < PrototypeOCREngine.CONFIDENCE_THRESHOLD:
                continue
            bbox = result["bbox"]
            xs = [bbox[0], bbox[2]]
            ys = [bbox[1], bbox[3]]
            blocks.append({
                "text": result["text"].strip(),
                "cx": sum(xs) / 2,
                "cy": sum(ys) / 2,
                "h": ys[1] - ys[0],
            })
        if not blocks:
            return ""
        blocks.sort(key=lambda b: b['cy'])
        lines = [[blocks[0]]]
        for blk in blocks[1:]:
            anchor = lines[-1][0]
            threshold = 0.7 * max(blk['h'], anchor['h'])
            if abs(blk['cy'] - anchor['cy']) < threshold:
                lines[-1].append(blk)
            else:
                lines.append([blk])
        output = []
        for line in lines:
            line.sort(key=lambda b: b['cx'])
            output.append(" ".join(b['text'] for b in line))
        raw = "\n".join(output)
        return re.sub(r'-\n\s*', '', raw)

    @classmethod
    def _prepare_crop(cls, crop):
        """Pad the crop before OCR.

        Unlike the previous EasyOCR-tuned version, this no longer converts
        to grayscale or manually upscales narrow crops: RapidOCR expects a
        color BGR image and performs its own internal resize via
        Det.limit_side_len, so this reproduces the conditions the benchmark
        in ocr_benchmark/docs/OCR_MODEL_DECISION.md actually measured
        rather than a preprocessing pipeline tuned for a different engine.
        Padding is kept - it's a generic aid against edge-clipped text, not
        EasyOCR-specific. The returned metrics dict keeps every key the
        caller (main.py's [CAPTURE] log line) already expects.
        """
        height, width = crop.shape[:2]
        padded = cv2.copyMakeBorder(
            crop,
            cls.OCR_PADDING_PX,
            cls.OCR_PADDING_PX,
            cls.OCR_PADDING_PX,
            cls.OCR_PADDING_PX,
            cv2.BORDER_REPLICATE,
        )
        padded_height, padded_width = padded.shape[:2]
        return padded, {
            "crop_width": width,
            "crop_height": height,
            "padded_width": padded_width,
            "padded_height": padded_height,
            "scaled_width": padded_width,
            "scaled_height": padded_height,
        }

    @staticmethod
    def _parse_rapidocr_results(result):
        """Convert a RapidOCR RapidOCROutput into the internal representation:

            {bbox: [x1, y1, x2, y2], text: "...", confidence: float}

        RapidOCR returns parallel boxes/txts/scores sequences (quad points
        per box, like EasyOCR's polygon format) rather than EasyOCR's list
        of (polygon, text, confidence) tuples. `boxes`/`txts`/`scores` are
        all None when nothing was detected. Skips entries with empty text.
        """
        parsed = []
        if result is None or result.boxes is None or result.txts is None:
            return parsed
        for quad, text, confidence in zip(result.boxes, result.txts, result.scores):
            if not text:
                continue
            xs = [float(point[0]) for point in quad]
            ys = [float(point[1]) for point in quad]
            parsed.append({
                "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                "text": str(text).strip(),
                "confidence": float(confidence),
            })
        return parsed

    def ocr(self, crop):
        """Run one RapidOCR (PP-OCRv5 mobile) inference on a single document
        crop.

        Returns (ocr_result, inference_ms, layout_ms). ocr_result is the
        compact OCR-result schema - {"raw_text": str, "detections": [...]}
        - where each detection is the raw per-fragment
        {"bbox": [x1, y1, x2, y2], "text": str, "confidence": float}
        RapidOCR gave us. Both are preserved here rather than discarded
        after layout reconstruction, so a future postprocessing/NLP stage
        has the raw spatial/confidence data available, not just the
        flattened text.
        """
        if crop is None:
            return {"raw_text": "", "detections": []}, 0.0, 0.0

        engine = self._init_reader()
        profiler.count("ocr_calls")
        start = time.perf_counter()
        result = engine(crop)
        inference_ms = (time.perf_counter() - start) * 1000.0

        if DEBUG_OCR:
            print("[OCR DEBUG] result type:", type(result))
            print("[OCR DEBUG] boxes:", result.boxes)
            print("[OCR DEBUG] txts:", result.txts)
            print("[OCR DEBUG] scores:", result.scores)

        parsed = self._parse_rapidocr_results(result)

        layout_start = time.perf_counter()
        text = self._reconstruct_layout(parsed) if parsed else ""
        layout_ms = (time.perf_counter() - layout_start) * 1000.0
        return {"raw_text": text, "detections": parsed}, inference_ms, layout_ms


ocr_engine = PrototypeOCREngine()
