"""Cap the ONNX threads of the local OCR reader.

Measured on this machine (Core Ultra 5 125H, 18 logical CPUs, one dense 1280x720 page, same text out):
    default (all threads)  2.87 s     12 threads 2.89 s     8 threads 2.24 s     6 threads 1.77 s     4 threads 1.69 s
Using every logical CPU is slower on this hybrid CPU AND leaves nothing for the camera reader, the video encoder and the
browser, which is what made the live picture lag while a page was being read. 4 threads is faster and leaves 14 free.

prototype/ocr_engine.py is untouched: PrototypeOCREngine caches its reader on the class, so the web app builds that same
reader once, with the same parameters plus the thread caps, before the first page is read.
MAVIS_OCR_THREADS=0 restores the library default.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "prototype", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ocr_engine import PrototypeOCREngine  # noqa: E402

DEFAULT_THREADS = 4


def configured_threads(env=None):
    env = os.environ if env is None else env
    try:
        return int(env.get("MAVIS_OCR_THREADS", DEFAULT_THREADS))
    except ValueError:
        return DEFAULT_THREADS


def install_reader(threads=None):
    """Create PrototypeOCREngine's shared reader with capped threads. True if it installed one."""
    threads = configured_threads() if threads is None else threads
    if threads <= 0:
        return False
    from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR
    cls = PrototypeOCREngine
    with cls._init_lock:
        if cls._reader is not None:
            return False
        # identical to PrototypeOCREngine._init_reader, plus the two thread settings
        params = {
            "Det.lang_type": LangDet.CH, "Det.model_type": ModelType.MOBILE, "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.limit_side_len": cls.DET_LIMIT_SIDE_LEN, "Det.limit_type": cls.DET_LIMIT_TYPE,
            "Rec.lang_type": LangRec.EN, "Rec.model_type": ModelType.MOBILE, "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Global.model_root_dir": str(cls.MODEL_CACHE_DIR),
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
        start = time.perf_counter()
        cls._reader = RapidOCR(params=params)
        cls._init_time_ms = (time.perf_counter() - start) * 1000.0
    return True
