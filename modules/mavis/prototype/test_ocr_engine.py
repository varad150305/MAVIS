import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

PROTOTYPE_ROOT = Path(__file__).resolve().parent
if str(PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT))

from ocr_engine import PrototypeOCREngine


def make_crop():
    return np.full((40, 120, 3), 255, dtype=np.uint8)


# ---- OCR result parsing (Phase 7 / 18B; RapidOCR since Phase 2 Step 2) ----

def make_rapidocr_result(boxes, txts, scores):
    """Stand-in for RapidOCR's RapidOCROutput: parallel boxes/txts/scores
    sequences, boxes as a list of 4-point quads."""
    return types.SimpleNamespace(boxes=boxes, txts=txts, scores=scores)


def test_parse_rapidocr_results_basic():
    result = make_rapidocr_result(
        boxes=[[[0, 0], [10, 0], [10, 5], [0, 5]]],
        txts=("Hello",),
        scores=(0.91,),
    )
    parsed = PrototypeOCREngine._parse_rapidocr_results(result)
    assert parsed == [{"bbox": [0, 0, 10, 5], "text": "Hello", "confidence": 0.91}]


def test_parse_rapidocr_results_none_result():
    assert PrototypeOCREngine._parse_rapidocr_results(None) == []


def test_parse_rapidocr_results_no_detections():
    # RapidOCR sets boxes/txts/scores to None (not empty sequences) when
    # nothing is detected in the frame.
    result = make_rapidocr_result(boxes=None, txts=None, scores=None)
    assert PrototypeOCREngine._parse_rapidocr_results(result) == []


def test_parse_rapidocr_results_skips_blank_text():
    result = make_rapidocr_result(
        boxes=[[[0, 0], [10, 0], [10, 5], [0, 5]]],
        txts=("",),
        scores=(0.91,),
    )
    assert PrototypeOCREngine._parse_rapidocr_results(result) == []


def test_parse_rapidocr_results_malformed_box_raises():
    # A quad point missing its y-coordinate is expected to raise here; the
    # worker_loop exception handler is what turns that into a graceful error
    # result instead of crashing anything - see
    # test_worker_continues_after_error below.
    result = make_rapidocr_result(
        boxes=[[[5]]],
        txts=("Hello",),
        scores=(0.91,),
    )
    with pytest.raises(IndexError):
        PrototypeOCREngine._parse_rapidocr_results(result)


# ---- OCR-result schema (P1-B): raw detections + raw_text preserved ----

def test_ocr_returns_none_crop_schema_with_empty_detections():
    engine = PrototypeOCREngine()
    ocr_result, inference_ms, layout_ms = engine.ocr(None)

    assert ocr_result == {"raw_text": "", "detections": []}
    assert inference_ms == 0.0
    assert layout_ms == 0.0


def test_ocr_preserves_raw_detections_alongside_raw_text(monkeypatch):
    PrototypeOCREngine._reader = None

    class FakeEngine:
        def __call__(self, crop):
            return make_rapidocr_result(
                boxes=[
                    [[0, 0], [50, 0], [50, 20], [0, 20]],
                    [[0, 25], [60, 25], [60, 45], [0, 45]],
                ],
                txts=("Hello", "world"),
                scores=(0.95, 0.88),
            )

    monkeypatch.setattr(
        PrototypeOCREngine, "_init_reader", classmethod(lambda cls: FakeEngine())
    )
    engine = PrototypeOCREngine()
    crop = np.full((50, 60, 3), 255, dtype=np.uint8)

    ocr_result, inference_ms, layout_ms = engine.ocr(crop)

    # Compact: exactly these two keys at the ocr() level (page_number is
    # attached later, by worker_loop, once the page is known).
    assert set(ocr_result.keys()) == {"raw_text", "detections"}
    assert ocr_result["raw_text"] == "Hello\nworld"
    assert ocr_result["detections"] == [
        {"bbox": [0, 0, 50, 20], "text": "Hello", "confidence": 0.95},
        {"bbox": [0, 25, 60, 45], "text": "world", "confidence": 0.88},
    ]


def test_worker_loop_preserves_ocr_result_schema_through_results_queue(monkeypatch):
    engine = PrototypeOCREngine()

    def fake_ocr(self, crop):
        return {
            "raw_text": "Line one\nLine two",
            "detections": [
                {"bbox": [0, 0, 10, 5], "text": "Line", "confidence": 0.9},
            ],
        }, 4.0, 0.5

    monkeypatch.setattr(PrototypeOCREngine, "ocr", fake_ocr)

    worker = threading.Thread(target=engine.worker_loop, daemon=True)
    worker.start()
    assert engine.submit(7, make_crop(), {})
    engine.stop()
    engine.wait()
    worker.join(timeout=5)

    page_number, text, metrics, ocr_result, error = engine.poll_result()

    assert page_number == 7
    assert error is None
    assert text == "Line one\nLine two"
    assert ocr_result["raw_text"] == text
    assert ocr_result["page_number"] == 7
    assert ocr_result["detections"] == [
        {"bbox": [0, 0, 10, 5], "text": "Line", "confidence": 0.9}
    ]
    # Schema stays compact and stable: exactly these three keys, no
    # duplicated text fields and no per-word top-level blocks.
    assert set(ocr_result.keys()) == {"page_number", "raw_text", "detections"}


def test_worker_loop_error_result_carries_no_ocr_result(monkeypatch):
    engine = PrototypeOCREngine()

    def fake_ocr(self, crop):
        raise RuntimeError("boom")

    monkeypatch.setattr(PrototypeOCREngine, "ocr", fake_ocr)

    worker = threading.Thread(target=engine.worker_loop, daemon=True)
    worker.start()
    assert engine.submit(1, make_crop(), {})
    engine.stop()
    engine.wait()
    worker.join(timeout=5)

    page_number, text, metrics, ocr_result, error = engine.poll_result()

    assert error is not None
    assert ocr_result is None
    assert text == ""


# ---- Reader lifecycle (Phase 5 / 18B) ----

def test_reader_initialized_once(monkeypatch):
    PrototypeOCREngine._reader = None
    init_calls = []

    class FakeRapidOCR:
        def __init__(self, *args, **kwargs):
            init_calls.append((args, kwargs))

    fake_module = types.ModuleType("rapidocr")
    fake_module.RapidOCR = FakeRapidOCR
    # _init_reader only uses these as opaque param values passed straight
    # into the params dict - any attribute-bearing stand-in is sufficient.
    fake_module.LangDet = types.SimpleNamespace(CH="ch", EN="en")
    fake_module.LangRec = types.SimpleNamespace(EN="en")
    fake_module.ModelType = types.SimpleNamespace(MOBILE="mobile", SMALL="small")
    fake_module.OCRVersion = types.SimpleNamespace(PPOCRV5="PP-OCRv5", PPOCRV4="PP-OCRv4")
    monkeypatch.setitem(sys.modules, "rapidocr", fake_module)

    try:
        reader_a = PrototypeOCREngine._init_reader()
        reader_b = PrototypeOCREngine._init_reader()

        assert reader_a is reader_b
        assert len(init_calls) == 1
    finally:
        PrototypeOCREngine._reader = None


# ---- worker_loop lifecycle / error handling (Phase 5, 18B, 18C) ----

def test_worker_continues_after_error(monkeypatch):
    engine = PrototypeOCREngine()
    call_count = {"n": 0}

    def fake_ocr(self, crop):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        return {"raw_text": "recovered text", "detections": []}, 5.0, 1.0

    monkeypatch.setattr(PrototypeOCREngine, "ocr", fake_ocr)

    worker = threading.Thread(target=engine.worker_loop, daemon=True)
    worker.start()

    assert engine.submit(1, make_crop(), {})
    assert engine.submit(2, make_crop(), {})
    engine.stop()
    engine.wait()
    worker.join(timeout=5)

    assert not worker.is_alive()

    results = {}
    while True:
        result = engine.poll_result()
        if result is None:
            break
        page_number, text, metrics, ocr_result, error = result
        results[page_number] = (text, ocr_result, error)

    assert results[1][2] is not None  # page 1 errored...
    assert results[1][1] is None      # ...and carries no ocr_result
    assert results[2][2] is None      # ...but the worker kept going
    assert results[2][0] == "recovered text"
    assert results[2][1]["raw_text"] == "recovered text"


def test_worker_processes_jobs_in_fifo_order():
    engine = PrototypeOCREngine()

    def fake_ocr(self, crop):
        return {"raw_text": "ok", "detections": []}, 1.0, 1.0

    PrototypeOCREngine_ocr_backup = PrototypeOCREngine.ocr
    PrototypeOCREngine.ocr = fake_ocr
    try:
        for page_number in range(1, 6):
            assert engine.submit(page_number, make_crop(), {})

        # Accepted jobs must keep draining even after stop() is called, as
        # long as work remains queued (accepted pages survive shutdown).
        engine.stop()

        worker = threading.Thread(target=engine.worker_loop, daemon=True)
        worker.start()
        engine.wait()
        worker.join(timeout=5)

        assert not worker.is_alive()

        order = []
        while True:
            result = engine.poll_result()
            if result is None:
                break
            order.append(result[0])

        assert order == [1, 2, 3, 4, 5]
    finally:
        PrototypeOCREngine.ocr = PrototypeOCREngine_ocr_backup


def test_queue_full_does_not_block_camera_and_increments_counter():
    from profiling import profiler

    engine = PrototypeOCREngine()
    before = profiler.counters.get("ocr_queue_full", 0)

    for page_number in range(1, engine.OCR_QUEUE_MAXSIZE + 1):
        assert engine.submit(page_number, make_crop(), {})

    # Queue is now full; submission must return False immediately instead
    # of blocking the caller (the camera thread).
    start = time.perf_counter()
    accepted = engine.submit(999, make_crop(), {})
    elapsed = time.perf_counter() - start

    assert accepted is False
    assert elapsed < 0.5
    assert profiler.counters.get("ocr_queue_full", 0) == before + 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
