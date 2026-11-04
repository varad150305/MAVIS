"""Schema v2 of the OCR exports (raw_output.json / corrected_output.json / metadata.json): each fact stored once,
nothing lost, readers still work. Compares against the pre-cleanup exporter kept in _backups/."""
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from postprocess.formatter import OCRFormatter  # noqa: E402
from postprocess.pipeline import PostProcessingPipeline  # noqa: E402

SAMPLES = [
    "ACME Industrial Supplies Ltd.\nInvoice No. 4821\nTotal due: 419.58",
    "vvell systems and modern documents\nPayment is due within 30 days.",
    "  \n",                                           # whitespace-only page
    "Quarterly   Report\n\n\nNorth 120 150 180",       # normalization changes the spacing
    "using droidcam The quick brown fox",              # normalization strips the watermark
]


@pytest.fixture
def exported(tmp_path):
    pipeline = PostProcessingPipeline(export_dir=tmp_path)
    document = pipeline.process_document(SAMPLES)
    processing = {"status": "complete_with_errors", "pages_ok": len(SAMPLES), "pages_failed": [9]}
    paths = pipeline.export_results(document, processing=processing)
    load = lambda k: json.loads(Path(paths[k]).read_text(encoding="utf-8"))
    return document, paths, load("raw_output"), load("corrected_output"), load("metadata"), tmp_path


def test_each_fact_is_stored_once(exported):
    document, paths, raw, corrected, meta, _ = exported
    for payload in (raw, corrected, meta):
        assert payload["schema_version"] == 2 and payload["document_id"] == document["document_id"]
    assert "timestamp" not in raw and "timestamp" not in corrected and meta["timestamp"] == document["timestamp"]
    assert all(set(p) == {"page_number", "raw_text"} for p in raw["pages"])
    assert all("raw_text" not in p for p in corrected["pages"])                  # only copy lives in raw_output.json
    for p, original in zip(corrected["pages"], document["pages"]):
        assert ("normalized_text" in p) == (original["normalized_text"] != original["raw_text"])
    assert "document_id" not in meta["summary"] and "timestamp" not in meta["summary"]
    assert "export_directory" not in meta and "notes" not in meta
    assert meta["pipeline"] == {"correction": "wordfreq_token_level", "heuristic_only": True}


def test_processing_status_and_page_order_are_retained(exported):
    document, _, raw, corrected, meta, _ = exported
    assert meta["processing"] == {"status": "complete_with_errors", "pages_ok": 5, "pages_failed": [9]}
    nums = [p["page_number"] for p in raw["pages"]]
    assert nums == [p["page_number"] for p in corrected["pages"]] == list(range(1, len(SAMPLES) + 1))
    assert meta["summary"]["page_count"] == len(SAMPLES)


def test_default_processing_block_is_complete(tmp_path):
    pipeline = PostProcessingPipeline(export_dir=tmp_path)
    paths = pipeline.export_results(pipeline.process_document(["Hello world"]))
    meta = json.loads(Path(paths["metadata"]).read_text(encoding="utf-8"))
    assert meta["processing"] == {"status": "complete", "pages_ok": 1, "pages_failed": []}


def test_load_document_reproduces_every_page_field(exported):
    document, _, _, _, _, tmp = exported
    loaded = OCRFormatter.load_document(tmp)
    assert loaded["document_id"] == document["document_id"] and loaded["processing"]["pages_failed"] == [9]
    for page, original in zip(loaded["pages"], document["pages"]):
        for field in ("page_number", "raw_text", "normalized_text", "corrected_text", "confidence_score",
                      "confidence_label", "structure", "corrections_applied"):
            assert page[field] == original[field], field
    # also loadable from one of the files
    assert OCRFormatter.load_document(tmp / "corrected_output.json")["pages"] == loaded["pages"]


def test_exports_are_deterministic_for_the_same_input(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        pl = PostProcessingPipeline(export_dir=d)
        doc = pl.process_document(SAMPLES, document_id="doc_fixed")
        doc["timestamp"] = "2026-01-01T00:00:00"
        pl.export_results(doc)
    for name in ("raw_output.json", "corrected_output.json", "metadata.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_version1_exports_still_load(tmp_path):
    v1 = {"document_id": "d", "timestamp": "t", "pages": [
        {"page_number": 1, "raw_text": "Hello  world", "normalized_text": "Hello world", "corrected_text": "Hello world",
         "confidence_score": 60.0, "confidence_label": "MEDIUM", "structure": {}, "corrections_applied": 0}]}
    (tmp_path / "corrected_output.json").write_text(json.dumps(v1), encoding="utf-8")
    page = OCRFormatter.load_document(tmp_path)["pages"][0]
    assert page["raw_text"] == "Hello  world" and page["normalized_text"] == "Hello world"


def test_new_exports_are_smaller_than_the_pre_cleanup_exporter(tmp_path):
    backup = REPO / "_backups" / "pipeline.py.pre_cleanup"
    if not backup.exists():
        pytest.skip("pre-cleanup backup not available")
    import postprocess  # noqa: F401  (package must be imported for the relative imports)
    loader = importlib.machinery.SourceFileLoader("postprocess.pipeline_before_t", str(backup))
    spec = importlib.util.spec_from_loader("postprocess.pipeline_before_t", loader)
    old_mod = importlib.util.module_from_spec(spec)
    sys.modules["postprocess.pipeline_before_t"] = old_mod
    spec.loader.exec_module(old_mod)
    old = old_mod.PostProcessingPipeline(export_dir=tmp_path / "old")
    new = PostProcessingPipeline(export_dir=tmp_path / "new")
    size = lambda paths: sum(Path(p).stat().st_size for p in paths.values())
    s_old = size(old.export_results(old.process_document(SAMPLES * 3)))
    s_new = size(new.export_results(new.process_document(SAMPLES * 3)))
    assert s_new < s_old
