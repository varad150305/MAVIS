"""OCRFormatter.load_document against representative v1 and v2 exports, including missing optional files/fields,
page-number gaps from failed pages, and a path that does not exist."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "prototype"))

from llm_stage import DocumentAssistant, LLMConfig  # noqa: E402
from postprocess.formatter import OCRFormatter  # noqa: E402
from postprocess.pipeline import PostProcessingPipeline  # noqa: E402


def write(d, name, obj):
    Path(d).mkdir(parents=True, exist_ok=True)
    (Path(d) / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def v1_export(d):
    pages = [
        {"page_number": 1, "raw_text": "Invoice  4821", "normalized_text": "Invoice 4821", "corrected_text": "Invoice 4821",
         "confidence_score": 70.0, "confidence_label": "HIGH", "structure": {"headings": 0}, "corrections_applied": 0},
        {"page_number": 2, "raw_text": "vvell  done", "normalized_text": "vvell done", "corrected_text": "well done",
         "confidence_score": 60.0, "confidence_label": "MEDIUM", "structure": {"headings": 0}, "corrections_applied": 1},
    ]
    write(d, "corrected_output.json", {"document_id": "doc_v1", "timestamp": "t", "pages": pages})
    write(d, "raw_output.json", {"document_id": "doc_v1", "timestamp": "t",
                                 "pages": [{"page_number": p["page_number"], "raw_text": p["raw_text"]} for p in pages]})
    write(d, "metadata.json", {"document_id": "doc_v1", "timestamp": "t", "summary": {"page_count": 2},
                               "export_directory": "C:/x", "heuristic_only": True, "notes": ["n"]})
    return pages


def v2_export_with_failed_page(d):
    pl = PostProcessingPipeline(export_dir=d)
    doc = pl.process_document(["First  page text", "Third page text"], document_id="doc_v2")
    for page, number in zip(doc["pages"], (1, 3)):          # capture numbering: page 2 failed OCR
        page["page_number"] = number
    pl.export_results(doc, processing={"status": "complete_with_errors", "pages_ok": 2, "pages_failed": [2]})
    return doc


def test_v1_export_loads_every_field_and_has_no_processing_block(tmp_path):
    pages = v1_export(tmp_path)
    doc = OCRFormatter.load_document(tmp_path)
    assert doc["document_id"] == "doc_v1" and "processing" not in doc
    assert [p["page_number"] for p in doc["pages"]] == [1, 2]
    for loaded, original in zip(doc["pages"], pages):
        for k, v in original.items():
            assert loaded[k] == v, k
        assert "raw_text_missing" not in loaded


def test_v1_single_file_loads_without_siblings(tmp_path):
    v1_export(tmp_path)
    doc = OCRFormatter.load_document(tmp_path / "corrected_output.json")
    assert [p["raw_text"] for p in doc["pages"]] == ["Invoice  4821", "vvell  done"]


def test_v2_export_with_a_failed_page_keeps_capture_numbering_and_status(tmp_path):
    original = v2_export_with_failed_page(tmp_path)
    doc = OCRFormatter.load_document(tmp_path)
    assert [p["page_number"] for p in doc["pages"]] == [1, 3]
    assert doc["processing"] == {"status": "complete_with_errors", "pages_ok": 2, "pages_failed": [2]}
    for loaded, orig in zip(doc["pages"], original["pages"]):
        for k in ("raw_text", "normalized_text", "corrected_text", "confidence_label", "corrections_applied"):
            assert loaded[k] == orig[k], k
    raw = json.loads((tmp_path / "raw_output.json").read_text(encoding="utf-8"))
    assert [p["page_number"] for p in raw["pages"]] == [1, 3]
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["processing"]["status"] == "complete_with_errors"


def test_missing_optional_pieces_do_not_crash_loading(tmp_path):
    # corrected only, no raw_output.json and no metadata.json (v2 pages carry no raw_text)
    write(tmp_path / "a", "corrected_output.json", {"schema_version": 2, "document_id": "d", "pages": [
        {"page_number": 1, "corrected_text": "Hello"}]})
    page = OCRFormatter.load_document(tmp_path / "a")["pages"][0]
    assert page["raw_text"] == "Hello" and page["raw_text_missing"] is True and page["normalized_text"] == "Hello"
    # raw only
    write(tmp_path / "b", "raw_output.json", {"document_id": "d", "pages": [{"page_number": 4, "raw_text": "Only raw"}]})
    page = OCRFormatter.load_document(tmp_path / "b")["pages"][0]
    assert page["page_number"] == 4 and page["raw_text"] == "Only raw" and "raw_text_missing" not in page
    # pages missing from raw_output.json are flagged individually
    write(tmp_path / "c", "corrected_output.json", {"document_id": "d", "pages": [
        {"page_number": 1, "corrected_text": "one"}, {"page_number": 2, "corrected_text": "two"}]})
    write(tmp_path / "c", "raw_output.json", {"document_id": "d", "pages": [{"page_number": 1, "raw_text": "ONE raw"}]})
    p1, p2 = OCRFormatter.load_document(tmp_path / "c")["pages"]
    assert p1["raw_text"] == "ONE raw" and "raw_text_missing" not in p1
    assert p2["raw_text_missing"] is True
    # empty pages list, no files at all, unknown extra fields and debug fields survive
    write(tmp_path / "d", "corrected_output.json", {"document_id": "d", "pages": []})
    assert OCRFormatter.load_document(tmp_path / "d")["pages"] == []
    (tmp_path / "e").mkdir()
    assert OCRFormatter.load_document(tmp_path / "e") == {"document_id": None, "pages": []}
    write(tmp_path / "f", "corrected_output.json", {"document_id": "d", "pages": [
        {"page_number": 1, "corrected_text": "x", "segmentation": {"raw_lines": []}, "future_field": 7}]})
    page = OCRFormatter.load_document(tmp_path / "f")["pages"][0]
    assert page["segmentation"] == {"raw_lines": []} and page["future_field"] == 7


def test_a_path_that_does_not_exist_is_an_error_not_the_parent_directorys_export(tmp_path):
    v1_export(tmp_path)                                            # a valid export sits in the parent
    with pytest.raises(FileNotFoundError):
        OCRFormatter.load_document(tmp_path / "typo_dir")
    with pytest.raises(FileNotFoundError):
        OCRFormatter.load_document(tmp_path / "typo.json")


def test_missing_raw_text_is_reported_to_the_user_by_the_llm_adapter(tmp_path):
    write(tmp_path, "corrected_output.json", {"document_id": "d", "pages": [{"page_number": 1, "corrected_text": "Hello there"}]})
    doc = OCRFormatter.load_document(tmp_path)

    class Rt:
        def chat(self, system, user, schema, max_tokens):
            body = ({"corrected_text": "Hello there", "uncertain_spans": []} if "repair OCR" in system else
                    {"answerable_from_document": True, "answer": "Hello", "general_knowledge": "",
                     "evidence": [{"page": 1, "quote": "Hello there"}]})
            return json.dumps(body), {"prompt_tokens": 1, "output_tokens": 1, "output_s": 1, "prompt_s": 1,
                                      "total_s": 1, "load_s": 0, "done_reason": "stop"}
    r = DocumentAssistant(LLMConfig(), Rt()).ask(doc, "greeting?")
    assert r["error"] is None and any("original OCR text is not in this export" in w for w in r["warnings"])


def test_failed_pages_reach_the_llm_as_missing_not_as_invented_text(tmp_path):
    v2_export_with_failed_page(tmp_path)
    doc = OCRFormatter.load_document(tmp_path)
    seen = {}

    class Rt:
        def chat(self, system, user, schema, max_tokens):
            if "repair OCR" in system:
                body = {"corrected_text": user.split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0], "uncertain_spans": []}
            else:
                seen["prompt"] = user
                body = {"answerable_from_document": False, "answer": "The captured document does not say.",
                        "general_knowledge": "", "evidence": []}
            return json.dumps(body), {"prompt_tokens": 1, "output_tokens": 1, "output_s": 1, "prompt_s": 1,
                                      "total_s": 1, "load_s": 0, "done_reason": "stop"}
    r = DocumentAssistant(LLMConfig(), Rt()).ask(doc, "what is on page 2?")
    assert "=== PAGE 1 ===" in seen["prompt"] and "=== PAGE 3 ===" in seen["prompt"] and "=== PAGE 2 ===" not in seen["prompt"]
    assert any("page(s) [2]" in w and "not part of this document" in w for w in r["warnings"])
