#!/usr/bin/env python3
"""Validation tests for the MAVIS V2 postprocessing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from postprocess.formatter import OCRFormatter
from postprocess.pipeline import PostProcessingPipeline
from postprocess.normalizer import OCRNormalizer
from postprocess.segmenter import OCRSegmenter
from postprocess.confidence import ConfidenceScorer
from postprocess.confusion_rules import OCRConfusionResolver


BASE_DIR = Path(__file__).resolve().parent
CONTEXT_BANK = BASE_DIR / "context_bank"
TEST_EXPORT_DIR = BASE_DIR / "exports" / "test_postprocessing"


def load_real_context_samples() -> list[str]:
    samples = []
    for path in sorted(CONTEXT_BANK.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            content = payload.get("content")
            if content:
                samples.append(content)
        except Exception as exc:
            print(f"[WARN] Skipping {path.name}: {exc}")
    return samples


def run_normalizer_tests(normalizer: OCRNormalizer) -> None:
    assert normalizer.normalize_text("whllee   expenditure=") == "whllee expenditure"
    assert normalizer.normalize_text("“smart quotes”") == '"smart quotes"'
    assert normalizer.normalize_text("line\x00with\x1fcontrols") == "linewithcontrols"


def run_segmenter_tests(segmenter: OCRSegmenter) -> None:
    sample = "TITLE\n\nFirst line of paragraph\ncontinues here.\n\n###\n"
    result = segmenter.segment(sample)
    assert "TITLE" in result["headings"]
    assert result["paragraphs"]
    assert result["raw_lines"]


def run_confidence_tests(confidence: ConfidenceScorer) -> None:
    high = confidence.score_line("This is a readable sentence with proper words.")
    low = confidence.score_line("1 l | 8 5 @@")
    assert confidence.classify_confidence(high) in {"HIGH", "MEDIUM"}
    assert confidence.classify_confidence(low) == "LOW"


def run_confusion_tests(resolver: OCRConfusionResolver) -> None:
    safe = resolver.resolve_token("vvell", confidence="HIGH")
    assert safe["corrected"] == "well"

    preserved = resolver.resolve_token("AB1234", confidence="HIGH")
    assert preserved["corrected"] == "AB1234"

    text_result = resolver.apply_rules("vvell systems and AB1234", confidence="HIGH")
    assert "well" in text_result["corrected_text"]
    assert "AB1234" in text_result["corrected_text"]


def run_pipeline_tests(samples: list[str]) -> None:
    TEST_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = PostProcessingPipeline(export_dir=TEST_EXPORT_DIR)
    document = pipeline.process_document(samples or ["[V2 OCR unreachable - check paddleocr installation]"])
    exports = pipeline.export_results(document)

    for key, value in exports.items():
        assert Path(value).exists(), f"Missing export: {key}"

    assert document["pages"]
    assert document["summary"]["page_count"] == len(document["pages"])

    with open(exports["corrected_output"], encoding="utf-8") as handle:
        corrected_payload = json.load(handle)

    for page in corrected_payload["pages"]:
        assert "segmentation" not in page, "default export must not carry full segmentation"
        assert "correction_trace" not in page, "default export must not carry the per-token trace"
        for field in ("page_number", "corrected_text", "confidence_score",
                      "confidence_label", "structure", "corrections_applied"):
            assert field in page, f"corrected export missing expected field: {field}"
        # schema v2: raw_text is stored once, in raw_output.json
        assert "raw_text" not in page, "raw_text must live only in raw_output.json"

    with open(exports["raw_output"], encoding="utf-8") as handle:
        raw_payload = json.load(handle)
    assert [p["page_number"] for p in raw_payload["pages"]] == [p["page_number"] for p in corrected_payload["pages"]]
    assert all("raw_text" in p for p in raw_payload["pages"])

    # the merged view still has every field the pre-v2 corrected export had
    loaded = OCRFormatter.load_document(TEST_EXPORT_DIR)
    for page, original in zip(loaded["pages"], document["pages"]):
        for field in ("raw_text", "normalized_text", "corrected_text", "confidence_score",
                      "confidence_label", "structure", "corrections_applied"):
            assert page[field] == original[field], f"load_document lost/changed {field}"


def run_debug_mode_tests(samples: list[str]) -> None:
    """debug=True is the only way to get the full per-token trace / nested
    segmentation - it must still be available, just opt-in."""
    pipeline = PostProcessingPipeline(export_dir=TEST_EXPORT_DIR)
    text = (samples or ["vvell systems and modern documents"])[0]

    default_page = pipeline.process_page(text, page_number=1)
    assert "segmentation" not in default_page
    assert "correction_trace" not in default_page

    debug_page = pipeline.process_page(text, page_number=1, debug=True)
    assert "segmentation" in debug_page
    assert "correction_trace" in debug_page
    assert debug_page["corrected_text"] == default_page["corrected_text"]
    assert debug_page["corrections_applied"] == default_page["corrections_applied"]

    debug_document = pipeline.process_document([text], debug=True)
    debug_exports = pipeline.export_results(debug_document)
    with open(debug_exports["corrected_output"], encoding="utf-8") as handle:
        debug_payload = json.load(handle)
    assert "segmentation" in debug_payload["pages"][0]
    assert "correction_trace" in debug_payload["pages"][0]


def main() -> int:
    print("=" * 60)
    print("MAVIS V2 POSTPROCESSING TESTS")
    print("=" * 60)

    samples = load_real_context_samples()
    print(f"[OK] Loaded {len(samples)} real context_bank sample(s)")

    normalizer = OCRNormalizer()
    segmenter = OCRSegmenter()
    confidence = ConfidenceScorer()
    resolver = OCRConfusionResolver()

    run_normalizer_tests(normalizer)
    print("[OK] Normalizer tests passed")

    run_segmenter_tests(segmenter)
    print("[OK] Segmenter tests passed")

    run_confidence_tests(confidence)
    print("[OK] Confidence tests passed")

    run_confusion_tests(resolver)
    print("[OK] Conservative correction tests passed")

    run_pipeline_tests(samples)
    print("[OK] Pipeline export tests passed")

    run_debug_mode_tests(samples)
    print("[OK] Debug-mode export tests passed")

    print("=" * 60)
    print("ALL POSTPROCESSING TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
