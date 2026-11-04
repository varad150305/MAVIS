#!/usr/bin/env python3
"""Focused unit tests for the production WordfreqCorrector
(postprocess/wordfreq_correction.py) - the conservative, token-level
correction stage integrated from the postprocessing benchmark's winning
wordfreq candidate (postprocessing_benchmark/candidates/c05_wordfreq.py,
see postprocessing_benchmark/reports/FINAL_RECOMMENDATION.md).

Two things are tested separately:
  1. The token-shape gate (`is_correction_candidate`) and `correct_token` -
     unit-level, in isolation.
  2. `PostProcessingPipeline` wiring - that the pipeline actually uses this
     corrector by default, and that `enable_correction=False` disables it.
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

from postprocess.wordfreq_correction import WordfreqCorrector
from postprocess.pipeline import PostProcessingPipeline

TEST_EXPORT_DIR = BASE_DIR / "exports" / "test_wordfreq_correction"

# Tokens that must NEVER be modified, grouped by why - each is a real
# example encountered in postprocessing_benchmark/ or external_validation/
# where an earlier candidate (RapidFuzz, or wordfreq without these extra
# gates) corrupted it. See external_validation/reports/REPORT.md section 3.
UNCHANGED_ALL_CAPS_ABBREVIATIONS = ["BTECH", "CSE", "OLS", "GST", "IBAN", "MSCI", "EOD", "WHT", "RSP"]
UNCHANGED_NUMBERS = ["12345", "100", "2026"]
UNCHANGED_ALPHANUMERIC = ["AB1234", "OE407"]
UNCHANGED_SHORT_TOKENS = ["onc", "wth", "ltem", "Ne"]  # <5 chars - would otherwise be plausible fixes
UNCHANGED_PROPER_NOUNS_ADDRESSES = ["Rogahn", "Nagar", "Vidya", "Lorillard", "JALAN", "BANDAR", "BARU", "JAYA"]
UNCHANGED_TECHNICAL_BRAND_TOKENS = ["iShares", "COVID"]

# Genuine OCR-like misspellings the corrector is expected to fix - verified
# directly against this exact module (not assumed) before being hardcoded
# here. All lowercase, >=5 chars, not already recognized by wordfreq at
# zero frequency, with a real edit-distance-1 word clearing the margin.
EXPECTED_CORRECTIONS = {
    "droido": "droid",
    "statstic": "statistic",
    "frequeney": "frequency",
    "dataser": "dataset",
    "asserion": "assertion",
}


def run_gate_tests(corrector: WordfreqCorrector) -> None:
    for token in (
        UNCHANGED_ALL_CAPS_ABBREVIATIONS
        + UNCHANGED_NUMBERS
        + UNCHANGED_ALPHANUMERIC
        + UNCHANGED_SHORT_TOKENS
        + UNCHANGED_PROPER_NOUNS_ADDRESSES
        + UNCHANGED_TECHNICAL_BRAND_TOKENS
    ):
        assert not corrector.is_correction_candidate(token), (
            f"{token!r} must be excluded by is_correction_candidate() but was not"
        )


def run_unchanged_tests(corrector: WordfreqCorrector) -> None:
    for token in (
        UNCHANGED_ALL_CAPS_ABBREVIATIONS
        + UNCHANGED_NUMBERS
        + UNCHANGED_ALPHANUMERIC
        + UNCHANGED_SHORT_TOKENS
        + UNCHANGED_PROPER_NOUNS_ADDRESSES
        + UNCHANGED_TECHNICAL_BRAND_TOKENS
    ):
        result = corrector.correct_token(token)
        assert result["changed"] is False, f"{token!r} was changed to {result['corrected']!r}"
        assert result["corrected"] == token

    # Same guarantee must hold end-to-end through apply_rules() on a
    # realistic mixed line, not just token-by-token.
    line = "BTECH CSE GST OLS IBAN MSCI 12345 AB1234 Rogahn JALAN iShares"
    result = corrector.apply_rules(line)
    assert result["corrected_text"] == line, f"apply_rules changed a should-be-untouched line: {result['corrected_text']!r}"
    assert result["trace"] == []


def run_correction_tests(corrector: WordfreqCorrector) -> None:
    for original, expected in EXPECTED_CORRECTIONS.items():
        result = corrector.correct_token(original)
        assert result["changed"] is True, f"{original!r} was expected to be corrected but was left unchanged"
        assert result["corrected"] == expected, f"{original!r} -> {result['corrected']!r}, expected {expected!r}"

    # Mixed line: correctable misspellings fixed, untouchable tokens preserved.
    line = "The droido system uses statstic methods for BTECH and GST reporting."
    result = corrector.apply_rules(line)
    assert "droid" in result["corrected_text"]
    assert "statistic" in result["corrected_text"]
    assert "BTECH" in result["corrected_text"]
    assert "GST" in result["corrected_text"]
    assert len(result["trace"]) == 2  # only droido and statstic changed


def run_pipeline_integration_tests() -> None:
    """The pipeline must use WordfreqCorrector by default, and honor
    enable_correction=False as the documented configuration toggle."""
    TEST_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    text = "The droido system failed. BTECH CSE GST OLS IBAN remain unchanged."

    pipeline = PostProcessingPipeline(export_dir=TEST_EXPORT_DIR)
    assert isinstance(pipeline.corrector, WordfreqCorrector)
    page = pipeline.process_page(text, page_number=1)
    assert "droid" in page["corrected_text"]
    for token in ("BTECH", "CSE", "GST", "OLS", "IBAN"):
        assert token in page["corrected_text"]
    assert page["corrections_applied"] >= 1

    disabled_pipeline = PostProcessingPipeline(export_dir=TEST_EXPORT_DIR, enable_correction=False)
    disabled_page = disabled_pipeline.process_page(text, page_number=1)
    assert disabled_page["corrected_text"] == disabled_page["normalized_text"]
    assert disabled_page["corrections_applied"] == 0


def main() -> int:
    print("=" * 60)
    print("WORDFREQ CORRECTION TESTS")
    print("=" * 60)

    corrector = WordfreqCorrector()
    assert corrector.available, "wordfreq package must be installed for these tests to be meaningful"
    print("[OK] WordfreqCorrector available (wordfreq installed)")

    run_gate_tests(corrector)
    print("[OK] Token-shape gate tests passed (all-caps/numbers/alphanumeric/short/proper-noun/brand excluded)")

    run_unchanged_tests(corrector)
    print("[OK] Unchanged-token tests passed")

    run_correction_tests(corrector)
    print("[OK] Genuine-misspelling correction tests passed")

    run_pipeline_integration_tests()
    print("[OK] Pipeline integration tests passed (default-on, enable_correction=False toggle)")

    print("=" * 60)
    print("ALL WORDFREQ CORRECTION TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
