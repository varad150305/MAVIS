from __future__ import annotations

import re
from typing import List

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover - optional dependency, see requirements-wordfreq.txt
    zipf_frequency = None


class WordfreqCorrector:
    """Conservative, token-level OCR correction using `wordfreq` frequency
    data - the winning candidate from the postprocessing benchmark
    (postprocessing_benchmark/candidates/c05_wordfreq.py and
    postprocessing_benchmark/reports/FINAL_RECOMMENDATION.md), ported here
    as a self-contained production implementation (this module does not
    import anything from postprocessing_benchmark/, which is benchmark-only
    tooling with its own isolated venv, not a production dependency).

    Same core rule as the benchmark candidate: a token is only ever
    replaced if wordfreq does not already recognize it at all, and a
    single-edit real-word replacement exists whose own frequency clears a
    floor - "strong evidence the original is wrong and the candidate is
    right", not a fuzzy guess.

    Adds a stricter token-shape gate on top of that, beyond what the
    benchmark candidate used - see `is_correction_candidate`. The benchmark
    and its follow-up ground-truth validation
    (postprocessing_benchmark/reports/WORDFREQ_VS_RAPIDFUZZ_DEEP_DIVE.md,
    external_validation/reports/REPORT.md) found the "already recognized"
    check alone still lets through damage to short/ALL-CAPS/proper-noun
    tokens that happen to have zero frequency in wordfreq's corpus (e.g.
    technical terms, names) - so here, only plain lowercase tokens of
    meaningful length are ever considered at all, before frequency is even
    checked. Gating is entirely per-token; there is no line- or
    confidence-level gate (the same evidence showed line-level confidence
    gating protects this poorly - see that report's section 8).
    """

    MIN_TOKEN_LENGTH = 5
    MIN_ZIPF_MARGIN = 3.0

    _TOKEN_RE = re.compile(r"\w+|[^\w\s]+|\s+")
    _ALPHABET = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self):
        self._available = zipf_frequency is not None

    @property
    def available(self) -> bool:
        """False when the `wordfreq` package isn't installed - callers can
        check this, but correct_token()/apply_rules() also no-op safely on
        their own either way (never raises for a missing dependency)."""
        return self._available

    @classmethod
    def is_correction_candidate(cls, token: str) -> bool:
        """True only for a token this corrector will ever consider
        replacing. Deliberately excludes, before any frequency lookup:
          - anything not purely alphabetic (numbers, alphanumeric ids,
            tokens containing symbols/punctuation)
          - short tokens (< MIN_TOKEN_LENGTH characters)
          - ALL-CAPS tokens (abbreviation/acronym/code-shaped - BTECH, CSE,
            GST, OLS, IBAN, MSCI, ...)
          - any token with an uppercase letter after the first position
            (camelCase/branded tokens - iShares, McDonald, ...)
          - Title-Case tokens (first letter upper, rest lower - the shape
            of proper nouns, names, and addresses - Rogahn, Nagar, Vidya,
            Lorillard, ...)
        Only a plain, fully-lowercase alphabetic word of reasonable length
        passes - exactly the shape of an ordinary garbled prose word, not
        an acronym, code, brand, or name.
        """
        if not token or len(token) < cls.MIN_TOKEN_LENGTH:
            return False
        if not token.isalpha():
            return False
        if token.isupper():
            return False
        if any(ch.isupper() for ch in token[1:]):
            return False
        if token[:1].isupper() and token[1:].islower():
            return False
        return True

    def _edits1(self, word: str):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [left + right[1:] for left, right in splits if right]
        transposes = [left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1]
        replaces = [left + c + right[1:] for left, right in splits if right for c in self._ALPHABET]
        inserts = [left + c + right for left, right in splits for c in self._ALPHABET]
        return set(deletes + transposes + replaces + inserts)

    def correct_token(self, token: str) -> dict:
        """Returns {"original", "corrected", "changed", "reasons"} - same
        shape as OCRConfusionResolver.resolve_token() so it's a drop-in
        value at each call site, even though the two use unrelated logic."""
        if not self._available or not self.is_correction_candidate(token):
            return {"original": token, "corrected": token, "changed": False, "reasons": ["not-a-candidate"]}

        lower = token.lower()
        if zipf_frequency(lower, "en") > 0:
            return {"original": token, "corrected": token, "changed": False, "reasons": ["already-recognized"]}

        best_word, best_score = None, 0.0
        for candidate in self._edits1(lower):
            score = zipf_frequency(candidate, "en")
            if score > best_score:
                best_word, best_score = candidate, score

        if best_word is None or best_score < self.MIN_ZIPF_MARGIN:
            return {"original": token, "corrected": token, "changed": False, "reasons": ["no-strong-candidate"]}

        return {
            "original": token,
            "corrected": best_word,
            "changed": True,
            "reasons": [f"wordfreq:{best_score:.2f}"],
        }

    def apply_rules(self, text: str) -> dict:
        """Same return shape as OCRConfusionResolver.apply_rules() -
        {"corrected_text": str, "trace": [...]} - a drop-in replacement at
        the pipeline call site. `trace` holds only tokens that actually
        changed (unlike the confusion resolver's trace, which records every
        token including unchanged ones) - process_page only ever reads
        `item.get("changed")` off trace entries, so this is compatible."""
        if not text:
            return {"corrected_text": "", "trace": []}

        corrected_parts: List[str] = []
        trace: List[dict] = []
        for part in self._TOKEN_RE.findall(text):
            if part.isspace() or not part.strip():
                corrected_parts.append(part)
                continue
            if part.isalpha():
                result = self.correct_token(part)
                corrected_parts.append(result["corrected"])
                if result["changed"]:
                    trace.append(result)
            else:
                corrected_parts.append(part)

        return {"corrected_text": "".join(corrected_parts), "trace": trace}
