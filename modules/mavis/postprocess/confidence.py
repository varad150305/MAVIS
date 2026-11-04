from __future__ import annotations

import re

try:
    from wordfreq import top_n_list, zipf_frequency
except ImportError:  # pragma: no cover - optional dependency
    top_n_list = None
    zipf_frequency = None


class ConfidenceScorer:
    """Heuristic confidence scoring for OCR text."""

    _token_re = re.compile(r"\S+")
    _repeated_chars_re = re.compile(r"(.)\1{2,}")
    _fallback_common_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "of", "on", "or", "that", "the",
        "their", "there", "they", "this", "to", "was", "we", "were", "well",
        "with", "you", "your", "systems", "modern", "documents", "text",
    }

    def __init__(self):
        self.common_words = set(self._fallback_common_words)
        if top_n_list is not None:
            try:
                self.common_words.update(top_n_list("en", 5000))
            except Exception:
                pass

    def score_token(self, token: str) -> float:
        if not token:
            return 0.0

        stripped = token.strip()
        if not stripped:
            return 0.0

        score = 50.0
        length = len(stripped)
        alpha_count = sum(1 for ch in stripped if ch.isalpha())
        digit_count = sum(1 for ch in stripped if ch.isdigit())
        symbol_count = sum(1 for ch in stripped if not ch.isalnum())

        if length == 1:
            score -= 25.0
        else:
            score += min(length, 12) * 2.0

        symbol_ratio = symbol_count / length
        score -= symbol_ratio * 40.0

        if self._repeated_chars_re.search(stripped):
            score -= 15.0

        if alpha_count and digit_count:
            score -= 20.0

        if stripped.isupper() and len(stripped) > 4:
            score += 4.0
        elif any(ch.isalpha() for ch in stripped):
            score += 3.0

        word = re.sub(r"[^A-Za-z]", "", stripped).lower()
        if word:
            if word in self.common_words:
                score += 20.0
            elif zipf_frequency is not None:
                try:
                    score += max(0.0, min(20.0, (zipf_frequency(word, "en") - 2.0) * 5.0))
                except Exception:
                    pass

        return max(0.0, min(100.0, score))

    def score_line(self, line: str) -> float:
        tokens = self._token_re.findall(line or "")
        if not tokens:
            return 0.0
        token_scores = [self.score_token(token) for token in tokens]
        return sum(token_scores) / len(token_scores)

    def classify_confidence(self, score: float) -> str:
        if score >= 70.0:
            return "HIGH"
        if score >= 40.0:
            return "MEDIUM"
        return "LOW"
