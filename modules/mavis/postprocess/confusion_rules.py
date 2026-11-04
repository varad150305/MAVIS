from __future__ import annotations

import re
from typing import List, Tuple

try:
    from rapidfuzz.distance import Levenshtein as RapidLevenshtein
except ImportError:  # pragma: no cover - optional dependency
    RapidLevenshtein = None

try:
    import Levenshtein as NativeLevenshtein
except ImportError:  # pragma: no cover - optional dependency
    NativeLevenshtein = None

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover - optional dependency
    zipf_frequency = None


class OCRConfusionResolver:
    """Conservative OCR confusion repair with traceability."""

    _fallback_common_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "of", "on", "or", "that", "the",
        "their", "there", "they", "this", "to", "was", "we", "were", "well",
        "with", "you", "your", "systems", "modern", "documents", "text",
    }

    _url_re = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
    _email_re = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    _phone_re = re.compile(r"^\+?[\d\s().-]{7,}$")
    _mixed_id_re = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d._/-]+$")
    _serial_re = re.compile(r"^[A-Z0-9]{2,}([-_/][A-Z0-9]{2,})+$", re.IGNORECASE)
    _token_re = re.compile(r"\w+|[^\w\s]+|\s+")

    def resolve_token(self, token: str, confidence: str | float = "MEDIUM") -> dict:
        if not token:
            return {"original": token, "corrected": token, "changed": False, "reasons": []}

        if not self.is_safe_to_modify(token):
            return {"original": token, "corrected": token, "changed": False, "reasons": ["unsafe-token"]}

        if self._confidence_value(confidence) < 0.4:
            return {"original": token, "corrected": token, "changed": False, "reasons": ["low-confidence"]}

        best_token = token
        best_score = self._frequency_score(token)
        reasons: List[str] = []

        for candidate, reason in self._generate_candidates(token):
            if candidate == token:
                continue
            if not self._within_edit_budget(token, candidate):
                continue

            candidate_score = self._frequency_score(candidate)
            if candidate_score > best_score + 0.4:
                best_token = candidate
                best_score = candidate_score
                reasons = [reason, "higher-frequency"]

        return {
            "original": token,
            "corrected": best_token,
            "changed": best_token != token,
            "reasons": reasons,
            "score": best_score,
        }

    def is_safe_to_modify(self, token: str) -> bool:
        if not token or len(token) <= 1:
            return False
        if self._url_re.match(token) or self._email_re.match(token) or self._phone_re.match(token):
            return False
        if self._mixed_id_re.match(token) or self._serial_re.match(token):
            return False
        if any(ch in token for ch in ["@", "://", "/", "\\"]):
            return False
        if any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token):
            return False
        return True

    def apply_rules(self, text: str, confidence: str | float = "MEDIUM") -> dict:
        if not text:
            return {"corrected_text": "", "trace": []}

        corrected_parts: List[str] = []
        trace: List[dict] = []
        for part in self._token_re.findall(text):
            if part.isspace() or not part.strip():
                corrected_parts.append(part)
                continue

            if part.isalpha() or self.is_safe_to_modify(part):
                result = self.resolve_token(part, confidence=confidence)
                corrected_parts.append(result["corrected"])
                trace.append(result)
            else:
                corrected_parts.append(part)
                trace.append({"original": part, "corrected": part, "changed": False, "reasons": ["preserved"]})

        return {"corrected_text": "".join(corrected_parts), "trace": trace}

    def _generate_candidates(self, token: str) -> List[Tuple[str, str]]:
        candidates = [(token, "original")]
        replacements = [
            ("rn", "m", "rn->m"),
            ("vv", "w", "vv->w"),
            ("cl", "d", "cl->d"),
            ("0", "o", "0->o"),
            ("1", "l", "1->l"),
            ("5", "s", "5->s"),
            ("8", "b", "8->b"),
        ]

        variants = {token}
        for old, new, reason in replacements:
            updated = set()
            for variant in variants:
                if old.lower() in variant.lower():
                    candidate = self._replace_preserving_case(variant, old, new)
                    updated.add(candidate)
                    candidates.append((candidate, reason))
            variants.update(updated)

        return candidates

    def _replace_preserving_case(self, token: str, old: str, new: str) -> str:
        pattern = re.compile(re.escape(old), re.IGNORECASE)

        def repl(match):
            original = match.group(0)
            if original.isupper():
                return new.upper()
            if original[:1].isupper():
                return new.capitalize()
            return new

        return pattern.sub(repl, token)

    def _frequency_score(self, token: str) -> float:
        word = re.sub(r"[^A-Za-z]", "", token).lower()
        if not word:
            return 0.0

        if zipf_frequency is not None:
            try:
                return float(zipf_frequency(word, "en"))
            except Exception:
                pass
        if word in self._fallback_common_words:
            return 5.0
        return 0.0

    def _within_edit_budget(self, original: str, candidate: str) -> bool:
        distance = self._edit_distance(original.lower(), candidate.lower())
        if len(original) <= 4:
            return distance <= 1
        return distance <= 2

    def _edit_distance(self, left: str, right: str) -> int:
        if RapidLevenshtein is not None:
            return int(RapidLevenshtein.distance(left, right))
        if NativeLevenshtein is not None:
            return int(NativeLevenshtein.distance(left, right))

        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, start=1):
            current = [i]
            for j, right_char in enumerate(right, start=1):
                insert_cost = current[j - 1] + 1
                delete_cost = previous[j] + 1
                replace_cost = previous[j - 1] + (left_char != right_char)
                current.append(min(insert_cost, delete_cost, replace_cost))
            previous = current
        return previous[-1]

    def _confidence_value(self, confidence: str | float) -> float:
        if isinstance(confidence, (int, float)):
            return float(confidence)
        value = str(confidence).upper()
        if value == "HIGH":
            return 0.9
        if value == "MEDIUM":
            return 0.6
        return 0.2
