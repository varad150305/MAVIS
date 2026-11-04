from __future__ import annotations

import re
import unicodedata

try:
    from ftfy import fix_text
except ImportError:  # pragma: no cover - optional dependency
    fix_text = None


class OCRNormalizer:
    """Safely normalize OCR text without inventing content."""

    _quote_map = str.maketrans({
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "´": "'",
        "`": "'",
        "–": "-",
        "—": "-",
        "−": "-",
        "­": "",
        "\u0000": "",
    })

    _dangling_symbol_re = re.compile(r"(?m)[\s]*[|=~_]+[\s]*$")
    _repeat_punct_re = re.compile(r"([!?.,;:])\1{2,}")
    _repeat_symbol_re = re.compile(r"([\-_/\\])\1{2,}")
    _space_before_punct_re = re.compile(r"\s+([,.;:!?])")
    _extra_space_re = re.compile(r"[ \t]{2,}")
    _multi_newline_re = re.compile(r"\n{3,}")

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""

        cleaned = self.clean_unicode(text)
        cleaned = self.remove_control_chars(cleaned)
        cleaned = self.cleanup_symbols(cleaned)
        cleaned = self.normalize_spacing(cleaned)
        return cleaned.strip()

    def clean_unicode(self, text: str) -> str:
        if fix_text is not None:
            try:
                text = fix_text(text)
            except Exception:
                pass

        text = unicodedata.normalize("NFKC", text)
        return text.translate(self._quote_map)

    def remove_control_chars(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned_chars = []
        for char in text:
            if char in {"\n", "\t"}:
                cleaned_chars.append(char)
                continue
            if unicodedata.category(char).startswith("C"):
                continue
            cleaned_chars.append(char)
        return "".join(cleaned_chars)

    def cleanup_symbols(self, text: str) -> str:
        if not text:
            return ""

        text = self._repeat_punct_re.sub(lambda match: match.group(1) * 2, text)
        text = self._repeat_symbol_re.sub(lambda match: match.group(1), text)
        text = self._space_before_punct_re.sub(r"\1", text)
        text = self._dangling_symbol_re.sub("", text)
        text = re.sub(r"(?m)[ \t]+$", "", text)
        return text

    def normalize_spacing(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace("\u00A0", " ")
        text = self._extra_space_re.sub(" ", text)
        text = self._multi_newline_re.sub("\n\n", text)

        lines = []
        for line in text.splitlines():
            collapsed = self._extra_space_re.sub(" ", line).strip()
            lines.append(collapsed)

        return "\n".join(lines)
