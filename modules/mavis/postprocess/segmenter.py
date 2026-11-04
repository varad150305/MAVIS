from __future__ import annotations

import re
from typing import List


class OCRSegmenter:
    """Heuristic text segmenter for OCR output."""

    _noise_re = re.compile(r"[^A-Za-z0-9\s]")
    _caps_re = re.compile(r"^[A-Z0-9\s,.:;\-()'\"]{3,}$")

    def segment(self, text: str) -> dict:
        raw_lines = [line.rstrip() for line in text.splitlines()]
        headings = self.detect_headings(raw_lines)
        noise_blocks = self.detect_noise_blocks(raw_lines)
        paragraphs = self.reconstruct_paragraphs(raw_lines, headings, noise_blocks)

        return {
            "headings": headings,
            "paragraphs": paragraphs,
            "noise_blocks": noise_blocks,
            "raw_lines": raw_lines,
        }

    def detect_headings(self, lines: List[str]) -> List[str]:
        headings = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) <= 80 and (
                self._caps_re.match(stripped)
                or stripped.endswith(":")
                or re.match(r"^(\d+[.)]|[IVXLC]+[.)])\s+[A-Z]", stripped)
            ):
                headings.append(stripped)
        return headings

    def detect_noise_blocks(self, lines: List[str]) -> List[str]:
        noise_blocks = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            alpha_count = sum(1 for ch in stripped if ch.isalpha())
            symbol_count = sum(1 for ch in stripped if self._noise_re.match(ch))
            length = max(len(stripped), 1)
            symbol_ratio = symbol_count / length
            alpha_ratio = alpha_count / length
            if len(stripped) <= 3:
                noise_blocks.append(stripped)
            elif symbol_ratio > 0.35 and alpha_ratio < 0.45:
                noise_blocks.append(stripped)
            elif re.search(r"(.)\1{3,}", stripped):
                noise_blocks.append(stripped)
        return noise_blocks

    def reconstruct_paragraphs(self, lines: List[str], headings: List[str] | None = None, noise_blocks: List[str] | None = None) -> List[str]:
        headings = set(headings or [])
        noise_blocks = set(noise_blocks or [])
        paragraphs = []
        current = []

        def flush() -> None:
            if current:
                paragraphs.append(" ".join(current).strip())
                current.clear()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                flush()
                continue
            if stripped in headings or stripped in noise_blocks:
                flush()
                paragraphs.append(stripped)
                continue

            if current and not current[-1].endswith((".", "!", "?", ":", ";")) and stripped[:1].islower():
                current.append(stripped)
            elif current and current[-1].endswith("-"):
                current[-1] = current[-1][:-1] + stripped
            else:
                if current:
                    flush()
                current.append(stripped)

        flush()
        return paragraphs
