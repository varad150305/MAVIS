"""Ignore the DroidCam app's burned-in watermark ("using droidcam.app") in OCR output.

The watermark is part of the video frame, so OCR reads it as text, usually garbled ("usEringdraidcamap", "droiccam.app",
"usig roicamoth.ir"). It is removed per detected text box BEFORE the line layout is rebuilt, so it can neither appear as a
line of its own nor be glued onto a neighbouring sentence by the layout step. Two cases are handled:

  * a box that is only the watermark (fuzzy match on its letters, short boxes only) is dropped;
  * a box where OCR fused the watermark onto real text ("using droidcaControlling Our ...") loses just the watermark
    at its start or end, when that edge matches the watermark closely.

Matching is fuzzy on letters only and deliberately strict for the fused case, so ordinary text is left alone.
"""
import difflib
import re

TARGETS = ("usingdroidcamapp", "droidcamapp", "droidcam")
MAX_LETTERS = 28          # a watermark-only box is short; a long box is real text
MIN_RATIO = 0.64          # whole-box match (a box that is nothing but the watermark, however garbled)
EDGE_RATIO = 0.80         # edge match inside a longer box (stricter: real words sit next to it)
EDGE_MIN_LETTERS = 8
EDGE_MAX_CHARS = 30
LEFTOVER_PIECES = ("using", "app", "usingdroid", "cam", "camapp")
_STRIP = re.compile(r"[^a-z0-9]")


def _letters(text):
    return _STRIP.sub("", (text or "").lower())


def _ratio(letters):
    return max(difflib.SequenceMatcher(None, letters, t).ratio() for t in TARGETS)


def is_watermark(text):
    letters = _letters(text)
    if len(letters) < 5 or len(letters) > MAX_LETTERS:
        return False
    if "droidcam" in letters:
        return True
    return _ratio(letters) >= MIN_RATIO


def _edge_cut(text, from_start):
    """Number of characters at the start (or end) of `text` that look like the watermark; 0 if none."""
    best_ratio, best_k = 0.0, 0
    for k in range(6, min(len(text), EDGE_MAX_CHARS) + 1):
        piece = text[:k] if from_start else text[len(text) - k:]
        letters = _letters(piece)
        if len(letters) < EDGE_MIN_LETTERS:
            continue
        r = _ratio(letters)
        if r >= EDGE_RATIO and r >= best_ratio:       # ties go to the longer piece
            best_ratio, best_k = r, k
    return best_k


def strip_edges(text):
    """`text` without a watermark fused onto its start or end."""
    k = _edge_cut(text, True)
    if k:
        text = text[k:].lstrip()
    k = _edge_cut(text, False) if text else 0
    if k:
        text = text[:len(text) - k].rstrip()
    return text


def clean_text_lines(text):
    """(text, removed_count) for a transcription that has no per-box detections (the Gemini reader): the same rules,
    applied to each line."""
    out, removed = [], 0
    for line in (text or "").splitlines():
        if is_watermark(line):
            removed += 1
            continue
        cleaned = strip_edges(line)
        if cleaned != line:
            removed += 1
        if cleaned or not line.strip():
            out.append(cleaned)
    return "\n".join(out).strip("\n"), removed


def split_detections(detections):
    """(kept, removed) for a list of {"bbox", "text", "confidence"} OCR detections.

    `removed` lists every box that lost any watermark text; a box that only lost an edge is kept with its text cleaned.
    """
    kept, removed = [], []
    for d in detections:
        text = d.get("text", "")
        if is_watermark(text):
            removed.append(d)
            continue
        cleaned = strip_edges(text)
        if cleaned != text:
            removed.append(d)
            if cleaned:
                kept.append({**d, "text": cleaned})
        else:
            kept.append(d)
    if removed:
        # The watermark is letter-spaced, so OCR sometimes returns it in pieces. Once one piece was found on this
        # page, a box that is nothing but "using" or "app" is another piece.
        pieces = [d for d in kept if _letters(d.get("text", "")) in LEFTOVER_PIECES]
        if pieces:
            kept = [d for d in kept if d not in pieces]
            removed.extend(pieces)
    return kept, removed
