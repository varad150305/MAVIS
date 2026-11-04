"""M2 Phase 4: word-level timing estimation and overlap-restricted alignment.

Tier (b) per the Phase 4 design review. Whisper currently surfaces no
word-level timestamps: `configs/m2.yaml`'s `asr.word_timestamps` is
`false`, and `asr/whisper_engine.py` discards per-word timing even when
faster-whisper computes it (deliberately not touched in this phase - see
that module). Word timings here are therefore ESTIMATED by deterministic
linear interpolation over each hypothesis's known window audio span
(`ProcessingWindow.audio_start_time`/`audio_end_time`, reused unchanged
from Phase 1). These are NOT Whisper word timestamps. Every public name
in this module that produces a timing value says "estimated" for this
reason - never call it measured, observed, or a Whisper timestamp.

This module is pure and stateless: it has no knowledge of ASRHypothesis,
utterances, or commitment state. `streaming/local_agreement.py` adapts
`ASRHypothesis` objects into the primitives used here. Kept independent
so it is testable in complete isolation, per the Phase 4 implementation
order (word_alignment.py and its tests come before local_agreement.py).

Research-integrity note (see the Phase 4 design review and CLAUDE.md
Section 5): this is a LocalAgreement-style alignment ADAPTED to this
project's overlapping sliding-window hypothesis stream. It is NOT a
reproduction of Macháček et al.'s Whisper-Streaming method, which
compares cumulative, same-start-point hypotheses via a plain prefix
match. Our windows slide (Phase 2's AdaptiveBuffer, unchanged), so two
consecutive hypotheses are decodes of different, only partially
overlapping audio - comparing their raw text with a whole-string prefix
match is a category error, not merely a fragile heuristic. This module
exists to restrict comparison to the audio region the two hypotheses
actually share, and to align what's said there with an order-preserving
method instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TimedWord:
    """One normalized word with an ESTIMATED (not measured) timestamp.

    Attributes:
        word: Normalized word text (see
            `evaluation.asr_evaluation.normalize_text`/`word_tokens`,
            reused unchanged by callers - this module does not
            normalize text itself).
        index: 0-based position within the hypothesis's word list.
        estimated_time: Deterministic linear-interpolation estimate of
            this word's position in time, in the same clock/units as
            `ProcessingWindow.audio_start_time`/`audio_end_time`. NOT a
            Whisper word timestamp - Whisper does not currently provide
            one to this project (see module docstring).
    """

    word: str
    index: int
    estimated_time: float


def estimate_word_timings(
    words: Sequence[str],
    audio_start_time: float,
    audio_end_time: float,
) -> tuple[TimedWord, ...]:
    """Deterministically estimate each word's timing within [start, end].

    Divides the span into `len(words)` equal slots and assigns each word
    the midpoint of its slot. Pure and deterministic: identical inputs
    always produce identical outputs - no randomness, no Whisper call.

    Args:
        words: Already-normalized words, in order.
        audio_start_time: Start of the hypothesis's window audio span
            (`ProcessingWindow.audio_start_time`).
        audio_end_time: End of the hypothesis's window audio span
            (`ProcessingWindow.audio_end_time`).

    Returns:
        One TimedWord per input word, in the same order. Empty if
        `words` is empty.

    Raises:
        ValueError: If `audio_end_time` precedes `audio_start_time`.
    """

    if audio_end_time < audio_start_time:
        raise ValueError(
            "audio_end_time cannot precede audio_start_time."
        )

    word_count = len(words)

    if word_count == 0:
        return ()

    span = audio_end_time - audio_start_time
    slot = span / word_count

    return tuple(
        TimedWord(
            word=word,
            index=index,
            estimated_time=audio_start_time + (index + 0.5) * slot,
        )
        for index, word in enumerate(words)
    )


def restrict_to_range(
    timed_words: Sequence[TimedWord],
    range_start: float,
    range_end: float,
) -> tuple[str, ...]:
    """Return words whose estimated_time lies in [range_start, range_end].

    Inclusive on both ends, in original order. Used to restrict a
    hypothesis's words to an overlap region computed from window
    metadata (see `align_overlap`).
    """

    return tuple(
        timed.word
        for timed in timed_words
        if range_start <= timed.estimated_time <= range_end
    )


def order_preserving_lcs_indices(
    sequence_a: Sequence[str],
    sequence_b: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    """Return index pairs for the longest order-preserving common subsequence.

    Standard O(len(a) * len(b)) dynamic programming LCS over word
    *values* (the caller normalizes). This is deliberately NOT:
        - an unordered set/multiset intersection (would mis-handle
          repeated words and reordering),
        - fuzzy/substring similarity,
        - a prefix match.
    Matched words must appear in increasing order in both inputs, so a
    repeated phrase (e.g. "it was evening it was evening") cannot be
    matched out of order, and reordered words only match as much as
    their order actually permits.

    A plain DP table is used rather than a suffix tree or approximate
    LCS - inputs here are at most a few dozen words (one window's
    hypothesis), so this is simple, fast, and easy to verify. See the
    Phase 4 CPU profiling results (`LocalAgreementTracker`) before
    considering anything more elaborate; premature optimization here
    was explicitly avoided per the Phase 4 implementation instructions.

    Returns:
        A tuple of (index_in_a, index_in_b) pairs, in increasing order
        of both indices, one pair per matched word. Empty if either
        input is empty or nothing matches.
    """

    len_a = len(sequence_a)
    len_b = len(sequence_b)

    if len_a == 0 or len_b == 0:
        return ()

    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            if sequence_a[i - 1] == sequence_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    pairs: list[tuple[int, int]] = []
    i, j = len_a, len_b

    while i > 0 and j > 0:
        if sequence_a[i - 1] == sequence_b[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    pairs.reverse()

    return tuple(pairs)


def order_preserving_lcs(
    sequence_a: Sequence[str],
    sequence_b: Sequence[str],
) -> tuple[str, ...]:
    """Return the matched words (values only) of the order-preserving LCS.

    Convenience wrapper over `order_preserving_lcs_indices` for callers
    that only need the agreed word sequence, not the index alignment.
    """

    return tuple(
        sequence_a[i]
        for i, _ in order_preserving_lcs_indices(sequence_a, sequence_b)
    )


@dataclass(frozen=True)
class OverlapAlignment:
    """Result of aligning two hypotheses restricted to their shared overlap.

    Attributes:
        has_overlap: Whether a positive-duration overlap exists at all.
            When False, every other field is empty/zeroed and no
            agreement should be attempted by the caller.
        overlap_start: Start of the shared audio-time overlap (the
            later hypothesis's `audio_start_time`).
        overlap_end: End of the shared audio-time overlap (the earlier
            hypothesis's `audio_end_time`).
        previous_overlap_words: Words from the earlier hypothesis
            estimated to fall within the overlap (its tail).
        current_overlap_words: Words from the later hypothesis
            estimated to fall within the overlap (its head).
        aligned_words: The order-preserving LCS of the two word lists
            above - the words both hypotheses agree appear, in order,
            within the shared audio region. This is the adapted
            analogue of LocalAgreement's "common prefix", generalized
            from an exact prefix to an overlap-restricted subsequence
            because our windows are not nested (see module docstring).
    """

    has_overlap: bool
    overlap_start: float
    overlap_end: float
    previous_overlap_words: tuple[str, ...]
    current_overlap_words: tuple[str, ...]
    aligned_words: tuple[str, ...]


def align_overlap(
    previous_words: Sequence[str],
    previous_audio_start_time: float,
    previous_audio_end_time: float,
    current_words: Sequence[str],
    current_audio_start_time: float,
    current_audio_end_time: float,
) -> OverlapAlignment:
    """Restrict two hypotheses to their audio-time overlap and align them.

    `overlap_start = current_audio_start_time`,
    `overlap_end = previous_audio_end_time` - the later hypothesis's
    start and the earlier hypothesis's end, exactly per the Phase 4
    design review. If `overlap_end <= overlap_start` there is no usable
    overlap and `has_overlap` is False; the caller must not fall back
    to comparing the full hypotheses (that would silently reintroduce
    the naive whole-string comparison this module exists to avoid).
    """

    overlap_start = current_audio_start_time
    overlap_end = previous_audio_end_time

    if overlap_end <= overlap_start:
        return OverlapAlignment(
            has_overlap=False,
            overlap_start=overlap_start,
            overlap_end=overlap_end,
            previous_overlap_words=(),
            current_overlap_words=(),
            aligned_words=(),
        )

    previous_timed = estimate_word_timings(
        previous_words,
        previous_audio_start_time,
        previous_audio_end_time,
    )

    current_timed = estimate_word_timings(
        current_words,
        current_audio_start_time,
        current_audio_end_time,
    )

    previous_overlap_words = restrict_to_range(
        previous_timed,
        overlap_start,
        overlap_end,
    )

    current_overlap_words = restrict_to_range(
        current_timed,
        overlap_start,
        overlap_end,
    )

    aligned_words = order_preserving_lcs(
        previous_overlap_words,
        current_overlap_words,
    )

    return OverlapAlignment(
        has_overlap=True,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        previous_overlap_words=previous_overlap_words,
        current_overlap_words=current_overlap_words,
        aligned_words=aligned_words,
    )
