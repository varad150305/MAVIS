"""M2 research-alignment pass: LocalAgreement-2 prefix confirmation.

Reference: Macháček, Dabre, Bojar, "Turning Whisper into Real-Time
Transcription System" (Whisper-Streaming, 2023). The LocalAgreement-n
policy "outputs the longest common prefix of the model on n consecutive
[updates]" - a plain, order-preserving STRING PREFIX comparison, not a
general alignment. Default n=2 ("LocalAgreement-2"), used here too
(`agreement_depth=2`) - the literature's own standard setting, not
independently tuned.

This is a LocalAgreement-style implementation aligned with the
Whisper-Streaming streaming formulation, not a claim of exact
reproduction - see the Phase 4C research-alignment report for the full
comparison, including the documented simplifications below.

`streaming.local_agreement.LocalAgreementTracker` (Phase 4-4C) is kept
UNCHANGED for historical evaluator reproducibility - it implements a
materially different algorithm (order-preserving LCS re-alignment
restricted to each pair's audio-time overlap, needed only because the
OLD `AdaptiveBuffer` produced disjoint/overlapping windows). This
module's `PrefixAgreementTracker` is what `main.py::run_m2()` now
drives, paired with `streaming.streaming_buffer.StreamingBuffer`.

============================================================
Why a plain prefix comparison suffices here (and LCS/overlap-alignment
does not)
============================================================

`StreamingBuffer` never produces disjoint windows: every window covers
the buffer's FULL currently-unconfirmed content, and the buffer is only
ever trimmed at a boundary this tracker itself confirms. Consequently,
for any two hypotheses belonging to the same utterance, comparing only
the words at or after the CURRENTLY confirmed boundary (`_unconfirmed_
tail`, computed from real Whisper word timestamps - see below) always
yields two sequences describing the *same* unconfirmed audio span
(modulo natural ASR variance), so a simple index-wise prefix match is
the correct comparison - there is no sliding-window overlap-restriction
problem to solve.

============================================================
Timestamps: measured, never interpolated
============================================================

Every timing decision here (which words are "confirmed" vs "not yet
seen by enough hypotheses", and the audio-time boundary passed to
`StreamingBuffer.trim_confirmed()`) uses `ASRHypothesis.words`
(`asr.whisper_engine.WordTiming`, MEASURED by faster-whisper when
`asr.word_timestamps=True`) - never `streaming.word_alignment`'s
interpolation (Tier (b), the Stage 4B decision for the OLD
LCS-based tracker). `streaming.word_alignment` is kept unchanged for
that old tracker's use; this module does not import it.

============================================================
Documented simplifications vs. the reference (see Phase 4C report)
============================================================

- The reference trims its audio buffer only at SENTENCE boundaries
  (punctuation + a language-specific sentence splitter), keeping some
  already-confirmed audio around for several updates. This tracker
  instead asks `StreamingBuffer.trim_confirmed()` to trim at EVERY
  newly confirmed boundary - simpler (no new sentence-segmentation
  dependency), keeps the buffer smaller on average, and is appropriate
  given this project's utterances are already VAD-bounded and short
  (seconds, not indefinite unsegmented speech) rather than a
  deviation from the core algorithmic contribution.
- The reference also carries the last ~200 confirmed words forward as
  Whisper's `initial_prompt` for cross-decode consistency - this is
  implemented (see `confirmed_prompt_text()`), not omitted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import threading
import time
from typing import TYPE_CHECKING, Deque

from evaluation.asr_evaluation import normalize_text
from streaming.local_agreement import CommittedText
from streaming.streaming_state import MonotonicIdGenerator

if TYPE_CHECKING:
    from streaming.streaming_asr import ASRHypothesis

#: Longest-common-prefix "tail" entry: (normalized word, absolute
#: audio-time start seconds, absolute audio-time end seconds).
_TailWord = tuple[str, float, float]


@dataclass
class _UtteranceState:
    agreement_depth: int
    prompt_word_limit: int

    confirmed_words: list[str] = field(default_factory=list)
    confirmed_boundary_time: float | None = None

    recent_tails: Deque[list[_TailWord]] = field(init=False)

    hypotheses_seen: int = 0
    failed_hypotheses_seen: int = 0
    revision_count: int = 0
    commit_events: list[CommittedText] = field(default_factory=list)
    latest_hypothesis: "ASRHypothesis | None" = None
    utterance_start_audio_time: float | None = None
    frozen: bool = False

    def __post_init__(self) -> None:
        self.recent_tails = deque(maxlen=self.agreement_depth)


def _normalized_tail(
    hypothesis: "ASRHypothesis", boundary_time: float | None
) -> list[_TailWord]:
    """Words from `hypothesis` at/after `boundary_time`, normalized.

    Absolute audio-time: `WordTiming.start`/`.end` are relative to the
    audio array passed to that decode call
    (`hypothesis.audio_start_time`), so absolute time =
    `hypothesis.audio_start_time + word.start` (unaffected by whether
    or when the buffer was physically trimmed - see module docstring).
    Words that normalize to empty (pure punctuation) are dropped.
    """

    floor = boundary_time if boundary_time is not None else float("-inf")

    tail: list[_TailWord] = []

    for word in hypothesis.words:
        abs_start = hypothesis.audio_start_time + word.start
        abs_end = hypothesis.audio_start_time + word.end

        if abs_start < floor:
            continue

        normalized = normalize_text(word.word)

        if not normalized:
            continue

        tail.append((normalized, abs_start, abs_end))

    return tail


def _common_prefix_length(sequences: list[list[_TailWord]]) -> int:
    """Longest prefix length shared by every sequence (word text only)."""

    if not sequences:
        return 0

    shortest = min(len(seq) for seq in sequences)

    length = 0
    for index in range(shortest):
        reference_word = sequences[0][index][0]
        if all(seq[index][0] == reference_word for seq in sequences[1:]):
            length += 1
        else:
            break

    return length


class PrefixAgreementTracker:
    """LocalAgreement-2-style stable-prefix commitment over a persistent buffer.

    One instance tracks many utterances (keyed by `utterance_id`), same
    lifecycle contract as `streaming.local_agreement.LocalAgreementTracker`:
    feed hypotheses via `process()` in any order; `finalize_utterance()`
    force-commits whatever remains and freezes the utterance.
    """

    def __init__(
        self,
        agreement_depth: int = 2,
        prompt_word_limit: int = 200,
        logger: logging.Logger | None = None,
    ) -> None:
        """Create a tracker.

        Args:
            agreement_depth: Consecutive successful hypotheses that
                must share a common prefix before it is confirmed.
                Default 2 ("LocalAgreement-2"), the literature's own
                standard setting - see module docstring.
            prompt_word_limit: Max trailing confirmed words returned by
                `confirmed_prompt_text()` - mirrors the reference's
                "last 200 words" `initial_prompt` carry-over.

        Raises:
            ValueError: For invalid configuration.
        """

        if agreement_depth < 2:
            raise ValueError(
                "agreement_depth must be at least 2 (LocalAgreement "
                "requires comparing at least 2 hypotheses)."
            )

        if prompt_word_limit < 0:
            raise ValueError("prompt_word_limit must not be negative.")

        self.agreement_depth = agreement_depth
        self.prompt_word_limit = prompt_word_limit
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self._lock = threading.Lock()
        self._states: dict[int, _UtteranceState] = {}
        self._commit_id_generator = MonotonicIdGenerator()

    def process(self, hypothesis: "ASRHypothesis") -> CommittedText | None:
        """Process one ASRHypothesis, possibly producing a commitment.

        Never raises for a missing/failed hypothesis - a failed
        hypothesis is recorded and explicitly logged (never silently
        discarded), but contributes no words. Safe to call with
        hypotheses in any order; a result for an already-frozen
        utterance is ignored (logged) rather than corrupting state.
        """

        with self._lock:
            state = self._states.setdefault(
                hypothesis.utterance_id,
                _UtteranceState(self.agreement_depth, self.prompt_word_limit),
            )

            if state.frozen:
                self.logger.warning(
                    "prefix_agreement: utterance_id=%d is frozen; "
                    "ignoring generation=%d",
                    hypothesis.utterance_id,
                    hypothesis.generation,
                )
                return None

            if state.utterance_start_audio_time is None:
                state.utterance_start_audio_time = hypothesis.audio_start_time

            if not hypothesis.success:
                state.failed_hypotheses_seen += 1

                self.logger.warning(
                    "prefix_agreement: missing hypothesis (ASR failure) | "
                    "utterance_id=%d generation=%d error=%s - surfaced "
                    "explicitly, not silently discarded.",
                    hypothesis.utterance_id,
                    hypothesis.generation,
                    hypothesis.error,
                )

                if hypothesis.is_final:
                    return self._finalize_locked(
                        state,
                        hypothesis.utterance_id,
                        reason=(
                            "final ASR generation "
                            f"{hypothesis.generation} failed: "
                            f"{hypothesis.error}"
                        ),
                    )

                return None

            state.hypotheses_seen += 1
            state.latest_hypothesis = hypothesis

            tail = _normalized_tail(hypothesis, state.confirmed_boundary_time)

            if state.recent_tails:
                previous_words = [w for w, _, _ in state.recent_tails[-1]]
                current_words = [w for w, _, _ in tail]
                agreed = _common_prefix_length([previous_words, current_words])
                if agreed < min(len(previous_words), len(current_words)):
                    state.revision_count += 1

            state.recent_tails.append(tail)

            commit = None

            if len(state.recent_tails) >= self.agreement_depth:
                common_length = _common_prefix_length(list(state.recent_tails))

                if common_length > 0:
                    newly_confirmed = tail[:common_length]

                    commit = self._commit_locked(
                        state,
                        hypothesis.utterance_id,
                        newly_confirmed,
                        generation=hypothesis.generation,
                        is_final=False,
                    )

                    remaining = tail[common_length:]
                    state.recent_tails.clear()
                    if remaining:
                        state.recent_tails.append(remaining)

            if hypothesis.is_final:
                final_commit = self._finalize_locked(
                    state, hypothesis.utterance_id
                )
                return final_commit if final_commit is not None else commit

            return commit

    def _commit_locked(
        self,
        state: _UtteranceState,
        utterance_id: int,
        newly_confirmed: list[_TailWord],
        generation: int,
        is_final: bool,
    ) -> CommittedText:
        """Apply a commitment. Caller must already hold `self._lock`."""

        new_words = [word for word, _, _ in newly_confirmed]
        state.confirmed_words.extend(new_words)

        if newly_confirmed:
            state.confirmed_boundary_time = newly_confirmed[-1][2]

        commit_event = CommittedText(
            utterance_id=utterance_id,
            commit_id=self._commit_id_generator.next(),
            generation=generation,
            new_text=" ".join(new_words),
            total_committed_text=" ".join(state.confirmed_words),
            committed_at=time.perf_counter(),
            is_final=is_final,
        )

        state.commit_events.append(commit_event)

        if commit_event.new_text:
            self.logger.info(
                "prefix_agreement: committed | utterance_id=%d "
                "generation=%d | new_text=%r | total_committed_text=%r",
                utterance_id,
                generation,
                commit_event.new_text,
                commit_event.total_committed_text,
            )

        return commit_event

    def finalize_utterance(
        self, utterance_id: int, reason: str | None = None
    ) -> CommittedText | None:
        """Force-commit whatever remains and freeze this utterance.

        Unlike `streaming.local_agreement.LocalAgreementTracker
        .finalize_utterance()` (which silently no-ops for an
        utterance_id it never saw - the confirmed root cause of Phase
        4C-3D/4C-3E's "ghost utterance" finding), this ALWAYS uses the
        latest available hypothesis if one exists, even one that never
        reached `agreement_depth` agreement - "handle finalization
        without losing the latest available hypothesis." If genuinely
        no hypothesis was EVER received for this utterance (every ASR
        call failed, or - structurally, with `StreamingBuffer`'s
        single-decode-in-flight design - literally none was ever
        submitted), that is explicitly logged and reflected in the
        returned commit's empty text, never silently swallowed.
        """

        with self._lock:
            return self._finalize_locked(
                self._states.setdefault(
                    utterance_id,
                    _UtteranceState(self.agreement_depth, self.prompt_word_limit),
                ),
                utterance_id,
                reason=reason,
            )

    def _finalize_locked(
        self,
        state: _UtteranceState,
        utterance_id: int,
        reason: str | None = None,
    ) -> CommittedText | None:
        if state.frozen:
            self.logger.warning(
                "prefix_agreement: utterance_id=%d already finalized; "
                "ignoring repeat finalize call",
                utterance_id,
            )
            return None

        if state.latest_hypothesis is None and state.hypotheses_seen == 0:
            self.logger.warning(
                "prefix_agreement: utterance_id=%d finalized with no "
                "successful hypothesis ever received (failed_hypotheses"
                "_seen=%d) - nothing to commit. reason=%s",
                utterance_id,
                state.failed_hypotheses_seen,
                reason,
            )

        # Whatever remains in the latest hypothesis's tail beyond the
        # confirmed boundary is committed as the forced tail - using
        # the LATEST hypothesis unconditionally, not requiring it to
        # have reached agreement_depth agreement.
        remaining_tail: list[_TailWord] = []
        if state.recent_tails:
            remaining_tail = state.recent_tails[-1]

        commit = self._commit_locked(
            state,
            utterance_id,
            remaining_tail,
            generation=(
                state.latest_hypothesis.generation
                if state.latest_hypothesis is not None
                else 0
            ),
            is_final=True,
        )

        state.frozen = True

        if reason is not None:
            self.logger.warning(
                "prefix_agreement: utterance_id=%d force-finalized | "
                "reason=%s | forced_tail=%r",
                utterance_id,
                reason,
                commit.new_text,
            )
        else:
            self.logger.info(
                "prefix_agreement: utterance_id=%d finalized | "
                "forced_tail=%r | total_committed_text=%r",
                utterance_id,
                commit.new_text,
                commit.total_committed_text,
            )

        return commit

    # ------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------

    def committed_text_for(self, utterance_id: int) -> str:
        with self._lock:
            state = self._states.get(utterance_id)
            return " ".join(state.confirmed_words) if state else ""

    def confirmed_boundary_time(self, utterance_id: int) -> float | None:
        """Real audio-time (seconds) up to which content is confirmed -
        what the caller should pass to `StreamingBuffer.trim_confirmed()`."""

        with self._lock:
            state = self._states.get(utterance_id)
            return state.confirmed_boundary_time if state else None

    def confirmed_prompt_text(self, utterance_id: int) -> str | None:
        """Last `prompt_word_limit` confirmed words, for Whisper's
        `initial_prompt` (reference's cross-decode consistency
        mechanism) - None if nothing is confirmed yet."""

        with self._lock:
            state = self._states.get(utterance_id)

            if state is None or not state.confirmed_words:
                return None

            return " ".join(state.confirmed_words[-self.prompt_word_limit :])

    def is_frozen(self, utterance_id: int) -> bool:
        with self._lock:
            state = self._states.get(utterance_id)
            return bool(state and state.frozen)

    def hypothesis_count(self, utterance_id: int) -> int:
        with self._lock:
            state = self._states.get(utterance_id)
            return state.hypotheses_seen if state else 0

    def failed_hypothesis_count(self, utterance_id: int) -> int:
        with self._lock:
            state = self._states.get(utterance_id)
            return state.failed_hypotheses_seen if state else 0

    def revision_count(self, utterance_id: int) -> int:
        with self._lock:
            state = self._states.get(utterance_id)
            return state.revision_count if state else 0

    def commitment_count(self, utterance_id: int) -> int:
        with self._lock:
            state = self._states.get(utterance_id)
            return len(state.commit_events) if state else 0

    def commit_events_for(self, utterance_id: int) -> tuple[CommittedText, ...]:
        with self._lock:
            state = self._states.get(utterance_id)
            return tuple(state.commit_events) if state else ()

    def stable_prefix_length(self, utterance_id: int) -> int:
        with self._lock:
            state = self._states.get(utterance_id)
            return len(state.confirmed_words) if state else 0

    def tentative_length(self, utterance_id: int) -> int:
        with self._lock:
            state = self._states.get(utterance_id)
            if state is None or not state.recent_tails:
                return 0
            return len(state.recent_tails[-1])

    def stable_token_ratio(self, utterance_id: int) -> float:
        """committed / (committed + tentative), bounded [0, 1] by
        construction - same definition as
        `streaming.local_agreement.LocalAgreementTracker
        .stable_token_ratio()`."""

        with self._lock:
            state = self._states.get(utterance_id)

            if state is None:
                return 0.0

            committed = len(state.confirmed_words)
            tentative = len(state.recent_tails[-1]) if state.recent_tails else 0
            denominator = committed + tentative

            return committed / denominator if denominator > 0 else 0.0

    def time_to_first_commit(self, utterance_id: int) -> float | None:
        with self._lock:
            state = self._states.get(utterance_id)

            if state is None or state.utterance_start_audio_time is None:
                return None

            first_commit = next(
                (event for event in state.commit_events if event.new_text),
                None,
            )

            if first_commit is None:
                return None

            return (
                first_commit.committed_at - state.utterance_start_audio_time
            )
