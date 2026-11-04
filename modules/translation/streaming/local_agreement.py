"""M2 Phase 4 (corrected, Stage 4A): LocalAgreement-style stable-prefix commitment.

This module is a LOCALAGREEMENT-STYLE STABLE-PREFIX COMMITMENT LAYER
ADAPTED to this project's overlapping sliding-window ASR hypothesis
stream (Phase 2's `AdaptiveBuffer`, unchanged; Phase 3's
`StreamingASRDispatcher`, unchanged). It is NOT a reproduction of
Macháček et al.'s Whisper-Streaming LocalAgreement, which compares
cumulative, same-start-point hypotheses by plain prefix matching - our
windows slide instead of growing from a fixed start, so comparison is
restricted to each pair's actual audio-time overlap and aligned with an
order-preserving subsequence method (`streaming.word_alignment`) rather
than a whole-string prefix match. Do not describe this module as the
original Whisper-Streaming algorithm; it is a correctness/architecture
correction for this project's M2 baseline, not a novel contribution.

Word timings used for alignment are ESTIMATED by linear interpolation
(Tier (b); see `streaming.word_alignment`), not measured Whisper word
timestamps - Whisper currently surfaces none to this project.

Stability is NOT correctness. A word becoming "committed" here means it
has survived `agreement_depth` consecutive overlap-restricted
comparisons - nothing more. A consistently repeated ASR error can and
will become committed. This module never labels its output "correct"
or "verified"; call it stable, committed, or locally agreed. Accuracy
is a separate concern, measured via `committed_transcript_error_rate`
(which reuses the project's existing WER/CER implementation unchanged)
against ground truth - never invented here.

============================================================
Stage 4A correction: persistent tentative state
============================================================

The original Phase 4 implementation only ever compared the last
`agreement_depth` raw hypotheses, restricted to their mutual audio-time
overlap. Any word that appeared in an earlier hypothesis but fell
entirely outside the *next* hypothesis's audio span (because the
sliding window had already moved past it) was never represented in any
later comparison and was never reconciled at finalization either
(finalization only used the single latest hypothesis). It was silently
and permanently lost - confirmed against real microphone logs, where
final committed transcripts were disconnected fragments like
"of translation" or "how system continuous".

The fix: each utterance now carries a PERSISTENT `tentative` word list
that survives across arbitrarily many hypotheses, not just the last
`agreement_depth` of them. On each new hypothesis H with audio span
[S, E], the existing tentative content is split by audio time into:

    A. words with estimated_time < S - NOT covered by H at all; carried
       forward completely unchanged (this is the fix - previously this
       content vanished when the hypothesis that produced it aged out
       of a fixed-size window).
    B. words with estimated_time >= S - covered by both the previous
       hypothesis and H; re-aligned via the SAME order-preserving LCS
       used before, restricted to the overlap region. A word survives
       here only if H's overlap-restricted words still agree with it;
       otherwise it is revised away (this is legitimate revision based
       on newer evidence, not loss - the rule only protects content H
       cannot see at all).
    C. H's words beyond the previous hypothesis's end - genuinely new
       content, appended as fresh tentative material.

A word is promoted from tentative to committed only when it is part of
the leading CONTIGUOUS run of tentative words (immediately following
whatever is already committed) whose per-word agreement streak has
reached `agreement_depth`. This preserves "never commit a later word
while an earlier one remains unresolved" by construction: a single
low-streak word anywhere in the tentative list blocks promotion of
everything after it, exactly as intended.

Finalization simply force-commits whatever remains in `tentative` in
full, exactly once - no separate LCS-remainder reconstruction is
needed anymore, because `tentative` is now guaranteed to already hold
every not-yet-committed word regardless of how many hypotheses ago it
first appeared.

This module does not call Whisper, does not change how many ASR calls
happen, does not modify `asr/worker.py` or `asr/whisper_engine.py`, and
does not implement LocalAgreement's literal algorithm, translation
gating, or M3 scheduling. It is a lightweight downstream consumer of
`streaming.streaming_asr.ASRHypothesis` objects.

============================================================
Phase 4C-1: canonical metric names (observability only)
============================================================

Two different, previously same-named "revision count" quantities exist
in this codebase and must never be reported as one undifferentiated
"revision_count" in Phase 4C evaluation output:

    - `hypothesis_revision_count` = `streaming.streaming_asr
      .StreamingASRDispatcher.revision_count()` - observation-only,
      whole-hypothesis-text inequality between consecutive admitted
      generations. Never used for any LocalAgreement decision.
    - `alignment_revision_count` = `LocalAgreementTracker.revision_count()`
      (this class's method, below) - counts admitted generations whose
      overlap-restricted re-alignment discarded at least one previously-
      tentative or newly-hypothesized word. This is the quantity Phase 4
      always computed; only the name was ambiguous.

Neither method's implementation changed for Phase 4C-1 - only the name
under which each is reported. The pre-existing `revision_count` field
in the `local_agreement_utterance_summary` JSONL event (produced by
`main.py::run_m2`) is kept unchanged for backward compatibility with
historical logs (e.g. `evaluation/stage4a_replay.py` reads it); it is
always numerically identical to `alignment_revision_count`, which is
additionally logged under its own explicit name going forward. Do not
write bare "revision_count" in any new report or table - always say
which of the two is meant.

`commitment_delay_ms` (see `commitment_delays_ms`/
`commitment_delay_ms_for` below) is an ESTIMATED source-to-commit delay:
its "source" endpoint is a word's Tier (b) linearly-interpolated
`estimated_time` (see `streaming.word_alignment`), not a measured
Whisper word timestamp (word_timestamps remains False per the Stage 4B
decision - see `configs/m2.yaml`). Report it as such, never as a
ground-truth latency.

`stable_token_ratio` remains a STABILITY-state metric
(committed / (committed + tentative)), never a correctness metric - a
consistently repeated ASR error becomes "stable" exactly as easily as a
correct word. See `stable_token_ratio()` below.

All timestamps this module reads off `ASRHypothesis`/`ProcessingWindow`
(`audio_start_time`, `audio_end_time`, `window_created_at`,
`submitted_at`, `started_at`, `completed_at`) and produces itself
(`CommittedText.committed_at`) are `time.perf_counter()`-based: monotonic
and comparable to each other ONLY within one running `main.py` process.
They are never comparable across separate process runs, and are
unrelated to the wall-clock UTC `timestamp` field
`utils.logger.EventLogger.event()` adds to every JSONL line (that field
exists purely for human log ordering, never for duration arithmetic).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import threading
import time
from typing import TYPE_CHECKING, Deque

from evaluation.asr_evaluation import (
    character_tokens,
    corpus_error_rate,
    summarize_latency,
    word_tokens,
)
from streaming.streaming_state import MonotonicIdGenerator
from streaming.word_alignment import (
    estimate_word_timings,
    order_preserving_lcs_indices,
    restrict_to_range,
)

if TYPE_CHECKING:
    # Avoids importing asr.worker transitively through streaming_asr at
    # module load time purely for a type hint - mirrors the same
    # deliberate TYPE_CHECKING guard used in streaming/streaming_asr.py
    # (see that module's docstring for why). ASRHypothesis is frozen and
    # only read here, never constructed.
    from streaming.streaming_asr import ASRHypothesis


#: Alignment methods implemented. Only "lcs" exists in Phase 4; the name
#: is validated so a future method can be added without silently
#: reinterpreting an unrecognized value.
SUPPORTED_ALIGNMENT_METHODS = frozenset({"lcs"})

#: Word-timing methods implemented. Only "interpolated" (Tier (b))
#: exists in Phase 4; see the module docstring and
#: streaming/word_alignment.py for why real Whisper word timestamps
#: (Tier (c)) are not used yet.
SUPPORTED_TIMING_METHODS = frozenset({"interpolated"})


@dataclass(frozen=True)
class CommittedText:
    """One immutable stable-prefix commitment event.

    Attributes:
        utterance_id: Utterance this commitment belongs to.
        commit_id: Unique id for this commitment event (monotonic,
            `streaming.streaming_state.MonotonicIdGenerator`, reused).
        generation: The generation whose processing produced this
            commitment (the most recent of the generations that
            contributed to it - see
            `LocalAgreementTracker.contributing_generations` for the
            full set of generations that introduced the promoted
            words).
        new_text: Only the newly committed words (space-joined).
        total_committed_text: The cumulative committed transcript for
            this utterance so far, including `new_text`.
        committed_at: `time.perf_counter()` at the commit decision.
        is_final: True only for the commitment produced at utterance
            finalization (at most one such event per utterance).
    """

    utterance_id: int
    commit_id: int
    generation: int
    new_text: str
    total_committed_text: str
    committed_at: float
    is_final: bool


@dataclass(frozen=True)
class _HypothesisRecord:
    """Minimal per-hypothesis data the tracker needs to keep around."""

    generation: int
    window_id: int
    words: tuple[str, ...]
    audio_start_time: float
    audio_end_time: float
    is_final: bool


@dataclass
class _TentativeWord:
    """One not-yet-committed word in the persistent tentative buffer.

    Attributes:
        word: Normalized word text.
        estimated_time: Interpolated timestamp from whichever hypothesis
            FIRST introduced this word (kept fixed across reconfirmation
            rounds - see the module docstring's region B - so it stays a
            stable anchor for audio-time splitting and a stable proxy
            for "when this content's audio became available").
        streak: Number of consecutive admitted hypotheses that have
            reconfirmed this exact word at this position via the
            overlap-restricted alignment. Starts at 1 when first
            introduced (region B/C); incremented only by a successful
            reconfirmation (region B match).
        first_seen_generation: Generation of the hypothesis that first
            introduced this word - used for `contributing_generations`,
            never for any commitment decision.
    """

    word: str
    estimated_time: float
    streak: int
    first_seen_generation: int


@dataclass
class _CommitMetadata:
    """Tracker-internal bookkeeping for one commit event (not exposed directly)."""

    contributing_generations: tuple[int, ...]
    oldest_contributing_time: float


@dataclass
class _UtteranceState:
    """Mutable per-utterance tracker state. Never exposed directly."""

    agreement_depth: int
    history_size: int

    committed_words: list[str] = field(default_factory=list)

    # Audio-time point up to which content is fixed. None until the
    # first commitment. Advances monotonically; never used to decide
    # correctness, purely descriptive/loggable.
    committed_audio_boundary: float | None = None

    # PERSISTENT across arbitrarily many hypotheses - see module
    # docstring. This is the field whose absence was the Stage 4A bug:
    # it must never be reconstructed from only the last
    # `agreement_depth` hypotheses.
    tentative: list[_TentativeWord] = field(default_factory=list)

    commit_events: list[CommittedText] = field(default_factory=list)
    commit_metadata: dict[int, _CommitMetadata] = field(default_factory=dict)

    # Bounded to history_size - a pure memory safeguard for diagnostics
    # (hypothesis_count, revision_count). Trimming this can NEVER
    # invalidate committed_words/tentative/commit_events, which live
    # separately and are never derived from this log.
    diagnostic_log: Deque[_HypothesisRecord] = field(init=False)

    seen_generations: set[int] = field(default_factory=set)
    highest_admitted_generation: int = 0
    frozen: bool = False
    revision_count: int = 0
    hypotheses_seen: int = 0
    total_words_seen: int = 0
    latest_hypothesis: _HypothesisRecord | None = None
    utterance_start_audio_time: float | None = None
    version: int = 0

    def __post_init__(self) -> None:
        self.diagnostic_log = deque(maxlen=self.history_size)


def _promote_leading_stable(
    tentative: list[_TentativeWord],
    agreement_depth: int,
) -> tuple[list[_TentativeWord], list[_TentativeWord]]:
    """Split into (promoted, remaining).

    `promoted` is the leading CONTIGUOUS run of `tentative` whose streak
    has reached `agreement_depth` - a single word below depth anywhere
    stops promotion, so a later word can never be committed while an
    earlier one remains unresolved.
    """

    promote_count = 0

    for item in tentative:
        if item.streak >= agreement_depth:
            promote_count += 1
        else:
            break

    return tentative[:promote_count], tentative[promote_count:]


class LocalAgreementTracker:
    """Stable-prefix commitment tracker over a stream of ASRHypothesis.

    One instance tracks many utterances (keyed by `utterance_id`). Feed
    it successful and failed hypotheses via `process()`, in any call
    order - it internally enforces generation-ordered admission so
    out-of-order arrival, missing generations, and duplicate
    generations are all handled safely.

    Algorithm (per utterance, `agreement_depth` = N), see the module
    docstring for the full rationale:
        - Only hypotheses with `generation` strictly greater than any
          generation already admitted update the persistent `tentative`
          buffer. A late-arriving OLDER generation is recorded for
          diagnostics but cannot retroactively reopen already-decided
          state - rewinding would violate exactly-once/immutability.
        - Each admitted hypothesis folds into `tentative` via
          `_advance_tentative` (module-level function): content before
          the new hypothesis's audio start is preserved untouched;
          content in the shared overlap is re-aligned via
          order-preserving LCS (`streaming.word_alignment`) and only
          survives if reconfirmed; content beyond the previous
          hypothesis's end is appended as new tentative material.
        - `_promote_leading_stable` commits the leading contiguous
          tentative run whose per-word streak has reached N.

    Thread-safety: a single `threading.Lock` protects only state
    mutation, never the alignment computation itself. `process()`
    snapshots state, releases the lock, computes the tentative update,
    then re-acquires the lock to validate-and-commit; if another thread
    mutated the same utterance's state in the meantime, it retries
    (bounded) rather than risk corrupting committed state.
    """

    def __init__(
        self,
        agreement_depth: int = 2,
        history_size: int = 5,
        alignment_method: str = "lcs",
        timing_method: str = "interpolated",
        logger: logging.Logger | None = None,
    ) -> None:
        """Create a tracker.

        Args:
            agreement_depth: Number of consecutive (generation-ordered)
                successful hypotheses that must agree before new words
                are committed. Default 2, matching the common
                LocalAgreement-2 literature default - NOT claimed to be
                theoretically optimal for this project; an experimental
                parameter. Not tuned in Stage 4A.
            history_size: Bounded-memory safeguard for diagnostic
                introspection (hypothesis_count etc.) only. Never
                affects commitment correctness - see `_UtteranceState`.
            alignment_method: Must be "lcs" (the only method Phase 4
                implements).
            timing_method: Must be "interpolated" (Tier (b); the only
                method Phase 4 implements - see
                streaming/word_alignment.py).
            logger: Optional logger.

        Raises:
            ValueError: For invalid configuration.
        """

        if agreement_depth < 2:
            raise ValueError(
                "agreement_depth must be at least 2 (LocalAgreement "
                "requires comparing at least 2 hypotheses)."
            )

        if history_size < 1:
            raise ValueError("history_size must be at least 1.")

        if alignment_method not in SUPPORTED_ALIGNMENT_METHODS:
            raise ValueError(
                f"Unsupported alignment_method: {alignment_method!r}. "
                f"Supported: {sorted(SUPPORTED_ALIGNMENT_METHODS)}"
            )

        if timing_method not in SUPPORTED_TIMING_METHODS:
            raise ValueError(
                f"Unsupported timing_method: {timing_method!r}. "
                f"Supported: {sorted(SUPPORTED_TIMING_METHODS)}"
            )

        self.agreement_depth = agreement_depth
        self.history_size = history_size
        self.alignment_method = alignment_method
        self.timing_method = timing_method

        self.logger = (
            logger
            if logger is not None
            else logging.getLogger(__name__)
        )

        self._lock = threading.Lock()
        self._states: dict[int, _UtteranceState] = {}
        self._commit_id_generator = MonotonicIdGenerator()

        self._max_retries = 5

        # CPU profiling: tentative-update/alignment compute time,
        # outside the lock. Tracker-wide (not per-utterance) - this is
        # a system-level cost measurement.
        self._alignment_timing_samples_ms: list[float] = []

    # ------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------

    def process(self, hypothesis: "ASRHypothesis") -> CommittedText | None:
        """Process one ASRHypothesis, possibly producing a commitment.

        Safe to call with hypotheses in any order (including duplicates
        of an already-processed generation, or a generation older than
        one already admitted) - see the class docstring for exactly how
        each case is handled. Never raises for a missing/duplicate/
        out-of-order generation; only raises for programmer-contract
        violations (none currently defined beyond constructor
        validation).

        Args:
            hypothesis: A Phase 3 ASRHypothesis (successful or failed).

        Returns:
            A CommittedText if this call produced a new commitment,
            else None. When `hypothesis.is_final` is True and this same
            call ALSO produces a normal (non-final) commitment, only
            the finalization event is returned here - both are still
            recorded and visible via `commit_events_for()`, which is
            the authoritative source for the complete event history.
        """

        if not hypothesis.success:
            # Still register that this utterance/generation was seen, so
            # an utterance whose ASR calls all fail is distinguishable
            # from one this tracker never heard of at all (matters for
            # finalize_utterance()'s "no successful hypothesis" branch).
            with self._lock:
                state = self._states.setdefault(
                    hypothesis.utterance_id,
                    _UtteranceState(self.agreement_depth, self.history_size),
                )

                if not state.frozen:
                    state.seen_generations.add(hypothesis.generation)

            if hypothesis.is_final:
                return self.finalize_utterance(
                    hypothesis.utterance_id,
                    reason=(
                        "final ASR generation "
                        f"{hypothesis.generation} failed: "
                        f"{hypothesis.error}"
                    ),
                )

            self.logger.info(
                "local_agreement: skipping failed hypothesis | "
                "utterance_id=%d generation=%d error=%s",
                hypothesis.utterance_id,
                hypothesis.generation,
                hypothesis.error,
            )

            return None

        words = tuple(word_tokens(hypothesis.text))

        record = _HypothesisRecord(
            generation=hypothesis.generation,
            window_id=hypothesis.window_id,
            words=words,
            audio_start_time=hypothesis.audio_start_time,
            audio_end_time=hypothesis.audio_end_time,
            is_final=hypothesis.is_final,
        )

        for _ in range(self._max_retries):
            with self._lock:
                state = self._states.setdefault(
                    hypothesis.utterance_id,
                    _UtteranceState(self.agreement_depth, self.history_size),
                )

                if state.frozen:
                    self.logger.warning(
                        "local_agreement: utterance_id=%d is frozen; "
                        "ignoring generation=%d",
                        hypothesis.utterance_id,
                        hypothesis.generation,
                    )

                    return None

                if hypothesis.generation in state.seen_generations:
                    return None

                snapshot_tentative = list(state.tentative)
                snapshot_previous_hypothesis = state.latest_hypothesis
                snapshot_version = state.version
                is_new_admission = (
                    hypothesis.generation
                    > state.highest_admitted_generation
                )

            # -------- compute phase: NO LOCK HELD --------
            compute_start = time.perf_counter()

            new_tentative: list[_TentativeWord] = snapshot_tentative
            had_disagreement = False
            promoted: list[_TentativeWord] = []

            if is_new_admission:
                new_tentative, had_disagreement = _advance_tentative(
                    snapshot_tentative,
                    snapshot_previous_hypothesis,
                    record,
                )

                promoted, new_tentative = _promote_leading_stable(
                    new_tentative,
                    self.agreement_depth,
                )

            alignment_elapsed_ms = (
                time.perf_counter() - compute_start
            ) * 1000.0

            # -------- validate/update phase: lock held --------
            with self._lock:
                state = self._states[hypothesis.utterance_id]

                if state.frozen:
                    return None

                if hypothesis.generation in state.seen_generations:
                    return None

                if state.version != snapshot_version:
                    # State changed underneath us - retry with a fresh
                    # snapshot rather than applying a possibly-stale
                    # computation.
                    continue

                state.seen_generations.add(hypothesis.generation)
                state.hypotheses_seen += 1
                state.total_words_seen += len(record.words)
                state.diagnostic_log.append(record)

                if state.utterance_start_audio_time is None:
                    state.utterance_start_audio_time = (
                        record.audio_start_time
                    )

                if is_new_admission:
                    # `latest_hypothesis` must only ever advance to a
                    # genuinely newer generation - it is the
                    # "previous_hypothesis" reference the NEXT
                    # admission's _advance_tentative() call will diff
                    # against. Updating it for a stale/late admission
                    # (generation <= highest_admitted_generation) would
                    # silently corrupt that future comparison, even
                    # though the stale call itself correctly avoids
                    # touching `tentative`/`committed_words`.
                    state.latest_hypothesis = record
                    state.tentative = new_tentative
                    state.highest_admitted_generation = (
                        hypothesis.generation
                    )

                    if had_disagreement:
                        state.revision_count += 1

                self._alignment_timing_samples_ms.append(
                    alignment_elapsed_ms
                )

                commit_event: CommittedText | None = None

                if promoted:
                    commit_event = self._commit_locked(
                        state=state,
                        utterance_id=hypothesis.utterance_id,
                        promoted=promoted,
                        generation=hypothesis.generation,
                        is_final=False,
                    )

                state.version += 1

            if commit_event is not None:
                self.logger.info(
                    "local_agreement: committed | utterance_id=%d "
                    "window_id=%d generation=%d | new_text=%r | "
                    "total_committed_text=%r",
                    hypothesis.utterance_id,
                    hypothesis.window_id,
                    hypothesis.generation,
                    commit_event.new_text,
                    commit_event.total_committed_text,
                )

            if hypothesis.is_final:
                return self.finalize_utterance(hypothesis.utterance_id)

            return commit_event

        self.logger.warning(
            "local_agreement: gave up processing utterance_id=%d "
            "generation=%d after %d retries due to contention",
            hypothesis.utterance_id,
            hypothesis.generation,
            self._max_retries,
        )

        return None

    def _commit_locked(
        self,
        state: _UtteranceState,
        utterance_id: int,
        promoted: list[_TentativeWord],
        generation: int,
        is_final: bool,
    ) -> CommittedText:
        """Apply a commitment. Caller must already hold `self._lock`.

        `promoted` may be empty ONLY when `is_final` is True (an
        explicit "utterance ended, nothing new to add" marker event);
        the normal (non-final) call site never invokes this with an
        empty list.
        """

        new_words = [item.word for item in promoted]

        state.committed_words.extend(new_words)

        if promoted:
            state.committed_audio_boundary = promoted[-1].estimated_time

        commit_id = self._commit_id_generator.next()

        commit_event = CommittedText(
            utterance_id=utterance_id,
            commit_id=commit_id,
            generation=generation,
            new_text=" ".join(new_words),
            total_committed_text=" ".join(state.committed_words),
            committed_at=time.perf_counter(),
            is_final=is_final,
        )

        state.commit_events.append(commit_event)

        contributing_generations = tuple(
            sorted({item.first_seen_generation for item in promoted})
        )

        oldest_contributing_time = min(
            (item.estimated_time for item in promoted),
            default=0.0,
        )

        state.commit_metadata[commit_id] = _CommitMetadata(
            contributing_generations=contributing_generations,
            oldest_contributing_time=oldest_contributing_time,
        )

        return commit_event

    # ------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------

    def finalize_utterance(
        self,
        utterance_id: int,
        reason: str | None = None,
    ) -> CommittedText | None:
        """Force-commit ALL remaining tentative content and freeze an utterance.

        Call this directly when the caller knows an utterance has ended
        but no successful `is_final=True` hypothesis was ever available
        to drive finalization through `process()` (e.g. the final
        window's ASR call failed or was dropped by the queue).
        `process()` also calls this internally once it finishes
        handling a successful `is_final=True` hypothesis, so
        finalization always goes through this one method regardless of
        path.

        Because `tentative` is now a persistent buffer that already
        holds every not-yet-committed word regardless of how many
        hypotheses ago it first appeared (see module docstring), this
        simply commits `state.tentative` in full - no separate
        reconstruction against `latest_hypothesis` is needed, and no
        previously-retained tentative content can be missed.

        Idempotent in the sense that it is safe to call on an
        already-frozen utterance (logs a warning, returns None) rather
        than raising - finalization must occur at most once, and a
        caller accidentally calling it twice (e.g. once from `process()`
        and once explicitly) must not corrupt state or double-commit.

        Args:
            utterance_id: The utterance to finalize.
            reason: If given, logged as a warning - used when
                finalizing WITHOUT the utterance's own final hypothesis
                (never claim the missing audio was transcribed; only
                state that finalization proceeded from whatever
                tentative content already exists).

        Returns:
            The finalization CommittedText (is_final=True), or None if
            the utterance is unknown or already frozen.
        """

        with self._lock:
            state = self._states.get(utterance_id)

            if state is None:
                self.logger.warning(
                    "local_agreement: finalize_utterance called for "
                    "unknown utterance_id=%d",
                    utterance_id,
                )

                return None

            if state.frozen:
                self.logger.warning(
                    "local_agreement: utterance_id=%d already "
                    "finalized; ignoring repeat finalize_utterance call",
                    utterance_id,
                )

                return None

            if state.latest_hypothesis is None:
                self.logger.warning(
                    "local_agreement: utterance_id=%d finalized with "
                    "no successful hypothesis at all; nothing to "
                    "commit. reason=%s",
                    utterance_id,
                    reason,
                )

            promoted = state.tentative
            state.tentative = []

            generation = (
                state.latest_hypothesis.generation
                if state.latest_hypothesis is not None
                else 0
            )

            commit_event = self._commit_locked(
                state=state,
                utterance_id=utterance_id,
                promoted=promoted,
                generation=generation,
                is_final=True,
            )

            state.frozen = True

            if reason is not None:
                self.logger.warning(
                    "local_agreement: utterance_id=%d force-finalized "
                    "without its own successful final hypothesis | "
                    "reason=%s | forced_tail=%r",
                    utterance_id,
                    reason,
                    commit_event.new_text,
                )
            else:
                self.logger.info(
                    "local_agreement: utterance_id=%d finalized | "
                    "forced_tail=%r | total_committed_text=%r",
                    utterance_id,
                    commit_event.new_text,
                    commit_event.total_committed_text,
                )

            return commit_event

    # ------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------

    def committed_text_for(self, utterance_id: int) -> str:
        """Return the cumulative committed transcript for one utterance."""

        with self._lock:
            state = self._states.get(utterance_id)
            return " ".join(state.committed_words) if state else ""

    def tentative_text_for(self, utterance_id: int) -> str:
        """Return the current uncommitted/tentative transcript for one utterance.

        Empty once finalized (finalization drains `tentative` entirely
        into the final commit).
        """

        with self._lock:
            state = self._states.get(utterance_id)
            return (
                " ".join(item.word for item in state.tentative)
                if state
                else ""
            )

    def commit_events_for(
        self,
        utterance_id: int,
    ) -> tuple[CommittedText, ...]:
        """Return every commit event for one utterance, in commit order."""

        with self._lock:
            state = self._states.get(utterance_id)
            return tuple(state.commit_events) if state else ()

    def contributing_generations(
        self,
        utterance_id: int,
        commit_id: int,
    ) -> tuple[int, ...]:
        """Return which generations first introduced the words in a commit.

        Sorted, distinct generation numbers - not claimed to be every
        generation that ever "touched" the commit, only the ones that
        first introduced each promoted word (see `_TentativeWord
        .first_seen_generation`).
        """

        with self._lock:
            state = self._states.get(utterance_id)

            if state is None:
                return ()

            metadata = state.commit_metadata.get(commit_id)

            return metadata.contributing_generations if metadata else ()

    def is_frozen(self, utterance_id: int) -> bool:
        """Return whether an utterance has been finalized."""

        with self._lock:
            state = self._states.get(utterance_id)
            return bool(state and state.frozen)

    def reset_utterance(self, utterance_id: int) -> None:
        """Discard all tracker state for one utterance (full reset)."""

        with self._lock:
            self._states.pop(utterance_id, None)

    # ------------------------------------------------------------
    # Metrics (all are observational diagnostics, not stability/
    # commitment decisions, and none are claimed as novel research
    # contributions)
    # ------------------------------------------------------------

    def hypothesis_count(self, utterance_id: int) -> int:
        """Total successful hypotheses ever processed for this utterance."""

        with self._lock:
            state = self._states.get(utterance_id)
            return state.hypotheses_seen if state else 0

    def revision_count(self, utterance_id: int) -> int:
        """Count of admitted generations whose overlap re-alignment disagreed.

        Based on alignment/commitment behavior (whether the
        overlap-restricted agreement came out shorter than a perfect
        match would), not naive whole-string inequality between raw
        hypothesis texts.

        Canonical Phase 4C-1 name: `alignment_revision_count` - see the
        module docstring's "canonical metric names" section for why this
        must be distinguished from
        `streaming.streaming_asr.StreamingASRDispatcher.revision_count()`
        (`hypothesis_revision_count`), a different, whole-text-based
        quantity. This method's own behavior is unchanged.
        """

        with self._lock:
            state = self._states.get(utterance_id)
            return state.revision_count if state else 0

    def commitment_count(self, utterance_id: int) -> int:
        """Number of distinct commit events for this utterance."""

        return len(self.commit_events_for(utterance_id))

    def stable_prefix_length(self, utterance_id: int) -> int:
        """Word count of the committed prefix right now."""

        with self._lock:
            state = self._states.get(utterance_id)
            return len(state.committed_words) if state else 0

    def tentative_length(self, utterance_id: int) -> int:
        """Word count of the current persistent tentative buffer."""

        with self._lock:
            state = self._states.get(utterance_id)
            return len(state.tentative) if state else 0

    def stable_token_ratio(self, utterance_id: int) -> float:
        """committed word count / (committed word count + tentative word count).

        Corrected definition (Stage 4A): the previous formula divided
        the CUMULATIVE committed word count by a single local
        hypothesis's word count, which is unbounded above 1.0 once
        several commits have accumulated (observed values of 1.6-3.0 in
        real logs). This formula is bounded in [0, 1] BY CONSTRUCTION -
        the denominator is always >= the numerator, since it is the
        numerator plus a non-negative tentative count - not by clamping
        an incorrect value.

        Returns 0.0 when nothing has been said yet (both counts zero).
        """

        with self._lock:
            state = self._states.get(utterance_id)

            if state is None:
                return 0.0

            committed_count = len(state.committed_words)
            tentative_count = len(state.tentative)
            denominator = committed_count + tentative_count

            if denominator == 0:
                return 0.0

            return committed_count / denominator

    def unstable_token_count(self, utterance_id: int) -> int:
        """Count of raw ASR words that were proposed but then revised away.

        Defined as total words seen across all successful hypotheses
        minus everything currently accounted for (committed + still
        tentative). This precisely counts words that were proposed at
        some point (region B/C of `_advance_tentative`) and later
        dropped because a subsequent hypothesis's overlap re-check
        disagreed with them (module docstring region B) - not merely
        "not yet committed", since still-pending tentative content is
        excluded from this count.
        """

        with self._lock:
            state = self._states.get(utterance_id)

            if state is None:
                return 0

            accounted_for = len(state.committed_words) + len(state.tentative)

            return max(0, state.total_words_seen - accounted_for)

    def time_to_first_commit(self, utterance_id: int) -> float | None:
        """Elapsed time from utterance start to the first non-empty commit.

        Assumes `committed_at` (time.perf_counter()) and
        `utterance_start_audio_time` (from ProcessingWindow.audio_start_time,
        itself AudioFrame.capture_timestamp) are on the same clock - true
        in production (both are time.perf_counter()-based), not
        necessarily true for hand-constructed synthetic timestamps in
        tests.
        """

        with self._lock:
            state = self._states.get(utterance_id)

            if state is None or state.utterance_start_audio_time is None:
                return None

            first_commit = next(
                (
                    event
                    for event in state.commit_events
                    if event.new_text
                ),
                None,
            )

            if first_commit is None:
                return None

            return (
                first_commit.committed_at
                - state.utterance_start_audio_time
            )

    @staticmethod
    def _commitment_delay_ms(
        event: CommittedText,
        metadata: "_CommitMetadata",
    ) -> float:
        """Shared delay formula, used by every commitment-delay query below.

        Canonical Phase 4C-1 name: `commitment_delay_ms`. This is an
        ESTIMATED source-to-commit delay - `oldest_contributing_time` is
        a Tier (b) linearly-interpolated word timestamp (see
        `streaming.word_alignment` and the module docstring), not a
        measured Whisper word timestamp. There is exactly one
        implementation of this formula in the codebase; every public
        commitment-delay method below calls this one rather than
        recomputing it.
        """

        return (
            event.committed_at
            - metadata.oldest_contributing_time
        ) * 1000.0

    def commitment_delays_ms(
        self,
        utterance_id: int,
    ) -> tuple[float, ...]:
        """Per-commit-event delay: committed_at minus the oldest

        contributing word's ESTIMATED introduction time (Tier (b)
        interpolated - see `_commitment_delay_ms`), in milliseconds.
        Empty commit events (new_text == "") are excluded.
        """

        with self._lock:
            state = self._states.get(utterance_id)

            if state is None:
                return ()

            delays: list[float] = []

            for event in state.commit_events:
                if not event.new_text:
                    continue

                metadata = state.commit_metadata.get(event.commit_id)

                if metadata is None:
                    continue

                delays.append(
                    self._commitment_delay_ms(event, metadata)
                )

            return tuple(delays)

    def commitment_delay_ms_for(
        self,
        utterance_id: int,
        commit_id: int,
    ) -> float | None:
        """Return the commitment delay (ms) for exactly one commit event.

        Same formula as `commitment_delays_ms` (see `_commitment_delay_ms`),
        restricted to a single already-known `commit_id` instead of the
        whole utterance's history. Intended for a caller (e.g. `main.py`)
        that already holds the `CommittedText` returned by `process()`/
        `finalize_utterance()` and wants that specific event's delay
        without re-deriving the formula or re-scanning every commit.

        Returns:
            The delay in milliseconds, or None if the utterance/commit_id
            is unknown, or the commit had no new text (e.g. an empty
            finalization marker commit - there is nothing to time).
        """

        with self._lock:
            state = self._states.get(utterance_id)

            if state is None:
                return None

            metadata = state.commit_metadata.get(commit_id)

            if metadata is None:
                return None

            event = next(
                (
                    item
                    for item in state.commit_events
                    if item.commit_id == commit_id
                ),
                None,
            )

            if event is None or not event.new_text:
                return None

            return self._commitment_delay_ms(event, metadata)

    def per_word_confirmation_delays_ms(
        self,
        utterance_id: int,
    ) -> tuple[float, ...]:
        """One delay value per committed word (a commit's delay, repeated

        once per word it committed). The simplest honest expansion of
        commitment_delays_ms to per-word granularity given this tracker
        does not track individual words' first-seen time separately
        within one commit event.
        """

        with self._lock:
            state = self._states.get(utterance_id)

            if state is None:
                return ()

            delays: list[float] = []

            for event in state.commit_events:
                if not event.new_text:
                    continue

                metadata = state.commit_metadata.get(event.commit_id)

                if metadata is None:
                    continue

                delay_ms = (
                    event.committed_at
                    - metadata.oldest_contributing_time
                ) * 1000.0

                word_count = len(event.new_text.split())
                delays.extend([delay_ms] * word_count)

            return tuple(delays)

    # ------------------------------------------------------------
    # CPU profiling (alignment cost, tracker-wide)
    # ------------------------------------------------------------

    @property
    def alignment_timing_samples_ms(self) -> tuple[float, ...]:
        """Every alignment-computation duration observed, in milliseconds."""

        with self._lock:
            return tuple(self._alignment_timing_samples_ms)

    def cpu_profile_summary(self) -> dict[str, float]:
        """Return mean/p50/p95/p99/max alignment compute time, in ms.

        Reuses evaluation.asr_evaluation.summarize_latency (no
        reimplemented percentile math).
        """

        with self._lock:
            samples = list(self._alignment_timing_samples_ms)

        return summarize_latency(samples)


def _advance_tentative(
    previous_tentative: list[_TentativeWord],
    previous_hypothesis: "_HypothesisRecord | None",
    new_record: "_HypothesisRecord",
) -> tuple[list[_TentativeWord], bool]:
    """Fold one new hypothesis into the persistent tentative buffer.

    Pure function of its arguments (no shared state, no lock needed) -
    see the module docstring for the exact rule (regions A/B/C).

    Returns:
        (new_tentative, had_disagreement). `had_disagreement` is True
        only when a real overlap existed and the re-alignment discarded
        at least one previously-tentative or newly-hypothesized word
        (used only for the `revision_count` diagnostic, never for any
        commitment decision).
    """

    new_timed = estimate_word_timings(
        new_record.words,
        new_record.audio_start_time,
        new_record.audio_end_time,
    )

    if previous_hypothesis is None:
        # First-ever hypothesis for this utterance: nothing exists yet
        # to compare it against, so everything it says becomes
        # tentative directly (region C in full).
        tentative = [
            _TentativeWord(
                word=timed.word,
                estimated_time=timed.estimated_time,
                streak=1,
                first_seen_generation=new_record.generation,
            )
            for timed in new_timed
        ]

        return tentative, False

    overlap_start = new_record.audio_start_time
    previous_end = previous_hypothesis.audio_end_time

    # Region A: existing tentative content H cannot see at all. Carried
    # forward completely untouched - this is the Stage 4A fix.
    preserved = [
        item
        for item in previous_tentative
        if item.estimated_time < overlap_start
    ]

    # Existing tentative content that falls within H's potential view.
    overlapping = [
        item
        for item in previous_tentative
        if item.estimated_time >= overlap_start
    ]

    had_disagreement = False

    if previous_end <= overlap_start:
        # No audio-time overlap between the two hypotheses at all - no
        # comparison is possible. `overlapping` (if anything landed
        # here at all) is left completely untouched, per the "no
        # overlap -> do not attempt agreement" rule; every one of H's
        # words is new content.
        reconfirmed = overlapping
        new_words_source = list(new_timed)
    else:
        overlap_words = [item.word for item in overlapping]

        # Every word of H falls into exactly one of two buckets below -
        # "in the overlap candidate range" (subject to matching against
        # `overlapping`) or not. Tracking each candidate's ORIGINAL
        # index into `new_timed` (rather than re-deriving it from a
        # time comparison a second time) guarantees the two buckets
        # partition `new_timed` exactly, with no boundary gap a word
        # could fall through unclassified - see the Stage 4A boundary
        # bug this replaced, where a word at exactly `previous_end`
        # could be selected into the overlap candidate set (inclusive
        # upper bound) yet excluded from the new-tail set (which used a
        # strict `>` lower bound), silently vanishing when unmatched.
        new_overlap_candidates = [
            (index, timed)
            for index, timed in enumerate(new_timed)
            if overlap_start <= timed.estimated_time <= previous_end
        ]
        new_overlap_words = [timed.word for _, timed in new_overlap_candidates]

        pairs = order_preserving_lcs_indices(overlap_words, new_overlap_words)

        if len(pairs) < min(len(overlap_words), len(new_overlap_words)):
            had_disagreement = True

        # Region B: only words BOTH sides still agree on survive - a
        # legitimate revision based on newer evidence, not loss (loss
        # would be discarding `preserved`, which never happens here).
        reconfirmed = [
            _TentativeWord(
                word=overlapping[i].word,
                estimated_time=overlapping[i].estimated_time,
                streak=overlapping[i].streak + 1,
                first_seen_generation=overlapping[i].first_seen_generation,
            )
            for i, _ in pairs
        ]

        matched_local_positions = {j for _, j in pairs}
        matched_new_timed_indices = {
            new_overlap_candidates[j][0] for j in matched_local_positions
        }

        # Everything in H not matched above is new content: either
        # strictly beyond the previous hypothesis's reach, or within
        # the overlap candidate range but not corresponding to anything
        # previously guessed (e.g. a word right at the boundary that
        # the earlier, narrower hypothesis simply never proposed).
        new_words_source = [
            timed
            for index, timed in enumerate(new_timed)
            if index not in matched_new_timed_indices
        ]

    # Region C: genuinely new content beyond anything seen before.
    new_tail = [
        _TentativeWord(
            word=timed.word,
            estimated_time=timed.estimated_time,
            streak=1,
            first_seen_generation=new_record.generation,
        )
        for timed in new_words_source
    ]

    tentative = preserved + reconfirmed + new_tail

    return tentative, had_disagreement


def committed_transcript_error_rate(
    committed_text: str,
    reference_text: str,
) -> dict[str, float]:
    """WER/CER of a committed transcript against a reference.

    Reuses evaluation.asr_evaluation.word_tokens/character_tokens/
    corpus_error_rate unchanged - no normalization or error-rate math is
    reimplemented here.
    """

    reference_words = [word_tokens(reference_text)]
    hypothesis_words = [word_tokens(committed_text)]

    word_errors, reference_word_count, wer = corpus_error_rate(
        reference_words,
        hypothesis_words,
    )

    reference_chars = [character_tokens(reference_text)]
    hypothesis_chars = [character_tokens(committed_text)]

    char_errors, reference_char_count, cer = corpus_error_rate(
        reference_chars,
        hypothesis_chars,
    )

    return {
        "wer": wer,
        "word_errors": word_errors,
        "reference_word_count": reference_word_count,
        "cer": cer,
        "character_errors": char_errors,
        "reference_character_count": reference_char_count,
    }
