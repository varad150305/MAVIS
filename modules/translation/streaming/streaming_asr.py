"""M2 Phase 3: connect ProcessingWindow to the existing ASR infrastructure.

This module is an orchestration/correlation layer ONLY. It does not
load Whisper, does not implement ASR inference, and does not own a
queue of its own for ASR work - all of that remains exactly
`asr.whisper_engine.WhisperEngine` and `asr.worker.ASRWorker`,
unmodified. `StreamingASRDispatcher` exists purely to:

    1. submit each `streaming.streaming_state.ProcessingWindow`
       emitted by Phase 2's `AdaptiveBuffer` to the existing
       `ASRWorker`,
    2. correlate the asynchronous `asr.worker.ASRTaskResult`s that
       come back to the exact window/utterance/generation that
       produced them (never by list position or completion order),
    3. keep an ordered-by-generation hypothesis history per utterance,
    4. expose the workload/queue/ASR/window metrics Phase 3 requires
       to OBSERVE system behavior - it makes no scheduling, stability,
       or commitment decisions.

Phase 3 scope only. This module does NOT implement:
    - LocalAgreement or stable-prefix commitment (Phase 4),
    - any stability/commitment decision over the hypothesis history,
    - translation invocation or gating,
    - M3 stability-/compute-aware scheduling,
    - any ASR submission "optimization" policy (e.g. retrying or
      evicting entries from ASRWorker's own queue). When ASRWorker's
      bounded queue is full, the rejection is recorded as a
      measurement, not worked around.

============================================================
Why a small adapter, not ProcessingWindow itself
============================================================

`asr.worker.ASRWorker` is typed to accept a
`streaming.buffer.SpeechSegment`, and its internal `_process_task`
reads exactly these attributes off whatever is submitted: `.samples`,
`.sample_rate`, `.start_sequence_number`, `.end_sequence_number`,
`.duration_ms`, and `.end_timestamp`. `ProcessingWindow` already has
every one of these EXCEPT `.end_timestamp` - it names the equivalent
field `.audio_end_time` instead (a deliberate Phase 1 naming choice,
documented in `streaming.streaming_state`). Submitting a
`ProcessingWindow` directly would raise `AttributeError` inside
`ASRWorker._process_task` the first time it computed
`segment_to_transcript_ms`.

This is exactly the kind of "existing API cannot support Phase 3
without modification" case the Phase 3 instructions anticipate -
but the fix does not require touching `asr/worker.py`. `_WindowSegmentAdapter`
is a tiny, private, frozen dataclass that presents exactly the
attributes `ASRWorker` needs (mapping `.audio_end_time` onto
`.end_timestamp`) while also carrying the original `ProcessingWindow`,
so results can be correlated back by identity once `ASRWorker` hands
the `ASRTaskResult` back. `asr/worker.py` is therefore reused
completely unmodified, as required.

============================================================
Ordering / out-of-order safety
============================================================

`ASRWorker` results arrive via `poll_results()` in whatever order
they complete. `StreamingASRDispatcher` never assumes completion
order equals generation order: history is stored as
`{utterance_id: {generation: ASRHypothesis}}`, so a later generation
completing before an earlier one cannot overwrite or corrupt the
earlier entry's slot, and `hypotheses_for()` always returns the
ordered-by-generation view regardless of arrival order.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

import numpy as np

from streaming.backpressure import QueueBackpressureStatus, WindowDropReason
from streaming.streaming_state import MonotonicIdGenerator, ProcessingWindow

if TYPE_CHECKING:
    # Deferred to avoid a circular import: asr/worker.py imports
    # streaming.buffer (for a SpeechSegment type hint), which runs
    # streaming/__init__.py, which imports this module. Nothing here
    # calls ASRWorker/ASRTaskResult at runtime - both names are only
    # ever used as type hints, and `from __future__ import annotations`
    # (above) already defers all annotation evaluation, so this import
    # is safe to skip entirely outside of static type checking.
    from asr.worker import ASRTaskResult, ASRWorker


@dataclass(frozen=True)
class _WindowSegmentAdapter:
    """Minimal SpeechSegment-shaped view of a ProcessingWindow.

    See the module docstring for exactly why this exists. Not
    exported - it is purely an internal bridge to the unmodified
    `asr.worker.ASRWorker`/`asr.whisper_engine.WhisperEngine`.
    """

    samples: np.ndarray
    sample_rate: int
    start_sequence_number: int
    end_sequence_number: int
    duration_ms: float
    end_timestamp: float
    window: ProcessingWindow
    initial_prompt: str | None = None

    @classmethod
    def from_window(cls, window: ProcessingWindow) -> "_WindowSegmentAdapter":
        """Build the adapter ASRWorker needs from a ProcessingWindow."""

        return cls(
            samples=window.samples,
            sample_rate=window.sample_rate,
            start_sequence_number=window.start_sequence_number,
            end_sequence_number=window.end_sequence_number,
            duration_ms=window.duration_ms,
            end_timestamp=window.audio_end_time,
            window=window,
            initial_prompt=window.initial_prompt,
        )


@dataclass(frozen=True)
class ASRHypothesis:
    """One correlated ASR result for one ProcessingWindow (Phase 3).

    Fully answers: which utterance, which window, which generation,
    when the window was created/submitted, when ASR started/finished,
    what Whisper returned, how long the audio was, how much was
    context vs new audio, and whether it succeeded.

    Timing fields (`submitted_at`, `started_at`, `completed_at`,
    `queue_wait_ms`, `segment_to_transcript_ms`) are reused verbatim
    from the existing `asr.worker.ASRTaskResult` - they are not
    recomputed or duplicated here. `window_created_at`,
    `audio_start_time`, `audio_end_time`, `window_duration_ms`,
    `context_duration_ms`, `new_audio_duration_ms`, and `is_final` are
    reused verbatim from the originating `ProcessingWindow` (Phase 1).

    `text`, `inference_time_ms`, and `rtf` are `None` when
    `success` is False - a failed hypothesis is never silently
    represented as an empty-but-successful transcription.
    """

    hypothesis_id: int
    utterance_id: int
    window_id: int
    generation: int

    submitted_at: float
    started_at: float
    completed_at: float
    queue_wait_ms: float
    segment_to_transcript_ms: float

    window_created_at: float
    audio_start_time: float
    audio_end_time: float
    window_duration_ms: float
    context_duration_ms: float
    new_audio_duration_ms: float
    is_final: bool

    success: bool
    error: str | None

    text: str | None
    inference_time_ms: float | None
    rtf: float | None

    # Measured (never interpolated) per-word timestamps, only populated
    # when the engine was constructed with word_timestamps=True - see
    # asr.whisper_engine.ASRResult.words. Empty tuple default keeps
    # every existing ASRHypothesis(...) construction site valid.
    words: tuple = ()

    def __post_init__(self) -> None:
        """Validate hypothesis invariants."""

        if self.hypothesis_id <= 0:
            raise ValueError("hypothesis_id must be greater than zero.")

        if self.utterance_id <= 0:
            raise ValueError("utterance_id must be greater than zero.")

        if self.window_id <= 0:
            raise ValueError("window_id must be greater than zero.")

        if self.generation <= 0:
            raise ValueError("generation must be greater than zero.")

        if self.success:
            if self.error is not None:
                raise ValueError(
                    "A successful hypothesis cannot carry an error."
                )
            if self.text is None:
                raise ValueError(
                    "A successful hypothesis must have text "
                    "(possibly an empty string, never None)."
                )
        else:
            if self.error is None:
                raise ValueError(
                    "A failed hypothesis must carry an error message."
                )
            if self.text is not None:
                raise ValueError(
                    "A failed hypothesis must not carry text - "
                    "failure must never be represented as an empty "
                    "successful transcription."
                )


class StreamingASRDispatcher:
    """Correlation layer between ProcessingWindow and the existing ASR worker.

    See the module docstring for the full design. Every
    `ProcessingWindow` handed to `submit_window()` is submitted to the
    wrapped `asr.worker.ASRWorker` exactly once; every completed
    result polled via `poll_results()` is correlated back to its
    exact utterance/window/generation and recorded in an
    ordered-by-generation history, independent of completion order.
    """

    def __init__(
        self,
        asr_worker: ASRWorker,
        logger: logging.Logger | None = None,
    ) -> None:
        """Create a dispatcher wrapping an already-started ASRWorker.

        Args:
            asr_worker: The existing, unmodified ASR worker to submit
                windows to. Not owned/started/stopped by this class -
                the caller remains responsible for its lifecycle.
            logger: Optional logger.
        """

        self.asr_worker = asr_worker

        self.logger = (
            logger
            if logger is not None
            else logging.getLogger(__name__)
        )

        self._hypothesis_id_generator = MonotonicIdGenerator()

        # utterance_id -> {generation: ASRHypothesis}. A dict keyed by
        # generation (not a list) so out-of-order completion can never
        # overwrite or misplace an entry - see module docstring.
        self._history: dict[int, dict[int, ASRHypothesis]] = {}

        # Every window_id ever offered to submit_window(), regardless
        # of whether ASRWorker accepted it. Guards against duplicate
        # submission of the same window.
        self._offered_window_ids: set[int] = set()

        # Windows rejected by ASRWorker's bounded queue at submission
        # time (queue was full). Recorded, never silently discarded.
        self._dropped_before_asr: list[
            tuple[ProcessingWindow, WindowDropReason]
        ] = []

        self._mismatched_result_count = 0
        self._duplicate_result_count = 0
        self._max_observed_queue_size = 0

        # Raw samples for the caller (e.g. main.py) to summarize with
        # whatever statistics it prefers - this module deliberately
        # does not compute percentiles itself (see Phase 3 report for
        # why: avoiding a streaming -> evaluation dependency).
        self._queue_wait_samples_ms: list[float] = []
        self._inference_time_samples_ms: list[float] = []
        self._rtf_samples: list[float] = []

    # ------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------

    def submit_window(self, window: ProcessingWindow) -> bool:
        """Submit one ProcessingWindow to the existing ASRWorker.

        The audio submitted to Whisper is exactly `window.samples` -
        nothing is trimmed, padded, resampled, or otherwise altered.

        Args:
            window: The window to submit.

        Returns:
            True if ASRWorker accepted it, False if its bounded queue
            was full (recorded as a drop, not retried).

        Raises:
            ValueError: If `window.window_id` has already been
                offered to this dispatcher (a caller-contract
                violation - window ids must be unique - rather than a
                runtime/backpressure condition).
        """

        if window.window_id in self._offered_window_ids:
            raise ValueError(
                "Duplicate window_id submitted to StreamingASRDispatcher: "
                f"{window.window_id}"
            )

        self._offered_window_ids.add(window.window_id)

        adapter = _WindowSegmentAdapter.from_window(window)

        accepted = self.asr_worker.submit(adapter)

        self._max_observed_queue_size = max(
            self._max_observed_queue_size,
            self.asr_worker.pending_count,
        )

        if not accepted:
            self._dropped_before_asr.append(
                (window, WindowDropReason.QUEUE_FULL)
            )

            self.logger.warning(
                "M2 streaming ASR: window rejected by ASR queue | "
                "utterance_id=%d window_id=%d generation=%d | "
                "reason=%s",
                window.utterance_id,
                window.window_id,
                window.generation,
                WindowDropReason.QUEUE_FULL.value,
            )

        return accepted

    # ------------------------------------------------------------
    # Completion / correlation
    # ------------------------------------------------------------

    def poll_results(self) -> list[ASRHypothesis]:
        """Drain completed ASR results and correlate them to hypotheses.

        Returns:
            Newly completed ASRHypothesis records, in the order
            ASRWorker returned them (completion order, NOT generation
            order - use `hypotheses_for()` for the ordered history).
        """

        hypotheses: list[ASRHypothesis] = []

        for task_result in self.asr_worker.poll_results():
            hypothesis = self._build_hypothesis(task_result)

            if hypothesis is None:
                continue

            self._record_sample(hypothesis)

            generation_map = self._history.setdefault(
                hypothesis.utterance_id,
                {},
            )

            if hypothesis.generation in generation_map:
                self._duplicate_result_count += 1

                self.logger.warning(
                    "M2 streaming ASR: duplicate result for "
                    "utterance_id=%d generation=%d (window_id=%d); "
                    "keeping the first result, ignoring this one.",
                    hypothesis.utterance_id,
                    hypothesis.generation,
                    hypothesis.window_id,
                )

                continue

            generation_map[hypothesis.generation] = hypothesis

            hypotheses.append(hypothesis)

        return hypotheses

    def _build_hypothesis(
        self,
        task_result: ASRTaskResult,
    ) -> ASRHypothesis | None:
        """Correlate one ASRTaskResult back to its ProcessingWindow.

        Returns:
            The constructed ASRHypothesis, or None if `task_result`
            did not originate from this dispatcher (a defensive
            "processing-window mismatch" guard - e.g. the same
            ASRWorker instance being shared with non-M2 submissions).
        """

        segment = task_result.segment

        if not isinstance(segment, _WindowSegmentAdapter):
            self._mismatched_result_count += 1

            self.logger.warning(
                "M2 streaming ASR: received an ASRTaskResult that did "
                "not originate from a ProcessingWindow submission; "
                "skipping (mismatched_result_count=%d).",
                self._mismatched_result_count,
            )

            return None

        window = segment.window

        words: tuple = ()

        if task_result.error is not None:
            success = False
            error: str | None = task_result.error
            text: str | None = None
            inference_time_ms: float | None = None
            rtf: float | None = None
        elif task_result.result is None:
            success = False
            error = "ASR worker returned no result and no error."
            text = None
            inference_time_ms = None
            rtf = None
        else:
            success = True
            error = None
            text = task_result.result.text
            inference_time_ms = task_result.result.inference_time_ms
            rtf = task_result.result.rtf
            words = task_result.result.words

        return ASRHypothesis(
            hypothesis_id=self._hypothesis_id_generator.next(),
            utterance_id=window.utterance_id,
            window_id=window.window_id,
            generation=window.generation,
            submitted_at=task_result.submitted_at,
            started_at=task_result.started_at,
            completed_at=task_result.completed_at,
            queue_wait_ms=task_result.queue_wait_ms,
            segment_to_transcript_ms=(
                task_result.segment_to_transcript_ms
            ),
            window_created_at=window.created_at,
            audio_start_time=window.audio_start_time,
            audio_end_time=window.audio_end_time,
            window_duration_ms=window.duration_ms,
            context_duration_ms=window.context_duration_ms,
            new_audio_duration_ms=window.new_audio_duration_ms,
            is_final=window.is_final,
            success=success,
            error=error,
            text=text,
            inference_time_ms=inference_time_ms,
            rtf=rtf,
            words=words,
        )

    def _record_sample(self, hypothesis: ASRHypothesis) -> None:
        """Record raw timing samples for later caller-side summarization."""

        self._queue_wait_samples_ms.append(hypothesis.queue_wait_ms)

        if hypothesis.inference_time_ms is not None:
            self._inference_time_samples_ms.append(
                hypothesis.inference_time_ms
            )

        if hypothesis.rtf is not None:
            self._rtf_samples.append(hypothesis.rtf)

    # ------------------------------------------------------------
    # Hypothesis history (Phase 4 will consume this; Phase 3 does not
    # decide stability/commitment from it)
    # ------------------------------------------------------------

    def hypotheses_for(self, utterance_id: int) -> tuple[ASRHypothesis, ...]:
        """Return one utterance's hypotheses ordered by generation."""

        generation_map = self._history.get(utterance_id, {})

        return tuple(
            generation_map[generation]
            for generation in sorted(generation_map)
        )

    def latest_hypothesis(
        self,
        utterance_id: int,
    ) -> ASRHypothesis | None:
        """Return the highest-generation hypothesis for one utterance."""

        generation_map = self._history.get(utterance_id, {})

        if not generation_map:
            return None

        return generation_map[max(generation_map)]

    def hypothesis_count(self, utterance_id: int) -> int:
        """Return how many hypotheses exist for one utterance."""

        return len(self._history.get(utterance_id, {}))

    def revision_count(self, utterance_id: int) -> int:
        """Count consecutive-generation text changes (OBSERVATION ONLY).

        This is purely descriptive. Phase 3 does not use this value to
        make any stability, commitment, or scheduling decision - that
        is explicitly reserved for Phase 4/M3.

        Canonical Phase 4C-1 name: `hypothesis_revision_count` - a
        different quantity from
        `streaming.local_agreement.LocalAgreementTracker.revision_count()`
        (`alignment_revision_count`, overlap-alignment-based). See that
        module's docstring "canonical metric names" section. Never
        report either bare as "revision_count" without specifying which.
        This method's own behavior is unchanged.
        """

        ordered_successful = [
            hypothesis
            for hypothesis in self.hypotheses_for(utterance_id)
            if hypothesis.success
        ]

        return sum(
            1
            for previous, current in zip(
                ordered_successful[:-1],
                ordered_successful[1:],
            )
            if previous.text != current.text
        )

    def reset_utterance(self, utterance_id: int) -> None:
        """Discard hypothesis history for one utterance.

        Intended to be called by the caller after
        `AdaptiveBuffer.reset()` abandons an utterance, purely for
        memory hygiene - correctness does not depend on this being
        called, since history is already partitioned by utterance_id.
        """

        self._history.pop(utterance_id, None)

    # ------------------------------------------------------------
    # Metrics (observational only)
    # ------------------------------------------------------------

    @property
    def windows_offered_count(self) -> int:
        """Return how many distinct windows were ever offered for ASR."""

        return len(self._offered_window_ids)

    @property
    def dropped_window_count(self) -> int:
        """Return how many windows ASRWorker's bounded queue rejected."""

        return len(self._dropped_before_asr)

    @property
    def dropped_windows(self) -> tuple[tuple[ProcessingWindow, WindowDropReason], ...]:
        """Return a snapshot of (window, reason) for every rejected window."""

        return tuple(self._dropped_before_asr)

    @property
    def mismatched_result_count(self) -> int:
        """Return how many polled results could not be correlated."""

        return self._mismatched_result_count

    @property
    def duplicate_result_count(self) -> int:
        """Return how many polled results were duplicates of a known generation."""

        return self._duplicate_result_count

    @property
    def max_observed_queue_size(self) -> int:
        """Return the largest ASRWorker.pending_count seen at submission time."""

        return self._max_observed_queue_size

    @property
    def queue_status(self) -> QueueBackpressureStatus:
        """Return the current bounded ASR queue status (Phase 1 type, reused)."""

        return QueueBackpressureStatus.evaluate(
            capacity=self.asr_worker.queue_size,
            current_size=self.asr_worker.pending_count,
        )

    @property
    def queue_wait_samples_ms(self) -> tuple[float, ...]:
        """Return every observed queue_wait_ms sample, in completion order."""

        return tuple(self._queue_wait_samples_ms)

    @property
    def inference_time_samples_ms(self) -> tuple[float, ...]:
        """Return every observed inference_time_ms sample (successes only)."""

        return tuple(self._inference_time_samples_ms)

    @property
    def rtf_samples(self) -> tuple[float, ...]:
        """Return every observed RTF sample (successes only)."""

        return tuple(self._rtf_samples)

    def metrics_snapshot(self) -> dict[str, int]:
        """Return workload/queue counters (no computed statistics).

        Percentile/mean summaries over `queue_wait_samples_ms`,
        `inference_time_samples_ms`, and `rtf_samples` are left to the
        caller (e.g. main.py reuses
        `evaluation.asr_evaluation.summarize_latency`/`percentile`),
        so this module has no dependency on the evaluation package.
        """

        return {
            "windows_offered": self.windows_offered_count,
            "asr_submitted": self.asr_worker.submitted_count,
            "asr_completed": self.asr_worker.completed_count,
            "asr_failed": self.asr_worker.failed_count,
            "asr_queue_dropped": self.asr_worker.dropped_count,
            "rejected_at_submission": self.dropped_window_count,
            "mismatched_results": self.mismatched_result_count,
            "duplicate_results": self.duplicate_result_count,
            "current_queue_size": self.asr_worker.pending_count,
            "max_observed_queue_size": self.max_observed_queue_size,
            "asr_queue_capacity": self.asr_worker.queue_size,
        }
