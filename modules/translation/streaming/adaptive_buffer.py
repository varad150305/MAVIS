"""M2 Phase 2: deterministic adaptive audio buffer.

This module implements `AdaptiveBuffer`, which replaces M1's fixed
one-segment-per-utterance buffering with a policy that can emit many
overlapping `ProcessingWindow`s per utterance. It builds directly on
the Phase 1 data model (`streaming.streaming_state`) and the Phase 1
backpressure primitives (`streaming.backpressure`) - it does not
introduce a second, competing state representation.

Phase 2 scope only. This module does NOT:
    - call Whisper or any ASR engine,
    - implement LocalAgreement stable-prefix commitment (Phase 4),
    - implement M3 stability-/compute-aware translation scheduling,
    - implement VAD (it accepts frames and an explicit finalize()
      call/action; it does not decide when speech starts or ends).

Data flow::

    AudioFrame -> StreamingAudioState (Phase 1) -> AdaptiveBuffer
    policy (this module) -> ProcessingWindow (Phase 1)

============================================================
Window / context / step semantics
============================================================

Every `StreamingAudioState` already distinguishes two audio pools
(Phase 1):
    - `new_audio_since_last_asr`: audio that arrived since the
      previous window was built.
    - `context_audio`: audio retained specifically to give a future
      window left-context.

`AdaptiveBuffer` decides *when* to call `state.build_window()` (which
folds both pools into one `ProcessingWindow` and rotates
`new_audio_since_last_asr` into `context_audio`), and *how much*
`context_audio` to keep afterward. It never sends the whole
accumulated utterance as one window - only `context_audio +
new_audio_since_last_asr`, bounded by configuration.

============================================================
Emission policy (deterministic, content-free)
============================================================

On every `ingest(frame)` call, after folding the frame into state,
`AdaptiveBuffer` computes:
    - `context_ms` = `state.context_duration_ms`
    - `new_ms` = `state.new_audio_duration_ms`
    - `pending_total_ms` = `context_ms + new_ms` (the size the next
      window would have if built right now)

A window is emitted when, in order:

1. Hard ceiling (always wins, cannot be delayed by backpressure):
   if `pending_total_ms >= max_window_ms`, emit immediately. This is
   the M2 analogue of `FixedProcessingBuffer`'s forced max-duration
   split - it guarantees a window can never silently grow past
   `max_window_ms` and audio is never lost.

2. Otherwise, compute an adaptive threshold for how much NEW audio
   must have arrived:
       - `baseline_ms` = `initial_window_ms` for the utterance's very
         first window (`generation` 1), else `step_ms` for every
         later window.
       - If the window queue is currently full
         (`queue_status.is_full`), raise the threshold to
         `max(baseline_ms, max_window_ms - context_ms)`: instead of
         emitting many small windows that would just queue up stale
         behind a full downstream queue, the buffer waits and
         consolidates as much audio as `max_window_ms` allows into
         one larger, more valuable window. This is the only place
         backpressure state (Phase 1's `QueueBackpressureStatus`)
         feeds into the policy - never ASR content.
       - The threshold is never allowed to drop below
         `min_new_audio_ms` (an absolute floor against
         near-empty/noisy windows).
   Emission then requires BOTH `new_ms >= threshold_ms` AND
   `pending_total_ms >= min_window_ms` (the absolute minimum useful
   window size, matters mainly for the very first window before any
   context exists).

Near the limits: below `min_window_ms` nothing is ever emitted
(except at finalization, see below); at/above `max_window_ms`
emission is unconditional. Between them, the effective threshold only
ever moves *up* under backpressure, never down - the policy never
tries to emit sooner than configured to "catch up".

Finalization (`finalize()`) bypasses all of the above thresholds -
see the class docstring for its exact rule, since it must guarantee
the audio tail is never lost regardless of how small it is.

============================================================
Context retention (`overlap_ms` vs `max_context_ms`)
============================================================

The requested initial configuration includes both `overlap_ms` and
`max_context_ms`. In this design they are NOT two independently
enforced trims - that would make one of them silently dead: whichever
was smaller would always win, and the other would never do anything.
Instead:
    - `overlap_ms` is the trim TARGET: after every window is built,
      `context_audio` is trimmed (oldest frames dropped first, in
      strict sequence-number order) down to at most `overlap_ms` of
      audio.
    - `max_context_ms` is validated as an upper CEILING that
      `overlap_ms` must not exceed (`overlap_ms <= max_context_ms`).
      Phase 2 does not vary the trim target at runtime, so the two
      values coincide whenever `overlap_ms == max_context_ms` (the
      shipped default); `max_context_ms` exists as headroom for a
      later phase that might want to grow the retained context
      adaptively without also being free to violate an absolute cap.

This is documented here, per the Phase 2 instructions, as a
deliberate simplification of a redundant pair of controls rather than
two silently-overlapping trims.

============================================================
Backpressure / window queue (Phase 3 integration point)
============================================================

Phase 3 will connect a real ASR worker to consume `ProcessingWindow`s.
Duplicating `asr.worker.ASRWorker`'s own bounded `queue.Queue` here
would be wrong - that queue does not exist yet for windows, and
Phase 2 must not invent a fake one just to have something to point at
Whisper. Instead, `AdaptiveBuffer` tracks only a minimal, bounded
*view* of "windows produced but not yet acknowledged as consumed":

    - Every emitted window is appended to an internal bounded list
      (`window_queue_size` from config, default 4).
    - `queue_status` (a Phase 1 `QueueBackpressureStatus`) reports
      whether that view is full - this is what feeds the backpressure
      branch of the emission policy above.
    - When `window_queue_policy == "keep_latest"` (the only policy
      Phase 2 implements) and the view is full, the OLDEST pending
      window is superseded (evicted) to make room for the newest one.
      This is recorded via Phase 1's `WindowDropReason
      .SUPERSEDED_BY_NEWER_WINDOW`, exposed on
      `last_superseded_window`/`last_superseded_reason`, and counted
      in `superseded_count` - nothing is silently discarded without
      being observable.
    - `acknowledge(window_id)` removes a window from the pending view.
      **This is the Phase 3 integration point**: when the real ASR
      worker dequeues/starts processing a window, it should call
      `adaptive_buffer.acknowledge(window.window_id)`. Phase 2 never
      calls this itself, since nothing consumes windows yet.

No unbounded backlog is possible: the pending view can never hold
more than `window_queue_size` windows, by construction.
"""

from __future__ import annotations

import logging

from audio.audio_stream import AudioFrame
from streaming.backpressure import QueueBackpressureStatus, WindowDropReason
from streaming.streaming_state import (
    MonotonicIdGenerator,
    ProcessingWindow,
    StreamingAudioState,
    StreamingLifecycleState,
)


class AdaptiveBuffer:
    """Deterministic M2 adaptive buffer: AudioFrame -> ProcessingWindow.

    Lifecycle (per utterance): IDLE (no `StreamingAudioState` created
    yet, `state` is None) -> ACTIVE (`ingest()` has been called at
    least once) -> FINALIZED (`finalize()` has been called). Frames
    are rejected once FINALIZED (`state.append_frame` raises
    `RuntimeError`, propagated unchanged - see Phase 1). `reset()`
    abandons the current utterance (if any) without emitting anything
    and returns to IDLE; the next `ingest()` call starts a brand new
    utterance with a fresh `utterance_id` (the old id is never
    reused, avoiding any risk of conflating two different
    utterances under one id).

    See the module docstring for the full emission policy, context
    retention design, and backpressure handling.
    """

    #: Window-queue policies implemented in Phase 2. Any other value
    #: is rejected at construction time rather than silently ignored.
    SUPPORTED_WINDOW_QUEUE_POLICIES = frozenset({"keep_latest"})

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        initial_window_ms: int,
        min_window_ms: int,
        max_window_ms: int,
        step_ms: int,
        min_new_audio_ms: int,
        max_context_ms: int,
        overlap_ms: int,
        window_queue_size: int = 4,
        window_queue_policy: str = "keep_latest",
        logger: logging.Logger | None = None,
    ) -> None:
        """Create an AdaptiveBuffer.

        Args:
            sample_rate: Expected audio sample rate for every
                utterance this buffer will accumulate.
            channels: Expected channel count.
            initial_window_ms: Target new-audio duration before the
                first window (generation 1) of an utterance is
                emitted.
            min_window_ms: Absolute minimum total window size
                (context + new audio) for a non-final emission.
            max_window_ms: Absolute maximum total window size; always
                forces emission when reached, regardless of
                backpressure or thresholds.
            step_ms: Target new-audio duration between windows after
                the first one.
            min_new_audio_ms: Absolute floor on new audio required
                before any non-final emission, regardless of
                backpressure adjustments.
            max_context_ms: Upper ceiling that `overlap_ms` must not
                exceed (see module docstring).
            overlap_ms: Target amount of retained context audio kept
                after each window (the active trim target).
            window_queue_size: Maximum number of not-yet-acknowledged
                windows retained in the bounded pending view.
            window_queue_policy: Policy applied when the pending view
                is full. Only `"keep_latest"` is implemented.
            logger: Optional logger, matching the style of
                `streaming.buffer.FixedProcessingBuffer` and
                `vad.vad.VAD`.

        Raises:
            ValueError: If any configuration value is invalid or
                internally inconsistent (see `_validate_configuration`).
        """

        self._validate_configuration(
            sample_rate=sample_rate,
            channels=channels,
            initial_window_ms=initial_window_ms,
            min_window_ms=min_window_ms,
            max_window_ms=max_window_ms,
            step_ms=step_ms,
            min_new_audio_ms=min_new_audio_ms,
            max_context_ms=max_context_ms,
            overlap_ms=overlap_ms,
            window_queue_size=window_queue_size,
            window_queue_policy=window_queue_policy,
        )

        self.sample_rate = sample_rate
        self.channels = channels
        self.initial_window_ms = initial_window_ms
        self.min_window_ms = min_window_ms
        self.max_window_ms = max_window_ms
        self.step_ms = step_ms
        self.min_new_audio_ms = min_new_audio_ms
        self.max_context_ms = max_context_ms
        self.overlap_ms = overlap_ms
        self.window_queue_size = window_queue_size
        self.window_queue_policy = window_queue_policy

        self.logger = (
            logger
            if logger is not None
            else logging.getLogger(__name__)
        )

        self._utterance_id_generator = MonotonicIdGenerator()
        self._window_id_generator = MonotonicIdGenerator()

        self._state: StreamingAudioState | None = None
        self._pending_windows: list[ProcessingWindow] = []

        self._superseded_count = 0
        self.last_superseded_window: ProcessingWindow | None = None
        self.last_superseded_reason: WindowDropReason | None = None

        # Phase 4C-1: pure observation counter, never consulted by the
        # emission/backpressure policy above. Tracks this buffer's
        # lifetime, exactly like `_superseded_count` - see
        # `max_pending_occupancy` and `reset()` for why it is NOT
        # cleared per-utterance.
        self._max_pending_occupancy = 0

    @staticmethod
    def _validate_configuration(
        sample_rate: int,
        channels: int,
        initial_window_ms: int,
        min_window_ms: int,
        max_window_ms: int,
        step_ms: int,
        min_new_audio_ms: int,
        max_context_ms: int,
        overlap_ms: int,
        window_queue_size: int,
        window_queue_policy: str,
    ) -> None:
        """Validate AdaptiveBuffer configuration.

        Raises:
            ValueError: For any invalid or inconsistent combination.
        """

        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero.")

        if channels <= 0:
            raise ValueError("channels must be greater than zero.")

        for name, value in (
            ("initial_window_ms", initial_window_ms),
            ("min_window_ms", min_window_ms),
            ("max_window_ms", max_window_ms),
            ("step_ms", step_ms),
            ("min_new_audio_ms", min_new_audio_ms),
        ):
            if value <= 0:
                raise ValueError(
                    f"{name} must be greater than zero."
                )

        if max_context_ms < 0:
            raise ValueError("max_context_ms cannot be negative.")

        if overlap_ms < 0:
            raise ValueError("overlap_ms cannot be negative.")

        if min_window_ms > max_window_ms:
            raise ValueError(
                "min_window_ms cannot exceed max_window_ms."
            )

        if not min_window_ms <= initial_window_ms <= max_window_ms:
            raise ValueError(
                "initial_window_ms must be between min_window_ms "
                "and max_window_ms."
            )

        if overlap_ms > max_context_ms:
            raise ValueError(
                "overlap_ms cannot exceed max_context_ms "
                "(overlap_ms is the active trim target and must "
                "fit within the configured context ceiling)."
            )

        if overlap_ms >= max_window_ms:
            raise ValueError(
                "overlap_ms must be smaller than max_window_ms "
                "(retained context alone cannot be allowed to "
                "consume the entire window budget)."
            )

        if min_new_audio_ms >= max_window_ms:
            raise ValueError(
                "min_new_audio_ms must be smaller than max_window_ms."
            )

        if window_queue_size <= 0:
            raise ValueError(
                "window_queue_size must be greater than zero."
            )

        if (
            window_queue_policy
            not in AdaptiveBuffer.SUPPORTED_WINDOW_QUEUE_POLICIES
        ):
            raise ValueError(
                "Unsupported window_queue_policy: "
                f"{window_queue_policy!r}. Supported policies: "
                f"{sorted(AdaptiveBuffer.SUPPORTED_WINDOW_QUEUE_POLICIES)}"
            )

    # ------------------------------------------------------------
    # Read-only introspection
    # ------------------------------------------------------------

    @property
    def state(self) -> StreamingAudioState | None:
        """Return the current utterance state, or None if IDLE."""

        return self._state

    @property
    def active(self) -> bool:
        """Return whether an utterance is currently ACTIVE."""

        return self._state is not None and self._state.active

    @property
    def pending_window_count(self) -> int:
        """Return the number of not-yet-acknowledged pending windows."""

        return len(self._pending_windows)

    @property
    def pending_windows(self) -> tuple[ProcessingWindow, ...]:
        """Return a snapshot of currently pending (unacknowledged) windows."""

        return tuple(self._pending_windows)

    @property
    def queue_status(self) -> QueueBackpressureStatus:
        """Return the current bounded pending-window queue status."""

        return QueueBackpressureStatus.evaluate(
            capacity=self.window_queue_size,
            current_size=len(self._pending_windows),
        )

    @property
    def superseded_count(self) -> int:
        """Return how many pending windows have been superseded so far."""

        return self._superseded_count

    @property
    def max_pending_occupancy(self) -> int:
        """Return the largest pending-window count ever observed (Phase 4C-1).

        Purely observational - never influences emission or backpressure
        decisions (see the emission policy in the module docstring,
        unchanged). By construction this can never exceed
        `window_queue_size`, since `_enqueue_window` always evicts down
        to capacity before appending.

        Tracks this buffer's LIFETIME, exactly like `superseded_count`:
        NOT reset by `reset()` (see that method's docstring - the
        pending-window view and its counters intentionally survive
        utterance resets, since they track buffer-wide backpressure
        behavior, not any single utterance).

        A superseded/evicted pending window is NOT necessarily a saved
        ASR invocation - it only reflects AdaptiveBuffer's own pre-ASR
        view. Whether the evicted window had already reached ASR is a
        separate question, answered by `StreamingASRDispatcher`'s own
        counters, never by this one.
        """

        return self._max_pending_occupancy

    # ------------------------------------------------------------
    # Public input
    # ------------------------------------------------------------

    def ingest(self, frame: AudioFrame) -> list[ProcessingWindow]:
        """Fold one AudioFrame into the active utterance.

        Starts a new utterance automatically if none is active (IDLE
        -> ACTIVE). Never starts a new utterance implicitly once the
        current one has been finalized - call `reset()` first.

        Args:
            frame: The captured audio frame to fold in.

        Returns:
            A list containing the newly emitted ProcessingWindow, or
            an empty list if no window was emitted for this frame.

        Raises:
            ValueError: If `frame` does not match this buffer's
                configured sample_rate/channels.
            RuntimeError: If the active utterance has already been
                finalized (frames must not silently continue to be
                accepted after finalization).
        """

        if self._state is None:
            self._validate_frame_format(frame)
            self._start_new_utterance(frame)

        # Raises RuntimeError if FINALIZED, ValueError on a mismatch
        # that only manifests after the first frame (defense in depth
        # - _validate_frame_format already covers the first frame).
        self._state.append_frame(frame)

        window = self._evaluate_emission(force_final=False)

        return [window] if window is not None else []

    def finalize(self) -> list[ProcessingWindow]:
        """Signal that the active utterance has ended.

        Emits the final remaining useful audio exactly once: if there
        is unflushed new audio, or if no window has ever been built
        for this utterance, a final ProcessingWindow (`is_final=True`)
        is built from whatever audio remains. If the last regular
        window already consumed everything and no new audio has
        arrived since, nothing is emitted (there is nothing new to
        say, and CLAUDE.md's finality is instead a property of the
        already-`FINALIZED` `StreamingAudioState`, not a redundant
        window).

        Returns:
            A list containing the final ProcessingWindow, or an empty
            list if there was nothing left to flush.

        Raises:
            RuntimeError: If no utterance is active, or the active
                utterance has already been finalized (finalize() may
                be called at most once per utterance).
        """

        if self._state is None:
            raise RuntimeError(
                "Cannot finalize before any audio was ingested."
            )

        if self._state.state == StreamingLifecycleState.FINALIZED:
            raise RuntimeError(
                "This utterance has already been finalized."
            )

        self._state.finalize()

        window = self._evaluate_emission(force_final=True)

        return [window] if window is not None else []

    def reset(self) -> None:
        """Abandon the current utterance (if any) without emitting.

        The next `ingest()` call starts a brand new utterance with a
        fresh `utterance_id`. Does not affect the pending window
        queue, superseded-window counters, or id generators - those
        track this buffer's lifetime, not any single utterance.
        """

        if self._state is not None:
            self.logger.info(
                "M2 adaptive buffer: utterance abandoned via reset() "
                "| utterance_id=%d",
                self._state.utterance_id,
            )

        self._state = None

    def acknowledge(self, window_id: int) -> bool:
        """Remove one window from the pending (unacknowledged) view.

        This is the Phase 3 integration point: once a real ASR worker
        dequeues/starts processing a window, it should call this with
        that window's `window_id`. Phase 2 never calls this itself.

        Args:
            window_id: The `ProcessingWindow.window_id` to acknowledge.

        Returns:
            True if a matching pending window was found and removed,
            False if it was not present (e.g. it was already
            superseded).
        """

        for index, window in enumerate(self._pending_windows):
            if window.window_id == window_id:
                del self._pending_windows[index]
                return True

        return False

    # ------------------------------------------------------------
    # Internal policy
    # ------------------------------------------------------------

    def _validate_frame_format(self, frame: AudioFrame) -> None:
        """Validate a frame against this buffer's configured format.

        Raises:
            ValueError: If sample_rate or channels do not match.
        """

        if frame.sample_rate != self.sample_rate:
            raise ValueError(
                f"AudioFrame sample_rate ({frame.sample_rate}) does "
                "not match AdaptiveBuffer sample_rate "
                f"({self.sample_rate})."
            )

        if frame.channels != self.channels:
            raise ValueError(
                f"AudioFrame channels ({frame.channels}) does not "
                f"match AdaptiveBuffer channels ({self.channels})."
            )

    def _start_new_utterance(self, frame: AudioFrame) -> None:
        """Create a fresh StreamingAudioState for a new utterance."""

        utterance_id = self._utterance_id_generator.next()

        self._state = StreamingAudioState(
            utterance_id=utterance_id,
            sample_rate=self.sample_rate,
            channels=self.channels,
            start_timestamp=frame.capture_timestamp,
            last_update_timestamp=frame.capture_timestamp,
        )

        self.logger.info(
            "M2 adaptive buffer: utterance started | utterance_id=%d",
            utterance_id,
        )

    def _evaluate_emission(
        self,
        force_final: bool,
    ) -> ProcessingWindow | None:
        """Decide whether to build/emit a window right now.

        See the module docstring for the exact policy. `force_final`
        selects the finalize() rule (bypasses all thresholds except
        "is there anything unflushed to say").
        """

        state = self._state
        assert state is not None

        context_ms = state.context_duration_ms
        new_ms = state.new_audio_duration_ms
        pending_total_ms = context_ms + new_ms

        if force_final:
            has_unflushed_tail = (
                new_ms > 0.0
                or state.windows_emitted == 0
            )

            if not has_unflushed_tail:
                return None

        else:
            if pending_total_ms <= 0.0:
                return None

            hard_cap_reached = pending_total_ms >= self.max_window_ms

            if not hard_cap_reached:
                baseline_ms = (
                    self.initial_window_ms
                    if state.windows_emitted == 0
                    else self.step_ms
                )

                threshold_ms = baseline_ms

                if self.queue_status.is_full:
                    # Downstream is congested: consolidate into a
                    # larger window instead of emitting one that
                    # would just queue up stale. Never lowers the
                    # threshold, only raises it.
                    threshold_ms = max(
                        threshold_ms,
                        self.max_window_ms - context_ms,
                    )

                threshold_ms = max(
                    threshold_ms,
                    self.min_new_audio_ms,
                )

                if new_ms < threshold_ms:
                    return None

                if pending_total_ms < self.min_window_ms:
                    return None

        window = state.build_window(
            window_id=self._window_id_generator.next(),
            is_final=force_final,
        )

        self._trim_context(state)
        self._enqueue_window(window)

        self.logger.info(
            "M2 adaptive buffer: window emitted | "
            "utterance_id=%d window_id=%d generation=%d | "
            "context_ms=%.1f new_ms=%.1f duration_ms=%.1f "
            "is_final=%s",
            window.utterance_id,
            window.window_id,
            window.generation,
            window.context_duration_ms,
            window.new_audio_duration_ms,
            window.duration_ms,
            window.is_final,
        )

        return window

    def _trim_context(self, state: StreamingAudioState) -> None:
        """Trim retained context down to at most `overlap_ms`.

        Drops the oldest frames first (strict sequence order),
        matching the deterministic, sequence-number-based selection
        required for context/new-audio boundaries. Only ever removes
        frames from `context_audio` - `new_audio_since_last_asr` is
        never touched here, so new audio can never be lost through
        trimming.
        """

        while (
            state.context_audio
            and state.context_duration_ms > self.overlap_ms
        ):
            state.context_audio.pop(0)

    def _enqueue_window(self, window: ProcessingWindow) -> None:
        """Add a window to the bounded pending view.

        Applies the `keep_latest` policy when the view is already at
        capacity: the oldest pending window is superseded (evicted)
        to make room, and the eviction is recorded (never silent).
        """

        if len(self._pending_windows) >= self.window_queue_size:
            superseded = self._pending_windows.pop(0)

            self._superseded_count += 1
            self.last_superseded_window = superseded
            self.last_superseded_reason = (
                WindowDropReason.SUPERSEDED_BY_NEWER_WINDOW
            )

            self.logger.info(
                "M2 adaptive buffer: window superseded | "
                "utterance_id=%d window_id=%d generation=%d | "
                "reason=%s",
                superseded.utterance_id,
                superseded.window_id,
                superseded.generation,
                WindowDropReason.SUPERSEDED_BY_NEWER_WINDOW.value,
            )

        self._pending_windows.append(window)

        self._max_pending_occupancy = max(
            self._max_pending_occupancy,
            len(self._pending_windows),
        )
