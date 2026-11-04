"""M2 research-alignment pass: persistent streaming audio buffer.

Reference: Macháček, Dabre, Bojar, "Turning Whisper into Real-Time
Transcription System" (Whisper-Streaming, 2023) - a LocalAgreement-style
implementation aligned with that streaming formulation, not a claim of
exact reproduction. See CLAUDE.md and the Phase 4C research-alignment
report for the full literature comparison.

Reference architecture: an audio buffer PERSISTS for the duration of one
utterance and is fully re-decoded by Whisper on every update (not split
into disjoint/overlapping fixed-size windows). Consecutive hypotheses
are therefore repeated decodings of a growing (or confirmation-trimmed)
audio context, never independent windows. The buffer is only trimmed at
a REAL, Whisper-measured word-timestamp boundary once
`streaming.prefix_agreement.PrefixAgreementTracker` confirms a stable
prefix - never after every window (that was `streaming.adaptive_buffer
.AdaptiveBuffer`'s `_trim_context()` behavior, which this class does
NOT reproduce).

`streaming.adaptive_buffer.AdaptiveBuffer` (Phase 2-4C) is kept
UNCHANGED for historical evaluator reproducibility (Phase 4C-2a/4C-3/
4C-3D/4C-3E's reports characterize it, not this class) - `main.py
::run_m2()` now builds this class instead.

============================================================
Why `streaming.streaming_state.StreamingAudioState` needed no changes
============================================================

`StreamingAudioState.build_window()` already folds `context_audio +
new_audio_since_last_asr` into one `ProcessingWindow` and then rotates
ALL of that combined audio into `context_audio` (see its docstring) -
i.e. it already implements "the whole accumulated buffer, re-decoded"
exactly as the reference requires. The only reason the OLD
`AdaptiveBuffer` behaved as disjoint/overlapping windows instead of a
persistent buffer was its own `_trim_context()` call immediately after
every `build_window()`, unconditionally shrinking `context_audio` back
down to a small `overlap_ms`. This class simply never calls anything
like that - it only ever removes audio from `context_audio` inside
`trim_confirmed()`, and only up to a REAL confirmed boundary.

============================================================
Single decode in flight (why the old "ghost utterance" mechanism
cannot occur here)
============================================================

`AdaptiveBuffer` maintained a bounded QUEUE of up to `window_queue_size`
pending windows and evicted the oldest (`keep_latest`) when it filled -
the confirmed root cause (Phase 4C-3D/4C-3E) of utterances whose every
window was lost before producing a hypothesis, so
`LocalAgreementTracker.finalize_utterance()` silently no-opped for a
tracker state that was never created.

This class instead tracks a single boolean, `pending`: at most one
ProcessingWindow may be in flight per utterance at a time (matching the
reference's own single-decode-per-stream model - see the Phase 4C
research-alignment report's literature comparison for the exact
"MinChunkSize" trigger this mirrors). While `pending` is True, newly
ingested audio simply accumulates in `new_audio_since_last_asr` - it is
NEVER superseded or dropped. The next window built once the pending
decode completes covers everything accumulated meanwhile, whole. There
is no queue to overflow and nothing to evict.

============================================================
Speech-onset pre-roll (M2 fix, VAD input ladder finding)
============================================================

VAD only reports SPEECH after `minimum_speech_duration_ms` of
above-threshold subframes, so starting ingestion at that frame withholds
the utterance onset from ASR (evaluation/vad_segmentation_audit.py: median
72 ms per onset; evaluation/vad_input_ladder.py: restoring it recovered
~9 WER points on the 300-file corpus). The caller therefore reports every
NON-ingested frame via `observe_non_speech()`; up to `pre_roll_ms` of the
most recent such frames are prepended to the next utterance, before its
first window is built. Semantics are the M1 `FixedProcessingBuffer`
pre-roll (streaming/buffer.py), as validated by the ladder's Arm 2:

  - whole AudioFrames; the oldest frame is dropped only while the rest
    still total >= `pre_roll_ms` (3 x 100 ms frames at 300 ms);
  - collected only while no utterance is active, and cleared both when an
    utterance is finalized and when its pre-roll is consumed - so it holds
    only frames seen after the previous utterance ended and never carries
    across utterances;
  - only frames whose sequence numbers run contiguously up to the onset
    frame are used, and never a frame at or before the last ingested
    sequence number (no duplication of the previous utterance's audio);
  - `clear_pre_roll()` starts a new stream (file/session), so pre-roll
    can never reach before the stream's first frame.
"""

from __future__ import annotations

from collections import deque
import logging
from typing import Any, Deque

from audio.audio_stream import AudioFrame
from streaming.streaming_state import (
    MonotonicIdGenerator,
    ProcessingWindow,
    StreamingAudioState,
    StreamingLifecycleState,
)


class StreamingBuffer:
    """Persistent, single-utterance-at-a-time, single-decode-in-flight buffer.

    One instance is reused across an entire run (many utterances in
    sequence), exactly like `AdaptiveBuffer`. Only one utterance may be
    ACTIVE at a time - call `reset()` before starting a new one, mirroring
    `AdaptiveBuffer.reset()`'s own contract.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        min_new_audio_ms: float,
        max_buffer_ms: float,
        logger: logging.Logger | None = None,
        pre_roll_ms: float = 300.0,
    ) -> None:
        """Create a buffer.

        Args:
            min_new_audio_ms: Minimum new (unconfirmed) audio duration
                required before a non-forced window may be built - the
                reference's "MinChunkSize" trigger.
            max_buffer_ms: Bounded-memory safety ceiling (Keep: bounded
                memory). If the accumulated buffer ever reaches this
                size without a fresh LocalAgreement confirmation to
                trim it, a window is force-built anyway - the same role
                as the reference's ~30s buffer-overflow fallback,
                scaled to this project's much shorter VAD-bounded
                utterances rather than indefinite unsegmented audio.
            pre_roll_ms: Maximum audio preceding each speech onset that is
                prepended to the utterance (see module docstring). 0
                disables pre-roll.

        Raises:
            ValueError: For non-positive thresholds or negative pre-roll.
        """

        if min_new_audio_ms <= 0:
            raise ValueError("min_new_audio_ms must be greater than zero.")

        if max_buffer_ms <= 0:
            raise ValueError("max_buffer_ms must be greater than zero.")

        if pre_roll_ms < 0:
            raise ValueError("pre_roll_ms cannot be negative.")

        self.sample_rate = sample_rate
        self.channels = channels
        self.min_new_audio_ms = min_new_audio_ms
        self.max_buffer_ms = max_buffer_ms
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self._utterance_ids = MonotonicIdGenerator()
        self._window_ids = MonotonicIdGenerator()

        self._state: StreamingAudioState | None = None
        self._pending = False

        # Buffer-lifetime diagnostics (mirrors AdaptiveBuffer's own
        # counters for evaluator/report comparability).
        self.windows_built: int = 0
        self.max_buffer_forced_count: int = 0

        # Speech-onset pre-roll state and instrumentation.
        self.pre_roll_ms = pre_roll_ms
        self._pre_roll: Deque[AudioFrame] = deque()
        self._last_ingested_sequence: int | None = None
        self.utterances_started: int = 0
        self.utterances_with_pre_roll: int = 0
        self.total_pre_roll_ms: float = 0.0
        #: Frames withheld from pre-roll to keep the invariants (should stay 0
        #: in a gap-free stream; nonzero values are reported, never hidden).
        self.pre_roll_overlap_frames_dropped: int = 0
        self.pre_roll_discontinuity_frames_dropped: int = 0
        #: utterance_id -> {"pre_roll_ms", "pre_roll_frames",
        #: "first_sequence_number", "onset_sequence_number", "first_window_ms"}
        self.utterance_pre_roll: dict[int, dict[str, Any]] = {}

    @property
    def active(self) -> bool:
        """Whether an utterance is currently accumulating audio."""

        return self._state is not None and self._state.active

    @property
    def state(self) -> StreamingAudioState | None:
        """The current utterance's accumulation state, or None."""

        return self._state

    @property
    def pending(self) -> bool:
        """Whether a decode is currently in flight for this utterance."""

        return self._pending

    def mark_submitted(self) -> None:
        """Call immediately after successfully submitting a built window."""

        self._pending = True

    def mark_result_received(self) -> None:
        """Call once the pending window's ASRHypothesis has arrived
        (success or failure) - frees the single decode-in-flight slot."""

        self._pending = False

    def ingest(self, frame: AudioFrame) -> None:
        """Fold one AudioFrame into the active (or newly started) utterance.

        Does not itself decide whether to build a window - call
        `try_build_window()` afterward.
        """

        if self._state is None:
            pre_roll = self._take_pre_roll(frame)
            first = pre_roll[0] if pre_roll else frame

            self._state = StreamingAudioState(
                utterance_id=self._utterance_ids.next(),
                sample_rate=self.sample_rate,
                channels=self.channels,
                start_timestamp=first.capture_timestamp,
                last_update_timestamp=first.capture_timestamp,
            )

            for pre_frame in pre_roll:
                self._state.append_frame(pre_frame)

            pre_roll_ms = sum(f.duration_ms for f in pre_roll)
            self.utterances_started += 1
            if pre_roll:
                self.utterances_with_pre_roll += 1
            self.total_pre_roll_ms += pre_roll_ms
            self.utterance_pre_roll[self._state.utterance_id] = {
                "pre_roll_ms": pre_roll_ms,
                "pre_roll_frames": len(pre_roll),
                "first_sequence_number": first.sequence_number,
                "onset_sequence_number": frame.sequence_number,
                "last_sequence_number": frame.sequence_number,
                "first_window_ms": None,
            }

        self._state.append_frame(frame)
        self._last_ingested_sequence = frame.sequence_number

        record = self.utterance_pre_roll.get(self._state.utterance_id)
        if record is not None:
            record["last_sequence_number"] = frame.sequence_number

    def observe_non_speech(self, frame: AudioFrame) -> None:
        """Report a frame the caller did NOT ingest (VAD state not SPEECH).

        Retained as candidate pre-roll only while no utterance is active;
        ignored otherwise. Never builds a window.
        """

        if self.pre_roll_ms <= 0 or self.active:
            return

        self._pre_roll.append(frame)

        # FixedProcessingBuffer._update_pre_roll() trimming rule.
        total_ms = sum(f.duration_ms for f in self._pre_roll)
        while len(self._pre_roll) > 1 and total_ms - self._pre_roll[0].duration_ms >= self.pre_roll_ms:
            total_ms -= self._pre_roll.popleft().duration_ms

    def clear_pre_roll(self) -> None:
        """Start a new independent audio stream (new file or session):
        discard retained pre-roll and forget the last ingested sequence."""

        self._pre_roll.clear()
        self._last_ingested_sequence = None

    def _take_pre_roll(self, onset: AudioFrame) -> list[AudioFrame]:
        """Consume the retained pre-roll for the utterance starting at `onset`.

        Keeps only frames contiguous with the onset and strictly after the
        last ingested frame; anything else is dropped and counted.
        """

        candidates = list(self._pre_roll)
        self._pre_roll.clear()

        selected: list[AudioFrame] = []
        expected = onset.sequence_number - 1
        for candidate in reversed(candidates):
            if candidate.sequence_number != expected:
                self.pre_roll_discontinuity_frames_dropped += 1
                continue
            if (
                self._last_ingested_sequence is not None
                and candidate.sequence_number <= self._last_ingested_sequence
            ):
                self.pre_roll_overlap_frames_dropped += 1
                continue
            selected.append(candidate)
            expected -= 1

        selected.reverse()
        return selected

    def pre_roll_summary(self) -> dict[str, Any]:
        """Run-level pre-roll instrumentation."""

        return {
            "pre_roll_ms_config": self.pre_roll_ms,
            "utterances_started": self.utterances_started,
            "utterances_with_pre_roll": self.utterances_with_pre_roll,
            "total_pre_roll_ms": self.total_pre_roll_ms,
            "pre_roll_overlap_frames_dropped": self.pre_roll_overlap_frames_dropped,
            "pre_roll_discontinuity_frames_dropped": self.pre_roll_discontinuity_frames_dropped,
        }

    def try_build_window(self, initial_prompt: str | None = None) -> ProcessingWindow | None:
        """Build a window now, if the trigger threshold is met and no
        decode is currently pending.

        Returns None when: no active utterance, a decode is already
        pending, there is no accumulated audio at all, or
        `new_audio_duration_ms` has not yet reached `min_new_audio_ms`
        (unless the `max_buffer_ms` safety ceiling forces it anyway).
        """

        if self._state is None or not self._state.active:
            return None

        if self._pending:
            return None

        if not self._has_new_content():
            return None

        total_buffer_ms = (
            self._state.context_duration_ms + self._state.new_audio_duration_ms
        )

        ceiling_forced = total_buffer_ms >= self.max_buffer_ms

        if not ceiling_forced and self._state.new_audio_duration_ms < self.min_new_audio_ms:
            return None

        return self._build(ceiling_forced=ceiling_forced, initial_prompt=initial_prompt)

    def _has_new_content(self) -> bool:
        """Whether there is any audio not yet covered by a built window.

        NOT the same as "buffer is non-empty": once at least one window
        has been built, `context_audio` (the persistent, unconfirmed
        buffer) is never empty again by design - see module docstring.
        The correct "anything new to decode" signal is
        `new_audio_duration_ms`, except for the very first window
        (`windows_built == 0`), when the entire buffer is still sitting
        in `new_audio_since_last_asr` and counts as new.
        """

        if self.windows_built == 0:
            return (
                self._state.context_duration_ms
                + self._state.new_audio_duration_ms
            ) > 0.0

        return self._state.new_audio_duration_ms > 0.0

    def finalize(self) -> None:
        """Mark the active utterance as finalized.

        Does not itself build a window - the caller must still call
        `try_build_final_window()` (after draining any pending decode)
        to flush whatever remains.
        """

        if self._state is None:
            raise RuntimeError("Cannot finalize before any audio was ingested.")

        if self._state.state == StreamingLifecycleState.FINALIZED:
            raise RuntimeError("This utterance has already been finalized.")

        self._state.finalize()

        # Pre-roll for the next utterance may only come from frames
        # observed after this one ended.
        self._pre_roll.clear()

    def try_build_final_window(
        self, initial_prompt: str | None = None
    ) -> ProcessingWindow | None:
        """Force-build the final window for a finalized utterance.

        Returns None if a decode is still pending (caller must drain it
        first - see module docstring) or if there is no audio left to
        flush (the previous window already consumed everything and no
        new audio arrived since - matches `AdaptiveBuffer.finalize()`'s
        own "nothing new to say" contract).
        """

        if self._state is None or self._pending:
            return None

        if not self._has_new_content():
            return None

        return self._build(ceiling_forced=False, initial_prompt=initial_prompt)

    def _build(
        self, ceiling_forced: bool, initial_prompt: str | None
    ) -> ProcessingWindow:
        is_first_window = self._state.windows_emitted == 0
        window = self._state.build_window(window_id=self._window_ids.next())

        record = self.utterance_pre_roll.get(window.utterance_id)
        if is_first_window and record is not None:
            record["first_window_ms"] = (
                window.context_duration_ms + window.new_audio_duration_ms
            )

        if initial_prompt:
            import dataclasses

            window = dataclasses.replace(window, initial_prompt=initial_prompt)

        self.windows_built += 1

        if ceiling_forced:
            self.max_buffer_forced_count += 1

            self.logger.warning(
                "M2 streaming buffer: max_buffer_ms safety ceiling reached "
                "| utterance_id=%d window_id=%d buffer_ms=%.1f "
                "max_buffer_ms=%.1f - forcing a decode without a fresh "
                "LocalAgreement confirmation (bounded-memory safety "
                "valve, analogous to the Whisper-Streaming reference's "
                "buffer-overflow fallback).",
                window.utterance_id,
                window.window_id,
                window.context_duration_ms + window.new_audio_duration_ms,
                self.max_buffer_ms,
            )

        return window

    def trim_confirmed(self, confirmed_up_to_time: float) -> None:
        """Discard buffered audio strictly before a confirmed boundary.

        `confirmed_up_to_time` is a REAL, Whisper-measured word-end
        timestamp (see `streaming.prefix_agreement`), never an
        interpolated estimate - the caller is responsible for that
        distinction. A frame is dropped only if it ends at or before
        the boundary; a straddling frame is conservatively kept
        (whole-frame granularity, matching
        `AdaptiveBuffer._trim_context()`'s own approach), so confirmed
        audio can never be discarded along with still-unconfirmed
        content.
        """

        if self._state is None:
            return

        self._state.context_audio = [
            frame
            for frame in self._state.context_audio
            if frame.capture_timestamp + frame.duration_ms / 1000.0
            > confirmed_up_to_time
        ]

    def reset(self) -> None:
        """Abandon the current utterance's state (not finalization)."""

        self._state = None
        self._pending = False
