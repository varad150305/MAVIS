"""M2 Phase 1 streaming data model: per-utterance state and ASR windows.

This module defines the data structures needed to represent
incremental streaming audio for one candidate speech utterance. It
does NOT implement:
    - adaptive buffering (Phase 2)
    - incremental/repeated Whisper decoding (Phase 3)
    - LocalAgreement stable-prefix commitment (Phase 4)
    - M3 stability/compute-aware scheduling

It exists so those later phases share one consistent shape for
"audio accumulated so far for this utterance" (`StreamingAudioState`)
and "one immutable snapshot of audio handed to ASR"
(`ProcessingWindow`), instead of each phase inventing its own.

Design notes (see the Phase 1 implementation report for full detail):
    - Audio is stored as lists of the existing, unmodified
      `audio.audio_stream.AudioFrame`, mirroring how
      `streaming.buffer.FixedProcessingBuffer` accumulates frames and
      only concatenates samples when a segment/window is actually
      built. No new audio-frame type is introduced.
    - `StreamingAudioState.context_audio` and
      `.new_audio_since_last_asr` are the two audio pools requested by
      the M2 design: "new since last ASR" vs "retained context".
    - `ProcessingWindow` deliberately mirrors
      `streaming.buffer.SpeechSegment` field names
      (samples/sample_rate/channels/start_sequence_number/
      end_sequence_number) so a future phase can adapt
      `ASRWorker`/`WhisperEngine` with minimal change.
    - Neither type generates its own IDs from hidden global state.
      `utterance_id` is supplied by whatever owns the pool of active
      utterances (a future Phase 2 component); `window_id` is supplied
      by whatever dispatches windows to ASR. `MonotonicIdGenerator` is
      a small reusable counter Phase 2 may use for either, mirroring
      the private `_sequence_number` counter `AudioStream` already
      owns internally today.
    - Correlation without a Hypothesis type yet: a future ASR
      hypothesis only needs to carry `(utterance_id, window_id,
      generation)` plus its own timing to be fully traceable back
      through Audio/VAD -> ProcessingWindow -> ASR -> (future)
      LocalAgreement -> translation. No `Hypothesis` dataclass is
      introduced in Phase 1 because LocalAgreement (Phase 4) is not
      implemented yet and a premature shape would likely be wrong.
    - Timestamps reuse existing semantics: `AudioFrame.capture_timestamp`
      (perf_counter-based) feeds `audio_start_time`/`audio_end_time`.
      `created_at` uses `time.perf_counter()`, matching every other
      timing field in the codebase (`SpeechSegment`, `ASRTaskResult`,
      `TranslationTaskResult`). A separate "submitted to ASR" timestamp
      is deliberately NOT added to `ProcessingWindow` itself - exactly
      like `SpeechSegment` does not carry an ASR submission timestamp
      today. That belongs on a future ASRTask-equivalent wrapper (the
      same way `asr.worker.ASRTask.submitted_at` is separate from the
      `SpeechSegment` it wraps), so ProcessingWindow is not duplicating
      a concern that already has a correct home elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time

import numpy as np

from audio.audio_stream import AudioFrame


class StreamingLifecycleState(str, Enum):
    """Lifecycle of one StreamingAudioState.

    Mirrors the string-Enum style already used by `vad.vad.VADState`.
    """

    # Accumulating audio; may still receive frames and emit windows.
    ACTIVE = "ACTIVE"

    # VAD-driven (or otherwise externally decided) termination has
    # been recorded via finalize(). No further frames should be
    # appended, though a completed final ProcessingWindow may still be
    # built from whatever audio was already accumulated.
    FINALIZED = "FINALIZED"


class MonotonicIdGenerator:
    """Small per-instance monotonically increasing id generator.

    Not wired into any pipeline yet. It exists so a future component
    that owns a pool of `StreamingAudioState` (utterance ids) or a
    window dispatcher (window ids) does not need to reinvent the
    counter pattern `audio.audio_stream.AudioStream` already uses
    internally for `AudioFrame.sequence_number`. Deliberately has no
    module-level/global state so tests and independent pipeline
    instances cannot interfere with each other's numbering.
    """

    def __init__(self, start: int = 1) -> None:
        """Initialize the generator.

        Args:
            start: The first id that will be returned.

        Raises:
            ValueError: If start is not positive.
        """

        if start <= 0:
            raise ValueError("start must be greater than zero.")

        self._next_value = start

    def next(self) -> int:
        """Return the next id and advance the counter."""

        value = self._next_value
        self._next_value += 1
        return value


def _validate_frame_for_state(
    frame: AudioFrame,
    sample_rate: int,
    channels: int,
) -> None:
    """Validate one AudioFrame against a StreamingAudioState's format.

    Raises:
        ValueError: If the frame is empty, non-finite, or does not
            match the state's configured sample rate/channels.
    """

    if frame.samples.size == 0:
        raise ValueError(
            "Cannot append an AudioFrame with empty samples."
        )

    if not np.all(np.isfinite(frame.samples)):
        raise ValueError(
            "AudioFrame contains NaN or infinite samples."
        )

    if frame.sample_rate != sample_rate:
        raise ValueError(
            "AudioFrame sample_rate "
            f"({frame.sample_rate}) does not match "
            f"StreamingAudioState sample_rate ({sample_rate})."
        )

    if frame.channels != channels:
        raise ValueError(
            "AudioFrame channels "
            f"({frame.channels}) does not match "
            f"StreamingAudioState channels ({channels})."
        )


def _concatenate_frames(frames: list[AudioFrame]) -> np.ndarray:
    """Concatenate AudioFrame samples into one float32 array."""

    if not frames:
        return np.empty(0, dtype=np.float32)

    return np.concatenate(
        [item.samples for item in frames]
    ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class ProcessingWindow:
    """One immutable snapshot of audio submitted to ASR (M2).

    A ProcessingWindow is the M2 analogue of a fully-formed M1
    `streaming.buffer.SpeechSegment`, except a single utterance is
    expected to produce many ProcessingWindows over its lifetime (one
    per incremental ASR invocation in Phase 3) rather than exactly one
    SpeechSegment.

    Attributes:
        window_id: Identifier for this specific window, unique across
            the run. Supplied by whatever dispatches windows to ASR
            (not generated internally - see module docstring).
        utterance_id: Identifier of the `StreamingAudioState` this
            window was built from. Constant across every window
            belonging to the same utterance; the primary key used to
            trace Audio/VAD -> ProcessingWindow -> ASR -> (future)
            LocalAgreement -> translation.
        generation: 1-based count of this window within its
            utterance (the 1st window built for an utterance has
            generation=1, the 2nd has generation=2, ...). Cheap to
            compare for "is this window superseded by a newer one for
            the same utterance" without a global lookup; `window_id`
            alone does not convey that ordering.
        samples: Concatenated mono float32 audio samples covering both
            retained context and new audio, in chronological order.
        sample_rate: Audio sample rate, matching
            `WhisperEngine.SUPPORTED_SAMPLE_RATE` expectations.
        channels: Number of audio channels.
        audio_start_time: `AudioFrame.capture_timestamp` of the first
            sample folded into this window (perf_counter semantics,
            same as `SpeechSegment.start_timestamp`).
        audio_end_time: Timestamp at the end of the last sample folded
            into this window (same semantics as
            `SpeechSegment.end_timestamp`).
        context_duration_ms: Total duration of the retained-context
            portion of `samples`.
        new_audio_duration_ms: Total duration of the newly-arrived
            portion of `samples` since the previous window for this
            utterance (or since utterance start, for generation=1).
        start_sequence_number: First AudioFrame sequence number folded
            into this window (same semantics as
            `SpeechSegment.start_sequence_number`).
        end_sequence_number: Last AudioFrame sequence number folded
            into this window (same semantics as
            `SpeechSegment.end_sequence_number`).
        created_at: `time.perf_counter()` timestamp when this window
            object was constructed. Stands in for "creation/emit time"
            in Phase 1, since there is not yet a separate queuing step
            between window construction and ASR submission. A future
            ASRTask-equivalent wrapper should carry its own
            `submitted_at` the same way `asr.worker.ASRTask.submitted_at`
            is separate from the `SpeechSegment` it wraps - do not
            duplicate that here.
        is_final: Whether this window represents the end of its
            utterance (utterance-final audio, driven by VAD/other
            finalization - not decided by this module). Analogous to
            `SpeechSegment.forced_split` but reflects utterance
            finality rather than a forced max-duration split.

    Note:
        `duration_ms` is intentionally a derived property, not a
        stored field, since it is always recoverable from
        `audio_start_time`/`audio_end_time`.
    """

    window_id: int
    utterance_id: int
    generation: int

    samples: np.ndarray
    sample_rate: int
    channels: int

    audio_start_time: float
    audio_end_time: float

    context_duration_ms: float
    new_audio_duration_ms: float

    start_sequence_number: int
    end_sequence_number: int

    created_at: float

    is_final: bool = False

    # Optional Whisper `initial_prompt` text (e.g. recently confirmed
    # transcript, for cross-decode consistency - see
    # streaming.prefix_agreement). None for every Phase 1-4 caller;
    # additive field, does not affect existing construction sites.
    initial_prompt: str | None = None

    def __post_init__(self) -> None:
        """Validate window invariants."""

        if self.window_id <= 0:
            raise ValueError("window_id must be greater than zero.")

        if self.utterance_id <= 0:
            raise ValueError("utterance_id must be greater than zero.")

        if self.generation <= 0:
            raise ValueError("generation must be greater than zero.")

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero.")

        if self.channels <= 0:
            raise ValueError("channels must be greater than zero.")

        if self.audio_end_time < self.audio_start_time:
            raise ValueError(
                "audio_end_time cannot precede audio_start_time."
            )

        if self.context_duration_ms < 0:
            raise ValueError("context_duration_ms cannot be negative.")

        if self.new_audio_duration_ms < 0:
            raise ValueError(
                "new_audio_duration_ms cannot be negative."
            )

        if self.end_sequence_number < self.start_sequence_number:
            raise ValueError(
                "end_sequence_number cannot precede "
                "start_sequence_number."
            )

    @property
    def duration_ms(self) -> float:
        """Return total window duration derived from its timestamps."""

        return (
            self.audio_end_time
            - self.audio_start_time
        ) * 1000.0


@dataclass
class StreamingAudioState:
    """Mutable per-utterance streaming accumulation state (M2).

    One instance tracks exactly one candidate speech utterance, from
    the moment audio is first folded into it until it is finalized
    (`finalize()`) or explicitly cleared (`reset()`). It is the M2
    analogue of the frame-accumulation half of
    `streaming.buffer.FixedProcessingBuffer`, except a
    `FixedProcessingBuffer` emits exactly one `SpeechSegment` per
    utterance while a `StreamingAudioState` is expected to emit many
    `ProcessingWindow`s per utterance via repeated `build_window()`
    calls (Phase 2/3).

    Attributes:
        utterance_id: Stable id for this utterance's lifetime, unique
            across the run. Supplied by the caller (not generated
            internally - see module docstring), and propagated onto
            every `ProcessingWindow` this state produces.
        sample_rate: Audio sample rate for this utterance. Fixed for
            the state's lifetime.
        channels: Number of audio channels. Fixed for the state's
            lifetime.
        start_timestamp: `AudioFrame.capture_timestamp` of the first
            frame folded into this utterance.
        last_update_timestamp: Timestamp at the end of the most
            recently appended frame. Advances on every
            `append_frame()` call.
        context_audio: Retained frames available to ASR as left
            context (e.g. audio already covered by a previous
            window). Empty until the first `build_window()` call.
        new_audio_since_last_asr: Frames accumulated since the last
            `ProcessingWindow` was built for this utterance (or since
            utterance start, if none has been built yet). Cleared -
            rotated into `context_audio` - by `build_window()`.
        start_sequence_number: First AudioFrame sequence number folded
            into this utterance so far.
        last_sequence_number: Most recent AudioFrame sequence number
            folded into this utterance so far.
        last_asr_emission_time: `time.perf_counter()` timestamp of the
            last `build_window()` call for this utterance, or None if
            no window has been built yet. Used to measure inter-window
            spacing (e.g. for a future minimum-interval policy).
        windows_emitted: Number of `ProcessingWindow`s built so far for
            this utterance; becomes the next window's `generation`.
        state: Lifecycle marker (ACTIVE / FINALIZED).
        is_final: True once this utterance has been finalized. Windows
            built after finalization default to `is_final=True` unless
            the caller overrides it.
    """

    utterance_id: int
    sample_rate: int
    channels: int
    start_timestamp: float
    last_update_timestamp: float

    context_audio: list[AudioFrame] = field(default_factory=list)
    new_audio_since_last_asr: list[AudioFrame] = field(
        default_factory=list
    )

    start_sequence_number: int | None = None
    last_sequence_number: int | None = None

    last_asr_emission_time: float | None = None
    windows_emitted: int = 0

    state: StreamingLifecycleState = StreamingLifecycleState.ACTIVE
    is_final: bool = False

    def __post_init__(self) -> None:
        """Validate state invariants."""

        if self.utterance_id <= 0:
            raise ValueError("utterance_id must be greater than zero.")

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero.")

        if self.channels <= 0:
            raise ValueError("channels must be greater than zero.")

    @property
    def active(self) -> bool:
        """Return whether this utterance can still accept audio."""

        return self.state == StreamingLifecycleState.ACTIVE

    @property
    def context_duration_ms(self) -> float:
        """Return total duration of retained context audio."""

        return sum(
            frame.duration_ms
            for frame in self.context_audio
        )

    @property
    def new_audio_duration_ms(self) -> float:
        """Return total duration of audio pending its first window."""

        return sum(
            frame.duration_ms
            for frame in self.new_audio_since_last_asr
        )

    def append_frame(self, frame: AudioFrame) -> None:
        """Fold one new AudioFrame into `new_audio_since_last_asr`.

        Args:
            frame: The captured audio frame to append.

        Raises:
            RuntimeError: If this state is not ACTIVE.
            ValueError: If the frame is invalid or does not match this
                state's configured sample rate/channels.
        """

        if self.state != StreamingLifecycleState.ACTIVE:
            raise RuntimeError(
                "Cannot append audio to a StreamingAudioState in "
                f"state {self.state.value}."
            )

        _validate_frame_for_state(
            frame,
            self.sample_rate,
            self.channels,
        )

        self.new_audio_since_last_asr.append(frame)

        self.last_update_timestamp = (
            frame.capture_timestamp
            + frame.duration_ms / 1000.0
        )

        if self.start_sequence_number is None:
            self.start_sequence_number = frame.sequence_number

        self.last_sequence_number = frame.sequence_number

    def build_window(
        self,
        window_id: int,
        is_final: bool | None = None,
    ) -> ProcessingWindow:
        """Build a ProcessingWindow snapshot from accumulated audio.

        All frames in `context_audio` and `new_audio_since_last_asr`
        are folded into the window. `new_audio_since_last_asr` is then
        rotated into `context_audio` and cleared, so the next window
        built for this utterance starts a fresh "new audio" pool.

        Args:
            window_id: Identifier for the produced window, supplied by
                the caller (see module docstring for rationale).
            is_final: Whether the produced window is utterance-final.
                Defaults to this state's own `is_final` flag when not
                given explicitly.

        Returns:
            The constructed ProcessingWindow.

        Raises:
            ValueError: If there is no accumulated audio to build a
                window from.
        """

        all_frames = [
            *self.context_audio,
            *self.new_audio_since_last_asr,
        ]

        if not all_frames:
            raise ValueError(
                "Cannot build a ProcessingWindow with no "
                "accumulated audio."
            )

        if is_final is None:
            is_final = self.is_final

        first = all_frames[0]
        last = all_frames[-1]

        created_at = time.perf_counter()

        window = ProcessingWindow(
            window_id=window_id,
            utterance_id=self.utterance_id,
            generation=self.windows_emitted + 1,
            samples=_concatenate_frames(all_frames),
            sample_rate=self.sample_rate,
            channels=self.channels,
            audio_start_time=first.capture_timestamp,
            audio_end_time=(
                last.capture_timestamp
                + last.duration_ms / 1000.0
            ),
            context_duration_ms=self.context_duration_ms,
            new_audio_duration_ms=self.new_audio_duration_ms,
            start_sequence_number=first.sequence_number,
            end_sequence_number=last.sequence_number,
            created_at=created_at,
            is_final=is_final,
        )

        self.context_audio = all_frames
        self.new_audio_since_last_asr = []
        self.windows_emitted += 1
        self.last_asr_emission_time = created_at

        return window

    def finalize(self) -> None:
        """Mark this utterance as finalized.

        Intended to be called once VAD (or other future finalization
        logic, not implemented here) reports the utterance has ended.
        No further frames should be appended afterward. Idempotent:
        calling finalize() more than once leaves the state finalized.
        """

        self.is_final = True
        self.state = StreamingLifecycleState.FINALIZED

    def reset(self) -> None:
        """Clear accumulated audio/timing for this same utterance id.

        This represents abandoning the current accumulation (e.g. a
        false-positive trigger), not finalization. `utterance_id` is
        left unchanged; whether the same state is reused for a new
        utterance is a decision for whatever owns this object, not
        this method.
        """

        self.context_audio = []
        self.new_audio_since_last_asr = []
        self.start_sequence_number = None
        self.last_sequence_number = None
        self.windows_emitted = 0
        self.last_asr_emission_time = None
        self.is_final = False
        self.state = StreamingLifecycleState.ACTIVE
