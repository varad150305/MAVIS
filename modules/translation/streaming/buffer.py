"""Fixed utterance buffer for the M1 baseline.

This module converts VAD-labelled AudioFrames into complete,
timestamped speech segments.

It does not implement:
    - adaptive buffering
    - LocalAgreement
    - stability control
    - translation stability
    - ASR
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import time
from typing import Deque

import numpy as np

from audio.audio_stream import AudioFrame
from vad.vad import VADResult, VADState


@dataclass(frozen=True)
class SpeechSegment:
    """A completed speech utterance.

    Attributes:
        samples: Concatenated mono audio samples.
        sample_rate: Audio sample rate.
        channels: Number of channels.
        start_timestamp: Monotonic timestamp of the first frame.
        end_timestamp: Monotonic timestamp at the end of the last frame.
        duration_ms: Segment duration based on frame timing.
        start_sequence_number: First AudioFrame sequence number.
        end_sequence_number: Last AudioFrame sequence number.
        frame_count: Number of AudioFrames included.
        forced_split: Whether the segment was emitted because the
            maximum configured utterance duration was reached.
    """

    samples: np.ndarray
    sample_rate: int
    channels: int
    start_timestamp: float
    end_timestamp: float
    duration_ms: float
    start_sequence_number: int
    end_sequence_number: int
    frame_count: int
    forced_split: bool = False


class FixedProcessingBuffer:
    """Fixed M1 utterance buffer driven by VAD state.

    The buffer maintains a fixed pre-roll and accumulates audio while
    VAD reports SPEECH. A segment is emitted when VAD transitions to
    SILENCE or when the fixed maximum utterance duration is reached.
    """

    def __init__(
        self,
        pre_roll_ms: int,
        max_utterance_duration_ms: int,
        min_utterance_duration_ms: int,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the fixed processing buffer.

        Args:
            pre_roll_ms: Fixed amount of recent audio retained before
                speech confirmation.
            max_utterance_duration_ms: Maximum segment duration before
                forced emission.
            min_utterance_duration_ms: Minimum segment duration required
                before normal emission.
            logger: Optional logger.

        Raises:
            ValueError: If configuration values are invalid.
        """

        if pre_roll_ms < 0:
            raise ValueError(
                "pre_roll_ms cannot be negative."
            )

        if max_utterance_duration_ms <= 0:
            raise ValueError(
                "max_utterance_duration_ms must be greater than zero."
            )

        if min_utterance_duration_ms < 0:
            raise ValueError(
                "min_utterance_duration_ms cannot be negative."
            )

        if (
            min_utterance_duration_ms
            > max_utterance_duration_ms
        ):
            raise ValueError(
                "min_utterance_duration_ms cannot exceed "
                "max_utterance_duration_ms."
            )

        self.pre_roll_ms = pre_roll_ms
        self.max_utterance_duration_ms = (
            max_utterance_duration_ms
        )
        self.min_utterance_duration_ms = (
            min_utterance_duration_ms
        )

        self.logger = (
            logger
            if logger is not None
            else logging.getLogger(__name__)
        )

        # Recent frames retained for speech-onset recovery.
        self._pre_roll: Deque[AudioFrame] = deque()

        # Frames belonging to the active utterance.
        self._active_frames: list[AudioFrame] = []

        self._active = False

        self._last_sequence_number: int | None = None

        self._last_processing_time_ms = 0.0

        self._segments_emitted = 0

    @property
    def active(self) -> bool:
        """Return whether an utterance is currently being accumulated."""

        return self._active

    @property
    def buffered_duration_ms(self) -> float:
        """Return duration of the current active segment."""

        if not self._active_frames:
            return 0.0

        first = self._active_frames[0]
        last = self._active_frames[-1]

        return (
            (
                last.capture_timestamp
                + last.duration_ms / 1000.0
            )
            - first.capture_timestamp
        ) * 1000.0

    @property
    def segments_emitted(self) -> int:
        """Return the number of completed segments emitted."""

        return self._segments_emitted

    @property
    def last_processing_time_ms(self) -> float:
        """Return the latest buffer-processing duration."""

        return self._last_processing_time_ms

    def _validate_frame(
        self,
        frame: AudioFrame,
    ) -> None:
        """Validate an incoming AudioFrame."""

        if frame.samples.size == 0:
            raise ValueError(
                "Cannot buffer an AudioFrame with empty samples."
            )

        if frame.sample_rate <= 0:
            raise ValueError(
                "AudioFrame sample_rate must be positive."
            )

        if frame.channels <= 0:
            raise ValueError(
                "AudioFrame channels must be positive."
            )

        if frame.duration_ms <= 0:
            raise ValueError(
                "AudioFrame duration_ms must be positive."
            )

        if not np.all(
            np.isfinite(frame.samples)
        ):
            raise ValueError(
                "AudioFrame contains NaN or infinite samples."
            )

    def _update_pre_roll(
        self,
        frame: AudioFrame,
    ) -> None:
        """Append a frame while retaining fixed pre-roll duration.

        The current frame is included in the retained history. Since
        AudioFrames are indivisible, the buffer preserves whole frames
        and removes only the oldest frames when possible.
        """

        if self.pre_roll_ms <= 0:
            self._pre_roll.clear()
            return

        self._pre_roll.append(frame)

        total_duration_ms = sum(
            item.duration_ms
            for item in self._pre_roll
        )

        # Remove the oldest frame only when doing so still leaves at
        # least the configured pre-roll duration.
        while (
            len(self._pre_roll) > 1
            and total_duration_ms
            - self._pre_roll[0].duration_ms
            >= self.pre_roll_ms
        ):
            removed = self._pre_roll.popleft()

            total_duration_ms -= removed.duration_ms

    def _sequence_gap_detected(
        self,
        frame: AudioFrame,
    ) -> bool:
        """Detect a missing AudioFrame sequence number."""

        if self._last_sequence_number is None:
            return False

        return (
            frame.sequence_number
            != self._last_sequence_number + 1
        )

    def _start_segment(
        self,
        frame: AudioFrame,
    ) -> None:
        """Start an utterance using the retained pre-roll."""

        self._active_frames = list(
            self._pre_roll
        )

        # Avoid duplicate insertion if the current frame is already
        # the newest frame in the pre-roll.
        self._pre_roll.clear()
        if not self._active_frames or (
            self._active_frames[-1].sequence_number
            != frame.sequence_number
        ):
            self._active_frames.append(frame)

        self._active = True

        self.logger.info(
            "M1 fixed buffer: utterance started | "
            "start_sequence=%d",
            self._active_frames[0].sequence_number,
        )

    def _append_active_frame(
        self,
        frame: AudioFrame,
    ) -> None:
        """Append a frame to the active utterance."""

        if not self._active_frames or (
            self._active_frames[-1].sequence_number
            != frame.sequence_number
        ):
            self._active_frames.append(frame)

    def _build_segment(
        self,
        forced_split: bool,
    ) -> SpeechSegment | None:
        """Construct a SpeechSegment from accumulated frames."""

        if not self._active_frames:
            return None

        first = self._active_frames[0]
        last = self._active_frames[-1]

        start_timestamp = first.capture_timestamp

        end_timestamp = (
            last.capture_timestamp
            + last.duration_ms / 1000.0
        )

        duration_ms = (
            end_timestamp
            - start_timestamp
        ) * 1000.0

        if (
            duration_ms
            < self.min_utterance_duration_ms
            and not forced_split
        ):
            return None

        samples = np.concatenate(
            [
                frame.samples
                for frame in self._active_frames
            ]
        ).astype(
            np.float32,
            copy=False,
        )

        return SpeechSegment(
            samples=samples,
            sample_rate=first.sample_rate,
            channels=first.channels,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            duration_ms=duration_ms,
            start_sequence_number=(
                first.sequence_number
            ),
            end_sequence_number=(
                last.sequence_number
            ),
            frame_count=len(
                self._active_frames
            ),
            forced_split=forced_split,
        )

    def _reset_active_segment(self) -> None:
        """Clear the active utterance state."""

        self._active_frames = []
        self._active = False

    def _emit_segment(
        self,
        forced_split: bool,
    ) -> SpeechSegment | None:
        """Emit and reset the current active segment."""

        segment = self._build_segment(
            forced_split=forced_split
        )

        self._reset_active_segment()

        if segment is None:
            return None

        self._segments_emitted += 1

        self.logger.info(
            "M1 fixed buffer: utterance completed | "
            "start_seq=%d | end_seq=%d | "
            "duration_ms=%.1f | frames=%d | "
            "forced_split=%s",
            segment.start_sequence_number,
            segment.end_sequence_number,
            segment.duration_ms,
            segment.frame_count,
            segment.forced_split,
        )

        return segment

    def ingest(
        self,
        frame: AudioFrame,
        vad_result: VADResult,
    ) -> SpeechSegment | None:
        """Consume one AudioFrame and its corresponding VADResult.

        Args:
            frame: Captured audio frame.
            vad_result: VAD result corresponding to this frame.

        Returns:
            A completed SpeechSegment when an utterance ends or reaches
            the maximum duration; otherwise None.
        """

        processing_start = time.perf_counter()

        self._validate_frame(frame)

        if (
            frame.sequence_number
            != vad_result.sequence_number
            and vad_result.sequence_number is not None
        ):
            raise ValueError(
                "AudioFrame and VADResult sequence numbers do not match."
            )

        if self._sequence_gap_detected(frame):
            self.logger.warning(
                "M1 fixed buffer detected AudioFrame sequence gap: "
                "previous=%s current=%s",
                self._last_sequence_number,
                frame.sequence_number,
            )

        self._last_sequence_number = (
            frame.sequence_number
        )

        # Retain recent frames for speech-onset recovery.
        #self._update_pre_roll(frame)

        segment: SpeechSegment | None = None

        # --------------------------------------------------------
        # Start/continue utterance while VAD says SPEECH.
        # --------------------------------------------------------
        if vad_result.state == VADState.SPEECH:

            if not self._active:
                # The pre-roll contains only frames that occurred
                # before the current speech frame.
                self._start_segment(frame)

            else:
                self._append_active_frame(frame)

            # Fixed maximum-duration safety boundary.
            if (
                self.buffered_duration_ms
                >= self.max_utterance_duration_ms
            ):
                segment = self._emit_segment(
                    forced_split=True
                )

        # --------------------------------------------------------
        # End utterance on SPEECH -> SILENCE.
        # --------------------------------------------------------

        elif (
            vad_result.state == VADState.SILENCE
            and vad_result.state_changed
            and self._active
        ):
            # Include the transition frame because it contains the
            # audio used to confirm the utterance has ended.
            self._append_active_frame(frame)

            segment = self._emit_segment(
                forced_split=False
            )

        # --------------------------------------------------------
        # Maintain pre-roll only while there is no active utterance.
        # --------------------------------------------------------

        if not self._active:
            self._update_pre_roll(frame)

        self._last_processing_time_ms = (
            time.perf_counter()
            - processing_start
        ) * 1000.0

        return segment

    def flush(self) -> SpeechSegment | None:
        """Force emission of an active segment.

        This is intended for clean application shutdown.
        """

        processing_start = time.perf_counter()

        segment = self._emit_segment(
            forced_split=True
        )

        self._last_processing_time_ms = (
            time.perf_counter()
            - processing_start
        ) * 1000.0

        return segment

    def reset(self) -> None:
        """Clear pre-roll and active utterance state."""

        self._pre_roll.clear()
        self._reset_active_segment()
        self._last_sequence_number = None

        self.logger.info(
            "M1 fixed buffer reset."
        )