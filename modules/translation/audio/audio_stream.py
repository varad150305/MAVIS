"""Microphone audio capture for the live translation system.

This module is responsible only for microphone capture and bounded
audio-frame buffering.

It does not perform:
    - VAD
    - ASR
    - translation
    - adaptive buffering
    - LocalAgreement
    - stability control
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AudioFrame:
    """A timestamped chunk of captured microphone audio.

    Attributes:
        samples: Mono floating-point audio samples in [-1.0, 1.0].
        capture_timestamp: Monotonic timestamp from time.perf_counter().
        duration_ms: Duration of this frame in milliseconds.
        sequence_number: Increasing frame identifier.
        wall_timestamp_ns: Wall-clock capture timestamp in nanoseconds.
        sample_rate: Sample rate of the captured audio.
        channels: Number of audio channels.
    """

    samples: np.ndarray
    capture_timestamp: float
    duration_ms: float
    sequence_number: int
    wall_timestamp_ns: int
    sample_rate: int
    channels: int


class CircularAudioBuffer:
    """Thread-safe bounded circular buffer for audio frames.

    When the buffer is full, the oldest frame is removed before the
    newest frame is inserted.
    """

    def __init__(
        self,
        capacity: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the circular buffer.

        Args:
            capacity: Maximum number of frames retained.
            logger: Logger used for overflow warnings.

        Raises:
            ValueError: If capacity is less than one.
        """

        if capacity < 1:
            raise ValueError(
                "Audio buffer capacity must be at least 1."
            )

        self._capacity = capacity
        self._buffer: list[Optional[AudioFrame]] = [
            None
        ] * capacity

        self._read_index = 0
        self._write_index = 0
        self._size = 0

        self._lock = threading.Lock()

        self._logger = (
            logger
            if logger is not None
            else logging.getLogger(__name__)
        )

        self._dropped_frames = 0

    @property
    def capacity(self) -> int:
        """Return the maximum number of stored frames."""

        return self._capacity

    @property
    def size(self) -> int:
        """Return the current number of stored frames."""

        with self._lock:
            return self._size

    @property
    def dropped_frames(self) -> int:
        """Return the total number of frames dropped."""

        with self._lock:
            return self._dropped_frames

    def put(self, frame: AudioFrame) -> bool:
        """Insert a frame into the buffer.

        If the buffer is full, the oldest frame is dropped.

        Args:
            frame: Audio frame to insert.

        Returns:
            True if an older frame had to be dropped.
        """

        dropped = False

        with self._lock:
            if self._size == self._capacity:
                # Drop the oldest frame.
                self._buffer[self._read_index] = None

                self._read_index = (
                    self._read_index + 1
                ) % self._capacity

                self._size -= 1

                self._dropped_frames += 1
                dropped = True

            self._buffer[self._write_index] = frame

            self._write_index = (
                self._write_index + 1
            ) % self._capacity

            self._size += 1

        if dropped:
            self._logger.warning(
                "Audio buffer overflow: "
                "dropped oldest frame. "
                "total_dropped=%d",
                self.dropped_frames,
            )

        return dropped

    def get(self) -> Optional[AudioFrame]:
        """Remove and return the oldest frame.

        Returns:
            The oldest frame, or None when the buffer is empty.
        """

        with self._lock:
            if self._size == 0:
                return None

            frame = self._buffer[
                self._read_index
            ]

            self._buffer[
                self._read_index
            ] = None

            self._read_index = (
                self._read_index + 1
            ) % self._capacity

            self._size -= 1

            return frame

    def clear(self) -> None:
        """Remove all frames from the buffer."""

        with self._lock:
            self._buffer = [
                None
            ] * self._capacity

            self._read_index = 0
            self._write_index = 0
            self._size = 0


class AudioStream:
    """Continuous microphone audio producer.

    The stream captures fixed-duration microphone frames and stores
    them in a bounded circular buffer.

    This class intentionally knows nothing about VAD, ASR,
    translation, adaptive buffering, or stability control.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        chunk_duration_ms: int,
        buffer_size: int,
        device: Optional[int | str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Create a microphone audio stream.

        Args:
            sample_rate: Microphone sample rate in Hz.
            channels: Number of input channels.
            chunk_duration_ms: Duration of each captured frame.
            buffer_size: Maximum number of frames retained.
            device: Optional sounddevice input device.
            logger: Optional logger.

        Raises:
            ValueError: If configuration values are invalid.
        """

        if sample_rate <= 0:
            raise ValueError(
                "sample_rate must be greater than zero."
            )

        if channels <= 0:
            raise ValueError(
                "channels must be at least 1."
            )

        if chunk_duration_ms <= 0:
            raise ValueError(
                "chunk_duration_ms must be greater than zero."
            )

        if chunk_duration_ms > 10_000:
            raise ValueError(
                "chunk_duration_ms is unreasonably large."
            )

        if buffer_size <= 0:
            raise ValueError(
                "buffer_size must be greater than zero."
            )

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.buffer_size = buffer_size
        self.device = device

        self._logger = (
            logger
            if logger is not None
            else logging.getLogger(__name__)
        )

        self._buffer = CircularAudioBuffer(
            capacity=buffer_size,
            logger=self._logger,
        )

        self._stream = None

        self._running = False
        self._closed = False

        self._sequence_number = 0

        self._state_lock = threading.Lock()

        self._callback_count = 0
        self._callback_total_time = 0.0

        # Number of samples expected in every frame.
        self._chunk_samples = int(
            round(
                sample_rate
                * chunk_duration_ms
                / 1000.0
            )
        )

        if self._chunk_samples <= 0:
            raise ValueError(
                "chunk_duration_ms produces "
                "zero audio samples."
            )

    @property
    def chunk_samples(self) -> int:
        """Return the number of samples in each frame."""

        return self._chunk_samples

    @property
    def is_running(self) -> bool:
        """Return whether microphone capture is active."""

        with self._state_lock:
            return self._running

    @property
    def dropped_frames(self) -> int:
        """Return the number of frames dropped due to overflow."""

        return self._buffer.dropped_frames

    @property
    def buffered_frames(self) -> int:
        """Return the number of currently buffered frames."""

        return self._buffer.size

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Receive audio from sounddevice.

        This callback performs only lightweight copying and
        timestamping before placing the frame into the bounded
        buffer.
        """

        callback_start = time.perf_counter()

        if status:
            self._logger.warning(
                "Microphone callback status: %s",
                status,
            )

        try:
            # Capture timestamps as close to the callback as possible.
            capture_timestamp = time.perf_counter()
            wall_timestamp_ns = time.time_ns()

            # Copy because sounddevice reuses the callback buffer.
            samples = np.asarray(
                indata,
                dtype=np.float32,
            ).copy()

            if self.channels == 1:
                samples = samples.reshape(-1)

            duration_ms = (
                len(samples)
                / self.sample_rate
                * 1000.0
            )

            frame = AudioFrame(
                samples=samples,
                capture_timestamp=capture_timestamp,
                duration_ms=duration_ms,
                sequence_number=self._sequence_number,
                wall_timestamp_ns=wall_timestamp_ns,
                sample_rate=self.sample_rate,
                channels=self.channels,
            )

            self._sequence_number += 1

            self._buffer.put(frame)

            self._callback_count += 1

        except Exception:
            self._logger.exception(
                "Unexpected error inside audio callback."
            )

        finally:
            callback_elapsed = (
                time.perf_counter()
                - callback_start
            )

            self._callback_total_time += (
                callback_elapsed
            )

    def start(self) -> None:
        """Start continuous microphone capture.

        Raises:
            RuntimeError: If the stream is already running,
                closed, or the microphone cannot be opened.
        """

        with self._state_lock:
            if self._closed:
                raise RuntimeError(
                    "Cannot start a closed AudioStream."
                )

            if self._running:
                return

        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for microphone "
                "capture. Install it with: "
                "pip install sounddevice"
            ) from exc

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self._chunk_samples,
                device=self.device,
                callback=self._audio_callback,
            )

            self._stream.start()

        except Exception as exc:
            self._stream = None

            self._logger.error(
                "Unable to start microphone capture: %s",
                exc,
            )

            raise RuntimeError(
                "Unable to start microphone capture. "
                "Check the selected device, sample rate, "
                "channel count, and microphone permissions."
            ) from exc

        with self._state_lock:
            self._running = True

        self._logger.info(
            "Audio stream started: "
            "sample_rate=%d, channels=%d, "
            "chunk_duration_ms=%d, "
            "chunk_samples=%d, buffer_size=%d",
            self.sample_rate,
            self.channels,
            self.chunk_duration_ms,
            self._chunk_samples,
            self.buffer_size,
        )

    def stop(self) -> None:
        """Stop microphone capture.

        The buffered frames are retained so consumers can finish
        reading them after capture has stopped.
        """

        with self._state_lock:
            if not self._running:
                return

            self._running = False

        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                self._logger.exception(
                    "Error while stopping microphone stream."
                )

        self._logger.info(
            "Audio stream stopped."
        )

    def read(self) -> Optional[AudioFrame]:
        """Read the oldest available audio frame.

        Returns:
            The next AudioFrame, or None if no frame is available.
        """

        return self._buffer.get()

    def close(self) -> None:
        """Stop capture and release microphone resources."""

        with self._state_lock:
            if self._closed:
                return

        self.stop()

        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                self._logger.exception(
                    "Error while closing microphone stream."
                )

            self._stream = None

        with self._state_lock:
            self._closed = True

        self._logger.info(
            "Audio stream closed."
        )

    def get_capture_statistics(self) -> dict[str, float]:
        """Return lightweight capture statistics.

        Returns:
            Dictionary containing callback count, average callback
            execution time, buffered frames, and dropped frames.
        """

        callback_count = self._callback_count

        if callback_count:
            mean_callback_ms = (
                self._callback_total_time
                / callback_count
                * 1000.0
            )
        else:
            mean_callback_ms = 0.0

        return {
            "callback_count": float(
                callback_count
            ),
            "mean_callback_ms": mean_callback_ms,
            "buffered_frames": float(
                self.buffered_frames
            ),
            "dropped_frames": float(
                self.dropped_frames
            ),
        }