"""Silero VAD implementation for the M1 baseline.

This module is responsible only for voice activity detection.

It does not implement:
    - adaptive buffering
    - LocalAgreement
    - stability scoring
    - translation stability
    - ASR
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import time
from typing import Any

import numpy as np


class VADState(str, Enum):
    """Explicit states of the VAD state machine."""

    IDLE = "IDLE"
    SPEECH = "SPEECH"
    SILENCE = "SILENCE"


@dataclass(frozen=True)
class VADResult:
    """Result produced by one VAD processing call.

    Attributes:
        state: Current VAD state.
        speech_probability: Latest speech probability.
        timestamp: Monotonic timestamp associated with the
            captured audio frame.
        state_changed: Whether the state changed during this call.
        processing_time_ms: VAD computation time.
        sequence_number: Sequence number of the source audio frame.
    """

    state: VADState
    speech_probability: float
    timestamp: float
    state_changed: bool
    processing_time_ms: float
    sequence_number: int | None


class VAD:
    """Stateful Silero VAD processor.

    The model is loaded exactly once during initialization and
    reused for all subsequent audio frames.
    """

    SUPPORTED_SAMPLE_RATES = {8000, 16000}

    def __init__(
        self,
        model_name: str,
        sample_rate: int,
        speech_start_threshold: float,
        speech_end_threshold: float,
        minimum_speech_duration_ms: int,
        minimum_silence_duration_ms: int,
        logger: logging.Logger | None = None,
        model: Any | None = None,
    ) -> None:
        """Initialize the VAD engine.

        Args:
            model_name: Configured VAD model name.
            sample_rate: Expected audio sample rate.
            speech_start_threshold: Probability required to enter
                speech state.
            speech_end_threshold: Probability below which speech
                may end.
            minimum_speech_duration_ms: Minimum continuous speech
                duration before entering SPEECH.
            minimum_silence_duration_ms: Minimum continuous silence
                before entering SILENCE.
            logger: Optional logger.
            model: Optional pre-loaded model, primarily for tests.

        Raises:
            ValueError: For invalid configuration.
            RuntimeError: If the VAD model cannot be loaded.
        """

        self._validate_configuration(
            model_name=model_name,
            sample_rate=sample_rate,
            speech_start_threshold=speech_start_threshold,
            speech_end_threshold=speech_end_threshold,
            minimum_speech_duration_ms=(
                minimum_speech_duration_ms
            ),
            minimum_silence_duration_ms=(
                minimum_silence_duration_ms
            ),
        )

        self.model_name = model_name
        self.sample_rate = sample_rate

        self.speech_start_threshold = (
            speech_start_threshold
        )

        self.speech_end_threshold = (
            speech_end_threshold
        )

        self.minimum_speech_duration_ms = (
            minimum_speech_duration_ms
        )

        self.minimum_silence_duration_ms = (
            minimum_silence_duration_ms
        )

        self.logger = (
            logger
            if logger is not None
            else logging.getLogger(__name__)
        )

        # At 16 kHz Silero uses 512-sample frames.
        # At 8 kHz it uses 256-sample frames.
        self.model_frame_samples = (
            512 if sample_rate == 16000 else 256
        )

        self.model_frame_duration_ms = (
            self.model_frame_samples
            / sample_rate
            * 1000.0
        )

        # VAD state machine.
        self._state = VADState.IDLE

        # Duration accumulated while trying to enter SPEECH.
        self._speech_candidate_duration_ms = 0.0

        # Duration accumulated while trying to leave SPEECH.
        self._silence_candidate_duration_ms = 0.0

        # Latest probability.
        self._last_probability = 0.0

        # Partial audio from an incoming block that does not form
        # a complete Silero frame.
        self._audio_remainder = np.empty(
            0,
            dtype=np.float32,
        )

        # Load exactly once.
        self._model = (
            model
            if model is not None
            else self._load_model()
        )

    @property
    def state(self) -> VADState:
        """Return the current VAD state."""

        return self._state

    @property
    def speech_probability(self) -> float:
        """Return the most recent speech probability."""

        return self._last_probability

    def _validate_configuration(
        self,
        model_name: str,
        sample_rate: int,
        speech_start_threshold: float,
        speech_end_threshold: float,
        minimum_speech_duration_ms: int,
        minimum_silence_duration_ms: int,
    ) -> None:
        """Validate VAD configuration."""

        if model_name != "silero_vad":
            raise ValueError(
                f"Unsupported VAD model: {model_name}"
            )

        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                "Silero VAD supports 8000 or 16000 Hz "
                f"for this implementation; received {sample_rate}."
            )

        if not 0.0 <= speech_start_threshold <= 1.0:
            raise ValueError(
                "speech_start_threshold must be between 0 and 1."
            )

        if not 0.0 <= speech_end_threshold <= 1.0:
            raise ValueError(
                "speech_end_threshold must be between 0 and 1."
            )

        if speech_end_threshold >= speech_start_threshold:
            raise ValueError(
                "speech_end_threshold must be lower than "
                "speech_start_threshold to provide hysteresis."
            )

        if minimum_speech_duration_ms < 0:
            raise ValueError(
                "minimum_speech_duration_ms cannot be negative."
            )

        if minimum_silence_duration_ms < 0:
            raise ValueError(
                "minimum_silence_duration_ms cannot be negative."
            )

    def _load_model(self) -> Any:
        """Load Silero VAD once.

        Returns:
            Loaded Silero VAD model.

        Raises:
            RuntimeError: If the package/model cannot be loaded.
        """

        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "silero-vad is not installed. "
                "Install it with: "
                "pip install silero-vad==6.2.1"
            ) from exc

        try:
            model = load_silero_vad(
                onnx=False
            )

            model.eval()

            self.logger.info(
                "Silero VAD model loaded once."
            )

            return model

        except Exception as exc:
            raise RuntimeError(
                "Failed to load Silero VAD."
            ) from exc

    def reset(self) -> None:
        """Reset VAD state for a new processing session."""

        self._state = VADState.IDLE

        self._speech_candidate_duration_ms = 0.0

        self._silence_candidate_duration_ms = 0.0

        self._last_probability = 0.0

        self._audio_remainder = np.empty(
            0,
            dtype=np.float32,
        )

        # Silero's stateful wrapper maintains recurrent state.
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()

        self.logger.info(
            "VAD state reset to IDLE."
        )

    def _validate_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Validate and normalize incoming audio."""

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"VAD expected {self.sample_rate} Hz audio, "
                f"received {sample_rate} Hz."
            )

        array = np.asarray(
            audio,
            dtype=np.float32,
        )

        if array.size == 0:
            return np.empty(
                0,
                dtype=np.float32,
            )

        if array.ndim == 0:
            raise ValueError(
                "Audio input must be a one-dimensional array."
            )

        if array.ndim > 1:
            # AudioStream provides mono audio as 1-D.
            # We reject unexpected multi-channel input rather than
            # silently changing its semantics.
            raise ValueError(
                f"Expected one-dimensional mono audio; "
                f"received shape {array.shape}."
            )

        if not np.all(
            np.isfinite(array)
        ):
            raise ValueError(
                "Audio contains NaN or infinite values."
            )

        return array

    def _process_model_frame(
        self,
        frame: np.ndarray,
    ) -> float:
        """Run one model-compatible frame through Silero."""

        import torch

        tensor = torch.from_numpy(
            frame.astype(
                np.float32,
                copy=False,
            )
        )

        with torch.no_grad():
            probability = float(
                self._model(
                    tensor,
                    self.sample_rate,
                ).item()
            )

        return max(
            0.0,
            min(
                1.0,
                probability,
            ),
        )

    def _update_state(
        self,
        probability: float,
        duration_ms: float,
    ) -> tuple[bool, VADState, VADState]:
        """Update the hysteresis state machine.

        Returns:
            A tuple containing:
                - whether the state changed
                - previous state
                - current state
        """

        previous_state = self._state

        if self._state == VADState.IDLE:

            self._silence_candidate_duration_ms = 0.0

            if probability >= self.speech_start_threshold:
                self._speech_candidate_duration_ms += duration_ms

                if (
                    self._speech_candidate_duration_ms
                    >= self.minimum_speech_duration_ms
                ):
                    self._state = VADState.SPEECH
                    self._speech_candidate_duration_ms = 0.0
            else:
                self._speech_candidate_duration_ms = 0.0

        elif self._state == VADState.SPEECH:

            if probability <= self.speech_end_threshold:
                self._silence_candidate_duration_ms += duration_ms

                if (
                    self._silence_candidate_duration_ms
                    >= self.minimum_silence_duration_ms
                ):
                    self._state = VADState.SILENCE
                    self._silence_candidate_duration_ms = 0.0
            else:
                self._silence_candidate_duration_ms = 0.0

        elif self._state == VADState.SILENCE:

            self._silence_candidate_duration_ms = 0.0

            if probability >= self.speech_start_threshold:
                self._speech_candidate_duration_ms += duration_ms

                if (
                    self._speech_candidate_duration_ms
                    >= self.minimum_speech_duration_ms
                ):
                    self._state = VADState.SPEECH
                    self._speech_candidate_duration_ms = 0.0
            else:
                self._speech_candidate_duration_ms = 0.0

        return (
            previous_state != self._state,
            previous_state,
            self._state,
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        timestamp: float | None = None,
        sequence_number: int | None = None,
    ) -> VADResult:
        """Process one captured audio block.

        Args:
            audio: One-dimensional mono floating-point audio.
            sample_rate: Audio sample rate.
            timestamp: Capture timestamp from AudioFrame.
            sequence_number: Source AudioFrame sequence number.

        Returns:
            Structured VADResult.

        Raises:
            ValueError: If the audio or sample rate is invalid.
        """

        processing_start = time.perf_counter()

        try:
            validated_audio = self._validate_audio(
                audio,
                sample_rate,
            )

            if timestamp is None:
                timestamp = time.perf_counter()

            # ------------------------------------------------
            # Empty audio
            # ------------------------------------------------
            if validated_audio.size == 0:

                processing_time_ms = (
                    time.perf_counter()
                    - processing_start
                ) * 1000.0

                return VADResult(
                    state=self._state,
                    speech_probability=self._last_probability,
                    timestamp=timestamp,
                    state_changed=False,
                    processing_time_ms=processing_time_ms,
                    sequence_number=sequence_number,
                )

            # ------------------------------------------------
            # Combine an incomplete frame from the previous
            # call with the new block.
            # ------------------------------------------------
            if self._audio_remainder.size > 0:
                validated_audio = np.concatenate(
                    (
                        self._audio_remainder,
                        validated_audio,
                    )
                )

                self._audio_remainder = np.empty(
                    0,
                    dtype=np.float32,
                )

            number_of_complete_frames = (
                len(validated_audio)
                // self.model_frame_samples
            )

            complete_samples = (
                number_of_complete_frames
                * self.model_frame_samples
            )

            if complete_samples < len(
                validated_audio
            ):
                self._audio_remainder = (
                    validated_audio[
                        complete_samples:
                    ].copy()
                )

            model_audio = validated_audio[
                :complete_samples
            ]

            state_changed_any = False

            # ------------------------------------------------
            # Silero inference in model-compatible frames.
            # ------------------------------------------------
            for start in range(
                0,
                len(model_audio),
                self.model_frame_samples,
            ):

                model_frame = model_audio[
                    start : start + self.model_frame_samples
                ]

                probability = self._process_model_frame(
                    model_frame
                )

                self._last_probability = probability

                (
                    frame_changed,
                    previous_state,
                    current_state,
                ) = self._update_state(
                    probability=probability,
                    duration_ms=self.model_frame_duration_ms,
                )

                # A transition may happen on ANY
                # Silero-compatible sub-frame inside
                # this AudioFrame.
                if frame_changed:
                    state_changed_any = True

                    self.logger.info(
                        "VAD state transition: "
                        "%s -> %s | "
                        "probability=%.4f | "
                        "sequence=%s",
                        previous_state.value,
                        current_state.value,
                        probability,
                        sequence_number,
                    )

            processing_time_ms = (
                time.perf_counter()
                - processing_start
            ) * 1000.0

            self.logger.debug(
                "VAD processed frame | "
                "sequence=%s | "
                "state=%s | "
                "probability=%.4f | "
                "processing_ms=%.3f",
                sequence_number,
                self._state.value,
                self._last_probability,
                processing_time_ms,
            )

            return VADResult(
                state=self._state,
                speech_probability=self._last_probability,
                timestamp=timestamp,
                state_changed=state_changed_any,
                processing_time_ms=processing_time_ms,
                sequence_number=sequence_number,
            )

        except Exception:
            self.logger.exception(
                "VAD processing failed."
            )
            raise