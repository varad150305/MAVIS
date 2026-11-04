"""Persistent faster-whisper ASR engine for M1 and M2."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np


@dataclass(frozen=True)
class WordTiming:
    """One measured (not interpolated) Whisper word-level timestamp.

    Only populated when the engine was constructed with
    `word_timestamps=True` - see `ASRResult.words`.
    """

    word: str
    start: float
    end: float


@dataclass(frozen=True)
class ASRResult:
    """Structured result produced by Whisper."""

    text: str

    # Duration of the audio passed to Whisper.
    audio_duration_ms: float

    # Wall-clock inference time for Whisper.
    inference_time_ms: float

    # Real-time factor:
    # inference_time / audio_duration.
    # < 1.0 means faster than real time.
    rtf: float

    # Detected language, when available.
    language: str | None

    # Number of Whisper segments returned internally.
    segment_count: int

    # Measured per-word timestamps (seconds, relative to the audio
    # passed to transcribe()), in order. Empty unless the engine was
    # constructed with word_timestamps=True - never interpolated or
    # estimated here. M1 never enables word_timestamps, so this is
    # always empty for M1 and this field changes no M1 behavior.
    words: tuple[WordTiming, ...] = field(default_factory=tuple)

    @property
    def duration_ms(self) -> float:
        """Backward-compatible alias for audio duration."""

        return self.audio_duration_ms


class WhisperEngine:
    """Persistent faster-whisper engine.

    The model is loaded once and reused for every utterance.
    """

    SUPPORTED_SAMPLE_RATE = 16000

    def __init__(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        language: str | None = None,
        beam_size: int = 1,
        temperature: float = 0.0,
        cpu_threads: int = 8,
        num_workers: int = 1,
        vad_filter: bool = False,
        word_timestamps: bool = False,
    ) -> None:
        """Load Whisper once and keep it in memory."""

        if not model_size:
            raise ValueError(
                "model_size must not be empty."
            )

        if not device:
            raise ValueError(
                "device must not be empty."
            )

        if not compute_type:
            raise ValueError(
                "compute_type must not be empty."
            )

        if beam_size <= 0:
            raise ValueError(
                "beam_size must be greater than zero."
            )

        if cpu_threads <= 0:
            raise ValueError(
                "cpu_threads must be greater than zero."
            )

        if num_workers <= 0:
            raise ValueError(
                "num_workers must be greater than zero."
            )

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed."
            ) from exc

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.temperature = temperature
        self.vad_filter = vad_filter
        self.word_timestamps = word_timestamps
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers

        load_start = time.perf_counter()

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )

        self.model_load_time_ms = (
            time.perf_counter() - load_start
        ) * 1000.0

    def _validate_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Validate and normalize one mono audio utterance."""

        if sample_rate != self.SUPPORTED_SAMPLE_RATE:
            raise ValueError(
                "WhisperEngine expects "
                f"{self.SUPPORTED_SAMPLE_RATE} Hz audio; "
                f"received {sample_rate} Hz."
            )

        audio_array = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio_array.ndim != 1:
            raise ValueError(
                "WhisperEngine expects a 1-D mono audio array."
            )

        if audio_array.size == 0:
            return audio_array

        if not np.all(
            np.isfinite(audio_array)
        ):
            raise ValueError(
                "Audio contains NaN or infinite values."
            )

        return audio_array

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        initial_prompt: str | None = None,
    ) -> ASRResult:
        """Transcribe one complete speech utterance.

        Args:
            initial_prompt: Optional text prompt passed straight to
                faster-whisper's `initial_prompt` (e.g. recently
                confirmed transcript text, for cross-decode
                consistency - see `streaming.prefix_agreement`). `None`
                (the default, and every M1 call site) is byte-identical
                to omitting the parameter entirely.
        """

        audio_array = self._validate_audio(
            audio,
            sample_rate,
        )

        audio_duration_ms = (
            audio_array.size
            / sample_rate
            * 1000.0
        )

        if audio_array.size == 0:
            return ASRResult(
                text="",
                audio_duration_ms=0.0,
                inference_time_ms=0.0,
                rtf=0.0,
                language=self.language,
                segment_count=0,
            )

        inference_start = time.perf_counter()

        segments, info = self.model.transcribe(
            audio_array,
            language=self.language,
            beam_size=self.beam_size,
            temperature=self.temperature,
            vad_filter=self.vad_filter,
            word_timestamps=self.word_timestamps,
            initial_prompt=initial_prompt,
        )

        # faster-whisper returns a lazy generator.
        # Actual inference occurs while consuming it,
        # therefore timing must include this loop.
        text_parts: list[str] = []
        segment_count = 0
        words: list[WordTiming] = []

        for segment in segments:
            segment_count += 1

            text = segment.text.strip()

            if text:
                text_parts.append(text)

            if self.word_timestamps and segment.words:
                words.extend(
                    WordTiming(
                        word=word.word,
                        start=word.start,
                        end=word.end,
                    )
                    for word in segment.words
                )

        inference_time_ms = (
            time.perf_counter()
            - inference_start
        ) * 1000.0

        audio_duration_seconds = (
            audio_duration_ms / 1000.0
        )

        if audio_duration_seconds > 0.0:
            rtf = (
                inference_time_ms / 1000.0
            ) / audio_duration_seconds
        else:
            rtf = 0.0

        detected_language = getattr(
            info,
            "language",
            self.language,
        )

        return ASRResult(
            text=" ".join(text_parts).strip(),
            audio_duration_ms=audio_duration_ms,
            inference_time_ms=inference_time_ms,
            rtf=rtf,
            language=detected_language,
            segment_count=segment_count,
            words=tuple(words),
        )

    def warm_up(
        self,
        sample_rate: int = 16000,
        duration_ms: int = 1000,
    ) -> ASRResult:
        """Run one warm-up inference.

        The result is intentionally returned so callers can
        record the warm-up behavior if required.
        """

        if duration_ms <= 0:
            raise ValueError(
                "duration_ms must be greater than zero."
            )

        sample_count = int(
            sample_rate
            * duration_ms
            / 1000
        )

        silence = np.zeros(
            sample_count,
            dtype=np.float32,
        )

        return self.transcribe(
            silence,
            sample_rate,
        )