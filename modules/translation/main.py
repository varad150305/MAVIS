"""Entry point for Audio + VAD integration."""

from __future__ import annotations

import argparse
import sys
import time
from asr import (
    ASRTaskResult,
    ASRWorker,
    WhisperEngine,
)

from translation import (
    TranslationTaskResult,
    TranslationEngine,
    TranslationWorker,
)
from audio.audio_stream import AudioStream
from utils.config import ConfigError, load_config
from utils.logger import EventLogger
from vad.vad import VAD, VADState
from streaming.buffer import (
    FixedProcessingBuffer,
    SpeechSegment,
)
from streaming.adaptive_buffer import AdaptiveBuffer
from streaming.local_agreement import CommittedText, LocalAgreementTracker
from streaming.prefix_agreement import PrefixAgreementTracker
from streaming.streaming_asr import ASRHypothesis, StreamingASRDispatcher
from streaming.streaming_buffer import StreamingBuffer
from streaming.streaming_state import ProcessingWindow

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="SAST Audio + VAD integration"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file.",
    )

    return parser.parse_args()


def build_logger(config) -> EventLogger:
    """Create the configured application logger."""

    logging_config = config.get(
        "logging",
        default={},
    )

    return EventLogger(
        level=logging_config["level"],
        console=logging_config["console"],
        file_enabled=logging_config["file_enabled"],
        file_path=logging_config["file"],
        json_enabled=logging_config["json_enabled"],
        json_path=logging_config["json_file"],
    )


def build_audio_stream(
    config,
    logger,
) -> AudioStream:
    """Create the configured audio stream."""

    audio_config = config.get("audio", default={})

    chunk_duration_ms = audio_config["chunk_duration_ms"]
    buffer_size_ms = audio_config["buffer_size_ms"]

    if buffer_size_ms < chunk_duration_ms:
        raise ValueError(
            "audio.buffer_size_ms must be greater than or equal "
            "to audio.chunk_duration_ms."
        )

    buffer_size_frames = max(
        1,
        int(round(buffer_size_ms / chunk_duration_ms)),
    )

    logger.info(
        f"Audio buffer configuration: "
        f"buffer_size_ms={buffer_size_ms}, "
        f"chunk_duration_ms={chunk_duration_ms}, "
        f"buffer_size_frames={buffer_size_frames}"
    )
    return AudioStream(
        sample_rate=audio_config["sample_rate"],
        channels=audio_config["channels"],
        chunk_duration_ms=chunk_duration_ms,
        buffer_size=buffer_size_frames,
        device=audio_config.get("device"),
        logger=logger.logger,
    )

def build_vad(
    config,
    logger,
) -> VAD:
    """Create the configured Silero VAD."""

    vad_config = config.get(
        "vad"
    )

    return VAD(
        model_name=vad_config[
            "model"
        ],
        sample_rate=vad_config[
            "sample_rate"
        ],
        speech_start_threshold=vad_config[
            "speech_start_threshold"
        ],
        speech_end_threshold=vad_config[
            "speech_end_threshold"
        ],
        minimum_speech_duration_ms=vad_config[
            "minimum_speech_duration_ms"
        ],
        minimum_silence_duration_ms=vad_config[
            "minimum_silence_duration_ms"
        ],
        logger=logger.logger,
    )

def build_fixed_buffer(
    config,
    logger,
) -> FixedProcessingBuffer:
    """Create the M1 fixed utterance buffer."""

    buffer_config = config.get(
        "fixed_buffer",
        default={},
    )

    if not buffer_config:
        raise ValueError(
            "Missing fixed_buffer configuration for M1."
        )

    if not buffer_config.get(
        "enabled",
        False,
    ):
        raise ValueError(
            "fixed_buffer.enabled must be true for M1."
        )

    if config.features.get(
        "use_adaptive_buffer",
        False,
    ):
        raise ValueError(
            "M1 fixed buffer cannot be used with "
            "adaptive buffering enabled."
        )

    return FixedProcessingBuffer(
        pre_roll_ms=buffer_config[
            "pre_roll_ms"
        ],
        max_utterance_duration_ms=buffer_config[
            "max_utterance_duration_ms"
        ],
        min_utterance_duration_ms=buffer_config[
            "min_utterance_duration_ms"
        ],
        logger=logger.logger,
    )


def build_adaptive_buffer(
    config,
    logger,
) -> AdaptiveBuffer:
    """Create the configured M2 adaptive buffer (Phase 2/3).

    Mirrors build_fixed_buffer()'s validation style. Reads
    `adaptive_buffer.*` for the window/step/context parameters and
    `streaming.window_queue_size`/`streaming.window_queue_policy` for
    the bounded pending-window view (the same values Phase 3 also
    uses for the real ASRWorker queue capacity - see
    build_asr_worker()).
    """

    buffer_config = config.get(
        "adaptive_buffer",
        default={},
    )

    if not buffer_config:
        raise ValueError(
            "Missing adaptive_buffer configuration for M2."
        )

    if not buffer_config.get(
        "enabled",
        False,
    ):
        raise ValueError(
            "adaptive_buffer.enabled must be true for M2."
        )

    if not config.features.get(
        "use_adaptive_buffer",
        False,
    ):
        raise ValueError(
            "M2 adaptive buffer requires "
            "features.use_adaptive_buffer=true."
        )

    audio_config = config.get("audio", default={})
    streaming_config = config.get("streaming", default={})

    buffer = AdaptiveBuffer(
        sample_rate=audio_config["sample_rate"],
        channels=audio_config["channels"],
        initial_window_ms=buffer_config["initial_window_ms"],
        min_window_ms=buffer_config["min_window_ms"],
        max_window_ms=buffer_config["max_window_ms"],
        step_ms=buffer_config["step_ms"],
        min_new_audio_ms=buffer_config["min_new_audio_ms"],
        max_context_ms=buffer_config["max_context_ms"],
        overlap_ms=buffer_config["overlap_ms"],
        window_queue_size=streaming_config.get(
            "window_queue_size",
            4,
        ),
        window_queue_policy=streaming_config.get(
            "window_queue_policy",
            "keep_latest",
        ),
        logger=logger.logger,
    )

    logger.info(
        "M2 adaptive buffer configured | "
        f"initial_window_ms={buffer.initial_window_ms} "
        f"min_window_ms={buffer.min_window_ms} "
        f"max_window_ms={buffer.max_window_ms} "
        f"step_ms={buffer.step_ms} "
        f"min_new_audio_ms={buffer.min_new_audio_ms} "
        f"overlap_ms={buffer.overlap_ms} "
        f"max_context_ms={buffer.max_context_ms} "
        f"window_queue_size={buffer.window_queue_size} "
        f"window_queue_policy={buffer.window_queue_policy}"
    )

    return buffer


def build_asr_engine(
    config,
    logger,
) -> WhisperEngine:
    """Create the persistent Faster-Whisper ASR engine."""

    asr_config = config.get(
        "asr",
        default={},
    )

    if not asr_config.get(
        "enabled",
        True,
    ):
        raise ValueError(
            "ASR must be enabled for M1 Phase 2B."
        )

    backend = asr_config.get(
        "backend",
        "faster-whisper",
    )

    if backend != "faster-whisper":
        raise ValueError(
            "Unsupported ASR backend for M1: "
            f"{backend}"
        )

    # CPU-thread wiring fix (M2-CPU audit finding): configs/*.yaml's
    # top-level `cpu:` section (threads/inter_op_threads/intra_op_threads)
    # was never read anywhere - WhisperEngine always fell back to the
    # hardcoded default below regardless of what `cpu:` declared (e.g.
    # configs/m2.yaml's cpu.threads=4 was silently ignored, and the
    # engine actually ran with cpu_threads=8). `asr.cpu_threads`, when
    # explicitly set (as configs/m1.yaml already does), still takes
    # precedence - this is purely an ADDITIONAL fallback source, not a
    # precedence change for any config that already sets asr.cpu_threads,
    # so M1's effective behavior is unchanged. `cpu.inter_op_threads`/
    # `cpu.intra_op_threads` remain unused: faster-whisper/CTranslate2
    # exposes only a single `cpu_threads` knob, with no equivalent of
    # PyTorch's separate inter-op/intra-op thread pools.
    cpu_config = config.get(
        "cpu",
        default={},
    )

    cpu_threads = asr_config.get(
        "cpu_threads",
        cpu_config.get(
            "threads",
            8,
        ),
    )

    logger.info(
        "Initializing M1 ASR | "
        f"backend={backend} | "
        f"model={asr_config['model_size']} | "
        f"device={asr_config['device']} | "
        f"compute_type={asr_config['compute_type']} | "
        f"cpu_threads={cpu_threads}"
    )

    engine = WhisperEngine(
        model_size=asr_config["model_size"],
        device=asr_config["device"],
        compute_type=asr_config["compute_type"],
        language=asr_config.get(
            "language"
        ),
        beam_size=asr_config.get(
            "beam_size",
            1,
        ),
        temperature=asr_config.get(
            "temperature",
            0.0,
        ),
        cpu_threads=cpu_threads,
        num_workers=asr_config.get(
            "num_workers",
            1,
        ),
        vad_filter=asr_config.get(
            "vad_filter",
            False,
        ),
        word_timestamps=asr_config.get(
            "word_timestamps",
            False,
        ),
    )

    logger.info(
        "M1 ASR model loaded | "
        f"load_ms={engine.model_load_time_ms:.1f}"
    )

    return engine
def build_asr_worker(
    config,
    engine: WhisperEngine,
    logger,
    queue_size_override: int | None = None,
) -> ASRWorker:
    """Create the background ASR worker (shared by M1 and M2).

    Args:
        config: Loaded configuration.
        engine: The persistent WhisperEngine to wrap.
        logger: Application logger.
        queue_size_override: When given, used instead of
            `asr.queue_size` from config. M2 (Phase 3) passes
            `streaming.window_queue_size` here so the real ASRWorker
            queue capacity matches AdaptiveBuffer's own pending-window
            view rather than introducing a second, possibly-diverging
            queue-size parameter. M1's call site never passes this,
            so M1 behavior is unchanged.
    """

    asr_config = config.get(
        "asr",
        default={},
    )

    queue_size = int(
        queue_size_override
        if queue_size_override is not None
        else asr_config.get(
            "queue_size",
            2,
        )
    )

    worker = ASRWorker(
        engine=engine,
        queue_size=queue_size,
        logger=logger.logger,
    )

    worker.start()

    logger.info(
        "M1 ASR worker configured | "
        f"queue_size={queue_size}"
    )

    return worker


def build_streaming_asr_dispatcher(
    asr_worker: ASRWorker,
    logger,
) -> StreamingASRDispatcher:
    """Create the Phase 3 M2 streaming-ASR orchestration/correlation layer.

    Thin wrapper kept for symmetry with the other build_*() helpers;
    StreamingASRDispatcher does not own asr_worker's lifecycle (the
    caller starts/stops it, exactly like M1 does today).
    """

    return StreamingASRDispatcher(
        asr_worker=asr_worker,
        logger=logger.logger,
    )


def build_local_agreement_tracker(
    config,
    logger,
) -> LocalAgreementTracker | None:
    """Create the Phase 4 LocalAgreement-style commitment tracker for M2.

    Returns None when `local_agreement.enabled` is false, in which case
    run_m2() runs exactly as in Phase 3 (hypotheses only, no commitment
    layer) - this keeps Phase 4 an additive, optional layer over the
    unchanged Phase 3 path.
    """

    local_agreement_config = config.get(
        "local_agreement",
        default={},
    )

    if not local_agreement_config.get(
        "enabled",
        False,
    ):
        logger.info(
            "M2 LocalAgreement tracker disabled "
            "(local_agreement.enabled=false)."
        )

        return None

    tracker = LocalAgreementTracker(
        agreement_depth=local_agreement_config.get(
            "agreement_depth",
            2,
        ),
        history_size=local_agreement_config.get(
            "history_size",
            5,
        ),
        alignment_method=local_agreement_config.get(
            "alignment_method",
            "lcs",
        ),
        timing_method=local_agreement_config.get(
            "timing_method",
            "interpolated",
        ),
        logger=logger.logger,
    )

    logger.info(
        "M2 LocalAgreement tracker configured | "
        f"agreement_depth={tracker.agreement_depth} "
        f"history_size={tracker.history_size} "
        f"alignment_method={tracker.alignment_method} "
        f"timing_method={tracker.timing_method}"
    )

    return tracker


def build_streaming_buffer(config, logger) -> StreamingBuffer:
    """Create the research-aligned M2 persistent streaming buffer.

    Reads `streaming_buffer.*` - see `configs/m2.yaml`'s
    "M2 STREAMING BUFFER" section and the Phase 4C research-alignment
    report for why this replaced `adaptive_buffer.*`/`AdaptiveBuffer`
    as `run_m2_aligned()`'s buffer. `build_adaptive_buffer()` above is
    kept, unused by the aligned path, for Phase 2-4C evaluator
    reproducibility.
    """

    buffer_config = config.get("streaming_buffer", default={})

    if not buffer_config:
        raise ValueError("Missing streaming_buffer configuration for M2.")

    if not buffer_config.get("enabled", False):
        raise ValueError("streaming_buffer.enabled must be true for M2.")

    audio_config = config.get("audio", default={})

    buffer = StreamingBuffer(
        sample_rate=audio_config["sample_rate"],
        channels=audio_config["channels"],
        min_new_audio_ms=buffer_config["min_new_audio_ms"],
        max_buffer_ms=buffer_config["max_buffer_ms"],
        logger=logger.logger,
        pre_roll_ms=buffer_config.get("pre_roll_ms", 300),
    )

    logger.info(
        "M2 streaming buffer configured | "
        f"min_new_audio_ms={buffer.min_new_audio_ms} "
        f"max_buffer_ms={buffer.max_buffer_ms} "
        f"pre_roll_ms={buffer.pre_roll_ms}"
    )

    return buffer


def build_prefix_agreement_tracker(config, logger) -> PrefixAgreementTracker:
    """Create the research-aligned M2 LocalAgreement-2 prefix tracker.

    Reads `prefix_agreement.*` - see `configs/m2.yaml`. Returns a
    tracker unconditionally (unlike `build_local_agreement_tracker()`,
    which can return None) - the aligned architecture has no
    "hypotheses only, no commitment" mode, matching the reference
    (LocalAgreement is not an optional add-on there).
    """

    prefix_config = config.get("prefix_agreement", default={})

    if not prefix_config:
        raise ValueError("Missing prefix_agreement configuration for M2.")

    tracker = PrefixAgreementTracker(
        agreement_depth=prefix_config.get("agreement_depth", 2),
        prompt_word_limit=prefix_config.get("prompt_word_limit", 200),
        logger=logger.logger,
    )

    logger.info(
        "M2 prefix agreement tracker configured | "
        f"agreement_depth={tracker.agreement_depth} "
        f"prompt_word_limit={tracker.prompt_word_limit}"
    )

    return tracker


def build_translation_engine(
    config,
    logger,
) -> TranslationEngine:
    """Create the persistent English-to-Hindi translation engine."""

    translation_config = config.get(
        "translation",
        default={},
    )

    if not translation_config.get(
        "enabled",
        True,
    ):
        raise ValueError(
            "Translation must be enabled for M1 Phase 3B."
        )

    backend = translation_config.get(
        "backend",
        "transformers",
    )

    if backend != "transformers":
        raise ValueError(
            "Unsupported translation backend for M1: "
            f"{backend}"
        )

    source_language = translation_config.get(
        "source_language",
        "en",
    )

    target_language = translation_config.get(
        "target_language",
        "hi",
    )

    if source_language != "en":
        raise ValueError(
            "Phase 3B expects English source text."
        )

    if target_language != "hi":
        raise ValueError(
            "Phase 3B expects Hindi target text."
        )

    logger.info(
        "Initializing M1 translation | "
        f"backend={backend} | "
        f"model={translation_config['model']} | "
        f"device={translation_config['device']} | "
        f"direction={source_language}->{target_language}"
    )

    engine = TranslationEngine(
        model_name=translation_config["model"],
        device=translation_config["device"],
        max_input_tokens=translation_config.get(
            "max_input_tokens",
            256,
        ),
        num_beams=translation_config.get(
            "num_beams",
            1,
        ),
        max_new_tokens=translation_config.get(
            "max_new_tokens",
            256,
        ),
    )

    logger.info(
        "M1 translation model loaded | "
        f"load_ms={engine.model_load_time_ms:.1f}"
    )

    return engine

def build_translation_worker(
    config,
    engine: TranslationEngine,
    logger,
) -> TranslationWorker:
    """Create the background M1 translation worker."""

    translation_config = config.get(
        "translation",
        default={},
    )

    queue_size = int(
        translation_config.get(
            "queue_size",
            2,
        )
    )

    worker = TranslationWorker(
        engine=engine,
        queue_size=queue_size,
        logger=logger.logger,
    )

    worker.start()

    logger.info(
        "M1 translation worker configured | "
        f"queue_size={queue_size}"
    )

    return worker

def run_m1(config, logger) -> int:
    """Run the M1 fixed-buffer pipeline.

    This is a mechanical extraction of the pre-Phase-3 main() body's
    M1-only code path - behaviorally unchanged. See run_m2() for the
    new M2 streaming-ASR path added in Phase 3.
    """

    features = config.features

    # --------------------------------------------------------
    # M1 safety check
    # --------------------------------------------------------

    enabled_future_features = [
        name
        for name, enabled in features.items()
        if enabled
    ]

    if enabled_future_features:
        logger.error(
            "M1 cannot enable future features: "
            f"{enabled_future_features}"
        )

        return 1

    # --------------------------------------------------------
    # Build modules
    # --------------------------------------------------------

    try:
        audio_stream = build_audio_stream(
            config,
            logger,
        )

        vad = build_vad(
            config,
            logger,
        )

        fixed_buffer = build_fixed_buffer(
            config,
            logger,
        )

        asr_engine = build_asr_engine(
            config,
            logger,
        )

        asr_worker = build_asr_worker(
            config,
            asr_engine,
            logger,
        )
        translation_engine = build_translation_engine(
            config,
            logger,
        )

        translation_worker = build_translation_worker(
            config,
            translation_engine,
            logger,
        )
        performance_config = config.get(
            "performance",
            default={},
        )

        if performance_config.get(
            "warmup_asr",
            True,
        ):
            logger.info(
                "Running M1 ASR warm-up..."
            )

            warmup_result = asr_engine.warm_up()

            logger.info(
                "M1 ASR warm-up completed | "
                f"inference_ms="
                f"{warmup_result.inference_time_ms:.1f}"
            )
        if performance_config.get(
            "warmup_translation",
            True,
        ):
            logger.info(
                "Running M1 translation warm-up..."
            )

            translation_warmup = (
                translation_engine.warm_up()
            )

            logger.info(
                "M1 translation warm-up completed | "
                f"inference_ms="
                f"{translation_warmup.inference_time_ms:.1f}"
            )
    except Exception as exc:
        logger.error(
            f"Failed to initialize M1 pipeline: {exc}"
        )

        return 1

    # --------------------------------------------------------
    # Runtime counters
    # --------------------------------------------------------

    processed_frames = 0

    state_transitions = 0
    segments_emitted = 0
    buffer_processing_times_ms: list[float] = []
    segment_durations_ms: list[float] = []

    audio_wait_times_ms: list[float] = []
    vad_times_ms: list[float] = []

    # ASR metrics.
    asr_queue_wait_times_ms: list[float] = []
    asr_inference_times_ms: list[float] = []
    asr_segment_to_transcript_times_ms: list[float] = []
    asr_rtf_values: list[float] = []

    asr_results = 0
    asr_failures = 0
    asr_queue_drops = 0
    # Translation metrics.
    translation_queue_wait_times_ms: list[float] = []
    translation_inference_times_ms: list[float] = []
    translation_asr_to_completion_times_ms: list[float] = []
    translation_segment_to_completion_times_ms: list[float] = []

    translation_results = 0
    translation_failures = 0
    translation_queue_drops = 0
    #helper function
    def handle_asr_results(
        results: list[ASRTaskResult],
    ) -> None:
        """Handle completed background ASR results."""

        nonlocal asr_results
        nonlocal asr_failures

        for task_result in results:
            segment = task_result.segment

            if task_result.error is not None:
                asr_failures += 1

                logger.event(
                    "asr_result",
                    start_sequence_number=(
                        segment.start_sequence_number
                    ),
                    end_sequence_number=(
                        segment.end_sequence_number
                    ),
                    text="",
                    error=task_result.error,
                    queue_wait_ms=(
                        task_result.queue_wait_ms
                    ),
                    segment_to_transcript_ms=(
                        task_result.segment_to_transcript_ms
                    ),
                )

                print(
                    "ASR ERROR | "
                    f"seq="
                    f"{segment.start_sequence_number}-"
                    f"{segment.end_sequence_number} | "
                    f"{task_result.error}",
                    flush=True,
                )

                continue

            result = task_result.result

            if result is None:
                asr_failures += 1
                continue

            asr_results += 1

            asr_queue_wait_times_ms.append(
                task_result.queue_wait_ms
            )

            asr_inference_times_ms.append(
                result.inference_time_ms
            )

            asr_segment_to_transcript_times_ms.append(
                task_result.segment_to_transcript_ms
            )

            asr_rtf_values.append(
                result.rtf
            )

            logger.event(
                "asr_result",
                start_sequence_number=(
                    segment.start_sequence_number
                ),
                end_sequence_number=(
                    segment.end_sequence_number
                ),
                duration_ms=(
                    segment.duration_ms
                ),
                text=result.text,
                language=result.language,
                segment_count=result.segment_count,
                inference_time_ms=(
                    result.inference_time_ms
                ),
                rtf=result.rtf,
                queue_wait_ms=(
                    task_result.queue_wait_ms
                ),
                segment_to_transcript_ms=(
                    task_result.segment_to_transcript_ms
                ),
            )

            print(
                "ASR RESULT | "
                f"seq="
                f"{segment.start_sequence_number}-"
                f"{segment.end_sequence_number} | "
                f"inference="
                f"{result.inference_time_ms:.1f} ms | "
                f"rtf="
                f"{result.rtf:.3f} | "
                f"latency="
                f"{task_result.segment_to_transcript_ms:.1f} ms",
                flush=True,
            )

            print(
                f"TRANSCRIPT: {result.text}",
                flush=True,
            )
            # ----------------------------------------------------
            # Submit successful ASR transcript to translation.
            # ----------------------------------------------------

            if not result.text.strip():
                logger.info(
                    "M1 translation skipped empty ASR transcript | "
                    f"seq="
                    f"{segment.start_sequence_number}-"
                    f"{segment.end_sequence_number}"
                )
            else:
                try:
                    accepted = translation_worker.submit(
                        text=result.text,
                        upstream_completed_at=(
                            task_result.completed_at
                        ),
                        segment_end_timestamp=(
                            segment.end_timestamp
                        ),
                        start_sequence_number=(
                            segment.start_sequence_number
                        ),
                        end_sequence_number=(
                            segment.end_sequence_number
                        ),
                    )

                    if not accepted:
                        translation_queue_drops += 1

                except Exception as exc:
                    translation_failures += 1

                    logger.error(
                        "Failed to submit transcript "
                        f"to translation worker: {exc}"
                    )
    def handle_translation_results(
        results: list[TranslationTaskResult],
    ) -> None:
        """Handle completed background translation results."""

        nonlocal translation_results
        nonlocal translation_failures

        for task_result in results:
            task = task_result.task

            if task_result.error is not None:
                translation_failures += 1

                logger.event(
                    "translation_result",
                    start_sequence_number=(
                        task.start_sequence_number
                    ),
                    end_sequence_number=(
                        task.end_sequence_number
                    ),
                    source_text=task.text,
                    translated_text="",
                    error=task_result.error,
                    queue_wait_ms=(
                        task_result.queue_wait_ms
                    ),
                    asr_to_translation_ms=(
                        task_result.asr_to_translation_ms
                    ),
                    segment_to_translation_ms=(
                        task_result.segment_to_translation_ms
                    ),
                )

                print(
                    "TRANSLATION ERROR | "
                    f"seq="
                    f"{task.start_sequence_number}-"
                    f"{task.end_sequence_number} | "
                    f"{task_result.error}",
                    flush=True,
                )

                continue

            result = task_result.result

            if result is None:
                translation_failures += 1
                continue

            translation_results += 1

            translation_queue_wait_times_ms.append(
                task_result.queue_wait_ms
            )

            translation_inference_times_ms.append(
                result.inference_time_ms
            )

            translation_asr_to_completion_times_ms.append(
                task_result.asr_to_translation_ms
            )

            translation_segment_to_completion_times_ms.append(
                task_result.segment_to_translation_ms
            )

            logger.event(
                "translation_result",
                start_sequence_number=(
                    task.start_sequence_number
                ),
                end_sequence_number=(
                    task.end_sequence_number
                ),
                source_text=(
                    result.source_text
                ),
                translated_text=(
                    result.translated_text
                ),
                source_language=(
                    result.source_language
                ),
                target_language=(
                    result.target_language
                ),
                input_token_count=(
                    result.input_token_count
                ),
                output_token_count=(
                    result.output_token_count
                ),
                chunk_count=(
                    result.chunk_count
                ),
                inference_time_ms=(
                    result.inference_time_ms
                ),
                queue_wait_ms=(
                    task_result.queue_wait_ms
                ),
                asr_to_translation_ms=(
                    task_result.asr_to_translation_ms
                ),
                segment_to_translation_ms=(
                    task_result.segment_to_translation_ms
                ),
            )

            print(
                "TRANSLATION RESULT | "
                f"seq="
                f"{task.start_sequence_number}-"
                f"{task.end_sequence_number} | "
                f"inference="
                f"{result.inference_time_ms:.1f} ms | "
                f"queue="
                f"{task_result.queue_wait_ms:.1f} ms",
                flush=True,
            )

            print(
                f"HINDI: {result.translated_text}",
                flush=True,
            )

    logger.info(
        "M1 Audio + VAD + Fixed Buffer + ASR initialized."
    )

    logger.info(
        f"ASR model load time: "
        f"{asr_engine.model_load_time_ms:.1f} ms"
    )

    logger.info(
        "Press Ctrl+C to stop."
    )

    try:

        audio_stream.start()

        while True:

            # ------------------------------------------------
            # Read AudioFrame
            # ------------------------------------------------

            read_start = time.perf_counter()

            frame = audio_stream.read()

            read_duration_ms = (
                time.perf_counter()
                - read_start
            ) * 1000.0

            if frame is None:
                continue

            audio_wait_times_ms.append(
                read_duration_ms
            )

            processed_frames += 1

            # ------------------------------------------------
            # VAD
            # ------------------------------------------------

            result = vad.process(
                audio=frame.samples,
                sample_rate=frame.sample_rate,
                timestamp=frame.capture_timestamp,
                sequence_number=frame.sequence_number,
            )
            print(
                f"Frame={frame.sequence_number:03d} | "
                f"state={result.state.value:<7} | "
                f"prob={result.speech_probability:.3f} | "
                f"changed={result.state_changed} | "
                f"vad={result.processing_time_ms:.2f} ms",
                flush=True,
            )
            vad_times_ms.append(
                result.processing_time_ms
            )

            if result.state_changed:
                state_transitions += 1
            buffer_start = time.perf_counter()

            segment = fixed_buffer.ingest(
                frame=frame,
                vad_result=result,
            )

            buffer_processing_ms = (
                time.perf_counter()
                - buffer_start
            ) * 1000.0

            buffer_processing_times_ms.append(
                buffer_processing_ms
            )
            if segment is not None:
                segments_emitted += 1
                segment_durations_ms.append(
                    segment.duration_ms
                )

                logger.event(
                    "speech_segment",
                    start_timestamp=(
                        segment.start_timestamp
                    ),
                    end_timestamp=(
                        segment.end_timestamp
                    ),
                    duration_ms=(
                        segment.duration_ms
                    ),
                    start_sequence_number=(
                        segment.start_sequence_number
                    ),
                    end_sequence_number=(
                        segment.end_sequence_number
                    ),
                    frame_count=segment.frame_count,
                    forced_split=segment.forced_split,
                    sample_count=len(
                        segment.samples
                    ),
                    sample_rate=segment.sample_rate,
                )

                print(
                    f"SEGMENT READY | "
                    f"seq="
                    f"{segment.start_sequence_number}-"
                    f"{segment.end_sequence_number} | "
                    f"duration="
                    f"{segment.duration_ms:.1f} ms | "
                    f"samples="
                    f"{len(segment.samples)} | "
                    f"forced="
                    f"{segment.forced_split}",
                    flush=True,
                )
                try:
                    accepted = asr_worker.submit(
                        segment
                    )

                    if not accepted:
                        asr_queue_drops += 1

                except Exception as exc:
                    asr_failures += 1

                    logger.error(
                        "Failed to submit segment "
                        f"to ASR worker: {exc}"
                    )
            handle_asr_results(
                asr_worker.poll_results()
            )
            handle_translation_results(
                translation_worker.poll_results()
            )
            # ------------------------------------------------
            # Structured event
            # ------------------------------------------------

            logger.event(
                "vad_result",
                sequence_number=frame.sequence_number,
                timestamp=frame.capture_timestamp,
                state=result.state.value,
                speech_probability=(
                    result.speech_probability
                ),
                state_changed=(
                    result.state_changed
                ),
                vad_processing_ms=(
                    result.processing_time_ms
                ),
                samples=len(frame.samples),
                frame_duration_ms=(
                    frame.duration_ms
                ),
            )

            # ------------------------------------------------
            # Human-readable output
            # ------------------------------------------------

            """print(
                f"\r"
                f"Frame={frame.sequence_number:03d} | "
                f"State={result.state.value:<7} | "
                f"Probability="
                f"{result.speech_probability:.3f} | "
                f"VAD="
                f"{result.processing_time_ms:.2f} ms",
                end="",
                flush=True,
            )"""

    except KeyboardInterrupt:

        logger.info(
            "Stopping Audio + VAD integration."
        )

    except Exception as exc:

        logger.error(
            f"Runtime error: {exc}"
        )

        return 1

    finally:
        final_segment = fixed_buffer.flush()

        if final_segment is not None:
            segments_emitted += 1
            segment_durations_ms.append(
                final_segment.duration_ms
            )

            logger.event(
                "speech_segment",
                start_timestamp=(
                    final_segment.start_timestamp
                ),
                end_timestamp=(
                    final_segment.end_timestamp
                ),
                duration_ms=(
                    final_segment.duration_ms
                ),
                start_sequence_number=(
                    final_segment.start_sequence_number
                ),
                end_sequence_number=(
                    final_segment.end_sequence_number
                ),
                frame_count=(
                    final_segment.frame_count
                ),
                forced_split=True,
                sample_count=len(
                    final_segment.samples
                ),
                sample_rate=(
                    final_segment.sample_rate
                ),
            )

            print(
                "FINAL SEGMENT READY | "
                f"seq="
                f"{final_segment.start_sequence_number}-"
                f"{final_segment.end_sequence_number} | "
                f"duration="
                f"{final_segment.duration_ms:.1f} ms",
                flush=True,
            )

            try:
                accepted = asr_worker.submit(
                    final_segment
                )

                if not accepted:
                    asr_queue_drops += 1

            except Exception as exc:
                asr_failures += 1

                logger.error(
                    "Failed to submit final segment "
                    f"to ASR worker: {exc}"
                )

        audio_stream.stop()
        audio_stream.close()

        # Finish all submitted ASR work.
        asr_worker.wait_until_idle(
            timeout_seconds=30.0
        )

        handle_asr_results(
            asr_worker.poll_results()
        )

        # Finish all translation work created by ASR results.
        translation_worker.wait_until_idle(
            timeout_seconds=60.0
        )

        handle_translation_results(
            translation_worker.poll_results()
        )

        translation_worker.stop(
            drain=True,
            timeout_seconds=60.0
        )

        handle_translation_results(
            translation_worker.poll_results()
        )

        asr_worker.stop(
            drain=True,
            timeout_seconds=30.0
        )

        handle_asr_results(
            asr_worker.poll_results()
        )

        # ASR results arriving during the final ASR shutdown
        # cannot be translated after the translation worker stops.
        # Therefore poll once more only if required by the worker
        # lifecycle in future revisions.

        # ----------------------------------------------------
        # Evaluation summary
        # ----------------------------------------------------

        logger.info(
            "Audio + VAD evaluation summary:"
        )

        logger.info(
            f"processed_frames={processed_frames}"
        )

        logger.info(
            f"state_transitions={state_transitions}"
        )

        logger.info(
            f"audio_dropped_frames="
            f"{audio_stream.dropped_frames}"
        )

        if vad_times_ms:

            sorted_vad = sorted(
                vad_times_ms
            )

            p50_index = int(
                0.50
                * (len(sorted_vad) - 1)
            )

            p95_index = int(
                0.95
                * (len(sorted_vad) - 1)
            )

            p99_index = int(
                0.99
                * (len(sorted_vad) - 1)
            )

            logger.info(
                "vad_latency_ms: "
                f"mean="
                f"{sum(vad_times_ms) / len(vad_times_ms):.3f}, "
                f"p50="
                f"{sorted_vad[p50_index]:.3f}, "
                f"p95="
                f"{sorted_vad[p95_index]:.3f}, "
                f"p99="
                f"{sorted_vad[p99_index]:.3f}"
            )

        if audio_wait_times_ms:

            sorted_wait = sorted(
                audio_wait_times_ms
            )

            p50_index = int(
                0.50
                * (len(sorted_wait) - 1)
            )

            p95_index = int(
                0.95
                * (len(sorted_wait) - 1)
            )

            logger.info(
                "audio_read_ms: "
                f"mean="
                f"{sum(audio_wait_times_ms) / len(audio_wait_times_ms):.3f}, "
                f"p50="
                f"{sorted_wait[p50_index]:.3f}, "
                f"p95="
                f"{sorted_wait[p95_index]:.3f}"
            )
        logger.info(
            f"segments_emitted={segments_emitted}"
        )

        if segment_durations_ms:
            logger.info(
                "segment_duration_ms: "
                f"mean="
                f"{sum(segment_durations_ms) / len(segment_durations_ms):.1f}, "
                f"min="
                f"{min(segment_durations_ms):.1f}, "
                f"max="
                f"{max(segment_durations_ms):.1f}"
            )

        if buffer_processing_times_ms:
            sorted_buffer_times = sorted(
                buffer_processing_times_ms
            )

            p95_index = int(
                0.95
                * (len(sorted_buffer_times) - 1)
            )

            logger.info(
                "fixed_buffer_latency_ms: "
                f"mean="
                f"{sum(buffer_processing_times_ms) / len(buffer_processing_times_ms):.3f}, "
                f"p95="
                f"{sorted_buffer_times[p95_index]:.3f}"
            )
        logger.info(
            f"asr_submitted="
            f"{asr_worker.submitted_count}"
        )

        logger.info(
            f"asr_completed="
            f"{asr_worker.completed_count}"
        )

        logger.info(
            f"asr_failures="
            f"{asr_worker.failed_count}"
        )

        logger.info(
            f"asr_queue_drops="
            f"{asr_worker.dropped_count}"
        )

        if asr_inference_times_ms:
            sorted_asr = sorted(
                asr_inference_times_ms
            )

            p50_index = int(
                0.50
                * (len(sorted_asr) - 1)
            )

            p95_index = int(
                0.95
                * (len(sorted_asr) - 1)
            )

            logger.info(
                "asr_inference_ms: "
                f"mean="
                f"{sum(asr_inference_times_ms) / len(asr_inference_times_ms):.1f}, "
                f"p50="
                f"{sorted_asr[p50_index]:.1f}, "
                f"p95="
                f"{sorted_asr[p95_index]:.1f}"
            )

        if asr_queue_wait_times_ms:
            logger.info(
                "asr_queue_wait_ms: "
                f"mean="
                f"{sum(asr_queue_wait_times_ms) / len(asr_queue_wait_times_ms):.1f}, "
                f"max="
                f"{max(asr_queue_wait_times_ms):.1f}"
            )

        if asr_segment_to_transcript_times_ms:
            sorted_latency = sorted(
                asr_segment_to_transcript_times_ms
            )

            p50_index = int(
                0.50
                * (len(sorted_latency) - 1)
            )

            p95_index = int(
                0.95
                * (len(sorted_latency) - 1)
            )

            logger.info(
                "segment_to_transcript_ms: "
                f"mean="
                f"{sum(asr_segment_to_transcript_times_ms) / len(asr_segment_to_transcript_times_ms):.1f}, "
                f"p50="
                f"{sorted_latency[p50_index]:.1f}, "
                f"p95="
                f"{sorted_latency[p95_index]:.1f}"
            )

        if asr_rtf_values:
            logger.info(
                "asr_rtf: "
                f"mean="
                f"{sum(asr_rtf_values) / len(asr_rtf_values):.3f}, "
                f"min="
                f"{min(asr_rtf_values):.3f}, "
                f"max="
                f"{max(asr_rtf_values):.3f}"
            )
        logger.info(
            f"translation_submitted="
            f"{translation_worker.submitted_count}"
        )

        logger.info(
            f"translation_completed="
            f"{translation_worker.completed_count}"
        )

        logger.info(
            f"translation_failures="
            f"{translation_worker.failed_count}"
        )

        logger.info(
            f"translation_queue_drops="
            f"{translation_worker.dropped_count}"
        )

        logger.info(
            f"translation_empty_skips="
            f"{translation_worker.skipped_empty_count}"
        )

        if translation_inference_times_ms:
            sorted_translation = sorted(
                translation_inference_times_ms
            )

            p50_index = int(
                0.50
                * (len(sorted_translation) - 1)
            )

            p95_index = int(
                0.95
                * (len(sorted_translation) - 1)
            )

            logger.info(
                "translation_inference_ms: "
                f"mean="
                f"{sum(translation_inference_times_ms) / len(translation_inference_times_ms):.1f}, "
                f"p50="
                f"{sorted_translation[p50_index]:.1f}, "
                f"p95="
                f"{sorted_translation[p95_index]:.1f}"
            )

        if translation_queue_wait_times_ms:
            logger.info(
                "translation_queue_wait_ms: "
                f"mean="
                f"{sum(translation_queue_wait_times_ms) / len(translation_queue_wait_times_ms):.1f}, "
                f"max="
                f"{max(translation_queue_wait_times_ms):.1f}"
            )

        if translation_asr_to_completion_times_ms:
            sorted_asr_to_translation = sorted(
                translation_asr_to_completion_times_ms
            )

            p50_index = int(
                0.50
                * (len(sorted_asr_to_translation) - 1)
            )

            p95_index = int(
                0.95
                * (len(sorted_asr_to_translation) - 1)
            )

            logger.info(
                "asr_to_translation_ms: "
                f"mean="
                f"{sum(translation_asr_to_completion_times_ms) / len(translation_asr_to_completion_times_ms):.1f}, "
                f"p50="
                f"{sorted_asr_to_translation[p50_index]:.1f}, "
                f"p95="
                f"{sorted_asr_to_translation[p95_index]:.1f}"
            )

        if translation_segment_to_completion_times_ms:
            sorted_segment_to_translation = sorted(
                translation_segment_to_completion_times_ms
            )

            p50_index = int(
                0.50
                * (len(sorted_segment_to_translation) - 1)
            )

            p95_index = int(
                0.95
                * (len(sorted_segment_to_translation) - 1)
            )

            logger.info(
                "segment_to_translation_ms: "
                f"mean="
                f"{sum(translation_segment_to_completion_times_ms) / len(translation_segment_to_completion_times_ms):.1f}, "
                f"p50="
                f"{sorted_segment_to_translation[p50_index]:.1f}, "
                f"p95="
                f"{sorted_segment_to_translation[p95_index]:.1f}"
            )
    return 0


def run_m2(config, logger) -> int:
    """Run the research-aligned M2 streaming path.

    Audio -> VAD -> StreamingBuffer (persistent, single-decode-in-flight)
    -> StreamingASRDispatcher -> existing ASRWorker/WhisperEngine ->
    PrefixAgreementTracker (LocalAgreement-2, measured word timestamps)
    -> committed transcript.

    Phase 4C research-alignment pass: this drives the same
    build_streaming_buffer()/build_prefix_agreement_tracker() factories
    (and therefore the same production classes) that
    evaluation/aligned_m2_evaluation.py already exercises offline over
    WAV files - see that module's docstring for the reference
    architecture and literature comparison. streaming.adaptive_buffer
    .AdaptiveBuffer and streaming.local_agreement.LocalAgreementTracker
    are no longer built by this function; both remain unchanged and
    importable so evaluation/streaming_m2_evaluation.py's historical,
    pre-alignment evaluator runs stay reproducible.

    One deliberate difference from the offline evaluator's control
    flow: finalizing an utterance here never blocks on
    asr_worker.wait_until_idle(). Blocking here, while a live
    sounddevice capture thread keeps filling AudioStream's bounded
    circular buffer, would risk exactly the dropped-audio failure mode
    Phase 4C-2b already fixed once for the old AdaptiveBuffer path -
    see finalize_active_utterance()/check_deferred_finalizations()
    below for the non-blocking deferred-finalization replacement.

    Translation is deliberately NOT connected here (unchanged from the
    pre-alignment M2 path; out of scope for this integration pass).
    Stability control and M3 scheduling are not implemented.
    """

    features = config.features

    if not features.get(
        "use_adaptive_buffer",
        False,
    ):
        logger.error(
            "M2 requires features.use_adaptive_buffer=true."
        )

        return 1

    forbidden_features = [
        name
        for name in (
            "use_stability_controller",
            "use_translation_stability",
        )
        if features.get(name, False)
    ]

    if forbidden_features:
        logger.error(
            "M2 (Phase 3) cannot enable M3 features yet: "
            f"{forbidden_features}"
        )

        return 1

    # --------------------------------------------------------
    # Build modules
    # --------------------------------------------------------

    try:
        audio_stream = build_audio_stream(
            config,
            logger,
        )

        vad = build_vad(
            config,
            logger,
        )

        streaming_buffer = build_streaming_buffer(
            config,
            logger,
        )

        asr_engine = build_asr_engine(
            config,
            logger,
        )

        streaming_config = config.get(
            "streaming",
            default={},
        )

        asr_worker = build_asr_worker(
            config,
            asr_engine,
            logger,
            queue_size_override=streaming_config.get(
                "window_queue_size",
                4,
            ),
        )

        dispatcher = build_streaming_asr_dispatcher(
            asr_worker,
            logger,
        )

        prefix_tracker = build_prefix_agreement_tracker(
            config,
            logger,
        )

        performance_config = config.get(
            "performance",
            default={},
        )

        if performance_config.get(
            "warmup_asr",
            True,
        ):
            logger.info(
                "Running M2 ASR warm-up..."
            )

            warmup_result = asr_engine.warm_up()

            logger.info(
                "M2 ASR warm-up completed | "
                f"inference_ms="
                f"{warmup_result.inference_time_ms:.1f}"
            )

        # Translation is intentionally not built here - Phase 3 does
        # not connect M2 to TranslationWorker/TranslationEngine.

    except Exception as exc:
        logger.error(
            f"Failed to initialize M2 pipeline: {exc}"
        )

        return 1

    # --------------------------------------------------------
    # Runtime counters
    # --------------------------------------------------------

    processed_frames = 0
    state_transitions = 0
    windows_submitted = 0
    windows_rejected_at_submission = 0
    processed_utterance_ids: set[int] = set()

    # Phase 4C-2b-style lifecycle bookkeeping (see
    # finalize_active_utterance() and check_deferred_finalizations()
    # below), adapted to StreamingBuffer's single-pending-decode-per-
    # utterance model: how many windows have been ACCEPTED by the ASR
    # queue for each utterance.
    windows_accepted_per_utterance: dict[int, int] = {}

    # Wiring-correction fix (post Phase-4C-research-alignment live
    # acceptance test): the utterance_id StreamingBuffer is still
    # "occupying" because a decode was pending when it was finalized -
    # streaming_buffer.reset() is deliberately NOT called yet for it
    # (see finalize_active_utterance()). None when the buffer is free.
    # At most one entry can ever be true at a time - StreamingBuffer
    # supports only one utterance's state at a time - so this is a
    # single value, not a set.
    pending_finalization_utterance_id: int | None = None

    # Frames (and finalize triggers) for whatever utterance starts
    # while pending_finalization_utterance_id is still unresolved -
    # StreamingBuffer's single-utterance-state contract means these
    # cannot be handed to it yet without either raising (ingest onto a
    # FINALIZED state) or silently discarding the still-undecided prior
    # utterance's trailing audio. Replayed in order, via
    # replay_deferred_buffer_actions(), once check_deferred_
    # finalizations() resolves the entry above and resets the buffer.
    deferred_buffer_actions: list[tuple[str, object]] = []

    # First hypothesis's window_created_at per utterance - used only to
    # compute an OBSERVATIONAL commitment_delay_ms for logging (never a
    # correctness signal). Mirrors
    # evaluation/aligned_m2_evaluation.py's utterance_start_time().
    utterance_start_times: dict[int, float] = {}

    # Observability pass: counts how each utterance's termination was
    # decided, by trigger - see the m2_utterance_finalization event
    # fired from finalize_active_utterance() and
    # check_deferred_finalizations() below. Logged in the shutdown
    # summary. Purely additive/observational - does not affect
    # PrefixAgreementTracker's own commit/finalization semantics.
    #   clean_immediate: finalize() found no decode pending and built a
    #     final window right there (the common, non-raced case).
    #   clean_no_new_audio: finalize() found no decode pending and
    #     nothing new to flush (previous window already covered
    #     everything) - finalized directly, no window built.
    #   deferred_trailing_catchup: a decode was still pending at
    #     finalize(); once it resolved, trailing audio HAD accumulated
    #     while waiting, so a supplementary final window was built.
    #   deferred_no_trailing_audio: a decode was still pending at
    #     finalize(); once it resolved, no further audio had
    #     accumulated - finalized directly, no window built.
    finalization_trigger_counts: dict[str, int] = {
        "clean_immediate": 0,
        "clean_no_new_audio": 0,
        "deferred_trailing_catchup": 0,
        "deferred_no_trailing_audio": 0,
    }

    def submit_single_window(
        window: ProcessingWindow,
        is_final: bool,
    ) -> None:
        """Submit one window to ASR, updating counters and StreamingBuffer's
        pending flag, and logging the decision."""

        nonlocal windows_submitted
        nonlocal windows_rejected_at_submission

        accepted = dispatcher.submit_window(window)
        processed_utterance_ids.add(window.utterance_id)

        if accepted:
            windows_submitted += 1
            streaming_buffer.mark_submitted()

            windows_accepted_per_utterance[window.utterance_id] = (
                windows_accepted_per_utterance.get(
                    window.utterance_id, 0
                )
                + 1
            )
        else:
            windows_rejected_at_submission += 1

            if is_final:
                # The final window was rejected by ASRWorker's bounded
                # queue before ASR ever ran on it - no ASRHypothesis
                # (success or failure) will ever arrive for this
                # generation. Finalize explicitly now, mirroring the
                # pre-alignment path's identical "dropped final window"
                # case.
                prefix_tracker.finalize_utterance(
                    window.utterance_id,
                    reason=(
                        "final window_id="
                        f"{window.window_id} "
                        "generation="
                        f"{window.generation} was rejected by "
                        "the ASR queue at submission; no ASR "
                        "result will ever arrive for it"
                    ),
                )

        logger.event(
            "streaming_asr_window_submitted",
            utterance_id=window.utterance_id,
            window_id=window.window_id,
            generation=window.generation,
            is_final=window.is_final,
            context_duration_ms=window.context_duration_ms,
            new_audio_duration_ms=window.new_audio_duration_ms,
            duration_ms=window.duration_ms,
            accepted=accepted,
        )

    def try_submit_next_window() -> None:
        """Build and submit StreamingBuffer's next window, if ready.

        StreamingBuffer.try_build_window() returns None (not a list)
        when no active utterance exists, a decode is already pending,
        or the min_new_audio_ms trigger has not yet been reached (see
        streaming/streaming_buffer.py) - unlike AdaptiveBuffer.ingest(),
        which could emit several windows at once.
        """

        if streaming_buffer.state is None:
            return

        prompt = prefix_tracker.confirmed_prompt_text(
            streaming_buffer.state.utterance_id
        )

        window = streaming_buffer.try_build_window(initial_prompt=prompt)

        if window is not None:
            submit_single_window(window, is_final=False)

    def handle_hypotheses(hypotheses: list[ASRHypothesis]) -> None:
        """Log/print every newly completed hypothesis, run it through
        PrefixAgreementTracker, and free/trim StreamingBuffer.

        Unlike the pre-alignment path, PrefixAgreementTracker is always
        present (build_prefix_agreement_tracker() never returns None -
        see its own docstring), so there is no "tracker disabled"
        branch here.
        """

        for hypothesis in hypotheses:
            if hypothesis.utterance_id not in utterance_start_times:
                utterance_start_times[hypothesis.utterance_id] = (
                    hypothesis.window_created_at
                )

            commit: CommittedText | None = prefix_tracker.process(hypothesis)

            stability_snapshot = {
                # committed_token_count/stable_prefix_length are the
                # same underlying value under two names: the former
                # matches CLAUDE.md's metric vocabulary, the latter
                # matches this tracker's own existing method name.
                "committed_token_count": (
                    prefix_tracker.stable_prefix_length(
                        hypothesis.utterance_id
                    )
                ),
                "tentative_token_count": (
                    prefix_tracker.tentative_length(
                        hypothesis.utterance_id
                    )
                ),
                "stable_token_ratio": (
                    prefix_tracker.stable_token_ratio(
                        hypothesis.utterance_id
                    )
                ),
                "stable_prefix_length": (
                    prefix_tracker.stable_prefix_length(
                        hypothesis.utterance_id
                    )
                ),
                "alignment_revision_count": (
                    prefix_tracker.revision_count(
                        hypothesis.utterance_id
                    )
                ),
                "commitment_count": (
                    prefix_tracker.commitment_count(
                        hypothesis.utterance_id
                    )
                ),
            }

            logger.event(
                "streaming_asr_hypothesis",
                utterance_id=hypothesis.utterance_id,
                window_id=hypothesis.window_id,
                generation=hypothesis.generation,
                is_final=hypothesis.is_final,
                success=hypothesis.success,
                error=hypothesis.error,
                text=hypothesis.text,
                window_duration_ms=hypothesis.window_duration_ms,
                context_duration_ms=hypothesis.context_duration_ms,
                new_audio_duration_ms=hypothesis.new_audio_duration_ms,
                queue_wait_ms=hypothesis.queue_wait_ms,
                inference_time_ms=hypothesis.inference_time_ms,
                segment_to_transcript_ms=(
                    hypothesis.segment_to_transcript_ms
                ),
                rtf=hypothesis.rtf,
                # time.perf_counter()-based - monotonic and comparable
                # to every other timestamp in this event ONLY within
                # this same process run; never comparable across runs,
                # and never to be mixed with the wall-clock UTC
                # "timestamp" field EventLogger.event() adds above
                # (that field is for log ordering only).
                audio_start_time=hypothesis.audio_start_time,
                audio_end_time=hypothesis.audio_end_time,
                window_created_at=hypothesis.window_created_at,
                submitted_at=hypothesis.submitted_at,
                started_at=hypothesis.started_at,
                completed_at=hypothesis.completed_at,
                # Per-hypothesis PrefixAgreement stability snapshot -
                # observational only, never a correctness signal. See
                # stability_snapshot above for field-by-field
                # provenance.
                **stability_snapshot,
            )

            status = "OK" if hypothesis.success else "ERROR"

            print(
                f"HYPOTHESIS {status} | "
                f"utt={hypothesis.utterance_id} "
                f"win={hypothesis.window_id} "
                f"gen={hypothesis.generation} "
                f"final={hypothesis.is_final} | "
                f"ctx={hypothesis.context_duration_ms:.0f} ms "
                f"new={hypothesis.new_audio_duration_ms:.0f} ms | "
                f"queue_wait={hypothesis.queue_wait_ms:.1f} ms "
                f"inference="
                f"{(hypothesis.inference_time_ms or 0.0):.1f} ms "
                f"rtf={(hypothesis.rtf or 0.0):.3f}",
                flush=True,
            )

            if hypothesis.success:
                print(
                    f"TRANSCRIPT: {hypothesis.text}",
                    flush=True,
                )
            else:
                print(
                    f"ASR ERROR: {hypothesis.error}",
                    flush=True,
                )

            # Free the single decode-in-flight slot and trim confirmed
            # audio out of StreamingBuffer - only meaningful while this
            # hypothesis's utterance is still the buffer's ACTIVE one
            # (a late result for an already-finalized/reset utterance
            # is still processed by prefix_tracker above, but has
            # nothing left in streaming_buffer to update).
            if (
                streaming_buffer.state is not None
                and streaming_buffer.state.utterance_id
                == hypothesis.utterance_id
            ):
                streaming_buffer.mark_result_received()

                boundary = prefix_tracker.confirmed_boundary_time(
                    hypothesis.utterance_id
                )

                if boundary is not None:
                    streaming_buffer.trim_confirmed(boundary)

            if commit is not None:
                utterance_start = utterance_start_times.get(
                    hypothesis.utterance_id
                )

                # OBSERVATIONAL real-time delay from this utterance's
                # first hypothesis to this commit - never a correctness
                # signal, and distinct from PrefixAgreementTracker's own
                # time_to_first_commit() (audio-time-based). None when
                # the commit had no new text (e.g. an empty finalization
                # marker) or the start time is unknown.
                commitment_delay_ms = (
                    (commit.committed_at - utterance_start) * 1000.0
                    if utterance_start is not None and commit.new_text
                    else None
                )

                logger.event(
                    "prefix_agreement_commit",
                    utterance_id=commit.utterance_id,
                    commit_id=commit.commit_id,
                    generation=commit.generation,
                    window_id=hypothesis.window_id,
                    new_text=commit.new_text,
                    total_committed_text=(
                        commit.total_committed_text
                    ),
                    committed_at=commit.committed_at,
                    is_final=commit.is_final,
                    commitment_delay_ms=commitment_delay_ms,
                )

                label = (
                    "FINAL COMMITTED TRANSCRIPT"
                    if commit.is_final
                    else "STABLE (not verified as correct)"
                )

                print(
                    f"{label} | utt={commit.utterance_id} | "
                    f"new={commit.new_text!r} | "
                    f"total={commit.total_committed_text!r}",
                    flush=True,
                )

    def finalize_active_utterance() -> None:
        """Finalize the current utterance without blocking on ASR.

        Builds and submits the final window if StreamingBuffer has one
        ready. StreamingBuffer.try_build_final_window() returns None
        when a decode is still pending (caller must drain it first) or
        when there is no new audio left to flush (see
        streaming/streaming_buffer.py).

        Fix A (wiring-correction pass): when a final window IS built,
        it is submitted BEFORE streaming_buffer.reset() - submission
        calls streaming_buffer.mark_submitted() (sets _pending=True),
        so reset() must always run LAST or a leaked _pending=True would
        silently block the utterance that starts next from ever
        building a window (confirmed live: this caused a whole
        utterance - "at nine fifteen in the morning" - to be dropped
        with zero ASR calls).

        Fix B (wiring-correction pass): when a decode is still pending,
        streaming_buffer is deliberately NOT reset here at all -
        resetting immediately would discard any audio ingested after
        the last submitted window and before this finalize() call (the
        confirmed cause of "...ik, and the train leaves from" being
        silently dropped mid-utterance). Instead pending_finalization_
        utterance_id records that streaming_buffer is still occupied;
        check_deferred_finalizations() resolves it - attempting one
        more final-window build over any trailing audio - once the
        pending decode's result arrives, and only then resets the
        buffer. Whatever starts while this is unresolved is queued via
        deferred_buffer_actions (see the SPEECH/SILENCE branches in the
        main loop below) rather than handed to streaming_buffer, which
        would otherwise raise (ingest onto a FINALIZED state).
        """

        nonlocal pending_finalization_utterance_id

        if streaming_buffer.state is None:
            return

        ended_utterance_id = streaming_buffer.state.utterance_id
        decode_was_pending = streaming_buffer.pending

        streaming_buffer.finalize()

        prompt = prefix_tracker.confirmed_prompt_text(ended_utterance_id)
        final_window = streaming_buffer.try_build_final_window(
            initial_prompt=prompt
        )

        if final_window is not None:
            finalization_trigger_counts["clean_immediate"] += 1

            logger.event(
                "m2_utterance_finalization",
                utterance_id=ended_utterance_id,
                trigger="clean_immediate",
                window_id=final_window.window_id,
                generation=final_window.generation,
            )

            submit_single_window(final_window, is_final=True)
            streaming_buffer.reset()
        elif decode_was_pending:
            pending_finalization_utterance_id = ended_utterance_id
        else:
            streaming_buffer.reset()

            if not prefix_tracker.is_frozen(ended_utterance_id):
                # Nothing pending and nothing new to flush - finalize
                # now, there is nothing left to wait for.
                finalization_trigger_counts["clean_no_new_audio"] += 1

                prefix_tracker.finalize_utterance(
                    ended_utterance_id,
                    reason=(
                        "no new audio to flush at finalize "
                        "(already fully committed)"
                    ),
                )

                logger.event(
                    "prefix_agreement_finalized_without_final_window",
                    utterance_id=ended_utterance_id,
                )

                logger.event(
                    "m2_utterance_finalization",
                    utterance_id=ended_utterance_id,
                    trigger="clean_no_new_audio",
                    window_id=None,
                    generation=None,
                )

    def replay_deferred_buffer_actions() -> None:
        """Feed every frame/finalize action queued while streaming_buffer
        was occupied by an unresolved Fix-B finalization into it now
        that check_deferred_finalizations() has reset it and it is
        free again.

        Nested-deferral fix: a replayed "finalize" action can itself
        re-enter the decode-still-pending branch of
        finalize_active_utterance() - e.g. a queued utterance spoke
        long enough, once replayed, to build and submit its OWN
        window before its own queued "finalize" action ran, and that
        window's decode is (necessarily) still pending since no new
        poll_results() call happens mid-replay. When that happens,
        pending_finalization_utterance_id becomes non-None again and
        streaming_buffer is held (not reset) for THAT utterance -
        continuing this loop would hand any further queued actions
        (e.g. a third utterance's frames) to streaming_buffer while it
        still belongs to the newly-deferred one, raising (ingest onto
        a FINALIZED state). So this checks pending_finalization_
        utterance_id after every single action and, the moment it
        becomes non-None, stops immediately and puts every remaining
        unprocessed action back at the FRONT of deferred_buffer_actions
        (preserving their exact order) - they are replayed again, from
        the top of this same function, once THAT deferral resolves
        (check_deferred_finalizations() always calls this function
        again after resolving one). Nesting depth is unbounded - each
        level just adds one more deferred replay pass.
        """

        actions = deferred_buffer_actions[:]
        deferred_buffer_actions.clear()

        for index, (action, payload) in enumerate(actions):
            if action == "ingest":
                streaming_buffer.ingest(payload)
                try_submit_next_window()
            elif action == "finalize" and streaming_buffer.active:
                finalize_active_utterance()
            elif action == "observe":
                streaming_buffer.observe_non_speech(payload)

            if pending_finalization_utterance_id is not None:
                deferred_buffer_actions[:0] = actions[index + 1 :]
                return

    def check_deferred_finalizations() -> None:
        """Resolve a Fix-B deferred finalization once its pending decode
        has arrived.

        Once the one outstanding hypothesis the held utterance was
        entitled to (tracked via windows_accepted_per_utterance) has
        been recorded by prefix_tracker (success or failure - already
        done by handle_hypotheses() above, called earlier this same
        frame, which also cleared streaming_buffer.pending via
        mark_result_received()), attempt ONE more final-window build
        over whatever trailing audio accumulated while waiting. If
        there is one, submit it and let its own (is_final=True)
        hypothesis finalize PrefixAgreementTracker naturally through
        the normal handle_hypotheses()/process() path - exactly like
        any other clean final window (see finalize_active_utterance()'s
        own final_window branch) - rather than duplicating that logic
        here. If there is nothing new, finalize the tracker directly
        with whatever the pending hypothesis already contributed.
        Either way, streaming_buffer.reset() only happens here, after
        this utterance is fully resolved, and any queued frames/
        finalize triggers for the utterance that started in the
        meantime are then replayed. Deliberately NON-BLOCKING: it only
        compares two counts already tracked elsewhere and piggybacks on
        the per-frame handle_hypotheses(dispatcher.poll_results()) call
        that already runs every iteration.
        """

        nonlocal pending_finalization_utterance_id

        if pending_finalization_utterance_id is None:
            return

        utterance_id = pending_finalization_utterance_id

        ready = (
            prefix_tracker.hypothesis_count(utterance_id)
            + prefix_tracker.failed_hypothesis_count(utterance_id)
        ) >= windows_accepted_per_utterance.get(utterance_id, 0)

        if not ready:
            return

        prompt = prefix_tracker.confirmed_prompt_text(utterance_id)
        trailing_window = streaming_buffer.try_build_final_window(
            initial_prompt=prompt
        )

        if trailing_window is not None:
            finalization_trigger_counts["deferred_trailing_catchup"] += 1

            logger.event(
                "m2_utterance_finalization",
                utterance_id=utterance_id,
                trigger="deferred_trailing_catchup",
                window_id=trailing_window.window_id,
                generation=trailing_window.generation,
            )

            # Fix A discipline applies here too: submit first, reset
            # last.
            submit_single_window(trailing_window, is_final=True)
        elif not prefix_tracker.is_frozen(utterance_id):
            finalization_trigger_counts["deferred_no_trailing_audio"] += 1

            prefix_tracker.finalize_utterance(
                utterance_id,
                reason=(
                    "StreamingBuffer.try_build_final_window() "
                    "returned None because a decode was still "
                    "pending at finalize time; forcing "
                    "PrefixAgreementTracker finalization now that "
                    "every ASR call this utterance was entitled to "
                    "has completed and no further trailing audio "
                    "accumulated while waiting."
                ),
            )

            logger.event(
                "prefix_agreement_finalized_without_final_window",
                utterance_id=utterance_id,
            )

            logger.event(
                "m2_utterance_finalization",
                utterance_id=utterance_id,
                trigger="deferred_no_trailing_audio",
                window_id=None,
                generation=None,
            )

        pending_finalization_utterance_id = None
        streaming_buffer.reset()
        replay_deferred_buffer_actions()

    logger.info(
        "M2 Audio + VAD + Streaming Buffer + streaming ASR initialized."
    )

    logger.info(
        "Translation is NOT connected in M2 "
        "(streaming-ASR-only path)."
    )

    logger.info(
        f"ASR model load time: "
        f"{asr_engine.model_load_time_ms:.1f} ms"
    )

    logger.info(
        "Press Ctrl+C to stop."
    )

    try:
        audio_stream.start()

        while True:
            frame = audio_stream.read()

            if frame is None:
                continue

            processed_frames += 1

            result = vad.process(
                audio=frame.samples,
                sample_rate=frame.sample_rate,
                timestamp=frame.capture_timestamp,
                sequence_number=frame.sequence_number,
            )

            print(
                f"Frame={frame.sequence_number:03d} | "
                f"state={result.state.value:<7} | "
                f"prob={result.speech_probability:.3f} | "
                f"changed={result.state_changed} | "
                f"vad={result.processing_time_ms:.2f} ms",
                flush=True,
            )

            if result.state_changed:
                state_transitions += 1

            # ------------------------------------------------
            # StreamingBuffer does not implement VAD itself - it only
            # accumulates audio while SPEECH is active and an explicit
            # finalize() action on the SPEECH -> SILENCE transition.
            # ------------------------------------------------

            if pending_finalization_utterance_id is not None:
                # Fix B: streaming_buffer is still occupied by a not-
                # yet-resolved finalization (see finalize_active_
                # utterance()/check_deferred_finalizations()) - queue
                # this frame's action instead of touching
                # streaming_buffer directly. Replayed in order once the
                # buffer is freed.
                if result.state == VADState.SPEECH:
                    deferred_buffer_actions.append(("ingest", frame))
                else:
                    if (
                        result.state == VADState.SILENCE
                        and result.state_changed
                    ):
                        deferred_buffer_actions.append(("finalize", None))
                    # Candidate speech-onset pre-roll, replayed in order.
                    deferred_buffer_actions.append(("observe", frame))

            elif result.state == VADState.SPEECH:
                streaming_buffer.ingest(frame)
                try_submit_next_window()

            else:
                if (
                    result.state == VADState.SILENCE
                    and result.state_changed
                    and streaming_buffer.active
                ):
                    finalize_active_utterance()
                # Non-ingested frame: candidate pre-roll for the next
                # speech onset (see streaming/streaming_buffer.py).
                streaming_buffer.observe_non_speech(frame)

            handle_hypotheses(
                dispatcher.poll_results()
            )

            try_submit_next_window()

            check_deferred_finalizations()

    except KeyboardInterrupt:
        logger.info(
            "Stopping M2 streaming ASR."
        )

    except Exception as exc:
        logger.error(
            f"Runtime error: {exc}"
        )

        return 1

    finally:
        audio_stream.stop()
        audio_stream.close()

        # Shutdown-order fix: resolve EVERY deferred finalization while the
        # ASR worker is still running, and only then stop it. Resolving one
        # deferral can replay queued actions that start (and defer) the
        # next utterance, or leave it active if the stream ended mid-speech;
        # the previous fixed wait/poll/check sequence could then stop the
        # worker and submit that utterance's trailing final window to the
        # stopped worker (RuntimeError, final transcript lost). Bounded:
        # stops when nothing is outstanding or a round makes no progress
        # (e.g. a worker that never finishes - left unresolved, not raised).
        # Only shutdown ordering changes; streaming behaviour is untouched.
        def shutdown_work_outstanding() -> bool:
            decodes_in_flight = any(
                accepted > (
                    prefix_tracker.hypothesis_count(utterance_id)
                    + prefix_tracker.failed_hypothesis_count(utterance_id)
                )
                for utterance_id, accepted in windows_accepted_per_utterance.items()
            )
            return (
                decodes_in_flight
                or pending_finalization_utterance_id is not None
                or bool(deferred_buffer_actions)
                or streaming_buffer.active
            )

        for _shutdown_round in range(100):
            if streaming_buffer.active and pending_finalization_utterance_id is None:
                finalize_active_utterance()

            if not shutdown_work_outstanding():
                break

            asr_worker.wait_until_idle(
                timeout_seconds=30.0
            )

            results = dispatcher.poll_results()
            handle_hypotheses(results)

            state_before = (
                pending_finalization_utterance_id,
                len(deferred_buffer_actions),
                streaming_buffer.active,
            )
            check_deferred_finalizations()
            state_after = (
                pending_finalization_utterance_id,
                len(deferred_buffer_actions),
                streaming_buffer.active,
            )

            if not results and state_before == state_after:
                logger.warning(
                    "M2 shutdown: no progress resolving outstanding ASR/"
                    "finalization work; stopping with it unresolved."
                )
                break

        asr_worker.stop(
            drain=True,
            timeout_seconds=30.0,
        )

        handle_hypotheses(
            dispatcher.poll_results()
        )

        if pending_finalization_utterance_id is not None:
            # Never submit to the stopped worker; report instead.
            logger.warning(
                "M2 shutdown: deferred finalization for utterance_id="
                f"{pending_finalization_utterance_id} left unresolved "
                "(ASR worker already stopped)."
            )

        # ----------------------------------------------------
        # Evaluation summary
        # ----------------------------------------------------

        logger.info(
            "M2 streaming ASR evaluation summary:"
        )

        logger.info(
            f"processed_frames={processed_frames}"
        )

        logger.info(
            f"state_transitions={state_transitions}"
        )

        logger.info(
            f"audio_dropped_frames="
            f"{audio_stream.dropped_frames}"
        )

        logger.info(
            f"windows_submitted={windows_submitted}"
        )

        logger.info(
            "windows_rejected_at_submission="
            f"{windows_rejected_at_submission}"
        )

        for key, value in dispatcher.metrics_snapshot().items():
            logger.info(f"{key}={value}")

        # ----------------------------------------------------
        # Phase 4C-1: backpressure/dropped-window observability.
        #
        # StreamingBuffer's own lifetime diagnostics (see
        # streaming/streaming_buffer.py) - replaces the pre-alignment
        # path's AdaptiveBuffer.superseded_count/max_pending_occupancy,
        # which have no equivalent here: StreamingBuffer never queues
        # or evicts multiple pending windows (single-decode-in-flight
        # design), so there is nothing to "supersede".
        #   - windows_built: total windows StreamingBuffer has built.
        #   - max_buffer_forced_count: how many of those were forced by
        #     the max_buffer_ms safety ceiling rather than a fresh
        #     LocalAgreement confirmation trigger.
        #   - asr_dropped_window_count: ASRWorker's bounded queue was
        #     full at submission time and rejected the window outright
        #     (== windows_rejected_at_submission /
        #     dispatcher.dropped_window_count, all the same quantity
        #     under one canonical name).
        # ----------------------------------------------------

        streaming_buffer_windows_built = streaming_buffer.windows_built
        streaming_buffer_max_buffer_forced_count = (
            streaming_buffer.max_buffer_forced_count
        )

        logger.info(
            f"streaming_buffer_windows_built={streaming_buffer_windows_built}"
        )

        logger.info(
            "streaming_buffer_max_buffer_forced_count="
            f"{streaming_buffer_max_buffer_forced_count}"
        )

        logger.info(
            f"asr_dropped_window_count={windows_rejected_at_submission}"
        )

        # Observability pass: per-trigger breakdown of how every
        # utterance in this run was terminated - see
        # finalization_trigger_counts's own declaration above for what
        # each trigger means.
        for trigger, count in finalization_trigger_counts.items():
            logger.info(
                f"m2_utterance_finalization_count[{trigger}]={count}"
            )

        # One additive, structured, run-level summary event - keeps the
        # existing plain-text logger.info lines above unchanged while
        # giving evaluation scripts a single JSONL line to parse
        # instead of scraping text output.
        logger.event(
            "m2_run_summary",
            windows_submitted=windows_submitted,
            asr_dropped_window_count=windows_rejected_at_submission,
            streaming_buffer_windows_built=streaming_buffer_windows_built,
            streaming_buffer_max_buffer_forced_count=(
                streaming_buffer_max_buffer_forced_count
            ),
            finalization_trigger_counts=dict(finalization_trigger_counts),
            **dispatcher.metrics_snapshot(),
        )

        try:
            from evaluation.asr_evaluation import (
                percentile,
                summarize_latency,
            )

            queue_wait_summary = summarize_latency(
                list(dispatcher.queue_wait_samples_ms)
            )

            inference_summary = summarize_latency(
                list(dispatcher.inference_time_samples_ms)
            )

            rtf_samples = list(dispatcher.rtf_samples)

            logger.info(
                "queue_wait_ms: "
                f"mean={queue_wait_summary['mean_ms']:.1f}, "
                f"p50={queue_wait_summary['p50_ms']:.1f}, "
                f"p95={queue_wait_summary['p95_ms']:.1f}, "
                f"p99={queue_wait_summary['p99_ms']:.1f}, "
                f"max={queue_wait_summary['max_ms']:.1f}"
            )

            logger.info(
                "inference_time_ms: "
                f"mean={inference_summary['mean_ms']:.1f}, "
                f"p50={inference_summary['p50_ms']:.1f}, "
                f"p95={inference_summary['p95_ms']:.1f}, "
                f"p99={inference_summary['p99_ms']:.1f}, "
                f"max={inference_summary['max_ms']:.1f}"
            )

            if rtf_samples:
                logger.info(
                    "rtf: "
                    f"mean={sum(rtf_samples) / len(rtf_samples):.3f}, "
                    f"p50={percentile(rtf_samples, 50.0):.3f}, "
                    f"p95={percentile(rtf_samples, 95.0):.3f}, "
                    f"p99={percentile(rtf_samples, 99.0):.3f}, "
                    f"max={max(rtf_samples):.3f}"
                )

        except Exception as exc:
            logger.warning(
                "Failed to summarize M2 latency statistics: "
                f"{exc}"
            )

        # ----------------------------------------------------
        # PrefixAgreementTracker (LocalAgreement-2) summary
        # ----------------------------------------------------

        logger.info(
            "M2 PrefixAgreement summary:"
        )

        for utterance_id in sorted(processed_utterance_ids):
            if prefix_tracker.hypothesis_count(
                utterance_id
            ) == 0:
                continue

            committed_text = (
                prefix_tracker.committed_text_for(
                    utterance_id
                )
            )

            logger.event(
                "prefix_agreement_utterance_summary",
                utterance_id=utterance_id,
                hypothesis_count=(
                    prefix_tracker.hypothesis_count(
                        utterance_id
                    )
                ),
                failed_hypothesis_count=(
                    prefix_tracker.failed_hypothesis_count(
                        utterance_id
                    )
                ),
                alignment_revision_count=(
                    prefix_tracker.revision_count(
                        utterance_id
                    )
                ),
                hypothesis_revision_count=(
                    dispatcher.revision_count(utterance_id)
                ),
                commitment_count=(
                    prefix_tracker.commitment_count(
                        utterance_id
                    )
                ),
                stable_prefix_length=(
                    prefix_tracker.stable_prefix_length(
                        utterance_id
                    )
                ),
                tentative_token_count=(
                    prefix_tracker.tentative_length(
                        utterance_id
                    )
                ),
                stable_token_ratio=(
                    prefix_tracker.stable_token_ratio(
                        utterance_id
                    )
                ),
                time_to_first_commit=(
                    prefix_tracker.time_to_first_commit(
                        utterance_id
                    )
                ),
                is_frozen=prefix_tracker.is_frozen(
                    utterance_id
                ),
                final_committed_transcript=committed_text,
            )

            print(
                "FINAL COMMITTED TRANSCRIPT SUMMARY | "
                f"utt={utterance_id} | "
                f"frozen={prefix_tracker.is_frozen(utterance_id)} | "
                f"text={committed_text!r}",
                flush=True,
            )

    return 0


def main() -> int:
    """Load configuration and dispatch to the configured system."""

    args = parse_args()

    try:
        config = load_config(
            args.config
        )

    except ConfigError as exc:
        print(
            f"Configuration error: {exc}",
            file=sys.stderr,
        )

        return 1

    logger = build_logger(
        config
    )

    logger.info(
        f"Starting system: "
        f"{config.system_name}"
    )

    features = config.features

    logger.info(
        f"Feature flags: {features}"
    )

    if config.system_name == "M1":
        return run_m1(config, logger)

    if config.system_name == "M2":
        return run_m2(config, logger)

    logger.error(
        "Unsupported system_name for main.py: "
        f"{config.system_name}"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
