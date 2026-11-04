"""M1 ASR evaluation utilities and benchmark runner."""

from __future__ import annotations

import argparse
import json
import math
import string
import wave
from pathlib import Path
from typing import Any

import numpy as np

from asr.whisper_engine import WhisperEngine
from utils.config import ConfigError, load_config


# ---------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------

PUNCTUATION_TABLE = str.maketrans(
    "",
    "",
    string.punctuation,
)


def normalize_text(text: str) -> str:
    """Normalize text for WER evaluation."""

    text = text.lower()
    text = text.translate(
        PUNCTUATION_TABLE
    )
    text = " ".join(
        text.split()
    )
    return text.strip()


def word_tokens(text: str) -> list[str]:
    """Return normalized word tokens."""

    normalized = normalize_text(text)

    if not normalized:
        return []

    return normalized.split()


def character_tokens(text: str) -> list[str]:
    """Return normalized character tokens.

    Whitespace is removed so CER is not affected by
    differences in spacing.
    """

    normalized = normalize_text(text)

    return [
        char
        for char in normalized
        if not char.isspace()
    ]


# ---------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------

def edit_distance(
    reference: list[str],
    hypothesis: list[str],
) -> int:
    """Compute Levenshtein edit distance."""

    previous = list(
        range(len(hypothesis) + 1)
    )

    for i, ref_item in enumerate(
        reference,
        start=1,
    ):
        current = [
            i
        ] + [
            0
            for _ in hypothesis
        ]

        for j, hyp_item in enumerate(
            hypothesis,
            start=1,
        ):
            deletion = (
                previous[j]
                + 1
            )

            insertion = (
                current[j - 1]
                + 1
            )

            substitution = (
                previous[j - 1]
                + (
                    0
                    if ref_item == hyp_item
                    else 1
                )
            )

            current[j] = min(
                deletion,
                insertion,
                substitution,
            )

        previous = current

    return previous[-1]


def corpus_error_rate(
    references: list[list[str]],
    hypotheses: list[list[str]],
) -> tuple[int, int, float]:
    """Calculate corpus-level error rate."""

    total_errors = 0
    total_reference_units = 0

    for reference, hypothesis in zip(
        references,
        hypotheses,
    ):
        total_errors += edit_distance(
            reference,
            hypothesis,
        )
        total_reference_units += len(
            reference
        )

    if total_reference_units == 0:
        return (
            total_errors,
            total_reference_units,
            0.0,
        )

    return (
        total_errors,
        total_reference_units,
        total_errors / total_reference_units,
    )


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    """Calculate linearly interpolated percentile."""

    if not values:
        return 0.0

    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError(
            "percentile_value must be between 0 and 100."
        )

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (
        (len(sorted_values) - 1)
        * percentile_value
        / 100.0
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return sorted_values[lower]

    weight = position - lower

    return (
        sorted_values[lower]
        * (1.0 - weight)
        + sorted_values[upper]
        * weight
    )


def summarize_latency(
    values: list[float],
) -> dict[str, float]:
    """Return latency summary statistics."""

    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }

    return {
        "count": len(values),
        "mean_ms": sum(values)
        / len(values),
        "min_ms": min(values),
        "p50_ms": percentile(
            values,
            50.0,
        ),
        "p95_ms": percentile(
            values,
            95.0,
        ),
        "p99_ms": percentile(
            values,
            99.0,
        ),
        "max_ms": max(values),
    }


# ---------------------------------------------------------------------
# WAV loading
# ---------------------------------------------------------------------

def load_wav(
    path: Path,
) -> tuple[np.ndarray, int]:
    """Load a mono PCM16 WAV file."""

    with wave.open(
        str(path),
        "rb",
    ) as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw = wav.readframes(
            frame_count
        )

    if channels != 1:
        raise ValueError(
            f"{path} must be mono."
        )

    if sample_width != 2:
        raise ValueError(
            f"{path} must be 16-bit PCM."
        )

    audio = (
        np.frombuffer(
            raw,
            dtype=np.int16,
        )
        .astype(np.float32)
        / 32768.0
    )

    return audio, sample_rate


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------

def load_manifest(
    path: Path,
) -> list[dict[str, str]]:
    """Load an ASR evaluation manifest."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    if not isinstance(
        manifest,
        list,
    ):
        raise ValueError(
            "ASR manifest must be a JSON list."
        )

    for index, item in enumerate(
        manifest
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                f"Manifest entry {index} "
                "must be an object."
            )

        if "audio" not in item:
            raise ValueError(
                f"Manifest entry {index} "
                "is missing 'audio'."
            )

        if "reference" not in item:
            raise ValueError(
                f"Manifest entry {index} "
                "is missing 'reference'."
            )

    return manifest


# ---------------------------------------------------------------------
# Offline benchmark
# ---------------------------------------------------------------------

def run_offline_benchmark(
    config_path: Path,
    manifest_path: Path,
    model_override: str | None = None,
) -> dict[str, Any]:
    """Run ASR against a reference-labelled corpus."""

    config = load_config(
        config_path
    )

    asr_config = config.get(
        "asr",
        default={},
    )

    model_size = (
        model_override
        if model_override
        else asr_config["model_size"]
    )

    engine = WhisperEngine(
        model_size=model_size,
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
        cpu_threads=asr_config.get(
            "cpu_threads",
            8,
        ),
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

    manifest = load_manifest(
        manifest_path
    )

    # Warm up once.
    warmup = engine.warm_up()

    references_words: list[list[str]] = []
    hypotheses_words: list[list[str]] = []

    references_chars: list[list[str]] = []
    hypotheses_chars: list[list[str]] = []

    inference_times: list[float] = []
    rtfs: list[float] = []
    audio_durations: list[float] = []

    utterance_results: list[dict[str, Any]] = []

    for index, item in enumerate(
        manifest,
        start=1,
    ):
        audio_path = Path(
            item["audio"]
        )

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: "
                f"{audio_path}"
            )

        reference = item[
            "reference"
        ]

        audio, sample_rate = load_wav(
            audio_path
        )

        result = engine.transcribe(
            audio,
            sample_rate,
        )

        ref_words = word_tokens(
            reference
        )
        hyp_words = word_tokens(
            result.text
        )

        ref_chars = character_tokens(
            reference
        )
        hyp_chars = character_tokens(
            result.text
        )

        references_words.append(
            ref_words
        )
        hypotheses_words.append(
            hyp_words
        )

        references_chars.append(
            ref_chars
        )
        hypotheses_chars.append(
            hyp_chars
        )

        inference_times.append(
            result.inference_time_ms
        )
        rtfs.append(
            result.rtf
        )
        audio_durations.append(
            result.audio_duration_ms
        )

        utterance_results.append(
            {
                "index": index,
                "audio": str(audio_path),
                "reference": reference,
                "hypothesis": result.text,
                "language": result.language,
                "audio_duration_ms": (
                    result.audio_duration_ms
                ),
                "inference_time_ms": (
                    result.inference_time_ms
                ),
                "rtf": result.rtf,
                "segment_count": (
                    result.segment_count
                ),
            }
        )

    word_errors, word_count, wer = (
        corpus_error_rate(
            references_words,
            hypotheses_words,
        )
    )

    char_errors, char_count, cer = (
        corpus_error_rate(
            references_chars,
            hypotheses_chars,
        )
    )

    result = {
        "model": model_size,
        "device": asr_config["device"],
        "compute_type": asr_config[
            "compute_type"
        ],
        "model_load_time_ms": (
            engine.model_load_time_ms
        ),
        "warmup_inference_ms": (
            warmup.inference_time_ms
        ),
        "utterance_count": len(
            manifest
        ),
        "total_audio_duration_ms": sum(
            audio_durations
        ),
        "accuracy": {
            "wer": wer,
            "wer_percent": wer * 100.0,
            "word_errors": word_errors,
            "reference_word_count": word_count,
            "cer": cer,
            "cer_percent": cer * 100.0,
            "character_errors": char_errors,
            "reference_character_count": char_count,
        },
        "latency": summarize_latency(
            inference_times
        ),
        "rtf": {
            "mean": (
                sum(rtfs) / len(rtfs)
                if rtfs
                else 0.0
            ),
            "min": (
                min(rtfs)
                if rtfs
                else 0.0
            ),
            "p50": percentile(
                rtfs,
                50.0,
            ),
            "p95": percentile(
                rtfs,
                95.0,
            ),
            "p99": percentile(
                rtfs,
                99.0,
            ),
            "max": (
                max(rtfs)
                if rtfs
                else 0.0
            ),
        },
        "utterances": utterance_results,
    }

    return result


# ---------------------------------------------------------------------
# Live log analysis
# ---------------------------------------------------------------------

def analyze_live_log(
    log_path: Path,
) -> dict[str, Any]:
    """Analyze asr_result events generated by main.py."""

    inference_times: list[float] = []
    queue_wait_times: list[float] = []
    segment_latency_times: list[float] = []
    rtfs: list[float] = []

    submitted = 0
    completed = 0
    failures = 0
    queue_drops = 0

    with log_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if "asr_result:" not in line:
                continue

            payload = line.split(
                "asr_result:",
                1,
            )[1].strip()

            # Logger writes Python-dict style text.
            # Safely parse only the expected primitive structure.
            import ast

            try:
                event = ast.literal_eval(
                    payload
                )
            except (SyntaxError, ValueError):
                continue

            if event.get(
                "error"
            ):
                failures += 1
                continue

            if "inference_time_ms" not in event:
                continue

            completed += 1

            inference_times.append(
                float(
                    event[
                        "inference_time_ms"
                    ]
                )
            )

            queue_wait_times.append(
                float(
                    event[
                        "queue_wait_ms"
                    ]
                )
            )

            segment_latency_times.append(
                float(
                    event[
                        "segment_to_transcript_ms"
                    ]
                )
            )

            rtfs.append(
                float(
                    event["rtf"]
                )
            )

    result = {
        "log": str(log_path),
        "asr_completed": completed,
        "asr_failures": failures,
        "asr_submitted": submitted,
        "asr_queue_drops": queue_drops,
        "inference": summarize_latency(
            inference_times
        ),
        "queue_wait": summarize_latency(
            queue_wait_times
        ),
        "segment_to_transcript": summarize_latency(
            segment_latency_times
        ),
        "rtf": {
            "count": len(rtfs),
            "mean": (
                sum(rtfs) / len(rtfs)
                if rtfs
                else 0.0
            ),
            "min": (
                min(rtfs)
                if rtfs
                else 0.0
            ),
            "p50": percentile(
                rtfs,
                50.0,
            ),
            "p95": percentile(
                rtfs,
                95.0,
            ),
            "p99": percentile(
                rtfs,
                99.0,
            ),
            "max": (
                max(rtfs)
                if rtfs
                else 0.0
            ),
        },
    }

    return result


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> int:
    """Run the selected evaluation."""

    parser = argparse.ArgumentParser(
        description="M1 ASR evaluation"
    )

    parser.add_argument(
        "--config",
        help="Path to M1 YAML config.",
    )

    parser.add_argument(
        "--manifest",
        help="JSON ASR evaluation manifest.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override, e.g. tiny or small.",
    )

    parser.add_argument(
        "--live-log",
        help="Analyze a live M1 JSON/text event log.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path.",
    )

    args = parser.parse_args()

    try:
        if args.manifest:
            if not args.config:
                parser.error(
                    "--config is required with --manifest."
                )

            result = run_offline_benchmark(
                config_path=Path(
                    args.config
                ),
                manifest_path=Path(
                    args.manifest
                ),
                model_override=args.model,
            )

            print(
                "\n===== M1 OFFLINE ASR EVALUATION ====="
            )
            print(
                f"model={result['model']}"
            )
            print(
                f"utterances="
                f"{result['utterance_count']}"
            )
            print(
                f"WER="
                f"{result['accuracy']['wer_percent']:.2f}%"
            )
            print(
                f"CER="
                f"{result['accuracy']['cer_percent']:.2f}%"
            )
            print(
                f"inference_mean_ms="
                f"{result['latency']['mean_ms']:.1f}"
            )
            print(
                f"inference_p50_ms="
                f"{result['latency']['p50_ms']:.1f}"
            )
            print(
                f"inference_p95_ms="
                f"{result['latency']['p95_ms']:.1f}"
            )
            print(
                f"inference_p99_ms="
                f"{result['latency']['p99_ms']:.1f}"
            )
            print(
                f"rtf_mean="
                f"{result['rtf']['mean']:.3f}"
            )

        elif args.live_log:
            result = analyze_live_log(
                Path(
                    args.live_log
                )
            )

            print(
                "\n===== M1 LIVE ASR EVALUATION ====="
            )
            print(
                f"asr_completed="
                f"{result['asr_completed']}"
            )
            print(
                f"asr_failures="
                f"{result['asr_failures']}"
            )
            print(
                f"inference_mean_ms="
                f"{result['inference']['mean_ms']:.1f}"
            )
            print(
                f"inference_p50_ms="
                f"{result['inference']['p50_ms']:.1f}"
            )
            print(
                f"inference_p95_ms="
                f"{result['inference']['p95_ms']:.1f}"
            )
            print(
                f"inference_p99_ms="
                f"{result['inference']['p99_ms']:.1f}"
            )
            print(
                f"queue_wait_mean_ms="
                f"{result['queue_wait']['mean_ms']:.1f}"
            )
            print(
                f"segment_to_transcript_p95_ms="
                f"{result['segment_to_transcript']['p95_ms']:.1f}"
            )
            print(
                f"rtf_mean="
                f"{result['rtf']['mean']:.3f}"
            )

        else:
            parser.error(
                "Provide either --manifest or --live-log."
            )

        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                result,
                handle,
                indent=2,
            )

        print(
            f"saved={output_path}"
        )

        return 0

    except (
        ConfigError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(
            f"Evaluation error: {exc}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )