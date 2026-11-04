"""CPU-friendly English-to-Hindi translation engine."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch


@dataclass(frozen=True)
class TranslationResult:
    """Structured result returned by the translation engine.

    `output_token_count` is the generated sequence length INCLUDING
    special tokens (decoder start / EOS) - kept unchanged for M1 log
    comparability. `target_token_count` counts only non-special generated
    tokens, the quantity MT-2/M3 computation accounting uses.
    """

    source_text: str
    translated_text: str

    source_language: str
    target_language: str

    input_token_count: int
    output_token_count: int

    chunk_count: int
    inference_time_ms: float

    source_word_count: int = 0
    target_token_count: int = 0

    @property
    def duration_ms(self) -> float:
        """Backward-compatible alias for inference time."""

        return self.inference_time_ms


@dataclass(frozen=True)
class TranslationRequest:
    """One MT invocation request.

    MT-1 uses only `source_text`. The remaining fields are carried through
    unchanged so that MT-2 (incremental stable-prefix input) and M3
    (scheduling: translation debt, last MT time) can attach their source
    state to each invocation without changing this interface. No
    scheduling logic lives here.
    """

    source_text: str
    timestamp: float | None = None
    is_final: bool = True
    stable_source_word_count: int | None = None
    newly_stabilized_words: int | None = None


@dataclass(frozen=True)
class TranslationResponse:
    """Result of one `TranslationEngine.translate_request()` call."""

    request: TranslationRequest
    result: TranslationResult
    started_at: float
    completed_at: float

    @property
    def target_text(self) -> str:
        return self.result.translated_text

    @property
    def inference_time_ms(self) -> float:
        return self.result.inference_time_ms

    @property
    def wall_time_ms(self) -> float:
        """Full call time, including input splitting/validation."""

        return (self.completed_at - self.started_at) * 1000.0

    @property
    def source_word_count(self) -> int:
        return self.result.source_word_count

    @property
    def target_token_count(self) -> int:
        return self.result.target_token_count


class TranslationEngine:
    """Persistent CPU-friendly English-to-Hindi translator.

    The pretrained translation model is loaded once and reused.

    Long input is split into token-safe chunks. Each chunk is
    translated independently and the results are joined in order.
    """

    SUPPORTED_SOURCE_LANGUAGE = "en"
    SUPPORTED_TARGET_LANGUAGE = "hi"

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-hi",
        device: str = "cpu",
        max_input_tokens: int = 256,
        num_beams: int = 1,
        max_new_tokens: int = 256,
        revision: str | None = None,
    ) -> None:
        if not model_name:
            raise ValueError(
                "model_name must not be empty."
            )

        if not device:
            raise ValueError(
                "device must not be empty."
            )

        if max_input_tokens <= 0:
            raise ValueError(
                "max_input_tokens must be greater than zero."
            )

        if num_beams <= 0:
            raise ValueError(
                "num_beams must be greater than zero."
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        try:
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed."
            ) from exc

        self.model_name = model_name
        self.device = torch.device(device)
        self.max_input_tokens = max_input_tokens
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.revision = revision

        # Only pass `revision` when pinned, keeping the default call
        # identical to the pre-MT-1 behavior.
        load_kwargs: dict[str, Any] = (
            {"revision": revision} if revision else {}
        )

        load_start = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **load_kwargs,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            **load_kwargs,
        )

        self.model.to(self.device)
        self.model.eval()

        self.model_load_time_ms = (
            time.perf_counter() - load_start
        ) * 1000.0

        model_limit = getattr(
            self.tokenizer,
            "model_max_length",
            max_input_tokens,
        )

        # Some tokenizers use an extremely large sentinel
        # instead of a real model limit.
        if (
            model_limit is None
            or model_limit > 10000
        ):
            model_limit = max_input_tokens

        self.effective_max_input_tokens = min(
            max_input_tokens,
            int(model_limit),
        )

    def _validate_text(
        self,
        text: str,
    ) -> str:
        """Validate and normalize source text."""

        if text is None:
            raise ValueError(
                "text must not be None."
            )

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string."
            )

        return " ".join(
            text.strip().split()
        )

    def _split_long_text(
        self,
        text: str,
    ) -> list[str]:
        """Split text into tokenizer-safe word chunks."""

        if not text:
            return []

        # Fast path: the whole text fits, which is exactly when the
        # incremental loop below would also return a single chunk - same
        # output, without re-tokenizing every growing prefix.
        if (
            len(self.tokenizer.encode(text, add_special_tokens=True))
            <= self.effective_max_input_tokens
        ):
            return [text]

        words = text.split()

        chunks: list[str] = []
        current_words: list[str] = []

        for word in words:
            candidate_words = (
                current_words
                + [word]
            )

            candidate_text = " ".join(
                candidate_words
            )

            token_count = len(
                self.tokenizer.encode(
                    candidate_text,
                    add_special_tokens=True,
                )
            )

            if (
                current_words
                and token_count
                > self.effective_max_input_tokens
            ):
                chunks.append(
                    " ".join(
                        current_words
                    )
                )

                current_words = [
                    word
                ]

            else:
                current_words = (
                    candidate_words
                )

        if current_words:
            chunks.append(
                " ".join(
                    current_words
                )
            )

        return chunks

    def _count_target_tokens(
        self,
        generated: Any,
    ) -> int:
        """Generated tokens excluding special tokens (pad/decoder-start, EOS)."""

        try:
            token_ids = generated[0].tolist()
        except AttributeError:
            return int(generated.shape[1])

        special_ids = set(
            getattr(self.tokenizer, "all_special_ids", []) or []
        )

        return sum(
            1 for token_id in token_ids if token_id not in special_ids
        )

    def _translate_chunk(
        self,
        chunk: str,
    ) -> tuple[str, int, int, int]:
        """Translate one safe-sized chunk."""

        encoded = self.tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=(
                self.effective_max_input_tokens
            ),
        )

        input_token_count = int(
            encoded["input_ids"].shape[1]
        )

        encoded = {
            key: value.to(
                self.device
            )
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
            )

        output_token_count = int(
            generated.shape[1]
        )

        translated = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ).strip()

        return (
            translated,
            input_token_count,
            output_token_count,
            self._count_target_tokens(generated),
        )

    def translate(
        self,
        text: str,
    ) -> TranslationResult:
        """Translate English text to Hindi."""

        source_text = self._validate_text(
            text
        )

        if not source_text:
            return TranslationResult(
                source_text="",
                translated_text="",
                source_language=self.SUPPORTED_SOURCE_LANGUAGE,
                target_language=self.SUPPORTED_TARGET_LANGUAGE,
                input_token_count=0,
                output_token_count=0,
                chunk_count=0,
                inference_time_ms=0.0,
            )

        chunks = self._split_long_text(
            source_text
        )

        translated_chunks: list[str] = []

        total_input_tokens = 0
        total_output_tokens = 0
        total_target_tokens = 0

        inference_start = time.perf_counter()

        for chunk in chunks:
            (
                translated,
                input_tokens,
                output_tokens,
                target_tokens,
            ) = self._translate_chunk(
                chunk
            )

            total_target_tokens += target_tokens

            if translated:
                translated_chunks.append(
                    translated
                )

            total_input_tokens += (
                input_tokens
            )

            total_output_tokens += (
                output_tokens
            )

        inference_time_ms = (
            time.perf_counter()
            - inference_start
        ) * 1000.0

        return TranslationResult(
            source_text=source_text,
            translated_text=" ".join(
                translated_chunks
            ).strip(),
            source_language=self.SUPPORTED_SOURCE_LANGUAGE,
            target_language=self.SUPPORTED_TARGET_LANGUAGE,
            input_token_count=total_input_tokens,
            output_token_count=total_output_tokens,
            chunk_count=len(chunks),
            inference_time_ms=inference_time_ms,
            source_word_count=len(source_text.split()),
            target_token_count=total_target_tokens,
        )

    def translate_request(
        self,
        request: TranslationRequest,
    ) -> TranslationResponse:
        """Translate one request; records full call timing.

        Thin wrapper over `translate()` (identical output) - the stable
        entry point MT-2/M3 will call.
        """

        started_at = time.perf_counter()
        result = self.translate(request.source_text)
        completed_at = time.perf_counter()

        return TranslationResponse(
            request=request,
            result=result,
            started_at=started_at,
            completed_at=completed_at,
        )

    def generation_settings(self) -> dict[str, Any]:
        """Effective decoding configuration, for evaluation artifacts."""

        generation_config = getattr(self.model, "generation_config", None)
        model_defaults = (
            generation_config.to_dict()
            if hasattr(generation_config, "to_dict")
            else None
        )

        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "device": str(self.device),
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "max_input_tokens": self.max_input_tokens,
            "effective_max_input_tokens": self.effective_max_input_tokens,
            "do_sample": False,
            "explicit_generate_kwargs": {
                "num_beams": self.num_beams,
                "max_new_tokens": self.max_new_tokens,
            },
            "model_generation_config_defaults": model_defaults,
        }

    def warm_up(
        self,
    ) -> TranslationResult:
        """Run a small warm-up translation."""

        return self.translate(
            "Hello."
        )