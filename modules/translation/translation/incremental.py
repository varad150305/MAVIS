"""MT-2: eager re-translation of growing stable source prefixes.

Wraps the frozen `TranslationEngine` (MT-1) - no second translator, no
caching, no change to decoding. Every `update()` with new stable source
words re-translates the FULL current stable prefix (the re-translation
baseline of the simultaneous-MT literature; not claimed as novel).

Translation debt is recorded as an OBSERVATION only:

    translation_debt = stable_source_words - source_words_already_processed_by_MT

measured at each update before the invocation. It never suppresses an
invocation here; that is MT-3's job (not implemented).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from translation.translator import TranslationEngine, TranslationRequest


@dataclass(frozen=True)
class MTInvocation:
    """One MT call on the full current stable source prefix."""

    sample_id: str
    invocation_id: int
    utterance_id: int | None
    source_available_time: float | None
    source_text: str
    source_word_count: int
    newly_stabilized_words: int
    source_words_already_processed: int
    translation_debt: int
    target_text: str
    target_token_count: int
    inference_ms: float
    wall_ms: float
    cpu_ms: float
    cumulative_inference_ms: float
    cumulative_wall_ms: float
    is_final: bool
    # time.perf_counter() at MT call start/completion (same-process clock;
    # needed for later paced 1x replay). Completion = Hindi emission time.
    mt_started_at: float | None = None
    mt_completed_at: float | None = None
    # Stream time at which the scheduler decided to call MT (update, deadline
    # timer or finalization). None = same as source_available_time (MT3-A/B).
    trigger_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FinalizeResult:
    sample_id: str
    final_source_text: str
    final_target_text: str
    invoked_mt: bool
    invocation: MTInvocation | None


@dataclass
class _SampleState:
    sample_id: str
    last_source_words: list[str] = field(default_factory=list)
    last_target_text: str = ""
    invocations: list[MTInvocation] = field(default_factory=list)
    cumulative_inference_ms: float = 0.0
    cumulative_wall_ms: float = 0.0
    finalized: bool = False


class IncrementalTranslator:
    """Eager re-translation over one sample's stable-prefix stream.

    Contract: stable source prefixes are monotonic - each new prefix's
    words must start with the previous prefix's words (M2's committed
    text is append-only). Violations raise instead of being repaired.
    """

    def __init__(self, engine: TranslationEngine) -> None:
        self.engine = engine
        self._state: _SampleState | None = None

    # ------------------------------------------------------------------

    def reset(self, sample_id: str) -> None:
        """Start a new independent sample; no state carries over."""

        self._state = _SampleState(sample_id=sample_id)

    @property
    def invocations(self) -> list[MTInvocation]:
        return list(self._state.invocations) if self._state else []

    def _require_state(self) -> _SampleState:
        if self._state is None:
            raise RuntimeError("call reset(sample_id) before update()/finalize()")
        if self._state.finalized:
            raise RuntimeError(f"sample {self._state.sample_id!r} already finalized; call reset()")
        return self._state

    def _invoke(self, state: _SampleState, words: list[str], *, utterance_id: int | None,
                source_available_time: float | None, is_final: bool,
                trigger_time: float | None = None) -> MTInvocation:
        source_text = " ".join(words)
        processed_before = len(state.last_source_words)

        cpu_start = time.process_time()
        response = self.engine.translate_request(
            TranslationRequest(
                source_text=source_text,
                timestamp=source_available_time,
                is_final=is_final,
                stable_source_word_count=len(words),
                newly_stabilized_words=len(words) - processed_before,
            )
        )
        cpu_ms = (time.process_time() - cpu_start) * 1000.0

        state.cumulative_inference_ms += response.inference_time_ms
        state.cumulative_wall_ms += response.wall_time_ms

        invocation = MTInvocation(
            sample_id=state.sample_id,
            invocation_id=len(state.invocations) + 1,
            utterance_id=utterance_id,
            source_available_time=source_available_time,
            source_text=source_text,
            source_word_count=len(words),
            newly_stabilized_words=len(words) - processed_before,
            source_words_already_processed=processed_before,
            translation_debt=len(words) - processed_before,
            target_text=response.target_text,
            target_token_count=response.target_token_count,
            inference_ms=response.inference_time_ms,
            wall_ms=response.wall_time_ms,
            cpu_ms=cpu_ms,
            cumulative_inference_ms=state.cumulative_inference_ms,
            cumulative_wall_ms=state.cumulative_wall_ms,
            is_final=is_final,
            mt_started_at=response.started_at,
            mt_completed_at=response.completed_at,
            trigger_time=trigger_time,
        )
        state.invocations.append(invocation)
        state.last_source_words = list(words)
        state.last_target_text = response.target_text
        return invocation

    # ------------------------------------------------------------------

    def update(self, stable_source_text: str, *, utterance_id: int | None = None,
               source_available_time: float | None = None,
               trigger_time: float | None = None) -> MTInvocation | None:
        """Re-translate the full stable prefix if it gained new words.

        Returns None (no MT call) when the prefix has no new words.
        """

        state = self._require_state()
        words = stable_source_text.split()
        previous = state.last_source_words

        if words[: len(previous)] != previous:
            raise ValueError(
                f"non-monotonic stable source for {state.sample_id!r}: "
                f"{' '.join(previous)!r} -> {stable_source_text!r}"
            )
        if len(words) == len(previous):
            return None

        return self._invoke(state, words, utterance_id=utterance_id,
                            source_available_time=source_available_time, is_final=False,
                            trigger_time=trigger_time)

    def finalize(self, final_source_text: str, *, utterance_id: int | None = None,
                 source_available_time: float | None = None,
                 trigger_time: float | None = None) -> FinalizeResult:
        """Close the sample. Translates only if the final source has words
        not yet translated; otherwise the last hypothesis is final (marked
        on its record) and no extra MT call is made."""

        state = self._require_state()
        words = final_source_text.split()

        if words[: len(state.last_source_words)] != state.last_source_words:
            raise ValueError(f"non-monotonic final source for {state.sample_id!r}")

        invocation = None
        if len(words) > len(state.last_source_words):
            invocation = self._invoke(state, words, utterance_id=utterance_id,
                                      source_available_time=source_available_time, is_final=True,
                                      trigger_time=trigger_time)
        elif state.invocations:
            last = state.invocations[-1]
            state.invocations[-1] = MTInvocation(**{**last.to_dict(), "is_final": True})

        state.finalized = True
        return FinalizeResult(
            sample_id=state.sample_id,
            final_source_text=" ".join(words),
            final_target_text=state.last_target_text,
            invoked_mt=invocation is not None,
            invocation=invocation,
        )
