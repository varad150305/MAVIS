"""Product live pipeline: frozen M2 stable-prefix events -> MT invocation
policy -> single FIFO MT worker -> Hindi display.

Architecture (one process, four threads):

  main thread      : the UNMODIFIED main.run_m2() loop (VAD, StreamingBuffer,
                     ASR worker, PrefixAgreementTracker). Its tracker is
                     observed, not changed: process()/finalize_utterance()
                     are wrapped so that every call that produced commits
                     emits one stable-update event - the same hook the
                     validated e2e harness (evaluation/e2e_mt3c.py) uses.
  ASR worker       : existing asr.worker.ASRWorker (started by run_m2).
  controller       : owns the scheduler (translation.scheduling, unchanged);
                     handles stable updates, fires deadline timers on time,
                     opens/closes translation segments. Never blocked by MT.
  MT worker        : one FIFO worker; every scheduled call runs once, in
                     order (frozen MT3-C execution model).

Product-specific behaviour, not part of the research specification:

  * Translation segments. The research scheduler re-translates the whole
    stable prefix of one evaluation sample. A live session has no sample
    boundary, so the controller closes a segment (one scheduler sample) at
    an utterance end once it holds enough words, after an idle gap, or at a
    hard word cap - otherwise every MT call would re-translate the entire
    session so far. Within a segment the policy is exactly the frozen one.
  * Display: the active segment's Hindi is a draft that may be revised by
    the next call; a segment's Hindi is final once the segment is closed and
    its last scheduled call has completed - or, with a final-quality
    translator configured (product/final_mt.py), once that translator has
    translated the closed segment's complete source. Drafts that complete
    after a segment's final are recorded but not shown.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, TextIO
from unittest.mock import patch

from translation.scheduling import (
    AdaptiveDeadlinePolicy,
    EagerPolicy,
    FixedThresholdPolicy,
    SourceUpdate,
    StreamingMTScheduler,
)
from translation.translator import TranslationRequest


# =======================================================================
# Policies
# =======================================================================


def load_mt3c_parameters(path: str | Path, expected_sha256: str | None) -> tuple[dict[str, Any], str]:
    """Read the frozen MT3-C parameters (read-only) and verify their hash."""

    import hashlib

    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(f"{path} SHA-256 {digest} does not match the expected frozen value {expected_sha256}")
    return json.loads(raw.decode("utf-8")), digest


def make_policy_factory(name: str, *, fixed_k: int = 8, mt3c_params: dict[str, Any] | None = None,
                        mt3c_sha256: str | None = None) -> Callable[[], Any]:
    if name == "eager":
        return EagerPolicy
    if name == "fixed":
        return lambda: FixedThresholdPolicy(fixed_k)
    if name == "mt3c":
        if mt3c_params is None:
            raise ValueError("mt3c policy needs the frozen parameters")
        return lambda: AdaptiveDeadlinePolicy.from_frozen(mt3c_params, mt3c_sha256)
    raise ValueError(f"unknown policy {name!r} (expected mt3c, eager or fixed)")


# =======================================================================
# MT execution
# =======================================================================


@dataclass
class ScheduledCall:
    """One MT call as scheduled (decision-time state)."""

    segment_id: int
    invocation_id: int
    utterance_id: int | None
    source_text: str
    source_word_count: int
    newly_stabilized_words: int
    source_words_already_processed: int
    is_final: bool
    trigger_time: float | None
    source_available_time: float | None
    enqueued_s: float


@dataclass
class CallResult:
    segment_id: int
    invocation_id: int
    source_word_count: int
    target_text: str
    inference_ms: float | None
    started_s: float
    completed_s: float
    queue_wait_s: float
    error: str | None = None


class MTWorker(threading.Thread):
    """Single MT worker; unbounded FIFO; every call executed exactly once.

    `torch_threads` is applied inside this thread: torch's intra-op thread
    count is per calling thread here, and silero_vad sets it to 1 at import
    for the main (VAD) thread. Setting 8 process-wide instead makes every
    32 ms VAD inference wake 8 spinning OpenMP workers (measured ~1.5 extra
    cores while idle), while MT on 1 thread is ~35% slower. The warm-up runs
    in this thread too, so the first real call does not pay pool start-up.
    """

    def __init__(self, engine: Any, clock: Callable[[], float], on_result: Callable[[ScheduledCall, CallResult], None],
                 torch_threads: int | None = None, warm_up: bool = False) -> None:
        super().__init__(daemon=True, name="product-mt-worker")
        self.engine = engine
        self.clock = clock
        self.on_result = on_result
        self.torch_threads = torch_threads
        self.warm_up = warm_up
        self.ready = threading.Event()
        self.warm_up_ms: float | None = None
        self.thread_torch_threads: int | None = None
        self.jobs: queue.Queue[ScheduledCall | None] = queue.Queue()
        self.max_queue_depth = 0

    def submit(self, call: ScheduledCall) -> None:
        self.jobs.put(call)
        self.max_queue_depth = max(self.max_queue_depth, self.jobs.qsize())

    def run(self) -> None:
        try:
            if self.torch_threads:
                import torch

                torch.set_num_threads(self.torch_threads)
                self.thread_torch_threads = torch.get_num_threads()
            if self.warm_up and hasattr(self.engine, "warm_up"):
                start = time.perf_counter()
                self.engine.warm_up()
                self.warm_up_ms = (time.perf_counter() - start) * 1000.0
        finally:
            self.ready.set()
        while True:
            call = self.jobs.get()
            try:
                if call is None:
                    return
                started = self.clock()
                try:
                    response = self.engine.translate_request(TranslationRequest(
                        source_text=call.source_text, timestamp=call.source_available_time, is_final=call.is_final,
                        stable_source_word_count=call.source_word_count,
                        newly_stabilized_words=call.newly_stabilized_words))
                    result = CallResult(call.segment_id, call.invocation_id, call.source_word_count, response.target_text,
                                        response.inference_time_ms, started, self.clock(), started - call.enqueued_s)
                except Exception as exc:  # recorded and displayed, never silently dropped
                    result = CallResult(call.segment_id, call.invocation_id, call.source_word_count, "", None,
                                        started, self.clock(), started - call.enqueued_s, error=repr(exc))
                try:
                    self.on_result(call, result)
                except Exception:  # a display failure must not kill the worker
                    pass
            finally:
                self.jobs.task_done()

    def shutdown(self, timeout_s: float = 120.0) -> bool:
        """Finish every queued call, then stop. Returns False on timeout."""

        deadline = time.perf_counter() + timeout_s
        while self.jobs.unfinished_tasks and time.perf_counter() < deadline:
            time.sleep(0.02)
        drained = not self.jobs.unfinished_tasks
        self.jobs.put(None)
        self.join(timeout=5)
        return drained


class AsyncSegmentTranslator:
    """The translator interface StreamingMTScheduler expects, but a call is
    SCHEDULED on the MT worker instead of executed synchronously (mirrors
    evaluation/e2e_mt3c.AsyncFIFOTranslator: same monotonicity checks and
    finalization semantics as translation.incremental.IncrementalTranslator)."""

    def __init__(self, worker: MTWorker, clock: Callable[[], float]) -> None:
        self.worker = worker
        self.clock = clock
        self.segment_id: int | None = None
        self._calls: list[ScheduledCall] = []
        self._last_words: list[str] = []
        self.finalized = False

    def reset(self, sample_id: str) -> None:
        self.segment_id = int(sample_id)
        self._calls, self._last_words, self.finalized = [], [], False

    @property
    def invocations(self) -> list[ScheduledCall]:
        return list(self._calls)

    def _schedule(self, words: list[str], *, utterance_id, source_available_time, trigger_time, is_final) -> ScheduledCall:
        call = ScheduledCall(self.segment_id, len(self._calls) + 1, utterance_id, " ".join(words), len(words),
                             len(words) - len(self._last_words), len(self._last_words), is_final, trigger_time,
                             source_available_time, self.clock())
        self._calls.append(call)
        self._last_words = list(words)
        self.worker.submit(call)
        return call

    def update(self, stable_source_text: str, *, utterance_id=None, source_available_time=None, trigger_time=None):
        if self.finalized:
            raise RuntimeError("segment already finalized")
        words = stable_source_text.split()
        if words[: len(self._last_words)] != self._last_words:
            raise ValueError("non-monotonic stable source")
        if len(words) == len(self._last_words):
            return None
        return self._schedule(words, utterance_id=utterance_id, source_available_time=source_available_time,
                              trigger_time=trigger_time, is_final=False)

    def finalize(self, final_source_text: str, *, utterance_id=None, source_available_time=None, trigger_time=None):
        from translation.incremental import FinalizeResult

        words = final_source_text.split()
        if words[: len(self._last_words)] != self._last_words:
            raise ValueError("non-monotonic final source")
        call = None
        if len(words) > len(self._last_words):
            call = self._schedule(words, utterance_id=utterance_id, source_available_time=source_available_time,
                                  trigger_time=trigger_time, is_final=True)
        self.finalized = True
        return FinalizeResult(str(self.segment_id), " ".join(words), "", call is not None, call)


class LiveScheduler(StreamingMTScheduler):
    """Adds `advance_to(t)`: fire due deadline timers at time t without an
    ASR update (same as the e2e harness's evaluation-only subclass)."""

    def advance_to(self, t: float) -> None:
        if self._log is not None:
            self._fire_timers(t, inclusive=True)


# =======================================================================
# Segments and display
# =======================================================================


@dataclass
class Segment:
    segment_id: int
    opened_s: float
    rate_origin_s: float
    words: list[str] = field(default_factory=list)
    word_times: list[float] = field(default_factory=list)
    utterance_ids: list[int] = field(default_factory=list)
    updates: int = 0
    last_update_s: float | None = None
    last_event_final: bool = False
    closed_s: float | None = None
    close_reason: str | None = None
    calls: list[ScheduledCall] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    final_call: ScheduledCall | None = None
    last_event_sentence_end: bool = False
    #: pending close of a segment at >= close_after_words (long_close_needs_pause)
    long_close_pending: bool = False


#: Printed once when listening starts (phase 4 audit: the markers were
#: documented only in the README, not on screen).
CONSOLE_LEGEND = (
    "How to read:  EN #n  | English of segment n (confirmed words; '.' = speaker paused)\n"
    "              HI #n ~| provisional Hindi draft (may change)\n"
    "              HI #n =| FINAL Hindi for segment n (shown once, after the segment closes)\n"
)


class Display:
    """Thread-safe console + JSONL transcript writer.

    The console shows the stable English of the active segment as it grows,
    each Hindi draft, and the final Hindi once a segment is closed and its
    last call completed. Everything is also appended to transcript.jsonl.
    """

    def __init__(self, console: TextIO | None, transcript_path: Path | None, clock: Callable[[], float],
                 sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.console = console
        #: structured-event consumer (product.session); receives a copy of
        #: every record, never console text. Exceptions in it are swallowed.
        self.sink = sink
        self.transcript_path = transcript_path
        self.clock = clock
        self._lock = threading.Lock()
        self._results: dict[int, dict[int, CallResult]] = {}
        self._closed: dict[int, int | None] = {}
        self._closed_s: dict[int, float] = {}
        self._final_shown: set[int] = set()
        self.finals: dict[int, str] = {}
        #: how each final was produced: "final_mt", "draft" or "draft_fallback"
        self.final_kinds: dict[int, str] = {}
        self._fq_pending: set[int] = set()
        self._fq_results: dict[int, CallResult] = {}
        #: every record written, with `render_s` = session clock right after
        #: the console write returned (phase 4.6; the terminal's own paint
        #: time is not observable). Event times (t_s, completed_s) are when
        #: the pipeline produced the output, render_s when it was printed.
        self.records: list[dict[str, Any]] = []
        if transcript_path is not None:
            # (phase 6) exists from the start, also for a session with no output
            transcript_path.touch(exist_ok=True)

    def _write(self, line: str, record: dict[str, Any]) -> None:
        if self.console is not None:
            try:
                self.console.write(line + "\n")
                self.console.flush()
            except Exception:
                pass
        record["render_s"] = self.clock()
        self._store(record)

    def _store(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if self.transcript_path is not None:
            with self.transcript_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.sink is not None:
            try:
                self.sink(dict(record))
            except Exception:
                pass

    def event(self, record: dict[str, Any]) -> None:
        """A structured record with no console line (ASR decode statistics,
        utterance timing): transcript + sink only."""

        with self._lock:
            self._store(record)

    def audio_health(self, record: dict[str, Any]) -> None:
        """audio_stall / audio_resumed / cleanup_error from the audio source
        (phase 6): one console warning line + transcript + sink."""

        kind, t = record.get("type"), record.get("t_s") or self.clock()
        if kind == "audio_stall":
            line = (f"[{t:7.1f}s] !! NO AUDIO from the microphone for {record.get('no_frames_for_s', 0):.1f} s "
                    f"- check the device (session still running)")
        elif kind == "audio_resumed":
            line = f"[{t:7.1f}s] !! audio resumed after {record.get('stalled_for_s', 0):.1f} s without frames"
        else:
            line = f"[{t:7.1f}s] !! audio cleanup failed ({record.get('op')}): {record.get('error')}"
        with self._lock:
            self._write(line, {k: v for k, v in record.items() if k != "traceback"})

    def status(self, message: str) -> None:
        with self._lock:
            self._write(f"[{self.clock():7.1f}s] {message}", {"type": "status", "t_s": self.clock(), "message": message})

    def english(self, segment: Segment, new_words: list[str], t: float, utterance_final: bool,
                utterance_id: int | None = None) -> None:
        with self._lock:
            mark = "." if utterance_final else " "
            self._write(f"[{t:7.1f}s] EN #{segment.segment_id:<3}{mark}| {' '.join(segment.words)}",
                        {"type": "en_update", "segment_id": segment.segment_id, "t_s": t, "new_text": " ".join(new_words),
                         "stable_text": " ".join(segment.words), "utterance_final": utterance_final,
                         "utterance_id": utterance_id})

    def provisional(self, utterance_id: int, text: str, t: float) -> None:
        """Display-only provisional English (product option
        `display_provisional_en`): the latest non-final Whisper hypothesis
        of the utterance, unconfirmed. Never translated, never committed."""

        with self._lock:
            self._write(f"[{t:7.1f}s] EN  ?  ?| {text}   (provisional, may change)",
                        {"type": "en_provisional", "utterance_id": utterance_id, "t_s": t, "text": text})

    def translation(self, call: ScheduledCall, result: CallResult) -> None:
        with self._lock:
            self._results.setdefault(call.segment_id, {})[call.invocation_id] = result
            if call.segment_id in self._final_shown:
                # The segment's final is already on screen (final-quality MT
                # finished first): never show an older-quality draft after it.
                # Recorded (transcript + sink -> hi_draft_late), never printed.
                self._store({"type": "hi_draft_after_final", "source_text": call.source_text, **asdict(result)})
                return
            if result.error:
                self._write(f"[{result.completed_s:7.1f}s] HI #{call.segment_id:<3}!| MT ERROR: {result.error}",
                            {"type": "mt_error", **asdict(result)})
            else:
                self._write(f"[{result.completed_s:7.1f}s] HI #{call.segment_id:<3}~| {result.target_text}",
                            {"type": "hi_draft", "source_text": call.source_text, **asdict(result)})
            self._maybe_final(call.segment_id)

    def segment_closed(self, segment: Segment, last_invocation_id: int | None,
                       final_quality_pending: bool = False) -> None:
        with self._lock:
            if final_quality_pending:
                self._fq_pending.add(segment.segment_id)
            self._closed[segment.segment_id] = last_invocation_id
            self._closed_s[segment.segment_id] = segment.closed_s or 0.0
            self._write(f"[{segment.closed_s:7.1f}s] -- segment #{segment.segment_id} closed ({segment.close_reason}, "
                        f"{len(segment.words)} words, {len(segment.calls)} MT calls)",
                        {"type": "segment_closed", "segment_id": segment.segment_id, "t_s": segment.closed_s,
                         "reason": segment.close_reason, "words": len(segment.words), "mt_calls": len(segment.calls),
                         "final_source": " ".join(segment.words), "utterance_ids": list(segment.utterance_ids)})
            self._maybe_final(segment.segment_id)

    def final_translation(self, call: ScheduledCall, result: CallResult) -> None:
        """Result of the final-quality translator for a closed segment."""

        with self._lock:
            self._fq_results[call.segment_id] = result
            if result.error:
                # no console line (the segment falls back to its draft, shown as final)
                self._store({"type": "final_mt_error", **asdict(result)})
            self._maybe_final(call.segment_id)

    def final_quality_result(self, segment_id: int) -> CallResult | None:
        with self._lock:
            return self._fq_results.get(segment_id)

    def _maybe_final(self, segment_id: int) -> None:
        if segment_id not in self._closed or segment_id in self._final_shown:
            return
        if segment_id in self._fq_pending:
            fq = self._fq_results.get(segment_id)
            if fq is None:
                return  # final-quality translation still running
            if fq.error is None:
                self._final_shown.add(segment_id)
                t = max(fq.completed_s, self._closed_s.get(segment_id, 0.0))
                self.finals[segment_id] = fq.target_text
                self.final_kinds[segment_id] = "final_mt"
                self._write(f"[{t:7.1f}s] HI #{segment_id:<3}=| {fq.target_text}",
                            {"type": "hi_final", "segment_id": segment_id, "t_s": t, "target_text": fq.target_text,
                             "invocation_id": None, "fallback_invocation_id": None, "kind": "final_mt",
                             "inference_ms": fq.inference_ms})
                return
            # final-quality call failed: fall back to the draft-based final below
        last_id = self._closed[segment_id]
        results = self._results.get(segment_id, {})
        if last_id is not None and last_id not in results:
            return  # last scheduled call still running
        final = results[last_id] if last_id is not None else None
        self._final_shown.add(segment_id)
        closed_s = self._closed_s.get(segment_id, 0.0)
        t = max(final.completed_s, closed_s) if final is not None else max(self.clock(), closed_s)
        fallback = None
        if final is not None and final.error:
            # Last call failed: keep the latest successful draft rather than
            # blanking the segment; flagged in the transcript.
            ok = [r for r in results.values() if not r.error]
            fallback = max(ok, key=lambda r: r.invocation_id) if ok else None
            final = fallback
        text = final.target_text if final is not None else ""
        self.finals[segment_id] = text
        kind = "draft_fallback" if fallback is not None or segment_id in self._fq_pending else "draft"
        self.final_kinds[segment_id] = kind
        note = "  (last MT call failed; showing latest successful draft)" if fallback is not None else ""
        if segment_id in self._fq_pending:
            note += "  (final-quality MT failed; showing draft MT)"
        self._write(f"[{t:7.1f}s] HI #{segment_id:<3}=| {text}{note}",
                    {"type": "hi_final", "segment_id": segment_id, "t_s": t, "target_text": text,
                     "invocation_id": last_id, "fallback_invocation_id": fallback.invocation_id if fallback else None,
                     "kind": kind})

    def result_for(self, segment_id: int, invocation_id: int) -> CallResult | None:
        with self._lock:
            return self._results.get(segment_id, {}).get(invocation_id)


# =======================================================================
# Controller
# =======================================================================


@dataclass(frozen=True)
class SegmentRules:
    close_after_words: int = 20
    idle_close_s: float = 2.5
    hard_max_words: int = 60
    #: Sentence-aware close (product, 2026-10-28 phase 3; None = off): an
    #: utterance whose final decode ends in . ? or ! closes the segment once
    #: it holds >= sentence_min_words and the speaker has been silent for
    #: sentence_gap_s (instead of idle_close_s).
    sentence_gap_s: float | None = None
    sentence_min_words: int = 3
    #: (phase 9; needs sentence_gap_s) a segment that reached close_after_words
    #: closes at an utterance end only after the same VAD-measured pause as a
    #: sentence end; if the speaker continues at once (the VAD split inside a
    #: sentence) it stays open, up to hard_max_words. Live 2026-10-29 AC:
    #: "... recognize the boundary" | "without cutting off the previous
    #: sentence" were closed and translated as two fragments.
    long_close_needs_pause: bool = False
    #: with long_close_needs_pause: from this many words on, close at the next
    #: utterance end without waiting for a pause (None = no limit). Reading
    #: pauses between sentences (0.2-1.6 s, phase 9 replays) overlap VAD
    #: splits inside a sentence (0.1-0.7 s), so without a limit sentences
    #: chain into 45+-word segments whose final Hindi comes ~20 s after
    #: their first word.
    long_close_max_words: int | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.close_after_words <= self.hard_max_words:
            raise ValueError("require 1 <= close_after_words <= hard_max_words")
        if self.idle_close_s <= 0:
            raise ValueError("idle_close_s must be > 0")
        if self.sentence_gap_s is not None and not 0 < self.sentence_gap_s <= self.idle_close_s:
            raise ValueError("require 0 < sentence_gap_s <= idle_close_s")


class SegmentController(threading.Thread):
    """Owns the scheduler; turns stable-update events into segments and MT calls."""

    def __init__(self, policy_factory: Callable[[], Any], translator: AsyncSegmentTranslator, clock: Callable[[], float],
                 rules: SegmentRules, display: Display, final_worker: "MTWorker | None" = None) -> None:
        super().__init__(daemon=True, name="product-segment-controller")
        #: optional final-quality translator: one call per closed segment, full source
        self.final_worker = final_worker
        self.scheduler = LiveScheduler(translator, policy_factory())
        self.translator = translator
        self.clock = clock
        self.rules = rules
        self.display = display
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.segments: list[Segment] = []
        self.active: Segment | None = None
        self.error: str | None = None
        self.timer_lateness_s: list[float] = []
        # VAD speech activity (observed). None = never reported (e.g. tests
        # that feed updates only): idle time then counts from the last update.
        self.speech_active: bool | None = None
        self.speech_ended_s: float | None = None
        self.speech_started_s: float | None = None

    # -- public API (thread-safe: events only) ---------------------------

    def submit_update(self, *, utterance_id: int, new_text: str, utterance_final: bool, t: float,
                      onset_s: float | None, sentence_end: bool = False, newer_utterances: int = 0) -> None:
        """`newer_utterances`: utterances the VAD has already started after
        this one (the speaker resumed before this commit arrived)."""

        self.events.put({"kind": "update", "utterance_id": utterance_id, "new_text": new_text,
                         "utterance_final": utterance_final, "t": t, "onset_s": onset_s,
                         "sentence_end": sentence_end, "newer_utterances": newer_utterances})

    def submit_speech(self, *, active: bool, t: float) -> None:
        """VAD speech start (active=True) or end (active=False) at stream time t."""

        self.events.put({"kind": "speech", "active": active, "t": t})

    def stop(self, t: float, timeout_s: float = 30.0) -> None:
        self.events.put({"kind": "stop", "t": t})
        self.join(timeout=timeout_s)

    # -- thread ----------------------------------------------------------

    def run(self) -> None:
        try:
            while True:
                timeout = self._next_timeout()
                try:
                    event = self.events.get(timeout=timeout)
                except queue.Empty:
                    self._tick(self.clock())
                    continue
                if event["kind"] == "update":
                    # Timers due strictly before the update fire inside
                    # scheduler.on_update(); only an overdue idle close
                    # must be applied first.
                    self._tick(event["t"], fire_timers=False)
                    self._on_update(event)
                elif event["kind"] == "speech":
                    self._tick(event["t"], fire_timers=False)
                    self.speech_active = event["active"]
                    if not event["active"]:
                        self.speech_ended_s = event["t"]
                    else:
                        self.speech_started_s = event["t"]
                        if self.active is not None:
                            # speech resumed before the sentence pause elapsed:
                            # a hesitation, not a sentence end
                            self.active.last_event_sentence_end = False
                            self.active.long_close_pending = False
                elif event["kind"] == "stop":
                    self._close(event["t"], "session end")
                    return
        except Exception as exc:  # surfaced in the session summary and on screen
            self.error = repr(exc)
            self.display.status(f"CONTROLLER ERROR: {self.error}")

    def _idle_deadline(self) -> float | None:
        """Close an idle segment once the speaker has been silent for
        idle_close_s after an utterance-final commit. Silence is measured from
        the VAD speech end, not from the last commit: the next utterance's
        first commit arrives seconds after its speech starts (min new audio +
        decode), so commit gaps overstate pauses."""

        seg = self.active
        if seg is None or not seg.last_event_final or seg.last_update_s is None:
            return None
        wait = self._sentence_gap(seg) or self.rules.idle_close_s
        if self.speech_active is None:
            return seg.last_update_s + wait
        if self.speech_active or self.speech_ended_s is None:
            return None
        return max(self.speech_ended_s + wait, seg.last_update_s)

    def _sentence_gap(self, seg: Segment) -> float | None:
        """sentence_gap_s when the segment ends at a sentence boundary, else None."""

        if self.rules.sentence_gap_s is None:
            return None
        if seg.long_close_pending:
            return self.rules.sentence_gap_s
        if not seg.last_event_sentence_end or len(seg.words) < self.rules.sentence_min_words:
            return None
        return self.rules.sentence_gap_s

    def _pause_reason(self, seg: Segment, gap: float) -> str:
        if seg.long_close_pending and not seg.last_event_sentence_end:
            return f"utterance end at >= {self.rules.close_after_words} words + {gap:g} s pause"
        return f"sentence end + {gap:g} s pause"

    def _next_timeout(self) -> float | None:
        candidates = [t for t in (self.scheduler.pending_timer if self.active else None, self._idle_deadline())
                      if t is not None]
        return max(0.0, min(candidates) - self.clock()) if candidates else None

    def _tick(self, now: float, fire_timers: bool = True) -> None:
        """Fire deadline timers due by `now` and close an idle segment."""

        if self.active is None:
            return
        timer = self.scheduler.pending_timer
        if fire_timers and timer is not None and timer <= now:
            n_before = len(self.scheduler.decisions)
            self.scheduler.advance_to(now)
            for d in self.scheduler.decisions[n_before:]:
                self.timer_lateness_s.append(self.clock() - d.state.now)
            self._sync_calls()
        idle = self._idle_deadline()
        if idle is not None and idle <= now:
            gap = self._sentence_gap(self.active)
            self._close(idle, self._pause_reason(self.active, gap) if gap else f"idle {self.rules.idle_close_s:g} s")

    def _on_update(self, event: dict[str, Any]) -> None:
        t = event["t"]
        new_words = event["new_text"].split()
        seg = self.active
        if seg is not None and len(seg.words) >= self.rules.hard_max_words:
            self._close(t, f"hard cap {self.rules.hard_max_words} words")
            seg = None
        if seg is None:
            if not new_words:
                return  # an empty finalization marker never opens a segment
            onset = event["onset_s"]
            rate_origin = min(onset, t - 1e-6) if onset is not None else max(0.0, t - 2.0)
            seg = Segment(segment_id=len(self.segments) + 1, opened_s=t, rate_origin_s=rate_origin)
            self.segments.append(seg)
            self.active = seg
            self.scheduler.start(str(seg.segment_id), rate_origin=rate_origin)

        seg.words += new_words
        seg.word_times += [t] * len(new_words)
        if event["utterance_id"] not in seg.utterance_ids:
            seg.utterance_ids.append(event["utterance_id"])
        seg.updates += 1
        seg.last_update_s = t
        seg.last_event_final = event["utterance_final"]
        seg.last_event_sentence_end = bool(event["utterance_final"] and event.get("sentence_end"))
        seg.long_close_pending = False
        if new_words:
            self.display.english(seg, new_words, t, event["utterance_final"], utterance_id=event["utterance_id"])
        self.scheduler.on_update(SourceUpdate(
            sample_id=str(seg.segment_id), update_id=seg.updates, stable_source_text=" ".join(seg.words),
            utterance_id=event["utterance_id"], source_available_time=t))
        self._sync_calls()
        if event["utterance_final"] and len(seg.words) >= self.rules.close_after_words:
            limit = self.rules.long_close_max_words
            if not (self.rules.long_close_needs_pause and self.rules.sentence_gap_s is not None) or \
                    (limit is not None and len(seg.words) >= limit):
                self._close(t, f"utterance end at >= {self.rules.close_after_words} words")
                return
            seg.long_close_pending = True      # decided by the pause below / the idle timer
        gap = self._sentence_gap(seg)
        if gap is None:
            return
        newer = event.get("newer_utterances", 0)
        resumed_now = (self.speech_active and self.speech_ended_s is not None and self.speech_started_s is not None
                       and self.speech_started_s > self.speech_ended_s)
        if newer or resumed_now:
            # The speaker resumed before this commit arrived (decodes lag
            # speech by ~1 s). If exactly the next utterance is under way,
            # the pause is known: a long one ends the sentence now, a short
            # one is a hesitation. Otherwise the pause is unknown: keep the
            # segment open under the ordinary rules.
            if newer <= 1 and resumed_now and self.speech_started_s - self.speech_ended_s >= gap:
                self._close(t, self._pause_reason(seg, gap))
            else:
                seg.last_event_sentence_end = False
                seg.long_close_pending = False

    def _sync_calls(self) -> None:
        if self.active is not None:
            self.active.calls = self.translator.invocations

    def _close(self, t: float, reason: str) -> None:
        seg = self.active
        if seg is None:
            return
        t_fin = max(t, seg.last_update_s or t)
        self.scheduler.finalize(" ".join(seg.words), source_available_time=seg.last_update_s, finalization_time=t_fin)
        seg.calls = self.translator.invocations
        seg.decisions = [d.to_dict() for d in self.scheduler.decisions]
        seg.closed_s = t_fin
        seg.close_reason = reason
        self.active = None
        fq = self.final_worker is not None and bool(seg.words)
        # mark the final-quality call pending BEFORE submitting it, so its
        # result can never arrive ahead of the close
        self.display.segment_closed(seg, seg.calls[-1].invocation_id if seg.calls else None, final_quality_pending=fq)
        if fq:
            seg.final_call = ScheduledCall(seg.segment_id, 0, seg.utterance_ids[-1] if seg.utterance_ids else None,
                                           " ".join(seg.words), len(seg.words), 0, 0, True, t_fin,
                                           seg.last_update_s, self.clock())
            self.final_worker.submit(seg.final_call)


# =======================================================================
# run_m2 integration
# =======================================================================


def collapse_repeats(words: list[str], max_ngram: int = 3, min_run: int = 4) -> list[str]:
    """Collapse a run of >= `min_run` consecutive copies of the same 1..max_ngram
    word sequence to one copy (Whisper repetition loops, e.g. "s s s s ...")."""

    out = list(words)
    for n in range(1, max_ngram + 1):
        i, result = 0, []
        while i < len(out):
            gram = out[i:i + n]
            run = 1
            while len(gram) == n and out[i + run * n:i + (run + 1) * n] == gram:
                run += 1
            if len(gram) == n and run >= min_run:
                result += gram
                i += run * n
            else:
                result.append(out[i])
                i += 1
        out = result
    return out


class RepetitionGuard:
    """Product-only filter on the committed English stream (audit R4).

    Whisper tiny can fall into a repetition loop on real speech (controlled
    corpus run 2026-10-28: one utterance committed "s" 112 times). The guard
    keeps the last words emitted per utterance and drops each new word that
    would extend a run of >= `min_run` copies of the same 1..max_ngram
    sequence, so a loop reaches the display and MT at most once. It never
    changes words already emitted and never touches the tracker.
    """

    def __init__(self, max_ngram: int = 3, min_run: int = 4) -> None:
        self.max_ngram, self.min_run = max_ngram, min_run
        self._tails: dict[int, list[str]] = {}
        self.dropped = 0

    def filter(self, utterance_id: int, words: list[str]) -> list[str]:
        tail = self._tails.setdefault(utterance_id, [])
        kept = []
        for w in words:
            candidate = tail + [w]
            if collapse_repeats(candidate[-(self.max_ngram * (self.min_run + 1)):], self.max_ngram, self.min_run) \
                    != candidate[-(self.max_ngram * (self.min_run + 1)):]:
                self.dropped += 1
                continue
            tail.append(w)
            kept.append(w)
        del tail[:-self.max_ngram * (self.min_run + 1)]
        return kept


@dataclass(frozen=True)
class BoundaryRepair:
    """Product-only repair of LocalAgreement commit boundaries (audit D7).

    The frozen tracker keeps only words whose measured start lies at or
    after the confirmed boundary. After the buffer is trimmed at that
    boundary, Whisper often places the first word of the next decode a few
    tens of milliseconds BEFORE it, and the word is silently lost
    (live session 2026-10-27: "while", "people", "without"). Whisper-
    Streaming's reference (HypothesisBuffer.insert) avoids this by
    accepting words that start slightly before the last commit and then
    removing a leading 1..5-gram that repeats the committed tail.

    `apply()` does the same on a COPY of the hypothesis before the
    unchanged tracker sees it: words starting within `tolerance_s` before
    the boundary are moved onto it, and a leading n-gram equal to the last
    n confirmed words is dropped. The tracker's code is not modified.
    """

    tolerance_s: float = 0.3
    max_ngram: int = 5

    def apply(self, tracker: Any, hypothesis: Any) -> tuple[Any, dict[str, Any] | None]:
        import dataclasses

        from streaming.prefix_agreement import normalize_text

        uid = hypothesis.utterance_id
        boundary = tracker.confirmed_boundary_time(uid)
        if boundary is None or not hypothesis.success or not hypothesis.words:
            return hypothesis, None
        base = hypothesis.audio_start_time
        kept, rescued = [], []
        for w in hypothesis.words:
            start = base + w.start
            if start < boundary - self.tolerance_s:
                kept.append(w)  # clearly inside confirmed audio: the tracker drops it as before
                continue
            if start < boundary:
                rescued.append(normalize_text(w.word))
                w = dataclasses.replace(w, start=boundary - base, end=max(w.end, boundary - base))
            kept.append(w)
        # leading words at/after the boundary, normalized (pure punctuation skipped)
        lead = [(i, normalize_text(w.word)) for i, w in enumerate(kept) if base + w.start >= boundary]
        lead = [(i, n) for i, n in lead if n]
        confirmed = tracker.committed_text_for(uid).split()
        dropped: list[str] = []
        if lead and base + kept[lead[0][0]].start < boundary + 1.0:
            for n in range(min(self.max_ngram, len(confirmed), len(lead)), 0, -1):
                if confirmed[-n:] == [x for _, x in lead[:n]]:
                    drop = {i for i, _ in lead[:n]}
                    dropped = [x for _, x in lead[:n]]
                    kept = [w for i, w in enumerate(kept) if i not in drop]
                    break
        if not rescued and not dropped:
            return hypothesis, None
        return dataclasses.replace(hypothesis, words=tuple(kept)), {"rescued": rescued, "dropped_overlap": dropped}


class StableUpdateHook:
    """Wraps the tracker run_m2 builds so each commit-producing call emits
    one stable-update event (tracker code is unchanged).

    Product-only options, each off unless configured:
      * `boundary_repair`: see BoundaryRepair (applied to a copy of each
        hypothesis before the tracker processes it);
      * `prompt_carry_words`: Whisper's initial_prompt is the utterance's own
        confirmed words (empty at an utterance start); with this set, the
        last N committed words of the preceding utterances are prepended,
        so short VAD fragments are decoded with the sentence's context;
      * `prompt_min_words` (phase 4.6, with trim_keep_s): when omitting
        the still-audible words leaves fewer than this many confirmed words,
        Whisper gets no prompt instead of the fragment. Saved session
        2026-10-29 11:24, utterance 17: confirmed "my normal piece in a",
        4 of them audible, prompt "my"; the final decode then collapsed to
        "cell phone." (13 of 30 quantization-level perturbations of the
        saved window; 0 of 30 with no prompt or the full prompt), all
        before the commit boundary, so the utterance was frozen and
        "quiet room using my usual microphone" was lost;
      * `trim_keep_s`: run_m2 trims the buffered audio at every newly
        confirmed boundary, so the next decode starts exactly at the first
        unconfirmed word with no acoustic left context (saved session
        2026-10-28 19:48: "helps" -> "health", "pace" -> "piece"). With this
        set, the trim point is moved `trim_keep_s` earlier, so the decode
        hears the end of the confirmed words again (Whisper-Streaming's
        reference keeps confirmed audio until a sentence end). The tracker
        still drops words before its boundary. The prompt then omits the
        confirmed words still audible in the window (the reference prompts
        with the text BEFORE its buffer): otherwise Whisper skips them in
        the audio and stamps the next word at the window start, where the
        tracker drops it (window replay: "boundary", "a normal" lost).
      * `final_window_full_max_s` (phase 9, with trim_keep_s): the audio a
        trim removes is remembered, and an utterance's FINAL window is the
        whole utterance with no prompt when that is at most this long (the
        research buffer's own max_buffer_ms is 8 s). Live 2026-10-29: the
        trimmed final windows gave "normal paste" (battery) and, with the
        prompt "i am speaking at my normal pace", "using my use of
        microphone" (AC); the same tiny model on the whole utterance gives
        "pace" and "usual" (phase 8A window_check). Non-final windows are
        unchanged.
      * `cross_utterance_dedup` (phase 9): drop the first committed word(s)
        of an utterance when they repeat the last committed word(s) of the
        previous utterance AND both copies touch the same cut: the windows
        are contiguous (VAD split after a dip shorter than the pre-roll)
        and the old word ends / the new word starts within 0.15 s of it.
        Live 2026-10-29 battery: "between people. | People who speak" at a
        29.40 s split, one "people" in the audio. A repetition the speaker
        separates by a real pause gives non-contiguous windows and is kept;
        repeats inside an utterance are never touched.
    Every hypothesis (text + measured word times) is recorded in
    `hypotheses` for diagnostics.
    """

    def __init__(self, main_module: Any, controller: SegmentController, clock: Callable[[], float], frame_s: float,
                 vad_reset_after_silence_s: float | None = None, boundary_repair: BoundaryRepair | None = None,
                 prompt_carry_words: int = 0, repetition_guard: RepetitionGuard | None = None,
                 trim_keep_s: float = 0.0, prompt_min_words: int = 0,
                 capture_to_session: Callable[[float], float] | None = None,
                 on_provisional: Callable[[int, str, float], None] | None = None,
                 on_record: Callable[[dict[str, Any]], None] | None = None,
                 final_window_full_max_s: float = 0.0, cross_utterance_dedup: bool = False) -> None:
        self.repetition_guard = repetition_guard
        self.final_window_full_max_s = final_window_full_max_s
        self._trimmed_audio: dict[int, list[Any]] = {}
        #: final windows rebuilt as the whole utterance (final_window_full_max_s)
        self.final_window_restores: list[dict[str, Any]] = []
        self.cross_utterance_dedup = cross_utterance_dedup
        #: duplicated words removed at an utterance cut (cross_utterance_dedup)
        self.cross_boundary_dedups: list[dict[str, Any]] = []
        #: structured records for product.session (asr_decode, utterance_timing)
        self.on_record = on_record
        self.trim_keep_s = trim_keep_s
        self.trims = 0
        self.prompt_min_words = prompt_min_words
        #: decoded windows whose trimmed prompt was dropped (prompt_min_words)
        self.prompts_dropped = 0
        self._prompt_dropped: dict[int, bool] = {}
        #: capture-clock timestamp of a frame -> session time of that frame's
        #: audio START (frames are stamped at their block end)
        self.capture_to_session = capture_to_session
        self.on_provisional = on_provisional
        #: per utterance: onset and first-output event times (session clock)
        self.utterance_times: dict[int, dict[str, Any]] = {}
        #: final decodes that committed nothing new and left >= 1 s of the
        #: window after their last word (diagnostic only; nothing is changed)
        self.final_coverage_gaps: list[dict[str, Any]] = []
        self._last_prompt: dict[int, str | None] = {}
        self.main = main_module
        self.controller = controller
        self.clock = clock
        self.frame_s = frame_s
        self.vad_reset_after_silence_s = vad_reset_after_silence_s
        self.boundary_repair = boundary_repair
        self.prompt_carry_words = prompt_carry_words
        self.vad_resets = 0
        self.buffer = None
        self.tracker = None
        self.events: list[dict[str, Any]] = []
        self.hypotheses: list[dict[str, Any]] = []
        self.repairs: list[dict[str, Any]] = []
        self._committed: list[tuple[int, str]] = []  # (utterance_id, word) in commit order

    def _onset_s(self, utterance_id: int) -> float | None:
        record = self.buffer.utterance_pre_roll.get(utterance_id, {}) if self.buffer is not None else {}
        seq = record.get("onset_sequence_number")
        return seq * self.frame_s if seq is not None else None

    def _emit(self, utterance_id: int, before: int) -> None:
        new = self.tracker.commit_events_for(utterance_id)[before:]
        if not new:
            return
        t = self.clock()
        text = " ".join(c.new_text for c in new if c.new_text).strip()
        final = any(c.is_final for c in new)
        if self.repetition_guard is not None:
            text = " ".join(self.repetition_guard.filter(utterance_id, text.split()))
        if self.cross_utterance_dedup and text:
            text = self._dedup_at_cut(utterance_id, text)
        if text and "first_commit_s" not in self._times(utterance_id):
            self._times(utterance_id)["first_commit_s"] = t
            self._publish_timing(utterance_id, t)
        self._committed += [(utterance_id, w) for w in text.split()]
        sentence_end = final and self._ends_sentence(utterance_id)
        self.events.append({"utterance_id": utterance_id, "new_text": text, "utterance_final": final, "t": t,
                            "commit_ids": [c.commit_id for c in new], "sentence_end": sentence_end})
        state = self.buffer.state if self.buffer is not None else None
        newer = max(0, state.utterance_id - utterance_id) if state is not None else 0
        self.controller.submit_update(utterance_id=utterance_id, new_text=text, utterance_final=final, t=t,
                                      onset_s=self._onset_s(utterance_id), sentence_end=sentence_end,
                                      newer_utterances=newer)

    def _dedup_at_cut(self, utterance_id: int, text: str, edge_s: float = 0.15, max_ngram: int = 2) -> str:
        """See `cross_utterance_dedup`. Only an utterance's first commit is checked."""

        from streaming.prefix_agreement import normalize_text

        if any(uid == utterance_id for uid, _ in self._committed):
            return text
        prev_words = [w for uid, w in self._committed if uid == utterance_id - 1]
        prev = next((h for h in reversed(self.hypotheses) if h["utterance_id"] == utterance_id - 1 and h["success"]
                     and h["words"]), None)
        cur = next((h for h in reversed(self.hypotheses) if h["utterance_id"] == utterance_id and h["success"]
                    and h["words"]), None)
        if not prev_words or prev is None or cur is None:
            return text
        prev_last = next((w for w in reversed(prev["words"]) if normalize_text(w[0])), None)
        cur_first = next((w for w in cur["words"] if normalize_text(w[0])), None)
        if prev_last is None or cur_first is None:
            return text
        contiguous = -0.05 <= cur["audio_start"] - prev["audio_end"] <= edge_s
        at_cut = prev["audio_end"] - prev_last[2] <= edge_s and cur_first[1] - cur["audio_start"] <= edge_s
        if not (contiguous and at_cut):
            return text
        words = text.split()
        for n in range(min(max_ngram, len(prev_words), len(words)), 0, -1):
            if prev_words[-n:] == words[:n]:
                self.cross_boundary_dedups.append({"utterance_id": utterance_id, "dropped": words[:n],
                                                   "cut_gap_s": round(cur["audio_start"] - prev["audio_end"], 3)})
                return " ".join(words[n:])
        return text

    def _ends_sentence(self, utterance_id: int) -> bool:
        """Whether the utterance's last successful decode ends in . ? or !
        (not an ellipsis, which Whisper uses for audio cut mid-phrase)."""

        for h in reversed(self.hypotheses):
            if h["utterance_id"] == utterance_id and h["success"]:
                text = (h["text"] or "").rstrip().rstrip('"\')')
                return text.endswith((".", "?", "!")) and not text.endswith(("..", "\u2026"))
        return False

    def build_vad(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """Observe VAD speech start/end transitions (results unchanged).

        Product-only option `vad_reset_after_silence_s`: reset the VAD's
        recurrent state after that much continuous non-speech. Silero's
        carried-over state can suppress quiet speech that follows loud
        speech (measured: a quiet clip after a loud one went undetected, and
        the next clip lost ~1.7 s of onset). The research harness reset the
        VAD for every file; a live session otherwise never resets. The reset
        only happens outside SPEECH, after run_m2 has already finalized the
        utterance, so no utterance is cut."""

        def factory(config, logger):
            vad = original(config, logger)
            process = vad.process
            reset_frames = (int(round(self.vad_reset_after_silence_s / self.frame_s))
                            if self.vad_reset_after_silence_s else None)
            silent = 0
            first_call = True

            def wrapped_process(*args, **kwargs):
                nonlocal silent, first_call
                if first_call:
                    # First frame of the run loop: all models are loaded.
                    # Pin the main (VAD) thread to 1 torch thread. With the
                    # MT worker at 8 threads and Whisper loaded in this
                    # thread, each 32 ms Silero call otherwise opens an
                    # 8-thread OpenMP region whose 7 workers spin between
                    # calls: measured 1.8-3.9 idle cores vs 0.2 with this,
                    # and MT latency unchanged (MT keeps its own setting).
                    first_call = False
                    import torch

                    torch.set_num_threads(1)
                result = process(*args, **kwargs)
                if result.state_changed and result.state.value in ("SPEECH", "SILENCE"):
                    self.controller.submit_speech(active=result.state.value == "SPEECH", t=self.clock())
                silent = 0 if result.state.value == "SPEECH" else silent + 1
                if reset_frames and silent == reset_frames and hasattr(vad, "reset"):
                    vad.reset()
                    self.vad_resets += 1
                return result

            vad.process = wrapped_process
            return vad
        return factory

    def build_streaming_buffer(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def factory(config, logger):
            self.buffer = original(config, logger)
            buffer = self.buffer
            if self.trim_keep_s > 0:
                trim = buffer.trim_confirmed

                def trim_keeping_context(confirmed_up_to_time):
                    self.trims += 1
                    cut = confirmed_up_to_time - self.trim_keep_s
                    state = buffer.state
                    if self.final_window_full_max_s > 0 and state is not None:
                        gone = [f for f in state.context_audio if f.capture_timestamp + f.duration_ms / 1000.0 <= cut]
                        self._trimmed_audio.setdefault(state.utterance_id, []).extend(gone)
                    return trim(cut)

                buffer.trim_confirmed = trim_keeping_context
            if self.trim_keep_s > 0 and self.final_window_full_max_s > 0:
                build_final = buffer.try_build_final_window

                def whole_utterance_final_window(initial_prompt=None):
                    state = buffer.state
                    if state is not None:
                        for uid in [u for u in self._trimmed_audio if u < state.utterance_id]:
                            del self._trimmed_audio[uid]          # earlier utterances: never needed again
                        gone = self._trimmed_audio.get(state.utterance_id)
                        if gone:
                            total_s = sum(f.duration_ms for f in [*gone, *state.context_audio,
                                                                  *state.new_audio_since_last_asr]) / 1000.0
                            if total_s <= self.final_window_full_max_s:
                                del self._trimmed_audio[state.utterance_id]
                                state.context_audio[:0] = gone
                                self.final_window_restores.append({"utterance_id": state.utterance_id,
                                                                   "window_s": round(total_s, 2),
                                                                   "prompt_dropped": initial_prompt})
                                initial_prompt = None   # all confirmed words are audible again
                    return build_final(initial_prompt=initial_prompt)

                buffer.try_build_final_window = whole_utterance_final_window
            return buffer
        return factory

    def build_prefix_agreement_tracker(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def factory(config, logger):
            tracker = original(config, logger)
            process, finalize = tracker.process, tracker.finalize_utterance
            own_prompt = tracker.confirmed_prompt_text

            def wrapped_process(hypothesis):
                before = len(tracker.commit_events_for(hypothesis.utterance_id))
                committed_before = tracker.committed_text_for(hypothesis.utterance_id)
                self._record(tracker, hypothesis)
                if self.boundary_repair is not None:
                    hypothesis, repair = self.boundary_repair.apply(tracker, hypothesis)
                    if repair is not None:
                        self.repairs.append({"utterance_id": hypothesis.utterance_id,
                                             "window_id": hypothesis.window_id, **repair})
                result = process(hypothesis)
                self._emit(hypothesis.utterance_id, before)
                self._check_final_coverage(tracker, hypothesis, committed_before)
                if (self.on_provisional is not None and hypothesis.success and not hypothesis.is_final
                        and (hypothesis.text or "").strip()):
                    self.on_provisional(hypothesis.utterance_id, hypothesis.text.strip(), self.clock())
                return result

            def wrapped_prompt(utterance_id):
                own = own_prompt(utterance_id)
                if own and self.trim_keep_s > 0:
                    own = self._prompt_before_audio(tracker, utterance_id, own)
                if not self.prompt_carry_words:
                    self._last_prompt[utterance_id] = own
                    return own
                carry = [w for uid, w in self._committed if uid != utterance_id][-self.prompt_carry_words:]
                words = carry + (own.split() if own else [])
                return " ".join(words[-tracker.prompt_word_limit:]) if words else None

            def wrapped_finalize(utterance_id, reason=None):
                before = len(tracker.commit_events_for(utterance_id))
                result = finalize(utterance_id, reason=reason)
                self._emit(utterance_id, before)
                return result

            tracker.process, tracker.finalize_utterance = wrapped_process, wrapped_finalize
            tracker.confirmed_prompt_text = wrapped_prompt
            self.tracker = tracker
            return tracker
        return factory

    def _prompt_before_audio(self, tracker: Any, utterance_id: int, prompt: str) -> str | None:
        """`prompt` without its last confirmed words that are still audible
        in the buffered audio (used with trim_keep_s). Their times come
        from the utterance's latest decode."""

        from streaming.prefix_agreement import normalize_text

        state = self.buffer.state if self.buffer is not None else None
        boundary = tracker.confirmed_boundary_time(utterance_id)
        if state is None or state.utterance_id != utterance_id or boundary is None:
            return prompt
        frames = [*state.context_audio, *state.new_audio_since_last_asr]
        latest = next((h for h in reversed(self.hypotheses)
                       if h["utterance_id"] == utterance_id and h["success"]), None)
        if not frames or latest is None:
            return prompt
        start = frames[0].capture_timestamp
        audible = sum(1 for word, a, b in latest["words"] if normalize_text(word) and b > start and a < boundary)
        words = prompt.split()
        words = words[:max(0, len(words) - audible)]
        dropped = bool(audible and 0 < len(words) < self.prompt_min_words)
        self._prompt_dropped[utterance_id] = dropped
        return None if dropped else (" ".join(words) or None)

    def _times(self, utterance_id: int) -> dict[str, Any]:
        times = self.utterance_times.setdefault(utterance_id, {})
        if times.get("onset_s") is None:
            times["onset_s"] = self._onset_s(utterance_id)
        return times

    def _record(self, tracker: Any, h: Any) -> None:
        base = h.audio_start_time
        t = self.clock()
        self.hypotheses.append({
            "utterance_id": h.utterance_id, "window_id": h.window_id, "is_final": h.is_final, "success": h.success,
            "t_s": t, "audio_start": base, "audio_end": h.audio_end_time,
            "boundary_before": tracker.confirmed_boundary_time(h.utterance_id), "text": h.text,
            # the prompt last requested for this utterance before this result
            # arrived (= the window's prompt: one decode is in flight at a time)
            "prompt": self._last_prompt.get(h.utterance_id),
            "words": [[w.word, base + w.start, base + w.end] for w in (h.words or ())]})
        if self._prompt_dropped.pop(h.utterance_id, False):
            self.prompts_dropped += 1
        times = self._times(h.utterance_id)
        before = dict(times)
        times.setdefault("first_hypothesis_s", t)
        if h.success and (h.text or "").strip():
            times.setdefault("first_text_hypothesis_s", t)
        start = self.capture_to_session(base) if self.capture_to_session is not None else None
        if start is not None and h.is_final:
            if h.success and h.words:
                times["last_word_end_s"] = start + max(w.end for w in h.words)
            times["vad_end_s"] = self.capture_to_session(h.audio_end_time) + self.frame_s
        if self.on_record is not None:
            completed, end = getattr(h, "completed_at", None), getattr(h, "audio_end_time", None)
            self.on_record({"type": "asr_decode", "t_s": t, "utterance_id": h.utterance_id, "window_id": h.window_id,
                            "is_final": h.is_final, "success": h.success,
                            "inference_ms": getattr(h, "inference_time_ms", None), "rtf": getattr(h, "rtf", None),
                            "queue_wait_ms": getattr(h, "queue_wait_ms", None),
                            "completion_lag_s": completed - end if completed and end else None})
            if times != before:
                self._publish_timing(h.utterance_id, t)

    def _publish_timing(self, utterance_id: int, t: float) -> None:
        if self.on_record is not None:
            times = self.utterance_times.get(utterance_id, {})
            self.on_record({"type": "utterance_timing", "t_s": t, "utterance_id": utterance_id,
                            **{k: times.get(k) for k in ("onset_s", "first_text_hypothesis_s", "first_commit_s",
                                                         "last_word_end_s", "vad_end_s")}})

    def _check_final_coverage(self, tracker: Any, h: Any, committed_before: str) -> None:
        """Record (only) a final decode that added no words although >= 1 s
        of its window lies after its last word."""

        if not (h.is_final and h.success):
            return
        if tracker.committed_text_for(h.utterance_id) != committed_before:
            return
        window_s = h.audio_end_time - h.audio_start_time
        last_end = max((w.end for w in (h.words or ())), default=0.0)
        if window_s - last_end >= 1.0:
            self.final_coverage_gaps.append({
                "utterance_id": h.utterance_id, "window_id": h.window_id, "text": h.text,
                "uncovered_s": window_s - last_end, "prompt": self._last_prompt.get(h.utterance_id)})


def run_m2_with_translation(main_module: Any, config: Any, logger: Any, source: Any, controller: SegmentController,
                            m2_console: TextIO | None, vad_reset_after_silence_s: float | None = None,
                            boundary_repair: BoundaryRepair | None = None, prompt_carry_words: int = 0,
                            repetition_guard: RepetitionGuard | None = None,
                            trim_keep_s: float = 0.0, prompt_min_words: int = 0,
                            on_provisional: Callable[[int, str, float], None] | None = None,
                            on_record: Callable[[dict[str, Any]], None] | None = None,
                            final_window_full_max_s: float = 0.0, cross_utterance_dedup: bool = False
                            ) -> tuple[int, StableUpdateHook]:
    """Run the unmodified main.run_m2() on `source`, feeding its stable
    commits to `controller`. run_m2's per-frame prints go to `m2_console`."""

    hook = StableUpdateHook(main_module, controller, source.clock, source.frame_s, vad_reset_after_silence_s,
                            boundary_repair, prompt_carry_words, repetition_guard, trim_keep_s,
                            prompt_min_words=prompt_min_words,
                            capture_to_session=lambda c: (c - source.t0 - source.frame_s
                                                          if getattr(source, "t0", None) is not None else None),
                            on_provisional=on_provisional, on_record=on_record,
                            final_window_full_max_s=final_window_full_max_s,
                            cross_utterance_dedup=cross_utterance_dedup)
    sink = m2_console if m2_console is not None else open(os.devnull, "w", encoding="utf-8")
    with patch.object(main_module, "build_audio_stream", return_value=source), \
            patch.object(main_module, "build_vad", side_effect=hook.build_vad(main_module.build_vad)), \
            patch.object(main_module, "build_streaming_buffer",
                         side_effect=hook.build_streaming_buffer(main_module.build_streaming_buffer)), \
            patch.object(main_module, "build_prefix_agreement_tracker",
                         side_effect=hook.build_prefix_agreement_tracker(main_module.build_prefix_agreement_tracker)), \
            contextlib.redirect_stdout(sink):
        exit_code = main_module.run_m2(config, logger)
    return exit_code, hook


# =======================================================================
# Session metrics
# =======================================================================


from product.latency import USER_LATENCY_DEFINITIONS, _dist_n, _pct, latency_core  # noqa: E402,F401 (phase 7: moved, re-exported)


def user_latency_metrics(hook: "StableUpdateHook", segments: list[Segment], display: Display) -> dict[str, Any]:
    """Viewer-facing latencies measured from speech onset (phase 4.6), for
    session_summary.json. Same code as the live panel (latency_core)."""

    segs = [{"segment_id": s.segment_id, "utterance_ids": list(s.utterance_ids), "closed_s": s.closed_s}
            for s in segments]
    return latency_core(display.records, hook.utterance_times, segs, hook.on_provisional is not None)


def _dist(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "mean": sum(values) / len(values) if values else None, "p50": _pct(values, 50),
            "p95": _pct(values, 95), "max": max(values) if values else None}


def translation_metrics(segments: list[Segment], display: Display) -> dict[str, Any]:
    """Word->Hindi latency: from the stream time a stable word became
    available to the completion of the first MT call that covered it."""

    word_latency, first_output, inference, queue_wait = [], [], [], []
    reasons: dict[str, int] = {}
    errors = 0
    for seg in segments:
        results = [display.result_for(seg.segment_id, c.invocation_id) for c in seg.calls]
        done = [(c, r) for c, r in zip(seg.calls, results) if r is not None and r.error is None]
        errors += sum(1 for r in results if r is not None and r.error)
        for r in (r for _, r in done):
            if r.inference_ms is not None:
                inference.append(r.inference_ms)
            queue_wait.append(r.queue_wait_s)
        for i, t_word in enumerate(seg.word_times):
            covering = next((r for c, r in done if c.source_word_count >= i + 1), None)
            if covering is not None:
                word_latency.append(covering.completed_s - t_word)
        if done and seg.word_times:
            first_output.append(min(r.completed_s for _, r in done) - seg.word_times[0])
        for d in seg.decisions:
            if d.get("invocation_id") is not None:
                reason = d["decision"]["reason"]
                reasons[reason] = reasons.get(reason, 0) + 1
    fq = [(s, display.final_quality_result(s.segment_id)) for s in segments if s.final_call is not None]
    fq_ok = [(s, r) for s, r in fq if r is not None and r.error is None]
    final_mt = {
        "calls": len(fq), "errors": sum(1 for _, r in fq if r is not None and r.error),
        "missing": sum(1 for _, r in fq if r is None),
        "inference_ms": _dist([r.inference_ms for _, r in fq_ok if r.inference_ms is not None]),
        "close_to_final_s": _dist([r.completed_s - s.closed_s for s, r in fq_ok if s.closed_s is not None]),
        "queue_wait_s": _dist([r.queue_wait_s for _, r in fq_ok]),
        "final_kinds": {k: sum(1 for v in display.final_kinds.values() if v == k) for k in set(display.final_kinds.values())},
    } if fq else None
    return {
        "final_mt": final_mt,
        "segments": len(segments),
        "segment_close_reasons": {r: sum(1 for s in segments if s.close_reason == r) for r in {s.close_reason for s in segments}},
        "stable_words": sum(len(s.words) for s in segments),
        "mt_calls": sum(len(s.calls) for s in segments),
        "mt_calls_by_reason": reasons,
        "mt_errors": errors,
        "word_to_hindi_s": _dist(word_latency),
        "segment_first_output_s": _dist(first_output),
        "mt_inference_ms": _dist(inference),
        "mt_queue_wait_s": _dist(queue_wait),
    }


def asr_metrics(events_path: Path) -> dict[str, Any]:
    """ASR decode statistics from the session's own m2_events.jsonl."""

    inference, rtf, lag, queue_wait, failures, summary = [], [], [], [], 0, None
    if events_path.exists():
        with events_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") == "streaming_asr_hypothesis":
                    if not event.get("success"):
                        failures += 1
                        continue
                    if event.get("inference_time_ms") is not None:
                        inference.append(event["inference_time_ms"])
                    if event.get("rtf") is not None:
                        rtf.append(event["rtf"])
                    if event.get("completed_at") is not None and event.get("audio_end_time") is not None:
                        lag.append(event["completed_at"] - event["audio_end_time"])
                    if event.get("queue_wait_ms") is not None:
                        queue_wait.append(event["queue_wait_ms"])
                elif event.get("event") == "m2_run_summary":
                    summary = event
    return {"decodes": len(inference), "failed_decodes": failures, "inference_ms": _dist(inference), "rtf": _dist(rtf),
            "decode_completion_lag_s": _dist(lag), "queue_wait_ms": _dist_n(queue_wait), "m2_run_summary": summary}
