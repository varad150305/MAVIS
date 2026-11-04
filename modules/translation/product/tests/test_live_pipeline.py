"""Product-only tests for the live translation pipeline (product/).

Fast and model-free: a fake MT engine, a real perf_counter clock and small
time constants. The run_m2 integration test drives the REAL main.run_m2()
with scripted frames/VAD/ASR (the same fake pattern as
tests/test_run_m2_finalization_fixes.py) plus the product hook, so it
exercises the production orchestration, StreamingBuffer and
PrefixAgreementTracker unchanged.
"""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from product import live_pipeline as lp
from product import runtime
from product.sources import LiveMicSource, PacedWavSource, check_input_device
from translation.scheduling import AdaptiveDeadlinePolicy, EagerPolicy


# =======================================================================
# Fakes and helpers
# =======================================================================


class FakeEngine:
    """translate_request() returns 'HI(<source>)'; optional delay/failure."""

    def __init__(self, delay_s: float = 0.0, fail_on: set[str] | None = None) -> None:
        self.delay_s = delay_s
        self.fail_on = fail_on or set()
        self.sources: list[str] = []

    def translate_request(self, request):
        self.sources.append(request.source_text)
        time.sleep(self.delay_s)
        if request.source_text in self.fail_on:
            raise RuntimeError("synthetic MT failure")

        class _Resp:
            target_text = f"HI({request.source_text})"
            inference_time_ms = self.delay_s * 1000.0

        return _Resp()


class Clock:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def __call__(self) -> float:
        return time.perf_counter() - self.t0


def build(policy_factory=EagerPolicy, rules=None, engine=None):
    clock = Clock()
    out = io.StringIO()
    display = lp.Display(out, None, clock)
    engine = engine or FakeEngine()
    worker = lp.MTWorker(engine, clock, display.translation)
    translator = lp.AsyncSegmentTranslator(worker, clock)
    controller = lp.SegmentController(policy_factory, translator, clock, rules or lp.SegmentRules(), display)
    worker.start()
    controller.start()
    return clock, display, worker, controller, engine, out


def finish(clock, controller, worker):
    controller.stop(clock())
    assert worker.shutdown(timeout_s=10)
    assert controller.error is None
    assert not controller.is_alive()


def mt3c_policy(t_max_s: float = 0.3):
    return lambda: AdaptiveDeadlinePolicy(k_base=8, k_max=24, t_max_s=t_max_s, w_r_s=4.0, rho=0.5,
                                          cost_a_s=0.06, cost_b_s_per_word=0.028)


def update(controller, clock, uid, text, final=False, onset=None):
    t = clock()
    controller.submit_update(utterance_id=uid, new_text=text, utterance_final=final, t=t,
                             onset_s=onset if onset is not None else max(0.0, t - 0.5))


# =======================================================================
# Controller / scheduler integration
# =======================================================================


def test_eager_translates_each_update_and_final_matches_last_call():
    clock, display, worker, controller, engine, out = build()
    update(controller, clock, 1, "hello there")
    update(controller, clock, 1, "my friend", final=True)
    time.sleep(0.1)
    finish(clock, controller, worker)

    assert engine.sources == ["hello there", "hello there my friend"]
    assert display.finals == {1: "HI(hello there my friend)"}
    seg = controller.segments[0]
    assert seg.close_reason == "session end"
    assert [d["decision"]["reason"] for d in seg.decisions if d["invocation_id"]] == ["new stable source words"] * 2


def test_segment_closes_at_utterance_end_and_next_segment_starts_fresh():
    rules = lp.SegmentRules(close_after_words=4, idle_close_s=5.0, hard_max_words=50)
    clock, display, worker, controller, engine, out = build(rules=rules)
    update(controller, clock, 1, "one two three")
    update(controller, clock, 1, "four five", final=True)     # 5 words at utterance end -> close
    update(controller, clock, 2, "six seven", final=True)
    time.sleep(0.1)
    finish(clock, controller, worker)

    assert [s.close_reason for s in controller.segments] == ["utterance end at >= 4 words", "session end"]
    # the second segment never re-translates the first segment's words
    assert engine.sources[-1] == "six seven"
    assert display.finals == {1: "HI(one two three four five)", 2: "HI(six seven)"}


def test_hard_cap_closes_segment_mid_utterance():
    rules = lp.SegmentRules(close_after_words=3, idle_close_s=5.0, hard_max_words=4)
    clock, display, worker, controller, engine, out = build(rules=rules)
    update(controller, clock, 1, "a b c d e")   # not final, already over the cap
    update(controller, clock, 1, "f g")
    time.sleep(0.1)
    finish(clock, controller, worker)
    assert controller.segments[0].close_reason == "hard cap 4 words"
    assert " ".join(controller.segments[1].words) == "f g"


def test_idle_gap_after_utterance_end_closes_segment_without_new_audio():
    rules = lp.SegmentRules(close_after_words=50, idle_close_s=0.2, hard_max_words=60)
    clock, display, worker, controller, engine, out = build(rules=rules)
    update(controller, clock, 1, "short phrase", final=True)
    time.sleep(0.6)
    assert controller.active is None, "idle timer should have closed the segment"
    assert controller.segments[0].close_reason == "idle 0.2 s"
    finish(clock, controller, worker)
    assert display.finals[1] == "HI(short phrase)"


def test_short_pause_with_late_next_commit_does_not_split_the_segment():
    """Regression (found on a real WAV run): the next utterance's first commit
    arrives seconds after its speech starts, so idle time must be measured
    from VAD silence, not from the last commit."""

    rules = lp.SegmentRules(close_after_words=50, idle_close_s=0.3, hard_max_words=60)
    clock, display, worker, controller, engine, out = build(rules=rules)
    controller.submit_speech(active=True, t=clock())
    controller.submit_speech(active=False, t=clock())          # 100 ms pause...
    update(controller, clock, 1, "mass car ownership also leads", final=True)
    time.sleep(0.05)
    controller.submit_speech(active=True, t=clock())           # ...speech resumes at once
    time.sleep(0.6)                                             # commit gap > idle_close_s
    assert controller.active is not None, "segment must stay open while the speaker talks"
    update(controller, clock, 2, "which leads to new techniques", final=True)
    controller.submit_speech(active=False, t=clock())
    time.sleep(0.6)                                             # real silence -> close
    assert controller.active is None
    assert len(controller.segments) == 1
    assert controller.segments[0].close_reason == "idle 0.3 s"
    finish(clock, controller, worker)
    assert display.finals[1] == "HI(mass car ownership also leads which leads to new techniques)"


def test_no_idle_close_while_utterance_still_open():
    rules = lp.SegmentRules(close_after_words=50, idle_close_s=0.1, hard_max_words=60)
    clock, display, worker, controller, engine, out = build(rules=rules)
    update(controller, clock, 1, "still talking")  # not utterance-final
    time.sleep(0.4)
    assert controller.active is not None
    finish(clock, controller, worker)


def test_mt3c_deadline_fires_without_new_update_while_mt_is_busy():
    engine = FakeEngine(delay_s=0.3)
    clock, display, worker, controller, engine, out = build(policy_factory=mt3c_policy(0.3), engine=engine)
    update(controller, clock, 1, "three short words")   # debt 3 < K_base 8 -> WAIT, timer armed
    time.sleep(0.6)                                        # deadline (0.3 s) passes with no update
    seg = controller.active
    reasons = [d.decision.reason for d in controller.scheduler.decisions]
    assert "deadline" in reasons
    assert len(seg.calls) == 1 and seg.calls[0].trigger_time is not None
    update(controller, clock, 1, "and nine more words arrive here right now ok", final=True)
    time.sleep(0.05)
    finish(clock, controller, worker)
    # FIFO: every scheduled call executed once, in order
    calls = controller.segments[0].calls
    assert engine.sources == [c.source_text for c in calls]
    assert display.finals[1] == f"HI({calls[-1].source_text})"


def test_mt_failure_is_reported_and_final_falls_back_to_latest_success():
    engine = FakeEngine(fail_on={"good start then"})
    clock, display, worker, controller, engine, out = build(engine=engine)
    update(controller, clock, 1, "good start")
    update(controller, clock, 1, "then")
    time.sleep(0.1)
    finish(clock, controller, worker)
    text = out.getvalue()
    assert "MT ERROR" in text
    assert display.finals[1] == "HI(good start)"
    metrics = lp.translation_metrics(controller.segments, display)
    assert metrics["mt_errors"] == 1


def test_empty_finalization_marker_does_not_open_a_segment():
    clock, display, worker, controller, engine, out = build()
    update(controller, clock, 1, "", final=True)
    time.sleep(0.05)
    finish(clock, controller, worker)
    assert controller.segments == [] and engine.sources == []


def test_mt_worker_applies_its_thread_count_and_warms_up_in_its_own_thread():
    """Regression: setting 8 torch threads in the main thread made each VAD
    inference wake 8 spinning OpenMP workers (~1.5 extra cores while idle).

    torch.get_num_threads() reports one process-wide value, but the OpenMP
    parallelism actually used is per calling thread (measured with matmul
    timings on this machine: the main thread stays single-threaded while the
    worker runs 8), so only the worker-side effects are asserted here."""

    clock = Clock()
    engine = FakeEngine()
    engine.warm_up = lambda: engine.sources.append("<warm-up>")
    worker = lp.MTWorker(engine, clock, lambda c, r: None, torch_threads=4, warm_up=True)
    worker.start()
    assert worker.ready.wait(5)
    assert worker.thread_torch_threads == 4
    assert engine.sources == ["<warm-up>"] and worker.warm_up_ms is not None
    assert worker.shutdown(timeout_s=5)


def test_translation_metrics_word_latency_uses_first_covering_call():
    seg = lp.Segment(segment_id=1, opened_s=0.0, rate_origin_s=0.0, words=["a", "b", "c"], word_times=[1.0, 1.0, 2.0])
    seg.calls = [lp.ScheduledCall(1, 1, 1, "a b", 2, 2, 0, False, 1.0, 1.0, 1.0),
                 lp.ScheduledCall(1, 2, 1, "a b c", 3, 1, 2, True, 2.0, 2.0, 2.0)]
    seg.decisions = [{"invocation_id": 1, "decision": {"reason": "x"}}, {"invocation_id": 2, "decision": {"reason": "y"}}]
    display = lp.Display(None, None, lambda: 0.0)
    display.translation(seg.calls[0], lp.CallResult(1, 1, 2, "h1", 100.0, 1.0, 1.5, 0.0))
    display.translation(seg.calls[1], lp.CallResult(1, 2, 3, "h2", 100.0, 2.0, 2.4, 0.0))
    m = lp.translation_metrics([seg], display)
    assert m["word_to_hindi_s"]["n"] == 3
    assert m["word_to_hindi_s"]["max"] == pytest.approx(0.5)   # words a,b -> 1.5-1.0; word c -> 2.4-2.0
    assert m["segment_first_output_s"]["p50"] == pytest.approx(0.5)
    assert m["mt_calls_by_reason"] == {"x": 1, "y": 1}


def test_mt3c_parameters_hash_is_enforced(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        lp.load_mt3c_parameters(path, "0" * 64)


def test_frozen_mt3c_parameters_load_unchanged():
    repo = Path(__file__).resolve().parents[2]
    params, digest = lp.load_mt3c_parameters(repo / "evaluation" / "mt3c_parameters.json",
                                             "56e86c68c791c7447e266c98de23d9c71068786a98e30404aea17617a0e42bb3")
    policy = lp.make_policy_factory("mt3c", mt3c_params=params, mt3c_sha256=digest)()
    assert policy.params()["T_max_s"] == 2.0 and policy.params()["K_base"] == 8


# =======================================================================
# Display / console robustness
# =======================================================================


def test_hindi_output_survives_a_non_utf8_redirected_stdout():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    with pytest.raises(UnicodeEncodeError):
        stream.write("नमस्ते")
    stream = runtime.utf8_console(io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    stream.write("नमस्ते\n")
    stream.flush()


def test_transcript_jsonl_records_every_display_event(tmp_path):
    path = tmp_path / "t.jsonl"
    clock, display = Clock(), None
    display = lp.Display(None, path, clock)
    seg = lp.Segment(segment_id=1, opened_s=0.0, rate_origin_s=0.0, words=["hi"], word_times=[0.0])
    call = lp.ScheduledCall(1, 1, 1, "hi", 1, 1, 0, False, 0.0, 0.0, 0.0)
    display.english(seg, ["hi"], 0.0, True)
    display.translation(call, lp.CallResult(1, 1, 1, "नमस्ते", 1.0, 0.0, 0.1, 0.0))
    seg.calls, seg.closed_s, seg.close_reason = [call], 0.2, "session end"
    display.segment_closed(seg, 1)
    kinds = [json.loads(l)["type"] for l in path.read_text(encoding="utf-8").splitlines()]
    assert kinds == ["en_update", "hi_draft", "segment_closed", "hi_final"]


# =======================================================================
# Sources
# =======================================================================


def write_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    import wave

    samples = (0.1 * np.sin(np.arange(int(seconds * rate)) / 10.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def drain(source, limit_s: float = 5.0):
    frames, start = [], time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        while time.perf_counter() - start < limit_s:
            frame = source.read()
            if frame is not None:
                frames.append(frame)
    return frames


def test_paced_wav_source_releases_at_1x_and_ends_with_keyboard_interrupt(tmp_path):
    path = tmp_path / "a.wav"
    write_wav(path, 0.55)
    source = PacedWavSource(path)
    source.start()
    t = time.perf_counter()
    frames = drain(source)
    elapsed = time.perf_counter() - t
    assert len(frames) == 6                      # 5 full frames + 1 zero-padded
    assert [f.sequence_number for f in frames] == list(range(6))
    assert elapsed >= 0.55
    assert source.stop_reason == "end of file"


def test_wav_source_rejects_wrong_sample_rate(tmp_path):
    path = tmp_path / "b.wav"
    write_wav(path, 0.2, rate=8000)
    with pytest.raises(ValueError, match="16000"):
        PacedWavSource(path)


def test_duration_limit_and_stop_event_end_a_session(tmp_path):
    path = tmp_path / "c.wav"
    write_wav(path, 2.0)
    source = PacedWavSource(path, max_duration_s=0.3)
    source.start()
    assert len(drain(source)) == 3 and "duration" in source.stop_reason

    stop = threading.Event()
    source = PacedWavSource(path, stop_event=stop)
    source.start()
    stop.set()
    with pytest.raises(KeyboardInterrupt):
        source.read()


class _EmptyStream:
    chunk_duration_ms = 100
    dropped_frames = 0

    def read(self):
        return None


def test_mic_source_waits_instead_of_spinning_when_no_frame_is_ready():
    source = LiveMicSource(_EmptyStream(), idle_wait_s=0.005)
    cpu0, t0 = time.process_time(), time.perf_counter()
    reads = 0
    while time.perf_counter() - t0 < 0.5:
        assert source.read() is None
        reads += 1
    cpu = time.process_time() - cpu0
    assert reads < 0.5 / 0.004          # bounded wake-ups, not a spin
    assert cpu < 0.25                    # well below one busy core over 0.5 s


def test_invalid_input_device_fails_fast_with_actionable_message():
    with pytest.raises(RuntimeError, match="--list-devices"):
        check_input_device(9999)


def test_vad_reset_after_sustained_silence_only_outside_speech():
    from vad.vad import VADState

    class FakeVAD:
        def __init__(self, states):
            self.states, self.i, self.last, self.resets = states, 0, None, []

        def process(self, *a, **k):
            state = self.states[self.i]
            self.i += 1
            changed, self.last = state != self.last, state

            class R:
                pass

            r = R()
            r.state, r.state_changed = state, changed
            return r

        def reset(self):
            self.resets.append(self.i)

    S, Q = VADState.SPEECH, VADState.SILENCE
    states = [S] * 5 + [Q] * 25 + [S] * 3 + [Q] * 5
    clock = Clock()
    controller = lp.SegmentController(EagerPolicy, lp.AsyncSegmentTranslator(lp.MTWorker(FakeEngine(), clock, lambda c, r: None), clock),
                                      clock, lp.SegmentRules(), lp.Display(None, None, clock))
    hook = lp.StableUpdateHook(None, controller, clock, 0.1, vad_reset_after_silence_s=2.0)
    vad = FakeVAD(states)
    wrapped = hook.build_vad(lambda c, l: vad)(None, None)
    results = [wrapped.process() for _ in states]
    assert [r.state for r in results] == states        # results pass through unchanged
    assert vad.resets == [25] and hook.vad_resets == 1  # once, after 20 non-speech frames (2.0 s)
    speech_events = [controller.events.get_nowait() for _ in range(controller.events.qsize())]
    assert [e["active"] for e in speech_events] == [True, False, True, False]


# =======================================================================
# Real run_m2 + product hook (scripted audio/VAD/ASR)
# =======================================================================


def test_real_run_m2_commits_flow_through_hook_to_hindi():
    from unittest.mock import patch

    import main
    from asr.whisper_engine import WordTiming
    from audio.audio_stream import AudioFrame
    from streaming.streaming_asr import ASRHypothesis
    from utils.config import Config
    from utils.logger import EventLogger
    from vad.vad import VADState

    # utterance 1: frames 0-9 speech, 10-11 silence; utterance 2: 12-17 speech, 18-19 silence
    states = [VADState.SPEECH] * 10 + [VADState.SILENCE] * 2 + [VADState.SPEECH] * 6 + [VADState.SILENCE] * 2
    words_by_utt = {1: ["the", "quick", "brown", "fox"], 2: ["jumps", "over"]}

    class Source:
        dropped_frames = 0
        frame_s = 0.1
        stop_reason = None

        def __init__(self):
            self.i = 0

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

        def clock(self):
            return self.i * 0.1

        def read(self):
            if self.i >= len(states):
                raise KeyboardInterrupt
            frame = AudioFrame(np.full(1600, 0.01, dtype=np.float32), 1000.0 + self.i * 0.1, 100.0, self.i, 0, 16000, 1)
            self.i += 1
            return frame

    class VADResult:
        def __init__(self, state, changed):
            self.state, self.state_changed = state, changed
            self.speech_probability, self.processing_time_ms = 0.5, 0.1

    class VAD:
        def __init__(self):
            self.i, self.last = 0, None

        def process(self, audio, sample_rate, timestamp, sequence_number):
            state = states[self.i]
            self.i += 1
            changed, self.last = state != self.last, state
            return VADResult(state, changed)

    # absolute (capture-clock) word spans: utterance 1 starts at frame 0, 2 at frame 12
    spans = {1: (1000.0, 0.25), 2: (1001.2, 0.3)}

    class Dispatcher:
        """Every window decodes immediately and reports exactly the words whose
        absolute time span lies inside the window's audio (timestamps relative
        to the window start, as faster-whisper reports them)."""

        def __init__(self):
            self.pending, self.n = [], 0

        def submit_window(self, window):
            self.pending.append(window)
            return True

        def poll_results(self):
            out, self.pending = self.pending, []
            hyps = []
            for w in out:
                self.n += 1
                start, width = spans[w.utterance_id]
                timings = tuple(
                    WordTiming(word=x, start=start + i * width - w.audio_start_time,
                               end=start + (i + 1) * width - w.audio_start_time)
                    for i, x in enumerate(words_by_utt[w.utterance_id])
                    if start + i * width >= w.audio_start_time - 1e-6
                    and start + (i + 1) * width <= w.audio_end_time + 1e-6)
                words = [t.word for t in timings]
                hyps.append(ASRHypothesis(
                    hypothesis_id=self.n, utterance_id=w.utterance_id, window_id=w.window_id, generation=w.generation,
                    submitted_at=0.0, started_at=0.0, completed_at=0.0, queue_wait_ms=0.0, segment_to_transcript_ms=0.0,
                    window_created_at=w.created_at, audio_start_time=w.audio_start_time, audio_end_time=w.audio_end_time,
                    window_duration_ms=w.duration_ms, context_duration_ms=w.context_duration_ms,
                    new_audio_duration_ms=w.new_audio_duration_ms, is_final=w.is_final, success=True, error=None,
                    text=" ".join(words), inference_time_ms=1.0, rtf=0.01, words=timings))
            return hyps

        def revision_count(self, uid):
            return 0

        def metrics_snapshot(self):
            return {}

        queue_wait_samples_ms = inference_time_samples_ms = rtf_samples = ()

    class Worker:
        def wait_until_idle(self, timeout_seconds):
            pass

        def stop(self, drain, timeout_seconds):
            pass

    config = Config(data={
        "system": {"name": "M2"},
        "features": {"use_adaptive_buffer": True, "use_local_agreement": True,
                     "use_stability_controller": False, "use_translation_stability": False},
        "audio": {"sample_rate": 16000, "channels": 1},
        "streaming": {"window_queue_size": 4, "window_queue_policy": "keep_latest"},
        "streaming_buffer": {"enabled": True, "min_new_audio_ms": 300.0, "max_buffer_ms": 5000.0, "pre_roll_ms": 0.0},
        "prefix_agreement": {"agreement_depth": 2, "prompt_word_limit": 200},
        "performance": {"warmup_asr": False, "warmup_translation": False},
    })
    logger = EventLogger(console=False, file_enabled=False, json_enabled=False)
    source = Source()
    clock = source.clock
    display = lp.Display(io.StringIO(), None, clock)
    engine = FakeEngine()
    worker = lp.MTWorker(engine, clock, display.translation)
    translator = lp.AsyncSegmentTranslator(worker, clock)
    controller = lp.SegmentController(EagerPolicy, translator, clock,
                                      lp.SegmentRules(close_after_words=50, idle_close_s=60, hard_max_words=60), display)
    worker.start()
    controller.start()

    with patch.object(main, "build_vad", return_value=VAD()), \
            patch.object(main, "build_asr_engine", return_value=type("E", (), {"model_load_time_ms": 0.0})()), \
            patch.object(main, "build_asr_worker", return_value=Worker()), \
            patch.object(main, "build_streaming_asr_dispatcher", return_value=Dispatcher()):
        exit_code, hook = lp.run_m2_with_translation(main, config, logger, source, controller, io.StringIO())

    controller.stop(clock())
    assert worker.shutdown(timeout_s=10)
    assert exit_code == 0 and controller.error is None
    committed = " ".join(hook.tracker.committed_text_for(u) for u in (1, 2))
    assert committed == "the quick brown fox jumps over"
    # one segment (idle/word rules not reached); final Hindi covers every committed word
    assert len(controller.segments) == 1
    assert " ".join(controller.segments[0].words) == committed
    assert display.finals[1] == f"HI({committed})"
    assert hook.buffer is not None and all(e["t"] >= 0 for e in hook.events)
