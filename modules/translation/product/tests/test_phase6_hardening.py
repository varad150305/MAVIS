"""Phase 6: the four defects found by the live-mic artifact audit of UI
session 20261029-143813_m2_mic_ui (product only; no research code involved).

  1. hi_draft_after_final / final_mt_error bypassed the Display sink, so a
     host never received hi_draft_late / final_mt_error;
  2. nothing about a session (environment, power, stop request) was on disk
     until it ended;
  3. a microphone that stopped delivering frames was reported as "ready";
  4. an exception from the audio stream's stop()/close() - called first in
     main.run_m2's finally - would skip the ASR drain and the summary.
"""

from __future__ import annotations

import io
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from audio.audio_stream import AudioFrame
from product import live_pipeline as lp
from product import live_translate as lt
from product import session as S
from product.lifecycle import LIFECYCLE_FILE, START_FILE, StopSignal, atomic_write_json, read_lifecycle
from product.sources import LiveMicSource
from product.ui_model import TranscriptModel


# -- 1. event delivery -------------------------------------------------------------


def _call(seg, inv, text="good morning everyone"):
    return lp.ScheduledCall(segment_id=seg, invocation_id=inv, utterance_id=1, source_text=text,
                            source_word_count=len(text.split()), newly_stabilized_words=1,
                            source_words_already_processed=0, is_final=True, trigger_time=None,
                            source_available_time=None, enqueued_s=1.0)


def _result(seg, inv, text, t, error=None):
    return lp.CallResult(segment_id=seg, invocation_id=inv, source_word_count=3, target_text=text,
                         inference_ms=100.0, started_s=t - 0.1, completed_s=t, queue_wait_s=0.0, error=error)


def _display(tmp_path):
    """A real Display wired to a real TranslationSession sink (no pipeline)."""

    console = io.StringIO()
    session = S.TranslationSession(S.SessionOptions(wav="x.wav"))
    display = lp.Display(console, tmp_path / "transcript.jsonl", lambda: 9.0, sink=session._sink)
    seg = lp.Segment(segment_id=1, opened_s=0.0, rate_origin_s=0.0, words="good morning everyone".split(),
                     utterance_ids=[1], closed_s=5.0, close_reason="sentence end")
    display.english(seg, seg.words, 4.0, True, utterance_id=1)
    display.translation(_call(1, 1, "good morning"), _result(1, 1, "DRAFT-1", 4.5))
    return console, session, display, seg


def _types(session):
    return [e["type"] for e in session.poll_events()]


def test_transcript_file_exists_even_without_output(tmp_path):
    lp.Display(None, tmp_path / "transcript.jsonl", lambda: 0.0)
    assert (tmp_path / "transcript.jsonl").read_text(encoding="utf-8") == ""


def test_late_draft_reaches_the_host_but_never_replaces_the_final(tmp_path):
    console, session, display, seg = _display(tmp_path)
    display.segment_closed(seg, last_invocation_id=2, final_quality_pending=True)
    display.final_translation(_call(1, 99), _result(1, 99, "FINAL-NLLB", 5.6))       # final first
    display.translation(_call(1, 2), _result(1, 2, "LATE-DRAFT", 5.7))               # then the last draft
    events = session.poll_events()
    assert [e["type"] for e in events] == ["en_committed", "hi_draft", "segment_closed", "hi_final", "hi_draft_late"]
    late = events[-1]
    assert late["target_text"] == "LATE-DRAFT" and late["segment_id"] == 1 and "render_s" not in late
    assert display.finals[1] == "FINAL-NLLB" and display.final_kinds[1] == "final_mt"
    assert "LATE-DRAFT" not in console.getvalue()                                     # console unchanged: not printed
    lines = [json.loads(l) for l in (tmp_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [l["type"] for l in lines][-1] == "hi_draft_after_final"                   # transcript record kept
    m = TranscriptModel(session.session_id)
    for e in events:
        m.apply(e)
    assert m.segments[1].hindi_final == "FINAL-NLLB" and m.ignored_late_drafts == 1
    # a stray hi_draft after the final is also ignored by the model
    m.apply({**events[1], "seq": 99, "target_text": "STRAY"})
    assert m.segments[1].hindi_final == "FINAL-NLLB" and m.segments[1].hindi_draft == "DRAFT-1"


def test_final_mt_error_reaches_the_host_and_the_draft_becomes_the_final(tmp_path):
    console, session, display, seg = _display(tmp_path)
    display.segment_closed(seg, last_invocation_id=1, final_quality_pending=True)
    display.final_translation(_call(1, 99), _result(1, 99, "", 5.6, error="RuntimeError: ct2 failed"))
    events = session.poll_events()
    types = [e["type"] for e in events]
    assert types == ["en_committed", "hi_draft", "segment_closed", "final_mt_error", "hi_final"]
    assert events[3]["error"] == "RuntimeError: ct2 failed" and events[3]["segment_id"] == 1
    assert events[4]["kind"] == "draft_fallback" and events[4]["target_text"] == "DRAFT-1"
    assert "ct2 failed" not in console.getvalue()          # console behaviour unchanged (fallback note only)
    m = TranscriptModel(session.session_id)
    for e in events:
        m.apply(e)
    assert m.segments[1].hindi_final == "DRAFT-1" and any("ct2 failed" in x for x in m.errors)


# -- 2. lifecycle evidence ---------------------------------------------------------


def _write_wav(path: Path, seconds: float = 1.0) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(np.zeros(int(16000 * seconds), dtype="<i2").tobytes())
    return path


def _only_session_dir(root: Path) -> Path:
    dirs = [d for d in root.iterdir() if d.is_dir()]
    assert len(dirs) == 1, dirs
    return dirs[0]


def test_stop_during_model_loading_leaves_start_and_stop_evidence_on_disk(tmp_path, monkeypatch):
    loading, release = threading.Event(), threading.Event()

    def slow_loading_run_m2(args, config, session, console, source, product_cfg, event_sink=None, on_display=None):
        loading.set()
        assert release.wait(20)          # "models loading"; the stop arrives meanwhile
        source.start()
        try:
            while True:
                source.read()            # the stop is seen at the first frame, as in main.run_m2
        except KeyboardInterrupt:
            pass
        return {"run_exit_code": 0}

    monkeypatch.setattr(lt, "run_m2", slow_loading_run_m2)
    wav = _write_wav(tmp_path / "in.wav")
    out = tmp_path / "sessions"
    seen: list[dict] = []
    s = S.TranslationSession(S.SessionOptions(wav=str(wav), isolation="thread", overrides=[f"product.output_root={out.as_posix()}"]),
                             on_event=seen.append)
    s.start()
    try:
        assert loading.wait(30)
        assert s.stop(timeout_s=0.2) is False                       # still loading
        folder = _only_session_dir(out)
        start = json.loads((folder / START_FILE).read_text(encoding="utf-8"))
        assert start["session_id"] == s.session_id
        assert "on_ac_power" in start["environment"]
        assert start["effective_config"]["product"]["prompt_min_words"] == 3
        assert "--session-id" in start["argv"]
        events = [r["event"] for r in read_lifecycle(folder / LIFECYCLE_FILE)]
        assert events == ["created", "loading", "stop_requested"]   # durable before the pipeline sees the stop
        assert not (folder / "session_summary.json").exists()
    finally:
        release.set()
    assert s.stop(timeout_s=30) is True
    assert s.state is S.SessionState.STOPPED
    recs = read_lifecycle(folder / LIFECYCLE_FILE)
    assert [r["event"] for r in recs] == ["created", "loading", "stop_requested", "ready", "pipeline_finished",
                                          "finalized"]
    assert {r["session_id"] for r in recs} == {s.session_id} and [r["n"] for r in recs] == list(range(1, 7))
    assert recs[3]["stop_already_requested"] is True
    states = [e["state"] for e in seen if e["type"] == "session_state"]
    assert states == ["starting", "stopping", "stopping", "stopped"]       # never "ready" after "stopping"
    summary = json.loads((folder / "session_summary.json").read_text(encoding="utf-8"))
    assert summary["session_id"] == s.session_id and summary["stop_reason"] == "stop requested"
    assert summary["exit_code"] == 0 and summary["cleanup"] == {"ok": True, "errors": []}


def test_pipeline_exception_still_writes_a_summary_and_a_failed_record(tmp_path, monkeypatch):
    def broken_run_m2(*a, **k):
        raise RuntimeError("model file missing")

    monkeypatch.setattr(lt, "run_m2", broken_run_m2)
    out = tmp_path / "sessions"
    code, summary = lt.run_session(["--wav", str(_write_wav(tmp_path / "in.wav")),
                                    "--set", f"product.output_root={out.as_posix()}"], io.StringIO())
    folder = _only_session_dir(out)
    assert code == 1 and summary["error"] == "RuntimeError: model file missing"
    assert json.loads((folder / "session_summary.json").read_text(encoding="utf-8"))["exit_code"] == 1
    events = [r["event"] for r in read_lifecycle(folder / LIFECYCLE_FILE)]
    assert events == ["created", "loading", "pipeline_error", "pipeline_finished", "failed"]


def test_torn_lifecycle_line_is_skipped_and_json_writes_are_atomic(tmp_path):
    p = tmp_path / LIFECYCLE_FILE
    p.write_text('{"n": 1, "event": "created"}\n{"n": 2, "ev', encoding="utf-8")   # killed mid-write
    assert [r["event"] for r in read_lifecycle(p)] == ["created"]
    target = tmp_path / "session_summary.json"
    atomic_write_json(target, {"a": 1})
    atomic_write_json(target, {"a": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}
    assert not list(tmp_path.glob("*.tmp"))


def test_stop_signal_records_once_and_late_subscribers_still_hear_it():
    sig, heard = StopSignal(), []
    sig.subscribe(lambda s: heard.append(("early", s.reason)))
    sig.set("stop requested")
    sig.set("again")                                        # idempotent: no second notification
    sig.subscribe(lambda s: heard.append(("late", s.reason)))
    assert heard == [("early", "stop requested"), ("late", "stop requested")]
    assert sig.is_set() and sig.requested_wall


# -- 3. microphone stall -----------------------------------------------------------


class FakeStream:
    """AudioStream stand-in: read() pops from a script (None = no frame yet)."""

    chunk_duration_ms = 100

    def __init__(self, stop_exc=None, close_exc=None):
        self.script: list = []
        self.dropped_frames = 0
        self.stop_exc, self.close_exc = stop_exc, close_exc
        self.seq = 0

    def frame(self):
        self.seq += 1
        return AudioFrame(samples=np.zeros(1600, dtype=np.float32), capture_timestamp=time.perf_counter(),
                          duration_ms=100.0, sequence_number=self.seq, wall_timestamp_ns=time.time_ns(),
                          sample_rate=16000, channels=1)

    def start(self):
        pass

    def read(self):
        return self.script.pop(0) if self.script else None

    def stop(self):
        if self.stop_exc:
            raise self.stop_exc

    def close(self):
        if self.close_exc:
            raise self.close_exc


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _mic(stream, clock, health):
    src = LiveMicSource(stream, threading.Event(), idle_wait_s=0.0, stall_after_s=2.0, time_fn=clock)
    src.on_audio_health = health.append
    src.start()
    return src


def test_no_frames_gives_exactly_one_stall_warning_and_frames_give_recovery():
    stream, clock, health = FakeStream(), FakeClock(), []
    src = _mic(stream, clock, health)
    for _ in range(10):                                  # 1 s of normal (silent) audio: frames keep arriving
        stream.script.append(stream.frame())
        clock.t += 0.1
        assert src.read() is not None
    assert health == []
    for _ in range(50):                                  # 5 s of no frames at all (device gone)
        clock.t += 0.1
        assert src.read() is None
    assert [h["type"] for h in health] == ["audio_stall"]
    assert health[0]["no_frames_for_s"] == pytest.approx(2.0, abs=0.11) and health[0]["frames_read"] == 10
    stream.script.append(stream.frame())                 # device back
    clock.t += 0.1
    assert src.read() is not None
    assert [h["type"] for h in health] == ["audio_stall", "audio_resumed"]
    assert health[1]["stalled_for_s"] == pytest.approx(5.1, abs=0.01)
    for _ in range(15):                                  # a short gap below the threshold: nothing new
        clock.t += 0.1
        src.read()
    assert len(health) == 2
    report = src.stall_report()
    assert len(report) == 1 and report[0]["duration_s"] == pytest.approx(5.1, abs=0.01)
    assert "_t" not in report[0]


def test_no_frame_ever_after_start_is_a_stall_and_an_open_stall_is_reported():
    stream, clock, health = FakeStream(), FakeClock(), []
    src = _mic(stream, clock, health)
    for _ in range(30):
        clock.t += 0.1
        src.read()
    assert [h["type"] for h in health] == ["audio_stall"] and health[0]["frames_read"] == 0
    assert src.stall_report()[0]["end_s"] is None


def test_stall_events_drive_the_ui_model_and_label():
    m = TranscriptModel("A")
    ev = lambda seq, t, **f: {"schema": S.SCHEMA, "session_id": "A", "seq": seq, "type": t, **f}
    m.apply(ev(1, "session_state", state="ready", ready_in_s=7.0))
    assert m.audio_warning is None
    m.apply(ev(2, "audio_stall", no_frames_for_s=2.0, threshold_s=2.0, frames_read=10))
    assert "No audio from the microphone" in m.audio_warning and m.audio_stalls == 1
    m.apply(ev(3, "audio_resumed", stalled_for_s=5.1))
    assert m.audio_warning is None
    assert {"audio_stall", "audio_resumed", "cleanup_error"} <= set(S.EVENT_TYPES)
    assert S.RECORD_TYPES["audio_stall"] == "audio_stall"


_UI_CHECK = r"""
import tkinter as tk
from product import session as S
from product.ui.app import TranslatorApp
from product.ui_model import TranscriptModel

ev = lambda seq, t, **f: {"schema": S.SCHEMA, "session_id": "A", "seq": seq, "type": t, **f}

class FakeSession:
    state, running = S.SessionState.READY, True
    events = [ev(10, "audio_stall", no_frames_for_s=2.0)]
    def poll_events(self):
        out, self.events = self.events, []
        return out

root = tk.Tk()
root.withdraw()
app = TranslatorApp(root)
app.session, app.model = FakeSession(), TranscriptModel("A")
app._poll()
assert app.state_lbl.cget("text") == "NO AUDIO", app.state_lbl.cget("text")       # phase 9 status pill
assert "No audio from the microphone" in app.status.cget("text")
assert app.status.winfo_manager() == "pack"                                       # the banner is shown
app.session.events = [ev(11, "audio_resumed", stalled_for_s=3.0)]
app._poll()
assert app.state_lbl.cget("text") == "LISTENING"
assert "audio resumed" in app.status.cget("text")                                  # green recovery notice
root.destroy()
print("UI-OK")
"""


def test_ui_label_shows_no_audio_while_stalled():
    """Run in a child process: a second Tk root in one pytest process fails on
    this machine (tcl_findLibrary) and would silently skip the check."""

    import subprocess
    import sys

    pytest.importorskip("tkinter")
    root = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, "-c", _UI_CHECK], cwd=root, capture_output=True, text=True, timeout=60)
    if "TclError" in r.stderr and "display" in r.stderr:
        pytest.skip(f"no display: {r.stderr[-200:]}")
    assert r.returncode == 0 and "UI-OK" in r.stdout, r.stderr[-2000:]


# -- 4. stream shutdown ------------------------------------------------------------


def test_stream_stop_and_close_errors_are_recorded_not_raised():
    stream = FakeStream(stop_exc=OSError("PortAudio: device unavailable"), close_exc=OSError("close failed"))
    health = []
    src = _mic(stream, FakeClock(), health)
    src.stop()
    src.close()                                           # neither raises
    assert [e["op"] for e in src.cleanup_errors] == ["audio_stream.stop", "audio_stream.close"]
    assert "device unavailable" in src.cleanup_errors[0]["error"]
    assert "OSError" in src.cleanup_errors[0]["traceback"]
    assert [h["type"] for h in health] == ["cleanup_error", "cleanup_error"]


def test_session_whose_stream_stop_raises_still_writes_a_summary_with_the_error(tmp_path, monkeypatch):
    """Mic session with a fake device: frames, then the device vanishes (stall)
    and stop() raises. run_m2 stand-in keeps main.run_m2's order: loop until
    KeyboardInterrupt, then stop()/close() in `finally`, then drain."""

    stream = FakeStream(stop_exc=OSError("PortAudio: device unavailable"))
    stream.script = [stream.frame() for _ in range(5)]
    drained = []

    def run_m2_like_main(args, config, session, console, source, product_cfg, event_sink=None, on_display=None):
        display = lp.Display(console, session / "transcript.jsonl", source.clock, sink=event_sink)
        on_display(display)
        source.start()
        try:
            while True:
                if source.read() is None and source.stalls and not source.stop_event.is_set():
                    source.stop_event.set()              # the user presses Stop after the warning
        except KeyboardInterrupt:
            pass
        finally:
            source.stop()
            source.close()
            drained.append(True)                         # the drain still runs
        return {"run_exit_code": 0}

    import audio.audio_stream
    import product.sources

    monkeypatch.setattr(lt, "run_m2", run_m2_like_main)
    monkeypatch.setattr(product.sources, "check_input_device", lambda *a, **k: "Fake mic")
    monkeypatch.setattr(audio.audio_stream, "AudioStream", lambda **kw: stream)
    out = tmp_path / "sessions"
    events = []
    s = S.TranslationSession(S.SessionOptions(isolation="thread", overrides=[f"product.output_root={out.as_posix()}",
                                                         "product.audio_stall_s=0.3"]), on_event=events.append)
    s.start()
    s._thread.join(30)                                   # the stand-in presses Stop after the stall
    assert drained == [True] and s.state is S.SessionState.STOPPED
    folder = _only_session_dir(out)
    summary = json.loads((folder / "session_summary.json").read_text(encoding="utf-8"))
    assert summary["session_id"] == s.session_id
    assert summary["cleanup"]["ok"] is False
    assert summary["cleanup"]["errors"][0]["op"] == "audio_stream.stop"
    assert "device unavailable" in summary["cleanup"]["errors"][0]["traceback"]
    assert len(summary["audio"]["stalls"]) == 1 and summary["audio"]["frames_consumed"] == 5
    lifecycle = [r["event"] for r in read_lifecycle(folder / LIFECYCLE_FILE)]
    assert "audio_stall" in lifecycle and "cleanup_error" in lifecycle and lifecycle[-1] == "finalized"
    types = [e["type"] for e in events]
    assert types.count("audio_stall") == 1 and "cleanup_error" in types
    stopped = [e for e in events if e["type"] == "session_state" and e["state"] == "stopped"]
    assert stopped and stopped[0]["cleanup_ok"] is False
    transcript = [json.loads(l)["type"] for l in (folder / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "audio_stall" in transcript and "cleanup_error" in transcript
    assert all(e["session_id"] == s.session_id for e in events)
