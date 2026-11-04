"""Phase 7: each session in its own worker process (product.session_worker,
product.session.ProcessRunner). Fast tests use a scripted fake worker that
speaks the same JSON-lines protocol; the real worker is exercised by
test_phase5_session.test_real_pipeline_through_session_api_on_a_saved_recording
and product_outputs/diagnostics/phase7/acceptance.py."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from product import session as S
from product.ui_model import TranscriptModel

REPO = Path(__file__).resolve().parents[2]

FAKE_WORKER = textwrap.dedent(r'''
    import json, sys, time
    mode = sys.argv[1]
    out = sys.stdout
    def send(**m):
        out.write(json.dumps(m) + "\n"); out.flush()
    send(k="console", text="Session folder: fake/session\n")
    if mode == "exit_loading":
        sys.stderr.write("model file is corrupt\n"); sys.exit(3)
    send(k="ready", ready_in_s=0.5, info={"on_ac_power": True})
    print("stray native output that is not protocol")          # must be ignored
    send(k="rec", r={"type": "en_update", "segment_id": 1, "utterance_id": 1, "t_s": 1.0, "new_text": "hello",
                     "stable_text": "hello", "utterance_final": True, "render_s": 1.0})
    send(k="rec", r={"type": "segment_closed", "segment_id": 1, "t_s": 1.2, "reason": "sentence end",
                     "utterance_ids": [1], "final_source": "hello", "render_s": 1.2})
    send(k="rec", r={"type": "hi_final", "segment_id": 1, "t_s": 1.5, "target_text": "FINAL", "kind": "final_mt",
                     "render_s": 1.5})
    send(k="rec", r={"type": "hi_draft_after_final", "segment_id": 1, "invocation_id": 2, "target_text": "LATE"})
    if mode == "crash":
        sys.stderr.write("Fatal Python error: access violation\n"); sys.stderr.flush(); sys.exit(-1073741819 & 0xFF)
    if mode == "fatal":
        send(k="fatal", error="RuntimeError: ct2 init failed", traceback="Traceback ...\nRuntimeError"); sys.exit(1)
    if mode == "hang":
        time.sleep(600)
    line = sys.stdin.readline()                                   # wait for the stop
    send(k="result", code=0, summary={"stop_reason": "stop requested" if line.strip() == "stop" else line,
                                      "argv": sys.argv[2:], "cleanup": {"ok": True, "errors": []}})
''')


@pytest.fixture()
def worker_script(tmp_path):
    p = tmp_path / "fake_worker.py"
    p.write_text(FAKE_WORKER, encoding="utf-8")
    return p


def _session(worker_script, mode, events):
    runner = S.ProcessRunner(command=[sys.executable, str(worker_script), mode])
    s = S.TranslationSession(S.SessionOptions(wav="x.wav"), runner=runner, on_event=events.append)
    s.start()
    return s


def _wait_ready(s, events, timeout=20):
    import time

    t0 = time.time()
    while time.time() - t0 < timeout:
        if any(e["type"] == "hi_draft_late" for e in events) or not s.running:
            return
        time.sleep(0.02)
    raise AssertionError("worker produced nothing")


def test_worker_events_keep_order_session_id_and_final_immutability(worker_script):
    events: list[dict] = []
    s = _session(worker_script, "normal", events)
    _wait_ready(s, events)
    assert s.state is S.SessionState.READY and s.worker_pid is not None
    assert s.stop(timeout_s=20) is True and s.stop(timeout_s=1) is True            # repeatable
    assert s.state is S.SessionState.STOPPED
    assert s.summary["stop_reason"] == "stop requested"                            # stop forwarded on stdin
    assert s.summary["argv"][-2:] == ["--session-id", s.session_id]
    types = [e["type"] for e in events]
    assert types == ["session_state", "session_state", "en_committed", "segment_closed", "hi_final", "hi_draft_late",
                     "session_state", "session_summary", "session_state"]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert {e["session_id"] for e in events} == {s.session_id}
    assert s.console.getvalue().startswith("Session folder: fake/session")
    m = TranscriptModel(s.session_id)
    for e in events:
        m.apply(e)
    m.apply({**events[2], "session_id": "someone-else", "seq": 999})               # other session: rejected
    assert m.segments[1].hindi_final == "FINAL" and m.ignored_late_drafts == 1 and m.ignored_stale == 1
    assert m.state == "stopped"
    assert s._worker.proc.poll() is not None


@pytest.mark.parametrize("mode, expect", [
    ("crash", "exited with code"),
    ("exit_loading", "exited with code 3"),
    ("fatal", "RuntimeError: ct2 init failed"),
])
def test_worker_failure_or_premature_exit_becomes_a_failed_state(worker_script, mode, expect):
    events: list[dict] = []
    s = _session(worker_script, mode, events)
    s._thread.join(20)
    assert not s.running and s.state is S.SessionState.FAILED
    assert expect in s.error and "fake/session" in s.error                          # useful, names the folder
    err = [e for e in events if e["type"] == "error"]
    assert err and err[0]["where"] == "worker"
    if mode == "exit_loading":
        assert "model file is corrupt" in err[0]["traceback"]                       # worker stderr tail kept
    m = TranscriptModel(s.session_id)
    for e in events:
        m.apply(e)
    assert m.state == "failed" and any(expect in x for x in m.errors)                # UI never stays "ready"
    s2 = S.TranslationSession(S.SessionOptions(wav="x.wav"),
                              runner=S.ProcessRunner(command=[sys.executable, str(worker_script), "normal"]))
    s2.start()                                                                        # lock was released
    assert s2.stop(timeout_s=20) and s2.state is S.SessionState.STOPPED


def test_hung_worker_can_be_terminated_and_ends_failed(worker_script):
    events: list[dict] = []
    s = _session(worker_script, "hang", events)
    _wait_ready(s, events)
    assert s.stop(timeout_s=0.5) is False and s.state is S.SessionState.STOPPING
    s.terminate()
    assert s.stop(timeout_s=20) is True
    assert s.state is S.SessionState.FAILED and "was terminated" in s.error


def test_ui_process_does_not_load_the_pipeline_or_torch():
    code = ("import sys; import product.ui.app, product.session; from product.metrics_panel import LiveMetrics; "
            "LiveMetrics('A').rows(); print('torch' in sys.modules, 'product.live_pipeline' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["False", "False"]


def test_default_isolation_is_a_worker_process():
    assert S.SessionOptions().isolation == "process"
    assert S.ProcessRunner().command[-2:] == ["-m", "product.session_worker"]


FAKE_RUN_SESSION = textwrap.dedent(r'''
    import subprocess, sys

    def run_session(argv, console, *, event_sink, stop_event, ready_callback):
        console.write("Session folder: fake/real-worker\n")
        # what platform.processor() does on Windows during environment_info():
        # a subprocess inheriting the standard handles while the stop channel is read
        subprocess.run([sys.executable, "-c", "print('child')"], capture_output=False, timeout=30)
        print("native-style noise on stdout")
        ready_callback(0.1, {"on_ac_power": None})
        event_sink({"type": "status", "t_s": 0.2, "message": "hello", "render_s": 0.2})
        assert stop_event.wait(30)
        return 0, {"stop_reason": stop_event.reason, "argv": argv}
''')


def test_real_worker_survives_subprocess_spawn_and_forwards_stop(tmp_path, monkeypatch):
    """Regression (phase 7): the worker deadlocked when a subprocess was started
    while its stop channel (stdin) was being read; it never loaded the models."""

    (tmp_path / "fake_run_session.py").write_text(FAKE_RUN_SESSION, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("MAVIS_SESSION_WORKER_RUNNER", "fake_run_session:run_session")
    events: list[dict] = []
    s = S.TranslationSession(S.SessionOptions(wav="x.wav"), on_event=events.append)   # default: real worker
    s.start()
    import time

    t0 = time.time()
    while not any(e["type"] == "status" for e in events) and time.time() - t0 < 60:
        time.sleep(0.05)
    assert any(e["type"] == "status" for e in events), s.error
    assert s.stop(timeout_s=30) is True and s.state is S.SessionState.STOPPED
    assert s.summary["stop_reason"] == "stop requested"
    assert "--session-id" in s.summary["argv"]
