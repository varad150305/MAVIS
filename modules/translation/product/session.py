"""Product-owned session API for live English -> Hindi translation.

This is the seam a host application (the standalone UI in product/ui, or
later the MAVIS shell) uses instead of the command line. It runs the
unchanged product pipeline (product.live_translate.run_session ->
frozen main.run_m2 + product MT layer) in a worker thread and publishes
structured events; nothing parses console text.

    from product.session import TranslationSession, SessionOptions

    session = TranslationSession(SessionOptions(wav="talk.wav"))
    session.start()                      # returns at once; models load in the worker
    for event in session.iter_events():  # or session.poll_events() from a UI timer
        ...                              # see EVENT CONTRACT below
    session.stop()                       # drains ASR/MT, closes the last segment
    summary = session.summary            # the same session_summary.json content

Phase 7: by default every session runs in its own worker process
(product.session_worker, ProcessRunner below); the worker exits once the
session's files are finalized, so its model memory is returned to the OS.
Run in-process, a finished session leaves ~1.3 GB of native memory behind
although no Python model object survives (product_outputs/diagnostics/phase7).
`isolation="thread"` keeps the Phase 5/6 in-process runner.

One session at a time per host: in-process, run_m2 is driven through module
patches and a stdout redirect (product.live_pipeline.run_m2_with_translation);
with workers, the microphone and CPU are still shared. A second start()
while one runs raises SessionBusy.

EVENT CONTRACT (schema "mavis.live_translation.v1"); every event is a dict:
  schema, session_id   identifies the session; a host must drop events whose
                       session_id is not the current one (no stale events)
  seq                  1, 2, 3 ... per session, in publication order
  type                 one of EVENT_TYPES
  t_s                  pipeline event time, session seconds (input audio
                       seconds since frame 0), or null for lifecycle events
  render_s             (display events) session clock when the CLI console
                       line was written; event time and render time differ by
                       the controller queue and console write
Types and their fields:
  session_state   state: starting|ready|stopping|stopped|failed; ready_in_s
                  (ready), stop_reason, cleanup_ok (stopped), error (failed),
                  info (starting/ready)
  en_provisional  utterance_id, text           (opt-in; unconfirmed, may change,
                                                never translated)
  en_committed    segment_id, utterance_id, new_text, stable_text,
                  utterance_final              (LocalAgreement-confirmed words)
  hi_draft        segment_id, invocation_id, source_text, source_word_count,
                  target_text, inference_ms, completed_s   (provisional Hindi)
  hi_draft_late   same fields: a draft that finished after the segment's final;
                  recorded only - a host must NOT show it
  hi_final        segment_id, target_text, kind (final_mt | draft |
                  draft_fallback)              (the segment's one final Hindi)
  segment_closed  segment_id, reason, words, utterance_ids, final_source
  mt_error / final_mt_error   segment_id, error
  asr_decode      utterance_id, window_id, is_final, success, inference_ms, rtf,
                  queue_wait_ms, completion_lag_s
  utterance_timing utterance_id, onset_s, first_text_hypothesis_s,
                  last_word_end_s, vad_end_s, first_commit_s (latest values)
  resources       cores_busy (this process), working_set_mb, dropped_frames
  status          message (console status line)
  audio_stall     no_frames_for_s, threshold_s, frames_read: the microphone
                  delivered NO frames for threshold_s (not silence - quiet
                  audio still arrives as frames); emitted once per stall
  audio_resumed   stalled_for_s, frames_read: frames arrive again
  cleanup_error   op, error: an audio stream stop/close failed (e.g. device
                  removed); the session still drains and writes its summary,
                  with summary.cleanup.ok = false
  session_summary summary (dict = session_summary.json)
  error           message, where (+ traceback); where="worker" when the worker
                  process failed or exited before finishing the session
Association rules the pipeline guarantees (and product/ui_model.py checks):
each hi_draft/hi_final carries the segment_id of the English it translates;
a segment gets exactly one hi_final, after its segment_closed; no hi_draft
is published as hi_draft after the hi_final of its segment.
"""

from __future__ import annotations

import collections
import io
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import traceback
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator

from product.lifecycle import StopSignal

SCHEMA = "mavis.live_translation.v1"
EVENT_TYPES = ("session_state", "en_provisional", "en_committed", "hi_draft", "hi_draft_late", "hi_final",
               "segment_closed", "mt_error", "final_mt_error", "asr_decode", "utterance_timing", "resources",
               "status", "session_summary", "error", "audio_stall", "audio_resumed", "cleanup_error")
#: transcript.jsonl record type -> event type (the transcript keeps its own names)
RECORD_TYPES = {"en_update": "en_committed", "en_provisional": "en_provisional", "hi_draft": "hi_draft",
                "hi_draft_after_final": "hi_draft_late", "hi_final": "hi_final", "segment_closed": "segment_closed",
                "mt_error": "mt_error", "final_mt_error": "final_mt_error", "asr_decode": "asr_decode",
                "utterance_timing": "utterance_timing", "resources": "resources", "status": "status",
                "audio_stall": "audio_stall", "audio_resumed": "audio_resumed", "cleanup_error": "cleanup_error"}


class SessionState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class SessionBusy(RuntimeError):
    """Another session is running in this process."""


class FeatureDisabled(RuntimeError):
    """The host feature flag product.integration.live_translation_enabled is off."""


class WorkerFailed(RuntimeError):
    """The session's worker process failed or exited before finishing."""


REPO_ROOT = Path(__file__).resolve().parent.parent


class ProcessRunner:
    """Runs one session in a product.session_worker process (run_session
    contract: argv, console, event_sink, stop_event, ready_callback ->
    (exit code, summary)). Records, ready, console text and the result come
    over the worker's stdout as JSON lines, in the order the worker produced
    them; a stop is forwarded on its stdin the moment it is requested.
    Returns only after the worker has exited."""

    def __init__(self, command: list[str] | None = None) -> None:
        self.command = command or [sys.executable, "-m", "product.session_worker"]
        self.proc: subprocess.Popen | None = None
        self.stderr_tail: collections.deque[str] = collections.deque(maxlen=60)
        self.session_dir: str | None = None
        self.terminated = False
        self._stdin_lock = threading.Lock()

    def _send_stop(self) -> None:
        with self._stdin_lock:
            try:
                if self.proc is not None and self.proc.stdin is not None and not self.proc.stdin.closed:
                    self.proc.stdin.write("stop\n")
                    self.proc.stdin.flush()
            except (OSError, ValueError):
                pass                      # the worker is already gone; the read loop reports it

    def _drain_stderr(self) -> None:
        for line in self.proc.stderr:
            self.stderr_tail.append(line.rstrip("\n"))

    def terminate(self) -> None:
        """Last resort (e.g. the host is closing and the drain timed out)."""

        if self.proc is not None and self.proc.poll() is None:
            self.terminated = True
            self.proc.kill()

    def __call__(self, argv, console, *, event_sink, stop_event, ready_callback):
        # same working directory as the host (relative --wav/--config resolve as in-process)
        path = os.pathsep.join(p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH")) if p)
        # handoff: the worker runs with the package root as its working
        # directory (`python -m` puts the cwd first on sys.path, so a host
        # running elsewhere can never shadow `product`); relative --wav /
        # --config paths are made absolute against the host's cwd first.
        argv = list(argv)
        for i, a in enumerate(argv[:-1]):
            if a in ("--wav", "--config"):
                argv[i + 1] = os.path.abspath(argv[i + 1])
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "PYTHONPATH": path}
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(self.command + list(argv), cwd=str(REPO_ROOT), env=env, creationflags=flags,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=self._drain_stderr, name="session-worker-stderr", daemon=True).start()
        if hasattr(stop_event, "subscribe"):
            stop_event.subscribe(lambda _sig: self._send_stop())
        else:
            threading.Thread(target=lambda: (stop_event.wait(), self._send_stop()), daemon=True).start()
        result = None
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self.stderr_tail.append("non-protocol output: " + line.rstrip()[:200])
                continue
            kind = msg.get("k")
            if kind == "rec":
                event_sink(msg["r"])
            elif kind == "ready":
                ready_callback(msg["ready_in_s"], msg["info"])
            elif kind == "console":
                text = msg.get("text", "")
                if text.startswith("Session folder: "):
                    self.session_dir = text[len("Session folder: "):].strip()
                console.write(text)
            elif kind in ("result", "fatal"):
                result = msg
        returncode = self.proc.wait()
        with self._stdin_lock:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        where = f" (session folder {self.session_dir})" if self.session_dir else ""
        tail = "\n".join(list(self.stderr_tail)[-15:])
        if result is None:
            why = "was terminated" if self.terminated else f"exited with code {returncode}"
            raise WorkerFailed(f"translation worker {why} before finishing the session{where}"
                               + (f"; last output:\n{tail}" if tail else ""))
        if result["k"] == "fatal":
            raise WorkerFailed(f"translation worker error: {result['error']}{where}\n{result.get('traceback', '')}")
        return result["code"], result["summary"]


@dataclass
class SessionOptions:
    """What a host can choose; everything else comes from the product config."""

    wav: str | None = None                  # replay a 16 kHz mono WAV at 1x; None = microphone
    device: str | int | None = None         # microphone (index or name); None = config/default
    save_audio: bool = False
    show_provisional_english: bool = False  # opt-in, labelled provisional
    session_name: str | None = None
    config: str | None = None               # default product/configs/live_m2_mt3c.yaml
    overrides: list[str] = field(default_factory=list)   # extra KEY=VALUE (as --set)
    duration_s: float | None = None
    #: "process" (default): the session runs in its own worker process, whose
    #: memory is returned to the OS when it ends; "thread": in this process
    isolation: str = "process"

    def argv(self) -> list[str]:
        args: list[str] = []
        if self.wav:
            args += ["--wav", str(self.wav)]
        if self.device is not None:
            args += ["--device", str(self.device)]
        if self.save_audio:
            args.append("--save-audio")
        if self.session_name:
            args += ["--session-name", self.session_name]
        if self.config:
            args += ["--config", self.config]
        if self.duration_s is not None:
            args += ["--duration", str(self.duration_s)]
        sets = list(self.overrides)
        if self.show_provisional_english:
            sets.append("product.display_provisional_en=true")
        for s in sets:
            args += ["--set", s]
        return args


_ACTIVE = threading.Lock()  # one running session per process


def feature_enabled(config_path: str | None = None) -> bool:
    """Host feature flag (default OFF): product.integration.live_translation_enabled."""

    from pathlib import Path

    from product.live_translate import DEFAULT_CONFIG
    from utils.config import load_config

    config = load_config(Path(config_path) if config_path else DEFAULT_CONFIG)
    return bool(((config.get("product", default={}) or {}).get("integration") or {}).get("live_translation_enabled"))


def create_host_session(options: SessionOptions | None = None, *, on_event: Callable[[dict], None] | None = None
                        ) -> "TranslationSession":
    """Entry point for a host application: refuses while the feature flag is off."""

    options = options or SessionOptions()
    if not feature_enabled(options.config):
        raise FeatureDisabled("set product.integration.live_translation_enabled: true to expose live translation")
    return TranslationSession(options, on_event=on_event)


class TranslationSession:
    def __init__(self, options: SessionOptions | None = None, *, on_event: Callable[[dict], None] | None = None,
                 runner: Callable[..., tuple[int, dict]] | None = None) -> None:
        self.options = options or SessionOptions()
        self.session_id = uuid.uuid4().hex[:12]
        self.state = SessionState.IDLE
        self.summary: dict[str, Any] | None = None
        self.exit_code: int | None = None
        self.error: str | None = None
        self._on_event = on_event
        self._runner = runner  # tests inject a fake pipeline
        self._worker: ProcessRunner | None = None
        self._events: queue.Queue[dict] = queue.Queue()
        self._seq = itertools.count(1)
        self._seq_lock = threading.Lock()
        #: records when the stop was requested; run_session writes it to the
        #: session's lifecycle.jsonl at that moment (also during model loading)
        self._stop = StopSignal()
        self._thread: threading.Thread | None = None
        self._holds_lock = False
        self.console = io.StringIO()  # the CLI's console lines (for logs/debugging only)

    # ------------------------------------------------------------------ events
    def _publish(self, type_: str, **fields: Any) -> dict:
        with self._seq_lock:
            event = {"schema": SCHEMA, "session_id": self.session_id, "seq": next(self._seq), "type": type_,
                     "t_s": fields.pop("t_s", None), **fields}
            self._events.put(event)
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:  # a host callback must never break the pipeline
                pass
        return event

    def _sink(self, record: dict) -> None:
        """Display/hook record -> event (called from pipeline threads)."""

        type_ = RECORD_TYPES.get(record.get("type"))
        if type_ is None:
            return
        fields = {k: v for k, v in record.items() if k != "type"}
        self._publish(type_, **fields)

    def poll_events(self, max_events: int = 1000) -> list[dict]:
        """Non-blocking: events published since the last poll (UI timers)."""

        out = []
        while len(out) < max_events:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                break
        return out

    def iter_events(self, timeout_s: float = 0.2) -> Iterator[dict]:
        """Blocking iterator that ends after the session has finished and
        every event was delivered."""

        while True:
            try:
                yield self._events.get(timeout=timeout_s)
            except queue.Empty:
                if self.state in (SessionState.STOPPED, SessionState.FAILED) and self._events.empty() and \
                        (self._thread is None or not self._thread.is_alive()):
                    return

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.state is not SessionState.IDLE:
            raise RuntimeError(f"a TranslationSession starts once (state {self.state.value}); create a new one")
        if not _ACTIVE.acquire(blocking=False):
            raise SessionBusy("another live-translation session is running in this process")
        self._holds_lock = True
        self.state = SessionState.STARTING
        self._publish("session_state", state="starting",
                      info={"source": self.options.wav or "microphone", "device": self.options.device,
                            "show_provisional_english": self.options.show_provisional_english})
        self._thread = threading.Thread(target=self._run, name=f"live-translation-{self.session_id}", daemon=True)
        self._thread.start()

    def _on_ready(self, ready_in_s: float, info: dict) -> None:
        if self.state is SessionState.STARTING:
            self.state = SessionState.READY
            self._publish("session_state", state="ready", ready_in_s=ready_in_s, info=info)
        else:
            # stopped while models were loading: capture starts only to deliver
            # the stop - never report "ready" after "stopping"
            self._publish("session_state", state=self.state.value, ready_in_s=ready_in_s, info=info)

    def _run(self) -> None:
        try:
            runner = self._runner
            if runner is None and self.options.isolation == "thread":
                from product.live_translate import run_session as runner
            elif runner is None:
                runner = ProcessRunner()
            if isinstance(runner, ProcessRunner):
                self._worker = runner
            argv = self.options.argv() + ["--session-id", self.session_id]   # same id in every session file
            code, summary = runner(argv, self.console, event_sink=self._sink,
                                   stop_event=self._stop, ready_callback=self._on_ready)
            self.exit_code, self.summary = code, summary
            if summary is not None:
                self._publish("session_summary", summary=summary)
            if code == 0:
                self.state = SessionState.STOPPED
                self._publish("session_state", state="stopped",
                              stop_reason=(summary or {}).get("stop_reason"),
                              cleanup_ok=((summary or {}).get("cleanup") or {}).get("ok"))
            else:
                self.state = SessionState.FAILED
                self.error = (summary or {}).get("error") or (summary or {}).get("controller_error") \
                    or f"pipeline exit code {code}"
                self._publish("session_state", state="failed", error=self.error)
        except BaseException as exc:  # noqa: BLE001 - report every failure as an event
            self.state = SessionState.FAILED
            worker = isinstance(exc, WorkerFailed)
            self.error = str(exc).split("\n", 1)[0] if worker else f"{type(exc).__name__}: {exc}"
            self._publish("error", message=self.error, where="worker" if worker else "session",
                          traceback=str(exc) if worker else traceback.format_exc())
            self._publish("session_state", state="failed", error=self.error)
        finally:
            if self._holds_lock:
                self._holds_lock = False
                _ACTIVE.release()

    def stop(self, timeout_s: float | None = 120.0) -> bool:
        """Request a stop (drain + finalize) and wait for it. True once the
        worker has exited. Safe to call more than once and before ready
        (the pipeline then stops at its first audio frame)."""

        if self._thread is None:
            self.state = SessionState.STOPPED
            return True
        if self.state in (SessionState.STARTING, SessionState.READY):
            self.state = SessionState.STOPPING
            self._publish("session_state", state="stopping")
        self._stop.set()
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def terminate(self) -> None:
        """Kill the worker process (last resort, e.g. after a stop() timeout
        when the host closes). The session then ends as failed; its files are
        whatever the worker had written (lifecycle.jsonl shows no finalized)."""

        if self._worker is not None:
            self._worker.terminate()

    @property
    def worker_pid(self) -> int | None:
        return self._worker.proc.pid if self._worker is not None and self._worker.proc is not None else None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
