"""Worker process for one translation session (phase 7).

    python -m product.session_worker <run_session arguments...>

product.session starts one of these per session (the default runner), so all
model memory is returned to the OS when the session's process exits: an
in-process session leaves ~1.3 GB private memory behind even after every
Python model object is gone (native allocator / library retention, see
product_outputs/diagnostics/phase7/).

Protocol (one JSON object per line):
  parent -> child, stdin   "stop"  request a drain-safe stop; EOF (the parent
                           went away) counts as a stop too
  child -> parent, the protocol pipe (the original stdout; fd 1 itself is
  pointed at stderr so stray native/print output cannot corrupt it):
    {"k": "rec", "r": record}             event_sink record
    {"k": "ready", "ready_in_s", "info"}  ready_callback
    {"k": "console", "text"}              CLI console text (debugging/logs)
    {"k": "result", "code", "summary"}    run_session returned; the session's
                                          files are finalized at this point
    {"k": "fatal", "error", "traceback"}  run_session raised
The child exits right after "result"/"fatal"; anything else (no result
line) is reported by the parent as a worker crash.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback


class _Channel:
    def __init__(self, stream) -> None:
        self._stream, self._lock = stream, threading.Lock()

    def send(self, msg: dict) -> None:
        line = json.dumps(msg, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            self._stream.write(line)
            self._stream.flush()


class _ConsoleToChannel:
    def __init__(self, channel: _Channel) -> None:
        self.channel = channel

    def write(self, text: str) -> int:
        if text:
            self.channel.send({"k": "console", "text": text})
        return len(text)

    def flush(self) -> None:
        pass


def main(argv: list[str]) -> int:
    proto_fd = os.dup(1)                 # the pipe to the parent (os.dup: not inheritable)
    os.dup2(2, 1)                        # fd 1 -> stderr: native writes can't reach the protocol
    sys.stdout = sys.stderr
    channel = _Channel(os.fdopen(proto_fd, "w", encoding="utf-8", newline="\n"))
    # The stop channel is read from a private, non-inheritable copy of stdin;
    # fd 0 itself becomes NUL. On Windows a subprocess started during the
    # session (platform.processor() runs one) duplicates the standard handles,
    # and duplicating a pipe handle that another thread is blocked reading
    # deadlocks - observed in phase 7: the worker hung before loading models.
    control = os.fdopen(os.dup(0), "r", encoding="utf-8")
    null_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null_fd, 0)
    os.close(null_fd)
    sys.stdin = open(os.devnull, "r", encoding="utf-8")

    from product.lifecycle import StopSignal

    stop = StopSignal()

    def watch_stdin() -> None:
        for line in control:
            if line.strip() == "stop":
                stop.set("stop requested")
                return
        stop.set("parent gone (worker stdin closed)")

    threading.Thread(target=watch_stdin, name="session-worker-stdin", daemon=True).start()
    try:
        from product.live_translate import run_session

        if os.environ.get("MAVIS_SESSION_WORKER_RUNNER"):      # tests only: "module:function"
            import importlib

            module, _, name = os.environ["MAVIS_SESSION_WORKER_RUNNER"].partition(":")
            run_session = getattr(importlib.import_module(module), name)

        code, summary = run_session(argv, _ConsoleToChannel(channel),
                                    event_sink=lambda r: channel.send({"k": "rec", "r": r}),
                                    stop_event=stop,
                                    ready_callback=lambda s, info: channel.send({"k": "ready", "ready_in_s": s,
                                                                                 "info": info}))
        channel.send({"k": "result", "code": code, "summary": summary})
        return 0
    except BaseException as exc:  # noqa: BLE001 - report every failure to the parent
        channel.send({"k": "fatal", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.stderr.flush()
    os._exit(exit_code)                # do not wait for daemon/library threads; all output is flushed
