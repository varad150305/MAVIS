"""Durable per-session lifecycle evidence (phase 6).

A session folder used to get its environment, power mode and summary only
when the session ended, so a session that stalled or a process that was
killed left no record of how it was started or whether a stop was ever
requested. This module writes that evidence as it happens:

  session_start.json  written once, atomically, right after the folder is
                      created: session_id, environment (incl. on_ac_power),
                      arguments and the effective options
  lifecycle.jsonl     append-only, one JSON object per line, flushed and
                      fsynced per line: created, loading, ready,
                      stop_requested, audio_stall/audio_resumed,
                      cleanup_error, pipeline_finished, finalized/failed

Crash safety: JSON documents are written to a temporary file and moved into
place with os.replace, so a reader sees the old file or the new one, never
half of one; a lifecycle line is a single write, so at most the last line
can be torn - read_lifecycle() skips it instead of failing.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

LIFECYCLE_FILE = "lifecycle.jsonl"
START_FILE = "session_start.json"


def atomic_write_json(path: Path, obj: Any, **dump_kw: Any) -> None:
    """Write JSON so that `path` is either the previous file or the complete new one."""

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=str, **dump_kw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_lifecycle(path: Path) -> list[dict[str, Any]]:
    """Lifecycle records in order; a torn (unparsable) line is skipped."""

    out = []
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


class SessionRecorder:
    """Thread-safe writer of session_start.json and lifecycle.jsonl.

    Recording never raises into the pipeline: a failed write is counted in
    `write_errors` (and reported in the session summary)."""

    def __init__(self, session_dir: Path, session_id: str) -> None:
        self.session_dir = Path(session_dir)
        self.session_id = session_id
        self.path = self.session_dir / LIFECYCLE_FILE
        self._lock = threading.Lock()
        self._n = 0
        self._t0 = time.perf_counter()
        self.write_errors: list[str] = []

    def write_start(self, meta: dict[str, Any]) -> None:
        try:
            atomic_write_json(self.session_dir / START_FILE, {"session_id": self.session_id, **meta}, indent=1)
        except Exception as exc:  # noqa: BLE001 - evidence writing must not break the session
            self.write_errors.append(f"{START_FILE}: {type(exc).__name__}: {exc}")

    def record(self, event: str, **fields: Any) -> None:
        with self._lock:
            self._n += 1
            rec = {"session_id": self.session_id, "n": self._n, "event": event,
                   "wall": datetime.now().isoformat(timespec="milliseconds"),
                   "elapsed_s": round(time.perf_counter() - self._t0, 3), **fields}
            try:
                line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as exc:  # noqa: BLE001
                self.write_errors.append(f"{LIFECYCLE_FILE} {event}: {type(exc).__name__}: {exc}")


class StopSignal(threading.Event):
    """A threading.Event that remembers when and why a stop was requested and
    notifies subscribers at that moment (product.session uses it, so a stop
    requested during model loading is on disk before the pipeline sees it)."""

    def __init__(self) -> None:
        super().__init__()
        self.requested_wall: str | None = None
        self.reason: str | None = None
        self._subscribers: list[Callable[["StopSignal"], None]] = []
        self._sub_lock = threading.Lock()

    def set(self, reason: str = "stop requested") -> None:  # type: ignore[override]
        with self._sub_lock:
            first = not self.is_set()
            if first:
                self.requested_wall = datetime.now().isoformat(timespec="milliseconds")
                self.reason = reason
            super().set()
            subscribers = list(self._subscribers) if first else []
        for cb in subscribers:
            try:
                cb(self)
            except Exception:  # noqa: BLE001
                pass

    def subscribe(self, cb: Callable[["StopSignal"], None]) -> None:
        """Call `cb` on the first set(); immediately if already set."""

        with self._sub_lock:
            already = self.is_set()
            if not already:
                self._subscribers.append(cb)
        if already:
            try:
                cb(self)
            except Exception:  # noqa: BLE001
                pass
