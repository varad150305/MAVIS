"""
events.py — the output contract of the gesture controller.

  Event        — one state transition (immutable)
  to_dict()    — the ONLY place an event is turned into its wire format
  to_json()    — compact JSON line
  EventBus     — fans events out to any number of subscribers (callables)
  JsonlWriter  — subscriber: JSON Lines on a stream (stdout), background thread
  QueueSink    — subscriber: puts Event objects on a queue.Queue

Transports are just callables taking an Event, so MAVIS can later subscribe
a Python function or a queue instead of (or next to) reading stdout,
without touching the classifier or the state machine.

Wire format (schema_version 1), common fields on every event:
  schema_version, source, event, gesture, state, timestamp
plus event-specific fields (see EVENT_FIELDS and CLAUDE.md).
"""

from __future__ import annotations

import json
import queue
import sys
import threading
from dataclasses import dataclass, field
from typing import Callable

SCHEMA_VERSION = 1
SOURCE = "mavis_gesture"

SELECTION_PENDING = "feature.selection_pending"
SELECTION_CANCELLED = "feature.selection_cancelled"
START_REQUESTED = "feature.start_requested"
STOP_REQUESTED = "feature.stop_requested"
RETURN_TO_SELECTION = "navigation.return_to_selection"

# event name -> event-specific fields (always present, may be null)
EVENT_FIELDS = {
    SELECTION_PENDING: ("feature", "label", "previous_feature"),
    SELECTION_CANCELLED: ("feature", "reason"),
    START_REQUESTED: ("feature", "label"),
    STOP_REQUESTED: ("feature", "reason"),
    RETURN_TO_SELECTION: ("previous_feature", "reason"),
}

Subscriber = Callable[["Event"], None]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class Event:
    name: str                       # one of EVENT_FIELDS
    timestamp: float                # Unix time, seconds
    gesture: str | None             # gesture that caused it; None for timeouts/shutdown
    state: str                      # controller state AFTER the transition
    fields: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name not in EVENT_FIELDS:
            raise ValueError(f"unknown event {self.name!r}")
        extra = set(self.fields) - set(EVENT_FIELDS[self.name])
        if extra:
            raise ValueError(f"{self.name}: unexpected fields {sorted(extra)}")

    def get(self, key: str, default=None):
        return self.fields.get(key, default)


def to_dict(event: Event) -> dict:
    """Serializes an event. Change the wire format here and nowhere else."""
    d = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "event": event.name,
        "gesture": event.gesture,
        "state": event.state,
    }
    for key in EVENT_FIELDS[event.name]:
        d[key] = event.fields.get(key)
    d["timestamp"] = round(event.timestamp, 3)
    return d


def to_json(event: Event) -> str:
    return json.dumps(to_dict(event), separators=(",", ":"))


class EventBus:
    """Synchronous fan-out. A failing subscriber is logged and never breaks
    the recognition loop or the other subscribers."""

    def __init__(self):
        self._subs: list[Subscriber] = []

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        self._subs.append(fn)
        return lambda: self._subs.remove(fn) if fn in self._subs else None

    def publish(self, event: Event) -> None:
        for fn in list(self._subs):
            try:
                fn(event)
            except Exception as e:          # noqa: BLE001 — isolate consumers
                _log(f"[events] subscriber {getattr(fn, '__name__', fn)!r} failed: {e!r}")


class JsonlWriter:
    """Subscriber that writes one JSON line per event to `stream`.

    Writing happens on a background thread so a slow or blocked reader on
    the other end of a pipe never stalls the camera loop.
    """

    def __init__(self, stream=None, maxsize: int = 256):
        self._stream = stream if stream is not None else sys.stdout
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._t = threading.Thread(target=self._run, name="jsonl-writer", daemon=True)
        self._t.start()

    def __call__(self, event: Event) -> None:
        try:
            self._q.put_nowait(to_json(event))
        except queue.Full:
            _log(f"[events] output queue full, dropped {event.name}")

    def _run(self) -> None:
        while True:
            line = self._q.get()
            if line is None:
                return
            try:
                self._stream.write(line + "\n")
                self._stream.flush()
            except (BrokenPipeError, OSError, ValueError):
                return

    def close(self, timeout: float = 1.0) -> None:
        self._q.put(None)
        self._t.join(timeout)


class QueueSink:
    """Subscriber that puts Event objects on a queue (for a consumer thread)."""

    def __init__(self, q: queue.Queue | None = None):
        self.queue: queue.Queue = q if q is not None else queue.Queue()

    def __call__(self, event: Event) -> None:
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            _log(f"[events] consumer queue full, dropped {event.name}")


def stderr_logger(event: Event) -> None:
    """Subscriber: human-readable one-liner on stderr (preview mode)."""
    extra = " ".join(f"{k}={v}" for k, v in event.fields.items() if v is not None)
    _log(f"[event] {event.name}  gesture={event.gesture}  {extra}  -> {event.state}")
