"""
feature_controller.py — MAVIS feature-selection state machine.

                 1/2/3 fingers                     thumbs up
  SELECTING_FEATURE ───────────► AWAITING_CONFIRMATION ───────────► FEATURE_ACTIVE
        ▲   ▲                        │  ▲  │ 1/2/3 fingers: replace     │
        │   │  open palm / timeout   │  └──┘ pending selection          │
        │   └────────────────────────┘                                  │
        │                        open palm (stop requested)             │
        └───────────────────────────────────────────────────────────────┘

Input: gesture *events* from the stabilizer — each continuous presentation
of a gesture arrives exactly once, already debounced and held for the
configured hold time. Hand loss never reaches this class (the stabilizer
turns it into "no event"), so losing the hand can never cancel, confirm or
stop anything.

Output: Event objects passed to `emit` (normally EventBus.publish) and also
returned from every call. This class performs no I/O and starts nothing;
"start" and "stop" are *requests* for an external application.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Callable

import events as ev
from gesture_mapping import GestureMapping


class State(str, Enum):
    SELECTING_FEATURE = "SELECTING_FEATURE"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    FEATURE_ACTIVE = "FEATURE_ACTIVE"


class FeatureController:
    def __init__(self, mapping: GestureMapping,
                 emit: Callable[[ev.Event], None] | None = None):
        self.mapping = mapping
        self._emit = emit
        self.state = State.SELECTING_FEATURE
        self.pending_feature: str | None = None
        self.pending_since = 0.0
        self.active_feature: str | None = None
        # for the preview: last transition and last ignored gesture
        self.last_event: ev.Event | None = None
        self.notice: tuple[str, float] | None = None

    # ---- inputs ------------------------------------------------------------

    def on_gesture(self, gesture: str, now: float | None = None) -> list[ev.Event]:
        """Handles one stabilized gesture event."""
        now = time.time() if now is None else now
        out = self.tick(now)                        # an expired selection goes first
        m = self.mapping
        spec = m.feature_for(gesture)

        if self.state is State.SELECTING_FEATURE:
            if spec:
                self._set_pending(spec.feature, gesture, now, out)
            else:
                self._note("Select a feature first", now)

        elif self.state is State.AWAITING_CONFIRMATION:
            if spec:
                if spec.feature == self.pending_feature:
                    self.pending_since = now        # same choice again: just refresh
                else:
                    self._set_pending(spec.feature, gesture, now, out)
            elif gesture == m.confirm_gesture:
                feature = self.pending_feature
                self.active_feature, self.pending_feature = feature, None
                self.state = State.FEATURE_ACTIVE
                self._event(out, ev.START_REQUESTED, gesture, now,
                            feature=feature, label=m.label(feature))
            elif gesture == m.back_gesture:
                self._cancel(gesture, "open_palm", now, out)

        elif self.state is State.FEATURE_ACTIVE:
            if gesture == m.back_gesture:
                feature = self.active_feature
                self.active_feature = None
                self.state = State.SELECTING_FEATURE
                self._event(out, ev.STOP_REQUESTED, gesture, now,
                            feature=feature, reason="open_palm")
                self._event(out, ev.RETURN_TO_SELECTION, gesture, now,
                            previous_feature=feature, reason="stopped")
            else:
                self._note("Show open palm to return to selection first", now)
        return out

    def tick(self, now: float | None = None) -> list[ev.Event]:
        """Call every frame: expires an unconfirmed selection."""
        now = time.time() if now is None else now
        out: list[ev.Event] = []
        if (self.state is State.AWAITING_CONFIRMATION and
                now - self.pending_since >= self.mapping.timings.confirmation_timeout_sec):
            self._cancel(None, "timeout", now, out)
        return out

    def shutdown(self, now: float | None = None) -> list[ev.Event]:
        """Application exit: ask the consumer to stop a still-active feature."""
        now = time.time() if now is None else now
        out: list[ev.Event] = []
        if self.state is State.FEATURE_ACTIVE:
            feature = self.active_feature
            self.active_feature = None
            self.state = State.SELECTING_FEATURE
            self._event(out, ev.STOP_REQUESTED, None, now, feature=feature, reason="shutdown")
        return out

    # ---- status for the preview -------------------------------------------

    def confirmation_time_left(self, now: float) -> float:
        if self.state is not State.AWAITING_CONFIRMATION:
            return 0.0
        return max(0.0, self.mapping.timings.confirmation_timeout_sec - (now - self.pending_since))

    # ---- internals ---------------------------------------------------------

    def _set_pending(self, feature: str, gesture: str, now: float, out: list) -> None:
        previous = self.pending_feature
        self.pending_feature, self.pending_since = feature, now
        self.state = State.AWAITING_CONFIRMATION
        self._event(out, ev.SELECTION_PENDING, gesture, now, feature=feature,
                    label=self.mapping.label(feature), previous_feature=previous)

    def _cancel(self, gesture: str | None, reason: str, now: float, out: list) -> None:
        feature = self.pending_feature
        self.pending_feature = None
        self.state = State.SELECTING_FEATURE
        self._event(out, ev.SELECTION_CANCELLED, gesture, now, feature=feature, reason=reason)
        self._event(out, ev.RETURN_TO_SELECTION, gesture, now, previous_feature=feature,
                    reason="cancelled" if reason == "open_palm" else reason)

    def _event(self, out: list, name: str, gesture: str | None, now: float, **fields) -> None:
        e = ev.Event(name, now, gesture, self.state.value, fields)
        out.append(e)
        self.last_event = e
        self.notice = None
        if self._emit:
            self._emit(e)

    def _note(self, text: str, now: float) -> None:
        self.notice = (text, now)
