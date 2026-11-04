"""MAVIS module controller: the gesture recognition module decides which MAVIS module runs.

Flow (agreed with the team):

    IDLE --(thumbs up)--> READY  (waits READY_SECONDS for a choice)
    READY --(1 finger)--> ACTIVE: OCR + LLM            (Varad's MAVIS OCR)
    READY --(2 fingers)-> ACTIVE: Live translation      (Yogita's translation module)
    READY --(3 fingers)-> ACTIVE: Drishti               (Harsh's Drishti)
    READY --(open palm or timeout)--> IDLE
    ACTIVE --(open palm)--> IDLE   (the module is turned off)

It consumes the *stabilised* gestures fired by the gesture project's own GestureStabilizer (one event per deliberate
showing). The gesture project's code is not changed; its own FeatureController keeps running alongside, unused for
navigation.
"""
from __future__ import annotations

READY_SECONDS = 8.0

MODULES = {
    "one_finger": {"id": "ocr", "label": "OCR + LLM", "owner": "Varad", "url": "/mavis/"},
    "two_fingers": {"id": "translation", "label": "Live Translation", "owner": "Yogita", "url": "/translate/?autostart=1"},
    "three_fingers": {"id": "drishti", "label": "Drishti", "owner": "Harsh", "url": "/drishti/?mode=describe"},
}
READY_GESTURE = "thumbs_up"
OFF_GESTURE = "open_palm"


class ModuleController:
    def __init__(self):
        self.state = "idle"          # idle | ready | active
        self.module = None           # MODULES entry while active
        self.ready_until = 0.0
        self.notice = None           # (text, time)

    def _say(self, text, now):
        self.notice = (text, now)

    def update(self, fired: str | None, now: float) -> list[dict]:
        """Feed one fired gesture (or None). Returns MAVIS events produced by this frame."""
        out: list[dict] = []
        if self.state == "ready" and now >= self.ready_until:
            self.state = "idle"
            self._say("Ready timed out. Show thumbs up again.", now)
            out.append({"event": "ready_timeout"})
        if not fired:
            return out
        if self.state == "idle":
            if fired == READY_GESTURE:
                self.state, self.ready_until = "ready", now + READY_SECONDS
                self._say("Ready. Show 1, 2 or 3 fingers.", now)
                out.append({"event": "ready"})
        elif self.state == "ready":
            if fired in MODULES:
                self.state, self.module = "active", MODULES[fired]
                self._say(f"Starting {self.module['label']}.", now)
                out.append({"event": "module_on", "module": self.module})
            elif fired == OFF_GESTURE:
                self.state = "idle"
                self._say("Cancelled.", now)
                out.append({"event": "ready_cancelled"})
            elif fired == READY_GESTURE:
                self.ready_until = now + READY_SECONDS
        elif self.state == "active":
            if fired == OFF_GESTURE:
                m = self.module
                self.state, self.module = "idle", None
                self._say(f"{m['label']} turned off.", now)
                out.append({"event": "module_off", "module": m})
        return out

    def describe(self, now: float) -> dict:
        return {
            "state": self.state,
            "module": self.module,
            "ready_left": round(max(0.0, self.ready_until - now), 1) if self.state == "ready" else None,
            "ready_seconds": READY_SECONDS,
            "notice": self.notice[0] if self.notice and now - self.notice[1] < 2.5 else None,
            "modules": {g: m for g, m in MODULES.items()},
        }
