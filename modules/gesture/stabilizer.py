"""
stabilizer.py — turns noisy per-frame classifications into discrete events.

Rules (all configurable):
  * a gesture is "confirmed" only while the latest `stable_frames`
    consecutive frames all show it with confidence >= `min_confidence`
    AND it has been shown continuously for its hold time (`hold_sec`,
    per gesture, default 0); any other frame (another gesture, UNKNOWN,
    no hand) clears it at once and restarts the hold;
  * an event fires only on a frame that itself confirms the gesture, so a
    stale gesture can never fire from earlier landmarks;
  * a confirmed gesture fires ONE event; holding it never repeats the event;
  * the same gesture can fire again only after it is released: neutral
    (UNKNOWN / low confidence) for `release_sec`, a different confirmed
    gesture, or no hand for `rearm_after_loss_sec`;
  * switching directly to a different gesture fires that gesture;
  * `cooldown_sec` is the minimum time between any two events;
  * losing the hand (NO_HAND) resets the stability counter and clears the
    confirmed gesture immediately; it is never treated as a gesture.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StabilitySettings:
    stable_frames: int = 5
    min_confidence: float = 0.55
    cooldown_sec: float = 1.0
    release_sec: float = 0.4
    rearm_after_loss_sec: float = 1.0
    hold_sec: dict[str, float] = field(default_factory=dict)   # per gesture

    @classmethod
    def from_dict(cls, d: dict | None) -> "StabilitySettings":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class GestureStabilizer:
    def __init__(self, settings: StabilitySettings | None = None):
        self.s = settings or StabilitySettings()
        self.reset()

    def reset(self) -> None:
        self.candidate: str | None = None   # what the recent frames show
        self.count = 0                      # consecutive frames of `candidate`
        self.candidate_since = 0.0          # time of the first of those frames
        self.stable: str | None = None      # confirmed on the latest frame
        self.last_fired: str | None = None  # blocks re-firing until released
        self.last_fire_time = float("-inf")
        self.neutral_since: float | None = None
        self.lost_since: float | None = None

    # ---- status for the overlay -------------------------------------------

    @property
    def progress(self) -> float:
        """0..1 frame-count progress of the current candidate."""
        if self.candidate is None:
            return 0.0
        return min(1.0, self.count / max(self.s.stable_frames, 1))

    def progress_at(self, now: float) -> float:
        """0..1 progress towards confirmation, counting frames and hold time."""
        if self.candidate is None:
            return 0.0
        hold = self.s.hold_sec.get(self.candidate, 0.0)
        held = 1.0 if hold <= 0 else (now - self.candidate_since) / hold
        return max(0.0, min(1.0, self.progress, held))

    def cooldown_left(self, now: float) -> float:
        return max(0.0, self.s.cooldown_sec - (now - self.last_fire_time))

    @property
    def armed(self) -> bool:
        """False while the last fired gesture is still being held."""
        return self.stable is None or self.stable != self.last_fired

    # ---- main entry --------------------------------------------------------

    def update(self, gesture: str | None, confidence: float, now: float,
               hand_present: bool = True) -> str | None:
        """Feeds one frame. Returns the gesture name when an event fires."""
        if not hand_present:
            self.candidate, self.count, self.stable = None, 0, None
            self.neutral_since = None
            if self.lost_since is None:
                self.lost_since = now
            if now - self.lost_since >= self.s.rearm_after_loss_sec:
                self.last_fired = None
            return None
        self.lost_since = None

        obs = gesture if (gesture and confidence >= self.s.min_confidence) else None
        if obs == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count, self.candidate_since = obs, 1, now

        if obs is None:
            self.stable = None
            if self.neutral_since is None:
                self.neutral_since = now
            if now - self.neutral_since >= self.s.release_sec:
                self.last_fired = None
            return None
        self.neutral_since = None

        held = now - self.candidate_since >= self.s.hold_sec.get(obs, 0.0)
        self.stable = obs if (self.count >= self.s.stable_frames and held) else None
        if self.stable != obs:
            return None
        if obs != self.last_fired and now - self.last_fire_time >= self.s.cooldown_sec:
            self.last_fired = obs
            self.last_fire_time = now
            return obs
        return None
