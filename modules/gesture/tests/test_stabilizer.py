"""Debounce / trigger behaviour of the gesture stabilizer."""

import unittest

from stabilizer import GestureStabilizer, StabilitySettings

FPS = 30.0


class Feeder:
    """Feeds frames at a fixed frame rate and collects fired events."""

    def __init__(self, **kw):
        self.st = GestureStabilizer(StabilitySettings(**kw))
        self.t = 0.0
        self.events: list[str] = []

    def feed(self, gesture, n=1, conf=0.9, present=True):
        for _ in range(n):
            self.t += 1.0 / FPS
            ev = self.st.update(gesture, conf, self.t, hand_present=present)
            if ev:
                self.events.append(ev)
        return self


def default(**kw):
    s = dict(stable_frames=5, min_confidence=0.55, cooldown_sec=1.0,
             release_sec=0.4, rearm_after_loss_sec=1.0)
    s.update(kw)
    return Feeder(**s)


class StabilizerTests(unittest.TestCase):
    def test_fires_after_stable_frames(self):
        f = default().feed("one_finger", 4)
        self.assertEqual(f.events, [])
        f.feed("one_finger", 1)
        self.assertEqual(f.events, ["one_finger"])

    def test_no_repeat_while_held(self):
        f = default().feed("open_palm", 300)        # 10 s hold
        self.assertEqual(f.events, ["open_palm"])

    def test_release_rearms(self):
        f = default().feed("thumbs_up", 10).feed(None, 15).feed("thumbs_up", 40)
        self.assertEqual(f.events, ["thumbs_up", "thumbs_up"])

    def test_short_unknown_flicker_does_not_rearm(self):
        f = default().feed("two_fingers", 10)
        for _ in range(10):
            f.feed(None, 3).feed("two_fingers", 8)
        self.assertEqual(f.events, ["two_fingers"])

    def test_cooldown_delays_then_fires_if_still_held(self):
        f = default().feed("one_finger", 6).feed("two_fingers", 6)
        self.assertEqual(f.events, ["one_finger"])        # still in cooldown
        f.feed("two_fingers", 30)                          # cooldown ends while held
        self.assertEqual(f.events, ["one_finger", "two_fingers"])

    def test_gesture_released_during_cooldown_never_fires(self):
        f = default().feed("one_finger", 6).feed("two_fingers", 6).feed(None, 40)
        self.assertEqual(f.events, ["one_finger"])

    def test_switching_gestures_fires_each(self):
        f = default(cooldown_sec=0.2)
        for g in ("one_finger", "two_fingers", "three_fingers", "thumbs_up"):
            f.feed(g, 15)
        self.assertEqual(f.events, ["one_finger", "two_fingers", "three_fingers", "thumbs_up"])

    def test_inconsistent_frames_never_stabilize(self):
        f = default()
        for _ in range(50):
            f.feed("one_finger", 3).feed("two_fingers", 3)
        self.assertEqual(f.events, [])

    def test_low_confidence_ignored(self):
        f = default().feed("open_palm", 60, conf=0.4)
        self.assertEqual(f.events, [])


class HoldDurationTests(unittest.TestCase):
    """Per-gesture hold time on top of stable_frames (selection / confirm holds)."""

    def test_hold_delays_confirmation(self):
        f = default(hold_sec={"one_finger": 0.5}).feed("one_finger", 14)   # 0.47 s
        self.assertEqual(f.events, [])
        f.feed("one_finger", 2)
        self.assertEqual(f.events, ["one_finger"])

    def test_hold_restarts_after_interruption(self):
        f = default(hold_sec={"thumbs_up": 0.5})
        f.feed("thumbs_up", 12).feed(None, 1).feed("thumbs_up", 12)
        self.assertEqual(f.events, [])          # 0.37 s since the restart
        f.feed("thumbs_up", 6)                 # 0.53 s
        self.assertEqual(f.events, ["thumbs_up"])

    def test_gestures_without_hold_use_frames_only(self):
        f = default(hold_sec={"open_palm": 2.0}).feed("one_finger", 5)
        self.assertEqual(f.events, ["one_finger"])

    def test_progress_includes_hold(self):
        f = default(hold_sec={"one_finger": 1.0}).feed("one_finger", 15)   # 0.5 s held
        self.assertAlmostEqual(f.st.progress_at(f.t), 14 / 30, places=2)


class HandLossTests(unittest.TestCase):
    """Losing the hand must never produce, complete or repeat a gesture."""

    def test_no_hand_never_fires(self):
        f = default().feed("open_palm", 300, present=False)
        self.assertEqual(f.events, [])
        f = default().feed(None, 300, present=False)
        self.assertEqual(f.events, [])

    def test_hand_loss_resets_counter_and_display(self):
        f = default().feed("open_palm", 4)                  # one frame short
        f.feed(None, 1, present=False)
        self.assertEqual((f.st.candidate, f.st.count, f.st.stable), (None, 0, None))
        f.feed("open_palm", 4)                               # needs a fresh 5
        self.assertEqual(f.events, [])
        f.feed("open_palm", 1)
        self.assertEqual(f.events, ["open_palm"])

    def test_confirmed_gesture_cleared_when_hand_lost(self):
        f = default().feed("thumbs_up", 10)
        self.assertEqual(f.st.stable, "thumbs_up")
        f.feed(None, 1, present=False)
        self.assertIsNone(f.st.stable)
        self.assertEqual(f.st.progress, 0.0)

    def test_brief_exit_with_same_gesture_does_not_refire(self):
        """The case from the webcam run: hand partly leaves the frame ~0.7 s."""
        f = default().feed("open_palm", 20).feed(None, 21, present=False).feed("open_palm", 30)
        self.assertEqual(f.events, ["open_palm"])

    def test_long_exit_rearms_with_fresh_frames(self):
        f = default().feed("open_palm", 20).feed(None, 40, present=False)
        f.feed("open_palm", 4)
        self.assertEqual(f.events, ["open_palm"])           # not yet: needs 5 fresh frames
        f.feed("open_palm", 1)
        self.assertEqual(f.events, ["open_palm", "open_palm"])

    def test_pending_gesture_not_fired_after_hand_lost(self):
        """Confirmed during cooldown, then hand vanishes: no stale event later."""
        f = default().feed("one_finger", 6).feed("two_fingers", 6)   # two_fingers pending
        f.feed(None, 60, present=False)                              # cooldown ends, no hand
        self.assertEqual(f.events, ["one_finger"])

    def test_pending_gesture_not_fired_on_a_different_frame(self):
        f = default().feed("one_finger", 6).feed("two_fingers", 6)   # pending until t=1.2
        f.feed(None, 27)                                             # neutral past cooldown
        f.feed("three_fingers", 1)
        self.assertEqual(f.events, ["one_finger"])
