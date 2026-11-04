"""Feature-selection state machine: unit tests and end-to-end workflows."""

import unittest

import events as ev
from feature_controller import FeatureController, State
from gesture_mapping import GestureMapping, load_mapping
from mavis_gesture import GesturePipeline, load_settings
from tests.helpers import detection
from tests.synthetic_hand import pose

S, A, F = State.SELECTING_FEATURE, State.AWAITING_CONFIRMATION, State.FEATURE_ACTIVE
MAPPING = load_mapping()
FEATURES = {"one_finger": "ocr", "two_fingers": "translation",
            "three_fingers": "price_comparison"}


def names(events):
    return [e.name for e in events]


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, e):
        self.events.append(e)


# ---------------------------------------------------------------- unit level

class ControllerUnitTests(unittest.TestCase):
    def setUp(self):
        self.rec = Recorder()
        self.c = FeatureController(MAPPING, self.rec)
        self.t = 100.0

    def g(self, gesture, dt=1.0):
        self.t += dt
        return self.c.on_gesture(gesture, self.t)

    def test_mapping_loaded_from_config(self):
        self.assertEqual({g: s.feature for g, s in MAPPING.selection.items()}, FEATURES)
        self.assertEqual(MAPPING.confirm_gesture, "thumbs_up")
        self.assertEqual(MAPPING.back_gesture, "open_palm")

    def test_full_workflow_each_feature(self):
        for gesture, feature in FEATURES.items():
            with self.subTest(feature=feature):
                self.rec.events.clear()
                self.assertIs(self.c.state, S)
                out = self.g(gesture)
                self.assertEqual(names(out), [ev.SELECTION_PENDING])
                self.assertEqual(out[0].get("feature"), feature)
                self.assertEqual(out[0].gesture, gesture)
                self.assertIs(self.c.state, A)
                self.assertEqual(self.c.pending_feature, feature)
                self.assertIsNone(self.c.active_feature)          # nothing started yet

                out = self.g("thumbs_up")
                self.assertEqual(names(out), [ev.START_REQUESTED])
                self.assertEqual(out[0].get("feature"), feature)
                self.assertEqual(out[0].state, F.value)
                self.assertEqual(self.c.active_feature, feature)

                out = self.g("open_palm")
                self.assertEqual(names(out), [ev.STOP_REQUESTED, ev.RETURN_TO_SELECTION])
                self.assertEqual(out[0].get("feature"), feature)
                self.assertEqual(out[0].get("reason"), "open_palm")
                self.assertEqual(out[1].get("previous_feature"), feature)
                self.assertEqual(out[1].get("reason"), "stopped")
                self.assertIs(self.c.state, S)
                self.assertIsNone(self.c.active_feature)
                self.assertEqual(self.rec.events, out[:0] + self.rec.events)  # emitted too
                self.assertEqual(len(self.rec.events), 4)

    def test_confirm_without_pending_is_ignored(self):
        self.assertEqual(self.g("thumbs_up"), [])
        self.assertIs(self.c.state, S)
        self.assertIsNotNone(self.c.notice)

    def test_palm_in_selection_is_ignored(self):
        self.assertEqual(self.g("open_palm"), [])
        self.assertIs(self.c.state, S)

    def test_palm_cancels_pending_without_stop_request(self):
        self.g("one_finger")
        out = self.g("open_palm")
        self.assertEqual(names(out), [ev.SELECTION_CANCELLED, ev.RETURN_TO_SELECTION])
        self.assertEqual(out[0].get("reason"), "open_palm")
        self.assertEqual(out[1].get("reason"), "cancelled")
        self.assertNotIn(ev.STOP_REQUESTED, names(self.rec.events))
        self.assertIs(self.c.state, S)
        self.assertIsNone(self.c.pending_feature)

    def test_switch_pending_ocr_to_translation_to_price(self):
        self.g("one_finger")
        out = self.g("two_fingers")
        self.assertEqual(names(out), [ev.SELECTION_PENDING])
        self.assertEqual((out[0].get("feature"), out[0].get("previous_feature")),
                         ("translation", "ocr"))
        out = self.g("three_fingers")
        self.assertEqual((out[0].get("feature"), out[0].get("previous_feature")),
                         ("price_comparison", "translation"))
        out = self.g("thumbs_up")
        self.assertEqual(out[0].get("feature"), "price_comparison")

    def test_same_selection_again_refreshes_without_event(self):
        self.g("one_finger")
        self.assertEqual(self.g("one_finger", dt=7.0), [])        # resets the timeout
        self.assertEqual(self.c.tick(self.t + 7.0), [])
        self.assertIs(self.c.state, A)

    def test_pending_timeout(self):
        self.g("two_fingers")
        self.assertEqual(self.c.tick(self.t + 7.9), [])
        out = self.c.tick(self.t + 8.0)
        self.assertEqual(names(out), [ev.SELECTION_CANCELLED, ev.RETURN_TO_SELECTION])
        self.assertEqual(out[0].get("reason"), "timeout")
        self.assertIsNone(out[0].gesture)
        self.assertIs(self.c.state, S)
        self.assertEqual(self.c.tick(self.t + 20), [])            # only once

    def test_late_confirmation_after_timeout_is_ignored(self):
        self.g("one_finger")
        out = self.g("thumbs_up", dt=9.0)
        self.assertEqual(names(out), [ev.SELECTION_CANCELLED, ev.RETURN_TO_SELECTION])
        self.assertIs(self.c.state, S)

    def test_repeated_thumbs_up_starts_once(self):
        self.g("one_finger")
        self.g("thumbs_up")
        self.assertEqual(self.g("thumbs_up"), [])
        self.assertEqual(self.g("thumbs_up"), [])
        self.assertEqual(names(self.rec.events).count(ev.START_REQUESTED), 1)

    def test_repeated_palm_stops_once(self):
        self.g("one_finger")
        self.g("thumbs_up")
        self.g("open_palm")
        self.assertEqual(self.g("open_palm"), [])
        self.assertEqual(names(self.rec.events).count(ev.STOP_REQUESTED), 1)

    def test_only_one_active_feature(self):
        self.g("one_finger")
        self.g("thumbs_up")
        for gesture in ("two_fingers", "three_fingers", "one_finger"):
            self.assertEqual(self.g(gesture), [])
        self.assertEqual(self.c.active_feature, "ocr")
        self.assertIs(self.c.state, F)
        self.assertIsNone(self.c.pending_feature)

    def test_no_timeout_while_active(self):
        self.g("one_finger")
        self.g("thumbs_up")
        self.assertEqual(self.c.tick(self.t + 3600), [])
        self.assertIs(self.c.state, F)

    def test_shutdown_stops_active_feature(self):
        self.g("three_fingers")
        self.g("thumbs_up")
        out = self.c.shutdown(self.t + 1)
        self.assertEqual(names(out), [ev.STOP_REQUESTED])
        self.assertEqual(out[0].get("reason"), "shutdown")
        self.assertEqual(self.c.shutdown(self.t + 2), [])

    def test_unknown_gestures_ignored(self):
        self.assertEqual(self.g("UNKNOWN"), [])
        self.assertEqual(self.g("NO_HAND"), [])
        self.assertIs(self.c.state, S)

    def test_custom_mapping(self):
        m = GestureMapping.from_dict({
            "selection_gestures": {"two_fingers": {"feature": "x", "label": "X"}},
            "confirm_gesture": "open_palm", "back_gesture": "thumbs_up"})
        c = FeatureController(m)
        self.assertEqual(c.on_gesture("one_finger", 1.0), [])
        self.assertEqual(names(c.on_gesture("two_fingers", 2.0)), [ev.SELECTION_PENDING])
        self.assertEqual(c.on_gesture("open_palm", 3.0)[0].get("feature"), "x")

    def test_invalid_mapping_rejected(self):
        with self.assertRaises(ValueError):
            GestureMapping.from_dict({"selection_gestures": {
                "one_finger": {"feature": "a", "label": "A"},
                "two_fingers": {"feature": "a", "label": "B"}}})
        with self.assertRaises(ValueError):
            GestureMapping.from_dict({"selection_gestures": {
                "thumbs_up": {"feature": "a", "label": "A"}}})


# ---------------------------------------------------------------- end to end

class NoTracker:
    def close(self):
        pass


class PipelineWorkflowTests(unittest.TestCase):
    """Synthetic detections through classifier -> stabilizer -> controller."""

    FPS = 20.0
    POSES = {
        "one_finger": lambda: detection(pose("one_finger"), roll=20),
        "two_fingers": lambda: detection(pose("two_fingers"), roll=-30, left=True),
        "three_fingers": lambda: detection(pose("three_fingers"), roll=35, pitch=15),
        "thumbs_up": lambda: detection(pose("thumbs_up"), roll=-66),
        "open_palm": lambda: detection(pose("open_palm"), roll=-20),
        "fist": lambda: detection(pose("fist")),
        "palm_half_out": lambda: detection(pose("open_palm"), center=(0.5, 0.08)),
    }

    def setUp(self):
        self.pipe = GesturePipeline(load_settings(), MAPPING, tracker=NoTracker())
        self.events = []
        self.pipe.bus.subscribe(self.events.append)
        self.t = 1000.0
        self.dets = {k: f() for k, f in self.POSES.items()}

    def show(self, what, seconds):
        """Feeds `seconds` of frames; what=None means no hand detected."""
        for _ in range(int(round(seconds * self.FPS))):
            self.t += 1.0 / self.FPS
            dets = [] if what is None else [self.dets[what]]
            self.pipe.process_detections(dets, self.t)

    def state(self):
        return self.pipe.controller.state

    def test_complete_workflow_each_feature(self):
        for gesture, feature in FEATURES.items():
            with self.subTest(feature=feature):
                self.events.clear()
                self.show(gesture, 1.0)
                self.assertEqual(names(self.events), [ev.SELECTION_PENDING])
                self.assertEqual(self.events[0].get("feature"), feature)
                self.assertIs(self.state(), A)
                self.show(None, 0.3)
                self.show("thumbs_up", 1.0)
                self.assertEqual(names(self.events)[-1], ev.START_REQUESTED)
                self.assertEqual(self.events[-1].get("feature"), feature)
                self.assertIs(self.state(), F)
                self.show(None, 0.3)
                self.show("open_palm", 1.0)
                self.assertEqual(names(self.events), [
                    ev.SELECTION_PENDING, ev.START_REQUESTED,
                    ev.STOP_REQUESTED, ev.RETURN_TO_SELECTION])
                self.assertEqual(self.events[2].get("feature"), feature)
                self.assertIs(self.state(), S)
                self.show(None, 1.2)

    def test_brief_flash_does_not_select(self):
        self.show("one_finger", 0.3)          # below the 0.5 s selection hold
        self.show(None, 0.5)
        self.assertEqual(self.events, [])

    def test_held_selection_fires_once(self):
        self.show("two_fingers", 5.0)
        self.assertEqual(names(self.events), [ev.SELECTION_PENDING])

    def test_direct_switch_between_pending_selections(self):
        self.show("one_finger", 1.0)
        self.show("two_fingers", 1.0)
        self.show("three_fingers", 1.0)
        self.assertEqual([e.get("feature") for e in self.events],
                         ["ocr", "translation", "price_comparison"])
        self.assertIs(self.state(), A)

    def test_hand_loss_while_awaiting_keeps_selection(self):
        self.show("one_finger", 1.0)
        self.show(None, 3.0)
        self.assertEqual(names(self.events), [ev.SELECTION_PENDING])
        self.assertIs(self.state(), A)
        self.show("thumbs_up", 1.0)
        self.assertEqual(names(self.events)[-1], ev.START_REQUESTED)

    def test_hand_loss_while_active_never_stops(self):
        self.show("one_finger", 1.0)
        self.show("thumbs_up", 1.0)
        self.show(None, 10.0)
        self.show("palm_half_out", 3.0)       # unreliable landmarks = NO_HAND
        self.assertEqual(names(self.events), [ev.SELECTION_PENDING, ev.START_REQUESTED])
        self.assertIs(self.state(), F)

    def test_no_hand_never_generates_stop_or_navigation(self):
        for start in (None, "fist"):
            self.events.clear()
            self.show(start, 0.5)
            self.show(None, 30.0)
            self.assertEqual(self.events, [])
        self.show("three_fingers", 1.0)
        self.show("thumbs_up", 1.0)
        n = len(self.events)
        self.show(None, 30.0)
        self.assertEqual(len(self.events), n)

    def test_palm_dropout_and_return_does_not_double_stop(self):
        self.show("one_finger", 1.0)
        self.show("thumbs_up", 1.0)
        self.show("open_palm", 1.0)
        self.show(None, 0.4)                  # hand briefly lost, palm shown again
        self.show("open_palm", 1.0)
        self.show("palm_half_out", 0.5)
        self.show("open_palm", 1.0)
        self.assertEqual(names(self.events).count(ev.STOP_REQUESTED), 1)
        self.assertIs(self.state(), S)

    def test_pending_timeout_end_to_end(self):
        self.show("two_fingers", 1.0)
        self.show("fist", 9.0)
        self.assertEqual(names(self.events), [ev.SELECTION_PENDING, ev.SELECTION_CANCELLED,
                                              ev.RETURN_TO_SELECTION])
        self.assertEqual(self.events[1].get("reason"), "timeout")

    def test_close_requests_stop_for_active_feature(self):
        self.show("one_finger", 1.0)
        self.show("thumbs_up", 1.0)
        out = self.pipe.close(self.t)
        self.assertEqual(names(out), [ev.STOP_REQUESTED])
        self.assertEqual(self.events[-1].get("reason"), "shutdown")

    def test_event_ordering_and_timestamps(self):
        self.show("one_finger", 1.0)
        self.show("two_fingers", 1.0)
        self.show("thumbs_up", 1.0)
        self.show("open_palm", 1.0)
        self.assertEqual(names(self.events), [
            ev.SELECTION_PENDING, ev.SELECTION_PENDING, ev.START_REQUESTED,
            ev.STOP_REQUESTED, ev.RETURN_TO_SELECTION])
        ts = [e.timestamp for e in self.events]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual([e.state for e in self.events],
                         [A.value, A.value, F.value, S.value, S.value])


if __name__ == "__main__":
    unittest.main()
