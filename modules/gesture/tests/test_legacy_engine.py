"""The legacy tray app's GestureEngine now uses the rotation-invariant geometry."""

import json
import unittest
from pathlib import Path

from gesture_engine import GestureEngine, finger_states, landmarks_to_features
from tests.synthetic_hand import pose, to_camera, to_image

CFG = json.loads((Path(__file__).resolve().parent.parent / "config" / "gestures.json")
                 .read_text(encoding="utf-8"))


def image_pts(name, **kw):
    return to_image(to_camera(pose(name), **kw), 640, 480, px_per_unit=1100.0)


class LegacyEngineTests(unittest.TestCase):
    def test_finger_states_rotation_invariant(self):
        expected = {"one_finger": [0, 1, 0, 0, 0], "two_fingers": [0, 1, 1, 0, 0],
                    "three_fingers": [0, 1, 1, 1, 0], "open_palm": [1, 1, 1, 1, 1],
                    "fist": [0, 0, 0, 0, 0]}
        for name, states in expected.items():
            for roll in (0, 45, 90, -90, 180):
                self.assertEqual(finger_states(image_pts(name, roll=roll)), states, (name, roll))

    def test_original_y_rule_would_fail_sideways(self):
        """Documents the bug that was fixed: a sideways 'point' is not 'up' in screen Y."""
        pts = image_pts("one_finger", roll=90)
        self.assertFalse(pts[8][1] < pts[6][1])            # old rule: tip above PIP
        self.assertEqual(finger_states(pts)[1], 1)

    def test_engine_patterns(self):
        eng = GestureEngine(CFG["gestures"])
        self.assertEqual(eng.recognize(image_pts("two_fingers", roll=60)), "peace")
        self.assertEqual(eng.recognize(image_pts("open_palm", roll=-45)), "open_palm")
        self.assertEqual(eng.recognize(image_pts("fist", roll=30)), "fist")

    def test_recorded_gesture_and_old_format_skipped(self):
        feats = landmarks_to_features(image_pts("l_shape"))
        cfg = {"rec": {"type": "recorded", "samples": [feats.tolist()]},
               "old": {"type": "recorded", "samples": [[0.1] * 63]}}
        eng = GestureEngine(cfg)
        self.assertEqual(eng.recognize(image_pts("l_shape", roll=70)), "rec")


if __name__ == "__main__":
    unittest.main()
