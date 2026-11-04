"""Regression tests on REAL MediaPipe landmarks captured from a webcam.

The fixture holds 95 labelled calibration samples (one tester, both hands,
varied rotation / tilt / distance). These tests check that

  * the rule classifier never turns a sample into the wrong gesture, and
  * applying synthetic transforms (translation, scale, in-plane rotation,
    moderate 3D rotation) to real landmarks does not change the result.

These are still landmark tests, not webcam tests: the transformed hands are
not re-detected by MediaPipe.
"""

import itertools
import json
import unittest
from pathlib import Path

import numpy as np

from gesture_classifier import (GESTURES, UNKNOWN, ClassifierSettings, GestureClassifier,
                                KnnModel)
from hand_geometry import image_to_metric
from hand_tracker import HandDetection

FIXTURE = Path(__file__).parent / "fixtures" / "real_calibration_samples.json"
DATA = json.loads(FIXTURE.read_text(encoding="utf-8"))
SAMPLES = DATA["samples"]


def to_detection(s: dict, R: np.ndarray | None = None, scale: float = 1.0,
                 shift=(0.0, 0.0)) -> HandDetection:
    """Sample -> HandDetection, optionally rotated by R about the hand centre."""
    w, h = s["frame_size"]
    world = np.asarray(s["world_landmarks"], dtype=np.float64)
    img = image_to_metric(np.asarray(s["image_landmarks"]), w, h)
    if R is not None:
        world = (world - world.mean(0)) @ R.T
        c = img.mean(0)
        img = (img - c) @ R.T * scale + c + np.array([shift[0] * w, shift[1] * h, 0.0])
    image = img / np.array([w, h, w])
    return HandDetection(image, world, s["handedness_label"], 0.99, (w, h))


def rot(roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    r, p, y = np.radians([roll, pitch, yaw])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return rz @ rx @ ry


def truth(s):
    return s["gesture"] if s["gesture"] in GESTURES else UNKNOWN


class RealLandmarkTests(unittest.TestCase):
    def test_no_false_gestures_on_real_samples(self):
        clf = GestureClassifier(mode="rules")
        correct, wrong = 0, []
        for s in SAMPLES:
            g = clf.classify(to_detection(s)).gesture
            correct += g == truth(s)
            if g != truth(s) and g != UNKNOWN:
                wrong.append((s["gesture"], g))
        self.assertEqual(wrong, [])
        self.assertGreaterEqual(correct / len(SAMPLES), 0.9)

    def test_real_samples_invariant_to_transforms(self):
        # orientation/distance gates are disabled: this checks the geometry
        # (scaling a sample that was captured far away trips "too small")
        clf = GestureClassifier(ClassifierSettings(thumbs_up_max_angle_deg=180,
                                                   max_tilt_deg=90, max_out_of_frame=21,
                                                   min_palm_px=0),
                                mode="rules")
        rolls = (0, 30, 45, 60, 90, 180, -45, -90)
        tilts = ((0, 0), (25, 0), (-25, 0), (0, 25), (0, -25))
        for s in SAMPLES:
            base = clf.classify(to_detection(s))
            for r, (p, y) in itertools.product(rolls, tilts):
                for scale, shift in ((1.0, (0, 0)), (0.6, (0.1, -0.05))):
                    res = clf.classify(to_detection(s, rot(r, p, y), scale, shift))
                    with self.subTest(sample=s["gesture"], roll=r, pitch=p, yaw=y, scale=scale):
                        self.assertEqual(res.gesture, base.gesture)
                        self.assertAlmostEqual(res.confidence, base.confidence, places=6)

    def test_left_and_right_hands_both_present_and_recognized(self):
        clf = GestureClassifier(mode="rules")
        for g in GESTURES:
            hands = {s["hand"] for s in SAMPLES
                     if s["gesture"] == g and clf.classify(to_detection(s)).gesture == g}
            self.assertEqual(hands, {"Left", "Right"}, g)

    def test_knn_leave_one_out(self):
        """KNN on real samples: each sample classified by all *other* samples."""
        clf_rules = GestureClassifier(mode="rules")
        feats = np.stack([clf_rules.classify(to_detection(s)).analysis.features
                          for s in SAMPLES])
        labels = [s["gesture"] for s in SAMPLES]
        ok, wrong = 0, 0
        for i, s in enumerate(SAMPLES):
            keep = [j for j in range(len(SAMPLES)) if j != i]
            knn = KnnModel([labels[j] for j in keep], feats[keep])
            g = GestureClassifier(knn=knn, mode="knn").classify(to_detection(s)).gesture
            ok += g == truth(s)
            wrong += g != truth(s) and g != UNKNOWN
        self.assertEqual(wrong, 0)
        self.assertGreaterEqual(ok / len(SAMPLES), 0.9)


if __name__ == "__main__":
    unittest.main()
