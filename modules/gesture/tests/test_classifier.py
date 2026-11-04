"""Synthetic classification tests: rules, KNN and hybrid modes."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gesture_classifier import (CALIBRATION_FORMAT, GESTURES, NEGATIVE, NO_HAND, UNKNOWN,
                                ClassifierSettings, GestureClassifier, KnnModel,
                                KnnSettings, analyze_hand)
from tests.helpers import detection
from tests.synthetic_hand import NEGATIVE_POSES, POSITIVE_POSES, pose, to_image

ROLLS = (0, 30, 45, 60, 90, 180, -30, -45, -60, -90)
TILTS = [(0, 0), (25, 0), (-25, 0), (0, 30), (0, -30), (20, 25), (-20, -25), (35, 0)]


def rules(**kw) -> GestureClassifier:
    return GestureClassifier(ClassifierSettings(**kw), mode="rules")


class RuleInvarianceTests(unittest.TestCase):
    """The expected gesture survives translation, scale, rotation, tilt, handedness."""

    def setUp(self):
        # the thumbs-up "pointing up" check is orientation-dependent by design;
        # it is disabled here and tested separately below
        self.clf = {src: rules(landmark_source=src, thumbs_up_max_angle_deg=180)
                    for src in ("world", "image")}

    def test_positive_poses_all_orientations(self):
        cases = itertools.product(POSITIVE_POSES, ROLLS, TILTS, (False, True), ("world", "image"))
        for name, roll, (pitch, yaw), left, src in cases:
            det = detection(pose(name), left=left, roll=roll, pitch=pitch, yaw=yaw)
            res = self.clf[src].classify(det)
            with self.subTest(gesture=name, roll=roll, pitch=pitch, yaw=yaw, left=left, src=src):
                self.assertEqual(res.gesture, name, res.reason)
                self.assertGreaterEqual(res.confidence, 0.5)

    def test_translation_and_scale(self):
        for name in POSITIVE_POSES:
            for center, ppu in [((0.4, 0.45), 900), ((0.6, 0.55), 900),
                                ((0.5, 0.5), 500), ((0.5, 0.5), 1400)]:
                det = detection(pose(name), center=center, px_per_unit=ppu, roll=30)
                with self.subTest(gesture=name, center=center, ppu=ppu):
                    self.assertEqual(self.clf["image"].classify(det).gesture, name)
                    self.assertEqual(self.clf["world"].classify(det).gesture, name)

    def test_negative_poses_rejected_in_all_orientations(self):
        for name, roll, (pitch, yaw), left in itertools.product(
                NEGATIVE_POSES, ROLLS, TILTS, (False, True)):
            det = detection(pose(name), left=left, roll=roll, pitch=pitch, yaw=yaw)
            res = self.clf["world"].classify(det)
            with self.subTest(pose=name, roll=roll, pitch=pitch, yaw=yaw, left=left):
                self.assertEqual(res.gesture, UNKNOWN)

    def test_classification_stable_under_landmark_noise(self):
        rng = np.random.default_rng(1)
        clf = self.clf["world"]
        for name in POSITIVE_POSES:
            base = detection(pose(name), roll=45, pitch=15)
            ok = 0
            for _ in range(100):
                det = detection(pose(name), roll=45, pitch=15)
                det.world = base.world + rng.normal(0, 0.0015, base.world.shape)
                ok += clf.classify(det).gesture == name
            with self.subTest(gesture=name):
                self.assertGreaterEqual(ok, 95)


class RuleGateTests(unittest.TestCase):
    def test_thumbs_up_must_point_up(self):
        clf = rules()
        # in the model the thumb points ~66 deg towards the thumb side; this roll
        # brings it to screen-up (mirrored for the left hand)
        for left, up in ((False, -66.0), (True, 66.0)):
            for off in (0, 30, -30, 60, -60):
                res = clf.classify(detection(pose("thumbs_up"), left=left, roll=up + off))
                self.assertEqual(res.gesture, "thumbs_up", (left, off, res.reason))
            for off in (110, 180, -110):   # sideways-down / thumbs-down
                res = clf.classify(detection(pose("thumbs_up"), left=left, roll=up + off))
                self.assertEqual(res.gesture, UNKNOWN, (left, off))
                self.assertIn("thumb not pointing up", res.reason)

    def test_open_palm_requires_spread(self):
        clf = rules()
        res = clf.classify(detection(pose("open_palm", spread_deg=1.0)))
        self.assertEqual(res.gesture, UNKNOWN)
        self.assertEqual(res.reason, "fingers not spread")

    def test_edge_on_rejected_except_thumbs_up(self):
        clf = rules()
        for name in ("one_finger", "two_fingers", "three_fingers", "open_palm"):
            res = clf.classify(detection(pose(name), yaw=85))
            self.assertEqual(res.gesture, UNKNOWN, name)
            self.assertIn("edge-on", res.reason)
        # a thumbs-up is normally shown with the side of the fist to the camera:
        # thumb up first (roll), then turn the fist around the vertical axis
        det = detection(pose("thumbs_up"), roll=-66)
        a = np.radians(80)
        ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        det.world = det.world @ ry.T
        det.image = to_image((det.world - det.world[0]) @ np.eye(3), 640, 480,
                             center=(0.5, 0.5), px_per_unit=1100.0)
        res = clf.classify(det)
        self.assertEqual(res.analysis.orientation.facing, "edge-on")
        self.assertEqual(res.gesture, "thumbs_up", res.reason)

    def test_low_handedness_score(self):
        res = rules().classify(detection(pose("open_palm"), label_score=0.3))
        self.assertEqual(res.gesture, NO_HAND)
        self.assertIn("tracking confidence", res.reason)

    def test_out_of_frame(self):
        res = rules().classify(detection(pose("open_palm"), center=(0.5, 0.05)))
        self.assertEqual(res.gesture, NO_HAND)
        self.assertIn("out of frame", res.reason)

    def test_too_small(self):
        res = rules().classify(detection(pose("open_palm"), px_per_unit=200))
        self.assertEqual(res.gesture, UNKNOWN)
        self.assertIn("too far", res.reason)

    def test_degenerate_landmarks(self):
        det = detection(pose("open_palm"))
        det.world[:] = det.world[0]
        res = rules().classify(det)
        self.assertEqual(res.gesture, NO_HAND)
        self.assertIsNotNone(res.analysis.reject_reason)

    def test_anatomical_hand_reported(self):
        clf = rules()
        self.assertEqual(clf.classify(detection(pose("open_palm"))).analysis.hand, "Left")
        self.assertEqual(clf.classify(detection(pose("open_palm"), left=True)).analysis.hand,
                         "Right")
        clf.mirrored_input = False
        self.assertEqual(clf.classify(detection(pose("open_palm"))).analysis.hand, "Right")


class HandLossPipelineTests(unittest.TestCase):
    """Classifier + stabilizer, as wired in mavis_gesture.run()."""

    def run_frames(self, frames, fps=20.0):
        from stabilizer import GestureStabilizer
        clf, st, events = rules(), GestureStabilizer(), []
        for i, det in enumerate(frames):
            res = clf.classify(det) if det is not None else None
            present = res is not None and res.gesture != NO_HAND
            ev = st.update(res.gesture if present else None,
                           res.confidence if present else 0.0, i / fps, present)
            if ev:
                events.append(ev)
        return events

    def test_partial_exit_does_not_refire_or_become_open_palm(self):
        palm = detection(pose("open_palm"))
        half_out = detection(pose("open_palm"), center=(0.5, 0.08))
        self.assertEqual(rules().classify(half_out).gesture, NO_HAND)
        events = self.run_frames([palm] * 20 + [half_out] * 14 + [palm] * 30)
        self.assertEqual(events, ["open_palm"])

    def test_no_detection_is_never_a_gesture(self):
        self.assertEqual(self.run_frames([None] * 100), [])

    def test_gesture_change_across_dropout_needs_fresh_frames(self):
        one = detection(pose("one_finger"))
        two = detection(pose("two_fingers"))
        events = self.run_frames([one] * 30 + [None] * 3 + [two] * 4)
        self.assertEqual(events, ["one_finger"])          # 4 fresh frames < 5
        events = self.run_frames([one] * 30 + [None] * 3 + [two] * 5)
        self.assertEqual(events, ["one_finger", "two_fingers"])


# ---------------------------------------------------------------- KNN

def _samples(rng, names, n=15, jitter=0.0012):
    """Calibration-style samples: varied orientation, both hands, landmark noise."""
    s = ClassifierSettings()
    labels, feats = [], []
    for name in names:
        model = pose(name if name != NEGATIVE else "fist")
        for i in range(n):
            det = detection(model, left=bool(i % 2), roll=rng.uniform(-90, 90),
                            pitch=rng.uniform(-30, 30), yaw=rng.uniform(-30, 30))
            det.world = det.world + rng.normal(0, jitter, det.world.shape)
            f = analyze_hand(det, s).features
            assert f is not None, (name, i)
            labels.append(name)
            feats.append(f)
    return labels, np.stack(feats)


class KnnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        labels, feats = _samples(rng, list(GESTURES) + [NEGATIVE])
        cls.knn = KnnModel(labels, feats, KnnSettings())

    def test_knn_recognizes_unseen_orientations(self):
        clf = GestureClassifier(ClassifierSettings(thumbs_up_max_angle_deg=180),
                                knn=self.knn, mode="knn")
        for name, roll, left in itertools.product(POSITIVE_POSES, (0, 45, -60, 180), (False, True)):
            res = clf.classify(detection(pose(name), roll=roll, pitch=10, left=left))
            with self.subTest(gesture=name, roll=roll, left=left):
                self.assertEqual(res.gesture, name, res.reason)

    def test_knn_rejects_far_poses(self):
        clf = GestureClassifier(knn=self.knn, mode="knn")
        for name in ("half_curled", "claw"):
            res = clf.classify(detection(pose(name)))
            self.assertEqual(res.gesture, UNKNOWN, (name, res.knn_gesture, res.knn_distance))

    def test_negative_class_is_not_a_gesture(self):
        clf = GestureClassifier(knn=self.knn, mode="knn")
        res = clf.classify(detection(pose("fist"), roll=20))
        self.assertEqual(res.knn_gesture, NEGATIVE)
        self.assertEqual(res.gesture, UNKNOWN)

    def test_hybrid_veto_on_disagreement(self):
        # KNN trained only on a deliberately wrong label for one_finger poses
        rng = np.random.default_rng(3)
        labels, feats = _samples(rng, ["one_finger"], n=6)
        knn = KnnModel(["two_fingers"] * len(labels), feats)
        clf = GestureClassifier(knn=knn, mode="hybrid")
        res = clf.classify(detection(pose("one_finger")))
        self.assertEqual(res.rule_gesture, "one_finger")
        self.assertEqual(res.knn_gesture, "two_fingers")
        self.assertEqual(res.gesture, UNKNOWN)

    def test_hybrid_fallback_respects_clear_fingers(self):
        """KNN may resolve ambiguity but cannot contradict a clearly-detected finger."""
        rng = np.random.default_rng(4)
        _, feats = _samples(rng, ["index_half"], n=8)
        # user calibrated a half-curled index as one_finger -> fallback allowed
        clf = GestureClassifier(knn=KnnModel(["one_finger"] * len(feats), feats), mode="hybrid")
        res = clf.classify(detection(pose("index_half")))
        self.assertEqual(res.rule_gesture, UNKNOWN)
        self.assertEqual(res.gesture, "one_finger", res.reason)
        # ...but labelling it open_palm contradicts the clearly folded fingers
        clf = GestureClassifier(knn=KnnModel(["open_palm"] * len(feats), feats), mode="hybrid")
        self.assertEqual(clf.classify(detection(pose("index_half"))).gesture, UNKNOWN)

    def test_hybrid_agrees_with_rules_on_clean_poses(self):
        clf = GestureClassifier(knn=self.knn, mode="hybrid")
        for name in POSITIVE_POSES:
            self.assertEqual(clf.classify(detection(pose(name), roll=-30)).gesture, name)

    def test_modes_need_samples(self):
        with self.assertRaises(ValueError):
            GestureClassifier(mode="hybrid")

    def test_calibration_file_roundtrip(self):
        det = detection(pose("two_fingers"), roll=30, left=True)
        sample = {"gesture": "two_fingers", "handedness_label": det.label,
                  "frame_size": list(det.frame_size), "world_landmarks": det.world.tolist(),
                  "image_landmarks": det.image.tolist()}
        data = {"format": CALIBRATION_FORMAT, "version": 1, "feature_version": 2,
                "landmark_source": "world", "samples": [sample] * 3}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cal.json"
            p.write_text(json.dumps(data))
            for src in ("world", "image"):
                knn = KnnModel.from_file(p, src)
                self.assertEqual(len(knn), 3)
                label, conf, dist = knn.predict(
                    analyze_hand(detection(pose("two_fingers"), roll=-60),
                                 ClassifierSettings(landmark_source=src)).features)
                self.assertEqual(label, "two_fingers")
                self.assertLess(dist, 1e-6)
            self.assertIsNone(KnnModel.from_file(Path(d) / "missing.json", "world"))


if __name__ == "__main__":
    unittest.main()
