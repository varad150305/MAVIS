"""Synthetic invariance and robustness tests for hand_geometry."""

import itertools
import unittest

import numpy as np

from hand_geometry import (DegenerateHand, build_palm_frame, finger_metrics,
                           hand_orientation, image_to_metric, landmark_sanity,
                           normalized_features)
from tests.synthetic_hand import (NEGATIVE_POSES, POSITIVE_POSES, pose, to_camera,
                                  to_image)

ROLLS = (0, 30, 45, 60, 90, 180, -30, -45, -60, -90)
TILTS = [(p, y) for p in (-35, -20, 0, 20, 35) for y in (-35, 0, 35)]


class PalmFrameTests(unittest.TestCase):
    def setUp(self):
        self.model = pose("two_fingers")
        self.ref = build_palm_frame(to_camera(self.model), 1.0).local

    def assertLocalEqual(self, cam, chirality=1.0, atol=1e-9):
        local = build_palm_frame(cam, chirality).local
        np.testing.assert_allclose(local, self.ref, atol=atol)

    def test_axes_orthonormal(self):
        f = build_palm_frame(to_camera(self.model, roll=37, pitch=20, yaw=-15), 1.0)
        np.testing.assert_allclose(f.axes @ f.axes.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(f.local[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(f.local[9], [0, 1, 0], atol=1e-12)

    def test_translation_invariance(self):
        for t in [(0.3, -0.2, 0.5), (-5.0, 7.0, 2.0), (1e3, 1e3, 1e3)]:
            self.assertLocalEqual(to_camera(self.model, translate=t), atol=1e-7)

    def test_scale_invariance(self):
        for s in (1e-4, 0.01, 0.085, 1.0, 640.0):
            self.assertLocalEqual(to_camera(self.model, scale=s), atol=1e-9)

    def test_in_plane_rotation_invariance(self):
        for r in ROLLS:
            with self.subTest(roll=r):
                self.assertLocalEqual(to_camera(self.model, roll=r))

    def test_3d_rotation_invariance(self):
        for (p, y), r in itertools.product(TILTS, (0, 45, -90, 180)):
            with self.subTest(pitch=p, yaw=y, roll=r):
                self.assertLocalEqual(to_camera(self.model, roll=r, pitch=p, yaw=y))

    def test_left_hand_maps_to_same_local_frame(self):
        for r in (0, 45, -90):
            cam = to_camera(self.model, roll=r, left=True)
            self.assertLocalEqual(cam, chirality=-1.0)

    def test_wrong_chirality_mirrors_z_only(self):
        cam = to_camera(self.model, left=True)
        local = build_palm_frame(cam, 1.0).local
        np.testing.assert_allclose(local[:, :2], self.ref[:, :2], atol=1e-9)
        np.testing.assert_allclose(local[:, 2], -self.ref[:, 2], atol=1e-9)

    def test_image_fallback_matches_world(self):
        """Isotropic image coordinates give the same local frame as world ones."""
        for r, p in [(0, 0), (45, 20), (-90, -25)]:
            cam = to_camera(self.model, roll=r, pitch=p)
            img = to_image(cam, 640, 480)
            local = build_palm_frame(image_to_metric(img, 640, 480), 1.0).local
            np.testing.assert_allclose(local, self.ref, atol=1e-9)

    def test_features_shape(self):
        f = normalized_features(self.ref)
        self.assertEqual(f.shape, (60,))
        self.assertTrue(np.all(np.isfinite(f)))


class DegenerateTests(unittest.TestCase):
    def test_wrong_shape(self):
        with self.assertRaises(DegenerateHand):
            build_palm_frame(np.zeros((20, 3)))

    def test_all_zero(self):
        with self.assertRaises(DegenerateHand):
            build_palm_frame(np.zeros((21, 3)))

    def test_nan_and_inf(self):
        for bad in (np.nan, np.inf):
            pts = to_camera(pose("open_palm"))
            pts[7, 1] = bad
            with self.assertRaises(DegenerateHand):
                build_palm_frame(pts)

    def test_collinear_mcps(self):
        pts = to_camera(pose("open_palm"))
        # put index and pinky MCPs on the wrist->middle MCP line
        for i, t in ((5, 0.95), (17, 0.8)):
            pts[i] = pts[0] + t * (pts[9] - pts[0])
        with self.assertRaises(DegenerateHand):
            build_palm_frame(pts)

    def test_wrist_on_middle_mcp(self):
        pts = to_camera(pose("open_palm"))
        pts[9] = pts[0]
        with self.assertRaises(DegenerateHand):
            build_palm_frame(pts)

    def test_sanity_rejects_collapsed_bone(self):
        pts = to_camera(pose("open_palm"))
        self.assertIsNone(landmark_sanity(pts))
        pts[8] = pts[7]
        self.assertIn("collapsed", landmark_sanity(pts))

    def test_sanity_rejects_overlong_bone(self):
        pts = to_camera(pose("open_palm"))
        pts[8] = pts[7] + 3.0 * (pts[9] - pts[0])
        self.assertIn("long", landmark_sanity(pts))

    def test_tiny_noise_is_stable(self):
        rng = np.random.default_rng(0)
        cam = to_camera(pose("three_fingers"), roll=30)
        ref = finger_metrics(build_palm_frame(cam).local)
        for _ in range(50):
            noisy = cam + rng.normal(0, 0.0015, cam.shape)    # ~1.5 mm on a 8.5 cm palm
            m = finger_metrics(build_palm_frame(noisy).local)
            for n in ref:
                # thumb clearance ramps over ~1.7 cm, so it is the most noise-sensitive
                self.assertLess(abs(m[n].score - ref[n].score), 0.2 if n == "thumb" else 0.12, n)


class FingerMetricTests(unittest.TestCase):
    def test_straight_finger_extended_in_any_direction(self):
        """Extension does not depend on where the finger points on screen."""
        for r in ROLLS:
            m = finger_metrics(build_palm_frame(to_camera(pose("one_finger"), roll=r)).local)
            self.assertGreater(m["index"].score, 0.9, r)
            self.assertLess(m["middle"].score, 0.1, r)

    def test_scores_identical_under_rotation(self):
        for name in POSITIVE_POSES + NEGATIVE_POSES:
            ref = finger_metrics(build_palm_frame(to_camera(pose(name))).local)
            for r, (p, y) in itertools.product(ROLLS, TILTS):
                m = finger_metrics(build_palm_frame(
                    to_camera(pose(name), roll=r, pitch=p, yaw=y)).local)
                for n in ref:
                    self.assertAlmostEqual(m[n].score, ref[n].score, places=9)

    def test_partial_curl_is_ambiguous(self):
        m = finger_metrics(build_palm_frame(to_camera(pose("half_curled"))).local)
        for n in ("index", "middle", "ring"):
            self.assertTrue(0.4 < m[n].score < 0.7, (n, m[n].score))


class OrientationTests(unittest.TestCase):
    def test_roll_reported(self):
        for r in (0, 30, 45, 60, 90, -45, -90, 170):
            o = hand_orientation(build_palm_frame(to_camera(pose("open_palm"), roll=r)))
            self.assertAlmostEqual(o.roll_deg, r, delta=0.5)

    def test_palm_vs_back_and_tilt(self):
        f = build_palm_frame(to_camera(pose("open_palm")), 1.0)
        self.assertEqual(hand_orientation(f).facing, "palm")
        f = build_palm_frame(to_camera(pose("open_palm"), yaw=180), 1.0)
        self.assertEqual(hand_orientation(f).facing, "back")
        f = build_palm_frame(to_camera(pose("open_palm"), left=True), -1.0)
        self.assertEqual(hand_orientation(f).facing, "palm")
        for p in (20, 40, 60):
            o = hand_orientation(build_palm_frame(to_camera(pose("open_palm"), pitch=p)))
            self.assertAlmostEqual(o.tilt_deg, p, delta=0.5)
        o = hand_orientation(build_palm_frame(to_camera(pose("open_palm"), yaw=85)))
        self.assertEqual(o.facing, "edge-on")


if __name__ == "__main__":
    unittest.main()
