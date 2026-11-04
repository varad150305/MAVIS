"""
gesture_engine.py — Intent parser.

Two gesture types:

1. "pattern" — defined by a combination of extended fingers
   [thumb, index, middle, ring, pinky], where 1 = extended.
   Example: [0,1,1,0,0] — "peace" (V sign). Editable right in config.json.

2. "recorded" — a custom gesture captured from the camera.
   Stored as a set of reference feature vectors; recognition is
   nearest-neighbor (KNN, k=1) by cosine distance.

Feature vector — the 21 landmarks expressed in a palm-local coordinate
frame (see hand_geometry.py): origin at the wrist, axes built from the
wrist, index/middle/pinky MCPs, unit = palm length. This is invariant to
translation, scale AND rotation of the hand (the original version only
removed translation and scale).

Finger states are judged from joint angles and palm-normalized distances
instead of screen-space Y comparisons, so a finger pointing sideways or
diagonally is still "extended".
"""

from __future__ import annotations

import sys

import numpy as np

from hand_geometry import (FINGER_NAMES, DegenerateHand, build_palm_frame, finger_metrics,
                           image_to_metric, normalized_features)

FEATURE_SIZE = 60
DEFAULT_FRAME = (640, 480)


def _metric(pts: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    return image_to_metric(np.asarray(pts, dtype=np.float64), *frame_size)


def landmarks_to_features(pts: np.ndarray,
                          frame_size: tuple[int, int] = DEFAULT_FRAME) -> np.ndarray:
    """(21,3) normalized image landmarks -> 60 palm-frame coordinates."""
    try:
        frame = build_palm_frame(_metric(pts, frame_size))
    except DegenerateHand:
        return np.zeros(FEATURE_SIZE)
    return normalized_features(frame.local)


def finger_states(pts: np.ndarray,
                  frame_size: tuple[int, int] = DEFAULT_FRAME) -> list[int]:
    """Returns [1/0]*5 — which fingers are extended (rotation-invariant)."""
    try:
        frame = build_palm_frame(_metric(pts, frame_size))
    except DegenerateHand:
        return [0] * 5
    m = finger_metrics(frame.local)
    return [1 if m[n].score >= 0.5 else 0 for n in FINGER_NAMES]


class GestureEngine:
    def __init__(self, gestures_cfg: dict, knn_threshold: float = 0.18):
        """
        gestures_cfg — dictionary from config.json:
        {
          "gesture_name": {
             "type": "pattern", "pattern": [0,1,1,0,0], ...
          },
          "gesture_name2": {
             "type": "recorded", "samples": [[...63 floats...], ...], ...
          }
        }
        knn_threshold — maximum cosine distance at which a recorded gesture
        counts as recognized (smaller = stricter).
        """
        self.threshold = knn_threshold
        self.pattern_gestures: dict[str, list[int]] = {}
        self._recorded_names: list[str] = []
        self._recorded_matrix: np.ndarray | None = None

        rec_vectors, rec_names = [], []
        for name, g in gestures_cfg.items():
            if g.get("type") == "pattern":
                self.pattern_gestures[name] = list(g["pattern"])
            elif g.get("type") == "recorded":
                for sample in g.get("samples", []):
                    if len(sample) != FEATURE_SIZE:
                        print(f"[gesture_engine] '{name}': sample recorded with the old "
                              "feature format, re-record it (--record)", file=sys.stderr)
                        break
                    rec_vectors.append(np.asarray(sample, dtype=np.float32))
                    rec_names.append(name)

        if rec_vectors:
            m = np.stack(rec_vectors)
            # pre-normalize rows for fast cosine distance
            self._recorded_matrix = m / np.linalg.norm(m, axis=1, keepdims=True)
            self._recorded_names = rec_names

    def recognize(self, pts: np.ndarray,
                  frame_size: tuple[int, int] = DEFAULT_FRAME) -> str | None:
        """Returns a gesture name or None.

        Recorded gestures take priority over pattern gestures:
        they are more specific (full hand shape, not just finger states).
        """
        feats = landmarks_to_features(pts, frame_size)

        # --- KNN over recorded gestures ---
        if self._recorded_matrix is not None:
            q = feats / max(float(np.linalg.norm(feats)), 1e-9)
            # cosine distance = 1 - cosine similarity
            dists = 1.0 - self._recorded_matrix @ q
            best = int(np.argmin(dists))
            if dists[best] < self.threshold:
                return self._recorded_names[best]

        # --- finger-combination gestures ---
        states = finger_states(pts, frame_size)
        for name, pattern in self.pattern_gestures.items():
            if states == pattern:
                return name

        return None
