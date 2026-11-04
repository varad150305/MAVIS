"""
hand_tracker.py — Vision module.

Wrapper around MediaPipe Hands: takes a BGR frame from OpenCV and
returns hand landmarks.

21 landmarks per hand, each is (x, y, z) with x/y normalized to frame size.
Index scheme: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
  0  — wrist
  4  — thumb tip
  8  — index finger tip
  12 — middle finger tip
  16 — ring finger tip
  20 — pinky tip

process()          — legacy API: list of (21,3) normalized image landmarks.
process_detailed() — list of HandDetection: image landmarks, metric world
                     landmarks (camera-aligned axes, metres) and handedness.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HandDetection:
    image: np.ndarray               # (21,3) normalized image landmarks
    world: np.ndarray | None        # (21,3) world landmarks in metres, if available
    label: str | None               # raw MediaPipe handedness label ("Left"/"Right")
    label_score: float              # handedness classification score 0..1
    frame_size: tuple[int, int]     # (width, height) of the processed frame


class HandTracker:
    def __init__(self, max_hands: int = 1,
                 detection_conf: float = 0.7,
                 tracking_conf: float = 0.6,
                 model_complexity: int = 1):
        self._mp_hands = mp.solutions.hands
        self._drawer = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._last_raw = None

    def process_detailed(self, frame_bgr) -> list[HandDetection]:
        """Returns one HandDetection per detected hand."""
        # MediaPipe expects RGB and a c_contiguous array; slicing with
        # [..., ::-1] would produce a negative-stride view — cvtColor copies
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        self._last_raw = result

        h, w = frame_bgr.shape[:2]
        hands: list[HandDetection] = []
        if not result.multi_hand_landmarks:
            return hands
        worlds = result.multi_hand_world_landmarks or []
        labels = result.multi_handedness or []
        for i, hand_lms in enumerate(result.multi_hand_landmarks):
            image = np.array([(lm.x, lm.y, lm.z) for lm in hand_lms.landmark],
                             dtype=np.float64)
            world = None
            if i < len(worlds):
                world = np.array([(lm.x, lm.y, lm.z) for lm in worlds[i].landmark],
                                 dtype=np.float64)
            label, score = None, 0.0
            if i < len(labels):
                cls = labels[i].classification[0]
                label, score = cls.label, float(cls.score)
            hands.append(HandDetection(image, world, label, score, (w, h)))
        return hands

    def process(self, frame_bgr) -> list[np.ndarray]:
        """Returns a list of hands; each hand is an array of shape (21, 3)."""
        return [d.image.astype(np.float32) for d in self.process_detailed(frame_bgr)]

    def draw(self, frame_bgr) -> None:
        """Draws the hand skeleton on the frame (Feedback layer). Call after process()."""
        result = self._last_raw
        if result and result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                self._drawer.draw_landmarks(
                    frame_bgr, hand_lms, self._mp_hands.HAND_CONNECTIONS
                )

    def close(self) -> None:
        self._hands.close()
