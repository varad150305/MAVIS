"""Builds HandDetection objects from synthetic camera-space landmarks."""

from __future__ import annotations

import numpy as np

from hand_tracker import HandDetection
from tests.synthetic_hand import to_camera, to_image

W, H = 640, 480


def detection(model: np.ndarray, left: bool = False, label_score: float = 0.98,
              center=(0.5, 0.5), px_per_unit: float = 1100.0, **pose) -> HandDetection:
    """Synthetic MediaPipe output for a hand placed with to_camera(**pose).

    MediaPipe labels assume mirrored input: a geometrically right hand in the
    image is labelled "Left" (see hand_geometry.CHIRALITY_BY_LABEL).
    """
    cam = to_camera(model, left=left, **pose)
    image = to_image(cam, W, H, center=center, px_per_unit=px_per_unit)
    world = cam - cam.mean(axis=0)          # world landmarks: origin at hand centre
    label = "Right" if left else "Left"
    return HandDetection(image=image, world=world, label=label,
                         label_score=label_score, frame_size=(W, H))
