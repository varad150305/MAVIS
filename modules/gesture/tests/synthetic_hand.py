"""
Synthetic 21-landmark hands for geometry tests.

A simple kinematic model of a RIGHT hand in "model" coordinates:
  +y = up the hand (wrist -> middle MCP, length 1)
  +x = towards the thumb / index side
  +z = out of the palm (fingers flex towards +z)

`to_camera()` places the hand in camera coordinates (x right, y down,
z away from the camera — the convention of MediaPipe image and world
landmarks) with a given roll / pitch / yaw, scale and translation.
A left hand is produced by mirroring the model x axis.
"""

from __future__ import annotations

import numpy as np

PALM = {
    0: (0.0, 0.0, 0.0),
    5: (0.32, 0.93, 0.0),
    9: (0.0, 1.0, 0.0),
    13: (-0.26, 0.94, 0.0),
    17: (-0.48, 0.82, 0.0),
}
LENGTHS = {
    "index": (0.45, 0.27, 0.20),
    "middle": (0.50, 0.30, 0.21),
    "ring": (0.47, 0.29, 0.20),
    "pinky": (0.37, 0.22, 0.18),
}
BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
SPREAD_SIGN = {"index": 1.0, "middle": 0.0, "ring": -1.0, "pinky": -2.0}

EXTENDED = (5.0, 5.0, 3.0)        # MCP, PIP, DIP flexion in degrees
FOLDED = (85.0, 100.0, 60.0)
HALF = (40.0, 55.0, 30.0)         # partially curled
HOOK = (0.0, 95.0, 70.0)

# Thumb joint positions (CMC, MCP, IP, TIP) in model coordinates.
THUMB_SPREAD = ((0.22, 0.18, 0.03), (0.48, 0.36, 0.06), (0.70, 0.50, 0.07), (0.88, 0.62, 0.07))
THUMB_UP = ((0.22, 0.18, 0.05), (0.45, 0.30, 0.15), (0.66, 0.40, 0.22), (0.85, 0.48, 0.27))
THUMB_ACROSS = ((0.22, 0.18, 0.05), (0.36, 0.38, 0.20), (0.28, 0.55, 0.35), (0.12, 0.66, 0.40))
THUMB_ON_RING = ((0.22, 0.18, 0.05), (0.36, 0.38, 0.20), (0.14, 0.56, 0.36), (-0.10, 0.66, 0.42))
THUMB_ON_PINKY = ((0.22, 0.18, 0.05), (0.34, 0.36, 0.18), (0.10, 0.52, 0.32), (-0.20, 0.60, 0.35))
THUMB_PALM = ((0.22, 0.18, 0.05), (0.36, 0.36, 0.12), (0.20, 0.50, 0.15), (0.00, 0.60, 0.12))
THUMB_ALONG = ((0.22, 0.18, 0.05), (0.40, 0.35, 0.15), (0.45, 0.58, 0.27), (0.42, 0.78, 0.37))


def _rot(axis: str, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_hand(fingers: dict[str, tuple[float, float, float]],
              thumb=THUMB_ACROSS,
              spread_deg: float = 9.0) -> np.ndarray:
    """Right hand in model coordinates, (21,3)."""
    P = np.zeros((21, 3))
    for i, p in PALM.items():
        P[i] = p
    z = np.array([0.0, 0.0, 1.0])
    for name, base in BASE.items():
        phi = np.radians(SPREAD_SIGN[name] * spread_deg)
        u = np.array([np.sin(phi), np.cos(phi), 0.0])
        flex = fingers.get(name, EXTENDED)
        pos, alpha = P[base].copy(), 0.0
        for j, (L, f) in enumerate(zip(LENGTHS[name], flex)):
            alpha += np.radians(f)
            pos = pos + L * (np.cos(alpha) * u + np.sin(alpha) * z)
            P[base + 1 + j] = pos

    P[1:5] = np.asarray(thumb, dtype=np.float64)
    return P


def to_camera(model: np.ndarray, roll: float = 0.0, pitch: float = 0.0,
              yaw: float = 0.0, scale: float = 0.085,
              translate=(0.0, 0.0, 0.0), left: bool = False) -> np.ndarray:
    """Model -> camera coordinates (x right, y down, z away from camera).

    roll  — in-plane rotation, positive = clockwise on screen
    pitch — palm tilting top-away (+) / top-towards (-) the camera
    yaw   — palm turning around the vertical axis
    With all angles 0 the hand is upright, palm facing the camera.
    """
    M = np.asarray(model, dtype=np.float64).copy()
    if left:
        M[:, 0] *= -1.0
    # upright, palm towards camera: model y -> -y_cam, model z -> -z_cam
    base = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    R = _rot("z", roll) @ _rot("x", -pitch) @ _rot("y", yaw) @ base
    return (M @ R.T) * scale + np.asarray(translate, dtype=np.float64)


def to_image(cam: np.ndarray, width: int = 640, height: int = 480,
             center=(0.5, 0.5), px_per_unit: float = 1800.0) -> np.ndarray:
    """Camera coordinates (metres) -> MediaPipe-style normalized image landmarks.

    Orthographic projection; z is expressed relative to the wrist on the same
    scale as x (as MediaPipe does).
    """
    c = np.asarray(cam, dtype=np.float64)
    xs = c[:, 0] * px_per_unit / width + center[0]
    ys = c[:, 1] * px_per_unit / height + center[1]
    zs = (c[:, 2] - c[0, 2]) * px_per_unit / width
    return np.stack([xs, ys, zs], axis=1)


# ---------------------------------------------------------------- poses

def pose(name: str, spread_deg: float = 9.0) -> np.ndarray:
    F, E = FOLDED, EXTENDED
    table = {
        "one_finger": ({"index": E, "middle": F, "ring": F, "pinky": F}, THUMB_ACROSS),
        "two_fingers": ({"index": E, "middle": E, "ring": F, "pinky": F}, THUMB_ON_RING),
        "three_fingers": ({"index": E, "middle": E, "ring": E, "pinky": F}, THUMB_ON_PINKY),
        "open_palm": ({"index": E, "middle": E, "ring": E, "pinky": E}, THUMB_SPREAD),
        "thumbs_up": ({"index": F, "middle": F, "ring": F, "pinky": F}, THUMB_UP),
        # negatives
        "fist": ({"index": F, "middle": F, "ring": F, "pinky": F}, THUMB_ACROSS),
        "fist_thumb_along": ({"index": F, "middle": F, "ring": F, "pinky": F}, THUMB_ALONG),
        "half_curled": ({"index": HALF, "middle": HALF, "ring": HALF, "pinky": HALF}, THUMB_ACROSS),
        "claw": ({"index": HOOK, "middle": HOOK, "ring": HOOK, "pinky": HOOK}, THUMB_SPREAD),
        "l_shape": ({"index": E, "middle": F, "ring": F, "pinky": F}, THUMB_SPREAD),
        "four": ({"index": E, "middle": E, "ring": E, "pinky": E}, THUMB_PALM),
        "pinky_only": ({"index": F, "middle": F, "ring": F, "pinky": E}, THUMB_ACROSS),
        "index_half": ({"index": HALF, "middle": F, "ring": F, "pinky": F}, THUMB_ACROSS),
    }
    fingers, thumb = table[name]
    return make_hand(fingers, thumb, spread_deg=spread_deg)


POSITIVE_POSES = ("one_finger", "two_fingers", "three_fingers", "open_palm", "thumbs_up")
NEGATIVE_POSES = ("fist", "fist_thumb_along", "half_curled", "claw", "l_shape",
                  "four", "pinky_only", "index_half")
