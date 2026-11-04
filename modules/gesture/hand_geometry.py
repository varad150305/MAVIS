"""
hand_geometry.py — rotation-invariant hand geometry.

Everything here works on MediaPipe's 21 hand landmarks and is independent
of where the hand is in the frame, how large it is and how it is rotated.

Palm coordinate frame
---------------------
  origin  — wrist (landmark 0)
  +y      — wrist -> middle-finger MCP (landmark 9): "up the hand"
  +x      — pinky MCP (17) -> index MCP (5), made orthogonal to +y:
            "towards the thumb side"
  +z      — palm normal, chosen so that +z always points OUT OF THE PALM
            for both left and right hands (see `chirality`)
  unit    — |wrist -> middle MCP| = 1 ("palm length")

Because x and y are built from the landmarks themselves, a hand rotated in
the image plane (or tilted in 3D) produces the same local coordinates.
A left hand is the mirror image of a right hand: x × y points out of the
palm for one and out of the back for the other, so z is multiplied by the
chirality sign to make both hands land in the same local frame.

Which coordinates go in
-----------------------
  * MediaPipe world landmarks (metres, camera-aligned axes) — preferred.
    Verified with mediapipe 0.10.21: the axes match the image landmarks.
  * Image landmarks, converted with `image_to_metric()` so x and y share
    one pixel scale (normalized x/y are anisotropic on a 640x480 frame).

Finger extension is judged from joint angles and palm-normalized distances,
never from screen-space Y, so a straight finger is "extended" whether it
points up, sideways or diagonally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WRIST = 0
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
# (base, joint1, joint2, tip) for every finger; thumb = CMC, MCP, IP, TIP
FINGER_JOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
# MediaPipe hand skeleton (bone list) — used for sanity checks
BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]

# Landmarks the thumb tip touches when it is tucked: folded finger joints
# and the palm surface near the MCPs (plus the palm centre, added below).
_THUMB_CONTACT_POINTS = [6, 7, 10, 11, 14, 15, 18, 19, MIDDLE_MCP, RING_MCP]
_PALM_POINTS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# Chirality per RAW MediaPipe handedness label. MediaPipe labels hands
# assuming a mirrored (selfie) image, so a hand labelled "Left" is
# geometrically a right hand in the image it was given, for which
# x × y already points out of the palm. This does not depend on whether
# *we* mirrored the frame: label and geometry always come from the same image.
CHIRALITY_BY_LABEL = {"Left": 1.0, "Right": -1.0}


def chirality_from_label(label: str | None) -> float:
    return CHIRALITY_BY_LABEL.get(label or "", 1.0)


def anatomical_hand(label: str | None, mirrored_input: bool) -> str:
    """The user's real hand. MediaPipe's label is correct for mirrored input."""
    if label not in ("Left", "Right"):
        return "Unknown"
    if mirrored_input:
        return label
    return "Right" if label == "Left" else "Left"


def image_to_metric(pts_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalized image landmarks -> isotropic pixel-scale coordinates.

    MediaPipe's z uses roughly the same scale as x, so it is scaled by width.
    """
    pts = np.asarray(pts_norm, dtype=np.float64)
    return pts * np.array([width, height, width], dtype=np.float64)


def ramp(value: float, lo: float, hi: float) -> float:
    """Linear 0..1 ramp: 0 at `lo`, 1 at `hi` (works for lo > hi too)."""
    if hi == lo:
        return 1.0 if value >= hi else 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def bend_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Flexion at joint b in degrees: 0 = a, b, c collinear (straight)."""
    u, v = b - a, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 180.0   # collapsed bone: treat as maximally bent (not extended)
    cos = float(np.dot(u, v) / (nu * nv))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))


# ---------------------------------------------------------------- palm frame

@dataclass
class PalmFrame:
    origin: np.ndarray        # wrist position in source coordinates
    axes: np.ndarray          # (3,3), rows = local x, y, z in source coordinates
    scale: float              # palm length in source units
    chirality: float          # +1 / -1, already folded into axes[2]
    local: np.ndarray         # (21,3) landmarks in the palm frame, palm length = 1

    @property
    def palm_normal(self) -> np.ndarray:
        """Unit vector pointing out of the palm, in source coordinates."""
        return self.axes[2]

    @property
    def up(self) -> np.ndarray:
        """Unit vector wrist -> middle MCP, in source coordinates."""
        return self.axes[1]


class DegenerateHand(ValueError):
    """Landmarks cannot define a reliable palm coordinate frame."""


def build_palm_frame(pts: np.ndarray, chirality: float = 1.0,
                     min_width_ratio: float = 0.25) -> PalmFrame:
    """Builds the palm frame and transforms all 21 landmarks into it.

    Raises DegenerateHand for NaN/inf input, a collapsed palm or index/pinky
    MCPs that are (nearly) collinear with the wrist -> middle-MCP axis.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (21, 3):
        raise DegenerateHand(f"expected (21,3) landmarks, got {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise DegenerateHand("non-finite landmark coordinates")

    origin = pts[WRIST]
    y = pts[MIDDLE_MCP] - origin
    scale = float(np.linalg.norm(y))
    spread = float(np.max(np.linalg.norm(pts - origin, axis=1)))
    if scale < 1e-9 or scale < 1e-3 * spread:
        raise DegenerateHand("palm length is zero")
    y /= scale

    t = pts[INDEX_MCP] - pts[PINKY_MCP]
    x = t - np.dot(t, y) * y
    width = float(np.linalg.norm(x))
    if width < min_width_ratio * scale:
        raise DegenerateHand("palm width collapsed (MCPs collinear with palm axis)")
    x /= width

    z = np.cross(x, y) * (1.0 if chirality >= 0 else -1.0)
    axes = np.stack([x, y, z])
    local = (pts - origin) @ axes.T / scale
    return PalmFrame(origin=origin, axes=axes, scale=scale,
                     chirality=1.0 if chirality >= 0 else -1.0, local=local)


def normalized_features(local: np.ndarray) -> np.ndarray:
    """Flat KNN feature vector: landmarks 1..20 in the palm frame (60 floats).

    The wrist is always (0,0,0) and is dropped.
    """
    return np.asarray(local, dtype=np.float64)[1:].reshape(-1)


# ---------------------------------------------------------------- sanity checks

def landmark_sanity(pts: np.ndarray) -> str | None:
    """Returns a reason string if the hand skeleton is implausible, else None.

    Catches landmark sets that are finite and non-collinear but still not a
    believable hand (e.g. a bone of zero length or a 3x-too-long finger),
    which happens with heavy occlusion or bad detections.
    """
    pts = np.asarray(pts, dtype=np.float64)
    scale = float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]))
    if scale < 1e-9:
        return "palm length is zero"
    for a, b in BONES:
        r = float(np.linalg.norm(pts[b] - pts[a])) / scale
        if r < 0.03:
            return f"bone {a}-{b} collapsed"
        if r > 1.6:
            return f"bone {a}-{b} implausibly long"
    width = float(np.linalg.norm(pts[INDEX_MCP] - pts[PINKY_MCP])) / scale
    if not 0.25 <= width <= 1.5:
        return "implausible palm proportions"
    return None


# ---------------------------------------------------------------- finger metrics

@dataclass
class FingerMetrics:
    name: str
    score: float              # 0 = clearly folded, 1 = clearly extended
    curl_deg: float           # sum of flexion at the two distal joints
    base_flex_deg: float      # flexion at the base joint (MCP; thumb: CMC->MCP)
    reach: float              # finger-specific palm-normalized distance
    extra: dict = field(default_factory=dict)


@dataclass
class ExtensionThresholds:
    """Ramps mapping raw geometry to a 0..1 extension score.

    Each pair is (fully-folded value, fully-extended value). Defaults were set
    from real webcam calibration data (world landmarks, 96 samples, two
    hands): extended fingers measured <= 87 deg total flexion and >= 1.58
    reach; folded fingers >= 131 deg and mostly <= 1.25 reach; tucked thumbs
    had clearance <= 0.30, extended thumbs >= 0.54.
    """
    finger_flex: tuple[float, float] = (140.0, 80.0)      # MCP+PIP+DIP flexion, deg
    finger_reach: tuple[float, float] = (1.20, 1.55)      # |tip-wrist| / |mcp-wrist|
    thumb_curl: tuple[float, float] = (100.0, 40.0)       # MCP+IP flexion, deg
    thumb_reach: tuple[float, float] = (0.35, 0.65)       # |thumb tip - index MCP|
    thumb_clearance: tuple[float, float] = (0.28, 0.52)   # distance to folded fingers/palm
    thumb_weights: tuple[float, float, float] = (0.2, 0.1, 0.7)   # curl, reach, clearance


def finger_metrics(local: np.ndarray,
                   th: ExtensionThresholds | None = None) -> dict[str, FingerMetrics]:
    """Per-finger extension scores from palm-frame landmarks.

    Fingers (index..pinky): total flexion along the finger (MCP + PIP + DIP)
    and how far the tip reaches from the wrist relative to that finger's
    metacarpal. Total flexion is used rather than PIP + DIP alone because
    when a fist is seen from the back the fingertips are hidden and MediaPipe
    underestimates the distal joints, while the knuckle bend stays visible.
    A half-curled finger lands in the middle of the scale.

    Thumb: straightness at MCP + IP, distance of the tip from the index MCP,
    and clearance from the folded fingers / palm surface. A straight thumb
    tucked along the side of the fist scores low on reach and clearance.
    """
    th = th or ExtensionThresholds()
    P = np.asarray(local, dtype=np.float64)
    out: dict[str, FingerMetrics] = {}

    for name in FINGER_NAMES[1:]:
        m, p, d, t = FINGER_JOINTS[name]
        base_flex = bend_angle(P[WRIST], P[m], P[p])
        curl = bend_angle(P[m], P[p], P[d]) + bend_angle(P[p], P[d], P[t])
        meta = float(np.linalg.norm(P[m] - P[WRIST]))
        reach = float(np.linalg.norm(P[t] - P[WRIST])) / max(meta, 1e-9)
        s_flex = ramp(base_flex + curl, *th.finger_flex)
        s_reach = ramp(reach, *th.finger_reach)
        out[name] = FingerMetrics(name, 0.5 * (s_flex + s_reach), curl, base_flex,
                                  reach, {"s_flex": s_flex, "s_reach": s_reach})

    c, m, i, t = FINGER_JOINTS["thumb"]
    base_flex = bend_angle(P[WRIST], P[c], P[m])
    curl = bend_angle(P[c], P[m], P[i]) + bend_angle(P[m], P[i], P[t])
    reach = float(np.linalg.norm(P[t] - P[INDEX_MCP]))
    contacts = np.vstack([P[_THUMB_CONTACT_POINTS], P[_PALM_POINTS].mean(axis=0)])
    clearance = float(np.min(np.linalg.norm(contacts - P[t], axis=1)))
    s_curl = ramp(curl, *th.thumb_curl)
    s_reach = ramp(reach, *th.thumb_reach)
    s_clear = ramp(clearance, *th.thumb_clearance)
    w = th.thumb_weights
    score = (w[0] * s_curl + w[1] * s_reach + w[2] * s_clear) / sum(w)
    out["thumb"] = FingerMetrics("thumb", score, curl, base_flex, reach,
                                 {"clearance": clearance, "s_curl": s_curl,
                                  "s_reach": s_reach, "s_clear": s_clear})
    return out


def finger_spread_deg(local: np.ndarray) -> dict[str, float]:
    """Angles between finger directions (MCP -> tip), in degrees."""
    P = np.asarray(local, dtype=np.float64)

    def direction(name: str) -> np.ndarray:
        b, _, _, t = FINGER_JOINTS[name]
        return P[t] - P[b]

    return {
        "thumb_index": _angle_between(P[4] - P[2], direction("index")),
        "index_middle": _angle_between(direction("index"), direction("middle")),
        "middle_ring": _angle_between(direction("middle"), direction("ring")),
        "ring_pinky": _angle_between(direction("ring"), direction("pinky")),
        "index_pinky": _angle_between(direction("index"), direction("pinky")),
    }


# ---------------------------------------------------------------- orientation

@dataclass
class Orientation:
    roll_deg: float     # in-plane rotation of the hand on screen; 0 = fingers up, + = clockwise
    tilt_deg: float     # angle between palm normal and camera axis; 0 = palm square to camera
    facing: str         # "palm" | "back" | "edge-on"

    def describe(self) -> str:
        # plain ASCII: OpenCV's Hershey fonts cannot draw a degree sign
        r = self.roll_deg
        if abs(r) < 15:
            roll = "upright"
        elif abs(r) > 165:
            roll = "upside-down"
        else:
            roll = f"rotated {'CW' if r > 0 else 'CCW'} {abs(r):.0f} deg"
        if self.facing == "edge-on":
            return f"{roll}, edge-on"
        tilt = "" if self.tilt_deg < 25 else f", tilted {self.tilt_deg:.0f} deg"
        return f"{roll}, {self.facing} to camera{tilt}"


def hand_orientation(frame: PalmFrame, edge_on_deg: float = 72.0) -> Orientation:
    """Orientation relative to the camera.

    Requires `frame` to be built from camera-aligned coordinates (world
    landmarks or image_to_metric output): x right, y down, z away from camera.
    """
    up = frame.up
    roll = float(np.degrees(np.arctan2(up[0], -up[1])))
    n = frame.palm_normal
    # camera looks along +z, so a palm facing the camera has normal z < 0
    tilt = float(np.degrees(np.arccos(np.clip(abs(n[2]), 0.0, 1.0))))
    if tilt > edge_on_deg:
        facing = "edge-on"
    else:
        facing = "palm" if n[2] < 0 else "back"
    return Orientation(roll_deg=roll, tilt_deg=tilt, facing=facing)
