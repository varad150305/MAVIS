"""
gesture_classifier.py — rotation-aware recognition of the five MAVIS gestures.

Pipeline for one hand:

  HandDetection --analyze_hand()--> HandAnalysis   (palm frame, finger scores,
                                                    orientation, quality check)
               --classify_rules()-> primary result (deterministic geometry)
               --KnnModel.predict-> optional refinement from calibration samples
               --combine (mode)---> Classification

Modes:
  rules  — geometry only (default when no calibration file exists)
  knn    — calibration samples only
  hybrid — geometry is primary; KNN may veto a rule result it confidently
           disagrees with, and may resolve a pose the rules found ambiguous,
           but only if it does not contradict any clearly-detected finger.

Anything that is not confidently one of the five gestures is UNKNOWN.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hand_geometry import (DegenerateHand, ExtensionThresholds, FingerMetrics,
                           Orientation, PalmFrame, anatomical_hand, build_palm_frame,
                           chirality_from_label, finger_metrics, finger_spread_deg,
                           hand_orientation, image_to_metric, landmark_sanity,
                           normalized_features)
from hand_tracker import HandDetection

UNKNOWN = "UNKNOWN"
NO_HAND = "NO_HAND"   # no hand, or landmarks too unreliable to count as a hand
NEGATIVE = "none"     # calibration class for "not a gesture" (fist, relaxed hand…)

GESTURES = ("one_finger", "two_fingers", "three_fingers", "open_palm", "thumbs_up")

# What each gesture *means* (features, labels, timings) lives in
# config/gesture_mapping.json / gesture_mapping.py, not here.

# Required finger states per gesture: E = extended, F = folded,
# T = tucked / not extended: score <= `loose_folded_max`. Used for the thumb
# (people rest it on the folded fingers in many slightly different ways) and
# for the pinky in three_fingers (ring and pinky share tendons, so with the
# ring raised the pinky often stays half-bent). A clearly extended T-finger
# still rejects the gesture.
TEMPLATES = {
    "one_finger":    {"thumb": "T", "index": "E", "middle": "F", "ring": "F", "pinky": "F"},
    "two_fingers":   {"thumb": "T", "index": "E", "middle": "E", "ring": "F", "pinky": "F"},
    "three_fingers": {"thumb": "T", "index": "E", "middle": "E", "ring": "E", "pinky": "T"},
    "open_palm":     {"thumb": "E", "index": "E", "middle": "E", "ring": "E", "pinky": "E"},
    "thumbs_up":     {"thumb": "E", "index": "F", "middle": "F", "ring": "F", "pinky": "F"},
}

FEATURE_VERSION = 2   # palm-frame features; bump when normalized_features changes


# ---------------------------------------------------------------- settings

@dataclass
class ClassifierSettings:
    landmark_source: str = "world"        # "world" or "image"
    extended_threshold: float = 0.60      # finger score >= this -> extended
    folded_threshold: float = 0.40        # finger score <= this -> folded
    loose_folded_max: float = 0.5         # "T" requirement: score <= this
    open_palm_min_spread_deg: float = 12.0    # index-vs-pinky direction angle
    open_palm_min_thumb_deg: float = 18.0     # thumb-vs-index direction angle
    thumbs_up_max_angle_deg: float = 80.0     # thumb vs screen-up; 180 disables
    thumbs_up_max_finger_reach: float = 1.2   # fist must be compact (rejects a "claw")
    max_tilt_deg: float = 72.0            # palm more edge-on than this -> UNKNOWN
    tilt_exempt: tuple[str, ...] = ("thumbs_up",)   # usually shown side-on
    min_handedness_score: float = 0.5
    min_palm_px: float = 28.0             # wrist->middle MCP in pixels ("too far")
    max_out_of_frame: int = 1             # landmarks allowed outside the image
    thresholds: ExtensionThresholds = field(default_factory=ExtensionThresholds)

    @classmethod
    def from_dict(cls, d: dict | None) -> "ClassifierSettings":
        d = dict(d or {})
        th = ExtensionThresholds(**{k: tuple(v) for k, v in d.pop("thresholds", {}).items()})
        if "tilt_exempt" in d:
            d["tilt_exempt"] = tuple(d["tilt_exempt"])
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            print(f"[classifier] ignoring unknown settings: {sorted(unknown)}", file=sys.stderr)
        return cls(thresholds=th, **{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------- analysis

@dataclass
class HandAnalysis:
    hand: str                              # anatomical "Left"/"Right"/"Unknown"
    reject_reason: str | None = None       # set -> hand cannot be classified at all
    tracking_lost: bool = False            # reject because tracking is unreliable
    frame: PalmFrame | None = None
    metrics: dict[str, FingerMetrics] = field(default_factory=dict)
    spread: dict[str, float] = field(default_factory=dict)
    orientation: Orientation | None = None
    thumb_screen_angle: float = 0.0        # thumb MCP->tip vs screen-up, degrees
    features: np.ndarray | None = None

    @property
    def scores(self) -> dict[str, float]:
        return {k: m.score for k, m in self.metrics.items()}


def _source_points(det: HandDetection, source: str) -> np.ndarray:
    w, h = det.frame_size
    if source == "world" and det.world is not None:
        return det.world
    return image_to_metric(det.image, w, h)


def analyze_hand(det: HandDetection, settings: ClassifierSettings,
                 mirrored_input: bool = True) -> HandAnalysis:
    """Builds the palm frame and all geometry needed to classify one hand."""
    a = HandAnalysis(hand=anatomical_hand(det.label, mirrored_input))
    w, h = det.frame_size
    img = image_to_metric(det.image, w, h)

    if det.label_score < settings.min_handedness_score:
        a.reject_reason = "low tracking confidence"
        a.tracking_lost = True
        return a
    outside = int(np.sum((det.image[:, 0] < -0.01) | (det.image[:, 0] > 1.01) |
                         (det.image[:, 1] < -0.01) | (det.image[:, 1] > 1.01)))
    if outside > settings.max_out_of_frame:
        a.reject_reason = "hand partially out of frame"
        a.tracking_lost = True
        return a
    # palm length or width in pixels; foreshortening rarely shrinks both
    palm_px = max(np.linalg.norm(img[9, :2] - img[0, :2]),
                  np.linalg.norm(img[5, :2] - img[17, :2]) * 1.25)
    if palm_px < settings.min_palm_px:
        a.reject_reason = "hand too far / too small"
        return a

    pts = _source_points(det, settings.landmark_source)
    reason = landmark_sanity(pts)
    if reason:
        a.reject_reason = f"unreliable landmarks ({reason})"
        a.tracking_lost = True
        return a
    chir = chirality_from_label(det.label)
    try:
        a.frame = build_palm_frame(pts, chir)
    except DegenerateHand as e:
        a.reject_reason = f"degenerate palm frame ({e})"
        a.tracking_lost = True
        return a

    a.metrics = finger_metrics(a.frame.local, settings.thresholds)
    a.spread = finger_spread_deg(a.frame.local)
    a.orientation = hand_orientation(a.frame)
    a.features = normalized_features(a.frame.local)

    thumb = img[4, :2] - img[2, :2]
    if np.linalg.norm(thumb) > 1e-9:
        cos = -thumb[1] / np.linalg.norm(thumb)          # screen-up is -y
        a.thumb_screen_angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return a


# ---------------------------------------------------------------- rules

def finger_state(name: str, score: float, s: ClassifierSettings) -> str:
    """E (extended), F (folded), T (loosely folded) or ? (ambiguous)."""
    if score >= s.extended_threshold:
        return "E"
    if score <= s.folded_threshold:
        return "F"
    if score <= s.loose_folded_max:
        return "T"
    return "?"


def _state_matches(required: str, state: str) -> bool:
    if required == "T":
        return state in ("F", "T")
    return state == required


def _certainty(required: str, score: float, s: ClassifierSettings) -> float:
    """0.5 at the decision threshold, 1.0 at the far end of the scale."""
    if required == "E":
        return 0.5 + 0.5 * float(np.clip((score - s.extended_threshold) /
                                         max(1 - s.extended_threshold, 1e-9), 0, 1))
    thr = s.loose_folded_max if required == "T" else s.folded_threshold
    return 0.5 + 0.5 * float(np.clip((thr - score) / max(thr, 1e-9), 0, 1))


def gesture_checks(gesture: str, a: HandAnalysis, s: ClassifierSettings) -> str | None:
    """Gesture-specific checks beyond finger states. Returns a reject reason."""
    if gesture == "open_palm":
        if a.spread["index_pinky"] < s.open_palm_min_spread_deg:
            return "fingers not spread"
        if a.spread["thumb_index"] < s.open_palm_min_thumb_deg:
            return "thumb not spread"
    if gesture == "thumbs_up":
        if a.thumb_screen_angle > s.thumbs_up_max_angle_deg:
            return "thumb not pointing up"
        if max(a.metrics[n].reach for n in ("index", "middle", "ring", "pinky")) >                 s.thumbs_up_max_finger_reach:
            return "fingers not curled into a fist"
    if (a.orientation is not None and a.orientation.tilt_deg > s.max_tilt_deg
            and gesture not in s.tilt_exempt):
        return "hand edge-on to camera"
    return None


def classify_rules(a: HandAnalysis, s: ClassifierSettings) -> tuple[str, float, str]:
    """Returns (gesture or UNKNOWN, confidence 0..1, reason)."""
    if a.reject_reason:
        return UNKNOWN, 0.0, a.reject_reason
    scores = a.scores
    states = {n: finger_state(n, sc, s) for n, sc in scores.items()}
    for gesture, template in TEMPLATES.items():
        if all(_state_matches(template[n], states[n]) for n in template):
            why = gesture_checks(gesture, a, s)
            if why:
                return UNKNOWN, 0.0, why
            conf = min(_certainty(template[n], scores[n], s) for n in template)
            return gesture, conf, "rules"
    if all(states[n] == "F" for n in ("index", "middle", "ring", "pinky")) and \
            states["thumb"] != "E":
        return UNKNOWN, 0.0, "fist"
    amb = [n for n, st in states.items() if st in ("?", "T")]
    if amb:
        return UNKNOWN, 0.0, "ambiguous: " + ", ".join(amb)
    return UNKNOWN, 0.0, "not a known gesture"


def template_compatible(gesture: str, a: HandAnalysis, s: ClassifierSettings) -> bool:
    """True if no clearly-detected finger contradicts the gesture template."""
    for n, req in TEMPLATES[gesture].items():
        st = finger_state(n, a.metrics[n].score, s)
        if st == "?":
            continue
        if req == "E" and st == "F":
            return False
        if req in ("F", "T") and st == "E":
            return False
    return True


# ---------------------------------------------------------------- KNN

@dataclass
class KnnSettings:
    k: int = 3
    max_distance: float = 0.30          # RMS palm-frame distance (palm lengths)
    min_margin: float = 1.15            # 2nd-best class distance / best distance
    fallback_min_confidence: float = 0.5

    @classmethod
    def from_dict(cls, d: dict | None) -> "KnnSettings":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class KnnModel:
    """Nearest-neighbour matcher over palm-frame features.

    Samples are compared by RMS landmark distance in palm lengths, so the
    threshold has a physical meaning ("on average every landmark is within
    0.3 palm lengths of a recorded sample").
    """

    def __init__(self, labels: list[str], features: np.ndarray,
                 settings: KnnSettings | None = None):
        self.settings = settings or KnnSettings()
        self.labels = np.asarray(labels)
        self.features = np.asarray(features, dtype=np.float64)
        self.classes = sorted(set(labels))
        self._n_points = self.features.shape[1] // 3 if len(self.features) else 20

    def __len__(self) -> int:
        return len(self.labels)

    def predict(self, feat: np.ndarray) -> tuple[str | None, float, float]:
        """Returns (label or None if no confident match, confidence, distance)."""
        if not len(self.labels):
            return None, 0.0, float("inf")
        d = np.linalg.norm(self.features - feat, axis=1) / np.sqrt(self._n_points)
        per_class = {}
        for c in self.classes:
            dc = np.sort(d[self.labels == c])[: self.settings.k]
            per_class[c] = float(np.mean(dc))
        ranked = sorted(per_class.items(), key=lambda kv: kv[1])
        best, d1 = ranked[0]
        if d1 > self.settings.max_distance:
            return None, 0.0, d1
        if len(ranked) > 1 and ranked[1][1] < d1 * self.settings.min_margin:
            return None, 0.0, d1
        conf = 1.0 - d1 / self.settings.max_distance
        return best, 0.5 + 0.5 * conf, d1

    # ---- persistence -------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path, landmark_source: str,
                  settings: KnnSettings | None = None) -> "KnnModel | None":
        path = Path(path)
        if not path.exists():
            return None
        data = load_calibration(path)
        if data.get("landmark_source") != landmark_source:
            print(f"[knn] {path.name} was recorded with landmark_source="
                  f"{data.get('landmark_source')!r}, current is {landmark_source!r}; "
                  "features are recomputed from the stored landmarks", file=sys.stderr)
        labels, feats = [], []
        for smp in data.get("samples", []):
            f = sample_features(smp, landmark_source)
            if f is not None:
                labels.append(smp["gesture"])
                feats.append(f)
        if not labels:
            return None
        return cls(labels, np.stack(feats), settings)


def sample_features(sample: dict, landmark_source: str) -> np.ndarray | None:
    """Recomputes palm-frame features from a stored calibration sample."""
    key = "world_landmarks" if landmark_source == "world" else "image_landmarks"
    pts = sample.get(key)
    if pts is None:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    if landmark_source == "image":
        w, h = sample.get("frame_size", [640, 480])
        pts = image_to_metric(pts, w, h)
    try:
        frame = build_palm_frame(pts, chirality_from_label(sample.get("handedness_label")))
    except DegenerateHand:
        return None
    return normalized_features(frame.local)


CALIBRATION_FORMAT = "mavis-gesture-calibration"


def load_calibration(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != CALIBRATION_FORMAT:
        raise ValueError(f"{path}: not a {CALIBRATION_FORMAT} file")
    return data


# ---------------------------------------------------------------- combined

@dataclass
class Classification:
    gesture: str                    # one of GESTURES, UNKNOWN or NO_HAND
    confidence: float
    reason: str
    analysis: HandAnalysis
    rule_gesture: str = UNKNOWN
    rule_confidence: float = 0.0
    knn_gesture: str | None = None  # None = KNN unavailable or abstained
    knn_confidence: float = 0.0
    knn_distance: float = float("inf")

    @property
    def is_gesture(self) -> bool:
        return self.gesture in GESTURES


class GestureClassifier:
    MODES = ("rules", "knn", "hybrid")

    def __init__(self, settings: ClassifierSettings | None = None,
                 knn: KnnModel | None = None, mode: str = "rules",
                 knn_settings: KnnSettings | None = None,
                 mirrored_input: bool = True):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if mode in ("knn", "hybrid") and knn is None:
            raise ValueError(f"mode {mode!r} needs calibration samples (run --calibrate)")
        self.settings = settings or ClassifierSettings()
        self.knn = knn
        self.knn_settings = knn_settings or (knn.settings if knn else KnnSettings())
        self.mode = mode
        self.mirrored_input = mirrored_input

    def classify(self, det: HandDetection) -> Classification:
        s = self.settings
        a = analyze_hand(det, s, self.mirrored_input)
        r_g, r_c, r_why = classify_rules(a, s)
        res = Classification(UNKNOWN, 0.0, r_why, a, r_g, r_c)
        if a.tracking_lost:
            res.gesture = NO_HAND
        if a.reject_reason:
            return res

        if self.knn is not None:
            k_g, k_c, k_d = self.knn.predict(a.features)
            res.knn_gesture, res.knn_confidence, res.knn_distance = k_g, k_c, k_d
            # a KNN match still has to pass the gesture-specific checks
            if k_g in GESTURES:
                why = gesture_checks(k_g, a, s)
                if why:
                    res.knn_gesture, res.knn_confidence = None, 0.0

        res.gesture, res.confidence, res.reason = self._combine(res, a)
        return res

    def _combine(self, r: Classification, a: HandAnalysis) -> tuple[str, float, str]:
        if self.mode == "rules":
            return r.rule_gesture, r.rule_confidence, r.reason
        k, kc = r.knn_gesture, r.knn_confidence
        if self.mode == "knn":
            if k in GESTURES:
                return k, kc, "knn"
            return UNKNOWN, 0.0, "knn: no confident match" if k is None else "knn: not a gesture"
        # hybrid
        if r.rule_gesture in GESTURES:
            if k is None or k == r.rule_gesture:
                return r.rule_gesture, max(r.rule_confidence, kc), \
                    "rules+knn" if k else "rules"
            return UNKNOWN, 0.0, f"rules say {r.rule_gesture}, knn says {k}"
        if (k in GESTURES and kc >= self.knn_settings.fallback_min_confidence
                and template_compatible(k, a, self.settings)):
            return k, kc, "knn fallback"
        return UNKNOWN, 0.0, r.reason
