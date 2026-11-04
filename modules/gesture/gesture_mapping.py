"""
gesture_mapping.py — the single place that knows what each gesture means.

Loads config/gesture_mapping.json:

  selection_gestures  gesture -> {feature, label, hint}   (one/two/three fingers)
  confirm_gesture     confirms the pending feature        (thumbs_up)
  back_gesture        cancels / stops, back to selection  (open_palm)
  timings             hold durations and the confirmation timeout

Nothing else in the code base hard-codes feature identifiers or labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from gesture_classifier import GESTURES

MAPPING_PATH = Path(__file__).parent / "config" / "gesture_mapping.json"


@dataclass(frozen=True)
class FeatureSpec:
    feature: str       # stable identifier used in events, e.g. "ocr"
    label: str         # human-readable name for the UI, e.g. "OCR"
    hint: str = ""     # how to select it, e.g. "1 finger"


@dataclass(frozen=True)
class InteractionTimings:
    selection_hold_sec: float = 0.5
    confirm_hold_sec: float = 0.5
    back_hold_sec: float = 0.5
    confirmation_timeout_sec: float = 8.0


@dataclass(frozen=True)
class GestureMapping:
    selection: dict[str, FeatureSpec]
    confirm_gesture: str = "thumbs_up"
    back_gesture: str = "open_palm"
    timings: InteractionTimings = field(default_factory=InteractionTimings)

    def __post_init__(self):
        used = list(self.selection) + [self.confirm_gesture, self.back_gesture]
        unknown = [g for g in used if g not in GESTURES]
        if unknown:
            raise ValueError(f"gesture_mapping: unknown gestures {unknown}; known: {GESTURES}")
        if len(set(used)) != len(used):
            raise ValueError("gesture_mapping: a gesture is assigned more than one role")
        features = [s.feature for s in self.selection.values()]
        if len(set(features)) != len(features):
            raise ValueError("gesture_mapping: duplicate feature identifiers")

    # ---- lookups -----------------------------------------------------------

    def feature_for(self, gesture: str) -> FeatureSpec | None:
        return self.selection.get(gesture)

    def label(self, feature: str | None) -> str:
        for spec in self.selection.values():
            if spec.feature == feature:
                return spec.label
        return feature or ""

    def meaning(self, gesture: str) -> str:
        """Short description of a gesture's role, for the preview."""
        if gesture in self.selection:
            return f"select {self.selection[gesture].label}"
        if gesture == self.confirm_gesture:
            return "confirm"
        if gesture == self.back_gesture:
            return "back to selection"
        return ""

    def hold_seconds(self) -> dict[str, float]:
        """Per-gesture hold durations for the stabilizer."""
        t = self.timings
        holds = {g: t.selection_hold_sec for g in self.selection}
        holds[self.confirm_gesture] = t.confirm_hold_sec
        holds[self.back_gesture] = t.back_hold_sec
        return holds

    # ---- loading -----------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "GestureMapping":
        selection = {g: FeatureSpec(**spec) for g, spec in d["selection_gestures"].items()}
        timings = InteractionTimings(**d.get("timings", {}))
        return cls(selection, d.get("confirm_gesture", "thumbs_up"),
                   d.get("back_gesture", "open_palm"), timings)


def load_mapping(path: str | Path = MAPPING_PATH) -> GestureMapping:
    with open(path, encoding="utf-8") as f:
        return GestureMapping.from_dict(json.load(f))
