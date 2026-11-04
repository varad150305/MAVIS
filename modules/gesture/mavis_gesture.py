"""
mavis_gesture.py — standalone MAVIS feature-selection gesture controller.

Pipeline (one camera, one loop):

  webcam -> HandTracker -> GestureClassifier -> GestureStabilizer
         -> FeatureController (state machine) -> EventBus -> JSONL on stdout
                                                          -> (later) MAVIS callback

  one finger    -> select OCR                  \\  selection is only *pending*
  two fingers   -> select Live Translation      >  until confirmed
  three fingers -> select Price Comparison     /
  thumbs up     -> confirm: feature.start_requested
  open palm     -> cancel pending selection, or feature.stop_requested,
                   then back to feature selection

Nothing is started or stopped here: the events are requests for an
external application (MAVIS). Diagnostics go to stderr only.

  python mavis_gesture.py                        demo window + events on stdout
  python mavis_gesture.py --jsonl                stdout carries ONLY JSON events
  python mavis_gesture.py --jsonl --no-preview   headless
  python mavis_gesture.py --calibrate            optional KNN samples (not needed)

Keys in the preview window: Q / Esc — quit.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import events as ev
from feature_controller import FeatureController, State
from gesture_classifier import (CALIBRATION_FORMAT, FEATURE_VERSION, GESTURES, NEGATIVE,
                                NO_HAND, UNKNOWN, Classification, ClassifierSettings,
                                GestureClassifier, KnnModel, KnnSettings, finger_state,
                                load_calibration)
from gesture_mapping import MAPPING_PATH, GestureMapping, load_mapping
from hand_geometry import FINGER_NAMES
from hand_tracker import HandTracker
from stabilizer import GestureStabilizer, StabilitySettings

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config" / "mavis_gestures.json"
WINDOW = "MAVIS gesture control  [Q = quit]"


def log(*args) -> None:
    """Diagnostics always go to stderr so stdout stays machine-readable."""
    print(*args, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- config

def load_settings(path: str | Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def build_classifier(cfg: dict, mode: str | None = None) -> GestureClassifier:
    ccfg = dict(cfg.get("classifier", {}))
    mode = mode or ccfg.pop("mode", "rules")
    ccfg.pop("mode", None)
    settings = ClassifierSettings.from_dict(ccfg)
    kcfg = dict(cfg.get("knn", {}))
    calib = resolve_path(kcfg.pop("calibration_file", "config/calibration_samples.json"))
    ksettings = KnnSettings.from_dict(kcfg)

    knn = None
    if mode != "rules":
        knn = KnnModel.from_file(calib, settings.landmark_source, ksettings)
        if knn is None and mode != "auto":
            raise SystemExit(f"mode {mode!r} needs calibration samples in {calib}; "
                             "run: python mavis_gesture.py --calibrate")
    if mode == "auto":
        mode = "hybrid" if knn is not None else "rules"
    if knn is not None:
        log(f"[mavis] KNN: {len(knn)} calibration samples, classes: {', '.join(knn.classes)}")
    mirror = cfg.get("camera", {}).get("mirror", True)
    return GestureClassifier(settings, knn, mode, ksettings, mirrored_input=mirror)


# ---------------------------------------------------------------- pipeline

@dataclass
class FrameResult:
    classification: Classification | None   # None = no hand detected
    hand_present: bool                       # False for NO_HAND (lost / unreliable)
    gesture_event: str | None                # stabilized gesture fired on this frame
    events: list[ev.Event] = field(default_factory=list)   # controller transitions


class GesturePipeline:
    """Recognition -> stabilization -> feature controller -> events.

    Frames are supplied by the caller, so a host application that already
    owns the camera (MAVIS) can feed its own frames instead of opening a
    second capture. Frames must be BGR and mirrored if camera.mirror is true
    (MediaPipe's handedness assumes a selfie view).

        pipe = GesturePipeline(cfg, mapping)
        pipe.bus.subscribe(my_callback)          # receives events.Event objects
        for frame in frames:
            pipe.process_frame(frame)
        pipe.close()                             # emits stop for an active feature
    """

    def __init__(self, cfg: dict, mapping: GestureMapping, mode: str | None = None,
                 bus: ev.EventBus | None = None, tracker: HandTracker | None = None):
        self.mapping = mapping
        self.classifier = build_classifier(cfg, mode)
        st = StabilitySettings.from_dict(cfg.get("stability"))
        st.hold_sec = mapping.hold_seconds()
        self.stabilizer = GestureStabilizer(st)
        self.bus = bus or ev.EventBus()
        self.controller = FeatureController(mapping, self.bus.publish)
        self.tracker = tracker or make_tracker(cfg)

    def process_frame(self, frame_bgr, now: float | None = None) -> FrameResult:
        now = time.time() if now is None else now
        dets = self.tracker.process_detailed(frame_bgr)
        return self.process_detections(dets, now)

    def process_detections(self, dets, now: float) -> FrameResult:
        res = self.classifier.classify(dets[0]) if dets else None
        # no detection and unreliable landmarks are both "no hand": they reset
        # stability and can never produce, complete or repeat a gesture
        present = res is not None and res.gesture != NO_HAND
        fired = self.stabilizer.update(res.gesture if present else None,
                                       res.confidence if present else 0.0, now, present)
        out = self.controller.tick(now)
        if fired:
            out += self.controller.on_gesture(fired, now)
        return FrameResult(res, present, fired, out)

    def close(self, now: float | None = None) -> list[ev.Event]:
        out = self.controller.shutdown(now)
        self.tracker.close()
        return out


# ---------------------------------------------------------------- preview

GREEN, YELLOW, GREY, RED, CYAN, WHITE = (60, 220, 60), (0, 210, 255), (170, 170, 170), \
    (70, 70, 235), (255, 220, 0), (255, 255, 255)
STATE_COLORS = {"E": GREEN, "F": GREY, "T": GREY, "?": YELLOW}
PANEL_W = 360


def _text(img, s, org, scale=0.6, color=WHITE, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)
    cv2.rectangle(img, (x + 1, y + 1), (x + 1 + int((w - 2) * max(0.0, min(1.0, frac))),
                                         y + h - 1), color, -1)


def draw_recognition(frame, res: Classification | None, stab: GestureStabilizer, now: float,
                     fps: float, classifier: GestureClassifier, mapping: GestureMapping) -> None:
    """Recognition status on top of the camera image."""
    h, w = frame.shape[:2]
    band = frame.copy()
    cv2.rectangle(band, (0, 0), (w, 118), (20, 20, 20), -1)
    cv2.addWeighted(band, 0.6, frame, 0.4, 0, frame)

    if res is None or res.gesture == NO_HAND:
        # tracking lost: show nothing from earlier frames
        _text(frame, "NO HAND", (12, 36), 1.0, GREY, 2)
        _text(frame, "show one hand to the camera" if res is None else
              f"why: {res.reason}", (12, 64), 0.55, GREY if res is None else YELLOW)
    else:
        g = res.gesture
        confirmed = stab.stable == g and g != UNKNOWN
        color = GREEN if confirmed else (YELLOW if g != UNKNOWN else GREY)
        _text(frame, g.replace("_", " ").upper(), (12, 36), 1.0, color, 2)
        if g != UNKNOWN:
            _text(frame, "-> " + mapping.meaning(g), (290, 34), 0.6, color, 2)
        a = res.analysis
        _text(frame, f"{a.hand} hand", (12, 62), 0.55)
        _text(frame, "conf", (120, 62), 0.5, GREY)
        _bar(frame, 160, 51, 90, 12, res.confidence, color)
        _text(frame, "stable", (262, 62), 0.5, GREY)
        _bar(frame, 318, 51, 90, 12, 1.0 if confirmed else stab.progress_at(now), color)
        status = "held (release to re-arm)" if not stab.armed else (
            f"cooldown {stab.cooldown_left(now):.1f}s" if stab.cooldown_left(now) > 0 else "armed")
        _text(frame, status, (420, 62), 0.5, GREY if stab.armed else YELLOW)
        if a.reject_reason or g == UNKNOWN:
            _text(frame, f"why: {res.reason}", (12, 86), 0.5, YELLOW)
        elif a.orientation:
            _text(frame, f"orientation: {a.orientation.describe()}", (12, 86), 0.5, CYAN)
        if a.metrics:
            x = 12
            for name in FINGER_NAMES:
                sc = a.metrics[name].score
                st = finger_state(name, sc, classifier.settings)
                _text(frame, name[0].upper(), (x, 110), 0.5, STATE_COLORS[st])
                _bar(frame, x + 14, 100, 46, 10, sc, STATE_COLORS[st])
                x += 72
    _text(frame, f"{fps:4.1f} FPS", (w - 90, h - 12), 0.5, GREY)
    _text(frame, "Q = quit", (12, h - 12), 0.5, GREY)


def draw_state_panel(height: int, ctl: FeatureController, stab: GestureStabilizer,
                     now: float) -> np.ndarray:
    """Side panel showing the interaction state (what a presenter needs to see)."""
    m = ctl.mapping
    p = np.full((height, PANEL_W, 3), 32, np.uint8)
    x, y = 18, 40
    last = ctl.last_event
    recent = last is not None and now - last.timestamp < 1.8

    if recent and last.name == ev.RETURN_TO_SELECTION:
        reason = last.get("reason")
        title = {"stopped": "STOP REQUESTED", "cancelled": "SELECTION CANCELLED",
                 "timeout": "CONFIRMATION TIMED OUT"}.get(reason, "BACK")
        cv2.rectangle(p, (0, 0), (PANEL_W, 92), (40, 40, 140), -1)
        _text(p, title, (x, 36), 0.7, WHITE, 2)
        _text(p, "RETURNING TO FEATURE SELECTION", (x, 70), 0.5, WHITE)
        y = 130

    if ctl.state is State.SELECTING_FEATURE:
        _text(p, "SELECT A FEATURE", (x, y), 0.85, CYAN, 2)
        y += 50
        for gesture, spec in m.selection.items():
            holding = stab.candidate == gesture
            color = YELLOW if holding else WHITE
            _text(p, spec.hint, (x, y), 0.5, GREY)
            _text(p, spec.label.upper(), (x + 90, y), 0.62, color, 2 if holding else 1)
            if holding:
                _bar(p, x + 90, y + 8, 200, 8, stab.progress_at(now), YELLOW)
            y += 46

    elif ctl.state is State.AWAITING_CONFIRMATION:
        cv2.rectangle(p, (0, y - 32), (PANEL_W, y + 108), (20, 70, 90), -1)
        _text(p, "CONFIRM FEATURE?", (x, y), 0.85, YELLOW, 2)
        _text(p, "Selected:", (x, y + 42), 0.55, GREY)
        _text(p, m.label(ctl.pending_feature).upper(), (x, y + 78), 0.8, WHITE, 2)
        y += 140
        _text(p, f"Show {m.confirm_gesture.replace('_', ' ')} to confirm", (x, y), 0.55, WHITE)
        if stab.candidate == m.confirm_gesture:
            _bar(p, x, y + 10, 250, 8, stab.progress_at(now), GREEN)
        y += 40
        _text(p, "Other finger count: change", (x, y), 0.5, GREY)
        _text(p, f"{m.back_gesture.replace('_', ' ').capitalize()}: cancel", (x, y + 24),
              0.5, GREY)
        left = ctl.confirmation_time_left(now)
        _text(p, f"expires in {left:.0f} s", (x, y + 58), 0.5, GREY)
        _bar(p, x, y + 66, 250, 6, left / m.timings.confirmation_timeout_sec, GREY)

    elif ctl.state is State.FEATURE_ACTIVE:
        cv2.rectangle(p, (0, y - 32), (PANEL_W, y + 70), (30, 90, 30), -1)
        _text(p, "FEATURE CONFIRMED:", (x, y), 0.7, WHITE, 2)
        _text(p, m.label(ctl.active_feature).upper(), (x, y + 44), 0.85, GREEN, 2)
        y += 110
        _text(p, "start requested", (x, y), 0.5, GREY)
        _text(p, "(no MAVIS module connected)", (x, y + 22), 0.45, GREY)
        y += 64
        _text(p, f"Show {m.back_gesture.replace('_', ' ')} to return", (x, y), 0.55, WHITE)
        _text(p, "to feature selection", (x, y + 24), 0.55, WHITE)
        if stab.candidate == m.back_gesture:
            _bar(p, x, y + 36, 250, 8, stab.progress_at(now), RED)

    if ctl.notice and now - ctl.notice[1] < 2.0:
        _text(p, ctl.notice[0], (x, height - 62), 0.45, YELLOW)
    if last is not None:
        _text(p, f"last event: {last.name}", (x, height - 16), 0.42, GREY)
    return p


# ---------------------------------------------------------------- camera

class FrameGrabber:
    """The single camera reader: a thread that always holds the newest frame.

    cap.read() blocks until the next frame arrives; doing that in the
    recognition loop serializes camera wait + inference. Here the loop picks
    up the latest frame as soon as it is free and stale frames are dropped.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._taken = 0
        self._running = True
        self._t = threading.Thread(target=self._run, name="camera", daemon=True)
        self._t.start()

    def _run(self) -> None:
        fails = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails in (30, 300):
                    log("[mavis] camera is not delivering frames")
                time.sleep(0.01)
                continue
            fails = 0
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 1.0):
        """Returns (ok, newest frame not returned before)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._taken or not self._running,
                                       timeout):
                return False, None
            self._taken = self._seq
            return self._frame is not None, self._frame

    def release(self) -> None:
        self._running = False
        self._t.join(1.0)
        self.cap.release()


def open_camera(cfg: dict) -> FrameGrabber:
    cam = cfg.get("camera", {})
    idx = cam.get("index", 0)
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise SystemExit(f"[mavis] camera {idx} unavailable")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.get("width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("height", 480))
    return FrameGrabber(cap)


def make_tracker(cfg: dict) -> HandTracker:
    t = cfg.get("tracking", {})
    return HandTracker(max_hands=t.get("max_hands", 1),
                       detection_conf=t.get("detection_conf", 0.7),
                       tracking_conf=t.get("tracking_conf", 0.6),
                       model_complexity=t.get("model_complexity", 1))


# ---------------------------------------------------------------- run

def run(cfg: dict, mapping: GestureMapping, args) -> None:
    mirror = cfg.get("camera", {}).get("mirror", True)
    out_stream = sys.stdout
    if args.jsonl:
        sys.stdout = sys.stderr          # anything else that prints goes to stderr

    bus = ev.EventBus()
    writer = ev.JsonlWriter(out_stream)  # JSON Lines transport (default)
    bus.subscribe(writer)
    if not args.jsonl:
        bus.subscribe(ev.stderr_logger)
    event_counts: Counter = Counter()
    bus.subscribe(lambda e: event_counts.update([e.name]))

    def _stop(*_):
        raise KeyboardInterrupt
    # a parent process stopping us (SIGTERM / Ctrl+Break) still shuts down cleanly
    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _stop)

    cap = open_camera(cfg)
    pipe = GesturePipeline(cfg, mapping, args.mode, bus)
    log(f"[mavis] mode={pipe.classifier.mode} source={pipe.classifier.settings.landmark_source} "
        f"preview={'off' if args.no_preview else 'on'}")
    fps, last_t = 0.0, time.time()
    beat_t, beat_frames, beat_hands = time.time(), 0, 0
    gesture_counts: Counter = Counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if mirror:
                frame = cv2.flip(frame, 1)
            now = time.time()
            r = pipe.process_frame(frame, now)
            if r.gesture_event:
                gesture_counts[r.gesture_event] += 1

            dt, last_t = now - last_t, now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            beat_frames += 1
            beat_hands += r.hand_present
            if args.no_preview and now - beat_t >= 10.0:
                res = r.classification
                log(f"[mavis] {beat_frames / (now - beat_t):.1f} FPS, hand in "
                    f"{100 * beat_hands / max(beat_frames, 1):.0f}% of frames, state "
                    f"{pipe.controller.state.value}, last: "
                    f"{res.gesture + ' (' + res.reason + ')' if res else 'no hand'}")
                beat_t, beat_frames, beat_hands = now, 0, 0

            if not args.no_preview:
                pipe.tracker.draw(frame)
                draw_recognition(frame, r.classification, pipe.stabilizer, now, fps,
                                 pipe.classifier, mapping)
                panel = draw_state_panel(frame.shape[0], pipe.controller, pipe.stabilizer, now)
                cv2.imshow(WINDOW, np.hstack([frame, panel]))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        pipe.close()                     # stop request for a still-active feature
        cv2.destroyAllWindows()
        writer.close()
        session_summary(gesture_counts, event_counts)


def session_summary(gestures: Counter, events: Counter) -> None:
    """Short stderr report after a demo / smoke test."""
    log("[mavis] session summary")
    log("  gestures: " + (", ".join(f"{g}={gestures[g]}" for g in GESTURES if gestures[g])
                          or "none"))
    log("  events:   " + (", ".join(f"{k}={v}" for k, v in events.items()) or "none"))
    missing = [g for g in GESTURES if not gestures[g]]
    if missing:
        log("  gestures not recognized this session: " + ", ".join(missing))


# ---------------------------------------------------------------- calibration

CALIBRATION_HINTS = {
    "one_finger": "index finger up, others folded, thumb tucked",
    "two_fingers": "index + middle up, thumb holds ring and pinky",
    "three_fingers": "index + middle + ring up, thumb holds pinky",
    "open_palm": "all five fingers extended and spread",
    "thumbs_up": "fist with thumb extended",
    NEGATIVE: "NOT a gesture: fist, relaxed / half-curled hand, other poses",
}


def calibration_sample(det, res: Classification, gesture: str) -> dict:
    o = res.analysis.orientation
    return {
        "gesture": gesture,
        "hand": res.analysis.hand,
        "handedness_label": det.label,
        "timestamp": round(time.time(), 3),
        "frame_size": list(det.frame_size),
        "world_landmarks": None if det.world is None else np.round(det.world, 5).tolist(),
        "image_landmarks": np.round(det.image, 5).tolist(),
        "features": np.round(res.analysis.features, 5).tolist(),
        "orientation": None if o is None else {"roll_deg": round(o.roll_deg, 1),
                                               "tilt_deg": round(o.tilt_deg, 1),
                                               "facing": o.facing},
        "rule_prediction": res.rule_gesture,
    }


def save_calibration(path: Path, samples: list[dict], source: str, replace: set[str]) -> None:
    old: list[dict] = []
    if path.exists():
        try:
            old = [s for s in load_calibration(path).get("samples", [])
                   if s.get("gesture") not in replace]
        except (ValueError, json.JSONDecodeError) as e:
            log(f"[calibrate] existing {path} unreadable ({e}); overwriting")
    data = {
        "format": CALIBRATION_FORMAT,
        "version": 1,
        "feature_version": FEATURE_VERSION,
        "landmark_source": source,
        "updated": round(time.time(), 3),
        "samples": old + samples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    tmp.replace(path)
    counts: dict[str, int] = {}
    for s in data["samples"]:
        counts[s["gesture"]] = counts.get(s["gesture"], 0) + 1
    log(f"[calibrate] saved {path}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


def calibrate(cfg: dict, args) -> None:
    """Interactive sample capture.

    SPACE — capture one sample   A — toggle auto-capture
    N — next gesture             B — previous gesture
    Q — save and quit            Esc — quit without saving
    Auto-capture takes a sample every 0.3 s, but only when the pose changed
    noticeably since the last one, which nudges the user to vary rotation,
    tilt and distance.
    """
    classifier = build_classifier(cfg, "rules")
    source = classifier.settings.landmark_source
    mirror = cfg.get("camera", {}).get("mirror", True)
    path = resolve_path(args.output or cfg.get("knn", {}).get(
        "calibration_file", "config/calibration_samples.json"))
    gestures = list(GESTURES) + ([] if args.no_negative else [NEGATIVE])
    if args.gestures:
        gestures = [g.strip() for g in args.gestures.split(",") if g.strip()]
        bad = [g for g in gestures if g not in GESTURES and g != NEGATIVE]
        if bad:
            raise SystemExit(f"unknown gestures: {bad}")
    target = args.samples
    samples: dict[str, list[dict]] = {g: [] for g in gestures}

    cap = open_camera(cfg)
    tracker = make_tracker(cfg)
    gi, auto, last_auto, last_feat = 0, False, 0.0, None
    save = True
    log("[calibrate] SPACE capture | A auto | N next | B back | Q save+quit | Esc abort")
    try:
        while gi < len(gestures):
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if mirror:
                frame = cv2.flip(frame, 1)
            dets = tracker.process_detailed(frame)
            res = classifier.classify(dets[0]) if dets else None
            usable = res is not None and res.analysis.reject_reason is None
            g = gestures[gi]
            got = samples[g]

            capture = False
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                capture = True
            elif key == ord("a"):
                auto = not auto
            elif key == ord("n"):
                gi, auto, last_feat = gi + 1, False, None
                continue
            elif key == ord("b"):
                gi, auto, last_feat = max(0, gi - 1), False, None
                continue
            elif key == ord("q"):
                break
            elif key == 27:
                save = False
                break
            now = time.time()
            if auto and usable and now - last_auto > 0.3:
                f = res.analysis.features
                if last_feat is None or np.linalg.norm(f - last_feat) / np.sqrt(20) > 0.04:
                    capture = True
                    last_auto = now
            if capture and usable:
                got.append(calibration_sample(dets[0], res, g))
                last_feat = res.analysis.features
                log(f"  {g}: {len(got)}/{target}  (rules saw: {res.rule_gesture})")
                if len(got) >= target:
                    gi, auto, last_feat = gi + 1, False, None
                    continue

            tracker.draw(frame)
            h, w = frame.shape[:2]
            panel = frame.copy()
            cv2.rectangle(panel, (0, 0), (w, 100), (20, 20, 20), -1)
            cv2.addWeighted(panel, 0.6, frame, 0.4, 0, frame)
            _text(frame, f"CALIBRATE {gi + 1}/{len(gestures)}: {g}   {len(got)}/{target}",
                  (12, 30), 0.75, CYAN, 2)
            _text(frame, CALIBRATION_HINTS[g], (12, 55), 0.55)
            _text(frame, "vary rotation, tilt, distance and hand between samples", (12, 77), 0.5, GREY)
            seen = "no hand" if res is None else (res.reason if not usable else
                                                  f"rules: {res.rule_gesture}")
            _text(frame, seen, (12, 96), 0.5, YELLOW if not usable else GREY)
            _text(frame, f"SPACE snap | A auto[{'ON' if auto else 'off'}] | N next | B back | "
                         "Q save+quit | Esc abort", (12, h - 12), 0.45, WHITE)
            cv2.imshow("MAVIS calibration", frame)
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    flat = [s for g in gestures for s in samples[g]]
    if save and flat:
        short = [g for g in gestures if 0 < len(samples[g]) < 12]
        if short:
            log(f"[calibrate] warning: fewer than 12 samples for {short}")
        save_calibration(path, flat, source, {g for g in gestures if samples[g]})
    else:
        log("[calibrate] nothing saved")


# ---------------------------------------------------------------- entrypoint

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="MAVIS feature-selection gesture controller")
    p.add_argument("--config", default=str(CONFIG_PATH), help="recognition settings JSON")
    p.add_argument("--mapping", default=str(MAPPING_PATH),
                   help="gesture-to-feature mapping and interaction timings JSON")
    p.add_argument("--jsonl", action="store_true",
                   help="stdout carries only JSON Lines events; diagnostics on stderr")
    p.add_argument("--no-preview", action="store_true", help="no camera window (headless)")
    p.add_argument("--mode", choices=["rules", "hybrid", "knn", "auto"],
                   help="classifier mode (default from config: rules)")
    p.add_argument("--source", choices=["world", "image"],
                   help="landmarks used for geometry (default from config: world)")
    p.add_argument("--calibrate", action="store_true",
                   help="optional: record KNN calibration samples (not needed)")
    p.add_argument("--samples", type=int, default=15, help="samples per gesture (calibration)")
    p.add_argument("--gestures", help="comma-separated subset to calibrate")
    p.add_argument("--no-negative", action="store_true",
                   help="skip the 'none' (not-a-gesture) calibration class")
    p.add_argument("--output", help="calibration file to write")
    args = p.parse_args(argv)

    cfg = load_settings(args.config)
    if args.source:
        cfg.setdefault("classifier", {})["landmark_source"] = args.source
    if args.calibrate:
        calibrate(cfg, args)
    else:
        run(cfg, load_mapping(args.mapping), args)


if __name__ == "__main__":
    main()
