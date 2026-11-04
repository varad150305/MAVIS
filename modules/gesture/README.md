# MAVIS Gesture Controller

A standalone webcam application for choosing a MAVIS feature with hand
gestures:

1. Pick a feature with a finger count.
2. Confirm it with a thumbs up.
3. Return to the menu with an open palm.

It recognizes the gestures in real time, including when the hand is
naturally rotated or tilted, and works with either hand. Every
transition goes out as a JSON event on stdout.

**It does not start or stop anything itself.** The events are *requests*
for an application such as MAVIS. No MAVIS code is included or called.

Developer and integration details live in [CLAUDE.md](CLAUDE.md).

## Gestures

| Gesture | Meaning |
|---|---|
| ☝️ one finger | select **OCR** (`ocr`) |
| ✌️ two fingers | select **Live Translation** (`translation`) |
| 🤟 three fingers (index, middle, ring) | select **Price Comparison** (`price_comparison`) |
| 👍 thumbs up | **confirm** the selected feature |
| 🖐️ open palm (fingers spread) | **back**: cancel a pending selection, or stop the confirmed feature |

The mapping, labels and timings are in `config/gesture_mapping.json`.

## How the interaction works

```
SELECT A FEATURE ──1/2/3 fingers──► CONFIRM FEATURE? ──thumbs up──► FEATURE CONFIRMED
      ▲    ▲                         │  (1/2/3 fingers again          │
      │    └──open palm / 8 s timeout┘   changes the selection)       │
      └──────────────────────── open palm (stop requested) ───────────┘
```

- Hold a gesture for about half a second. Each showing counts once;
  holding it longer does nothing more.
- A selection never starts a feature by itself; it always needs a thumbs up.
- Only one feature can be active. Show an open palm to stop it before
  choosing another.
- Taking your hand out of view never counts as a gesture. It cancels,
  confirms and stops nothing.

## Setup (Windows, Python 3.12)

```powershell
cd path\to\gesture-control
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```

Use Python **3.12**: MediaPipe has no wheels for 3.13+. `mediapipe` is
pinned to `0.10.21` because newer releases removed the API this code
uses. Avoid project paths with non-Latin characters.

macOS and Linux: `python3.12 -m venv .venv`, `source .venv/bin/activate`,
then the same `pip install`. Allow camera access for your terminal.

## Running the demo

```powershell
python mavis_gesture.py
```

The window shows:

- **left:** the camera with the hand skeleton, the recognized gesture
  and its meaning, which hand, a confidence bar, a stability/hold bar,
  per-finger state, hand orientation and FPS;
- **right:** the interaction state ("SELECT A FEATURE", "CONFIRM
  FEATURE? Selected: OCR", "FEATURE CONFIRMED: OCR", "STOP REQUESTED —
  RETURNING TO FEATURE SELECTION").

`Q` or `Esc` quits and prints a short summary to stderr.

## JSON Lines output

```powershell
python mavis_gesture.py --jsonl                 # stdout = events only (preview still shown)
python mavis_gesture.py --jsonl --no-preview    # headless
```

```json
{"schema_version":1,"source":"mavis_gesture","event":"feature.selection_pending","gesture":"one_finger","state":"AWAITING_CONFIRMATION","feature":"ocr","label":"OCR","previous_feature":null,"timestamp":1793372212.075}
{"schema_version":1,"source":"mavis_gesture","event":"feature.start_requested","gesture":"thumbs_up","state":"FEATURE_ACTIVE","feature":"ocr","label":"OCR","timestamp":1793372214.330}
{"schema_version":1,"source":"mavis_gesture","event":"feature.stop_requested","gesture":"open_palm","state":"SELECTING_FEATURE","feature":"ocr","reason":"open_palm","timestamp":1793372230.118}
{"schema_version":1,"source":"mavis_gesture","event":"navigation.return_to_selection","gesture":"open_palm","state":"SELECTING_FEATURE","previous_feature":"ocr","reason":"stopped","timestamp":1793372230.118}
```

Event types:

- `feature.selection_pending`
- `feature.selection_cancelled`
- `feature.start_requested`
- `feature.stop_requested`
- `navigation.return_to_selection`

Timestamps are Unix seconds. Diagnostics always go to stderr. The full
contract, event ordering and a parsing example are in
[CLAUDE.md § D](CLAUDE.md#d-output-interface).

## How recognition works

- **Palm-relative coordinates.** All 21 MediaPipe landmarks are
  re-expressed in a coordinate frame attached to the palm:
  - origin: wrist;
  - axes: towards the middle-finger knuckle, across the palm, and out of
    the palm;
  - left and right hands are mirrored into the same frame.

  This removes position, size and rotation. The default input is
  MediaPipe's metric 3D "world" landmarks.
- **Finger states from angles, not screen positions.** Each finger gets
  a 0–1 extension score:
  - fingers: total bend at the three joints, and how far the tip reaches
    from the wrist;
  - thumb: mainly its distance from the folded fingers and palm.

  A straight finger counts as extended whether it points up, sideways
  or diagonally. Thresholds were set from real webcam samples.
- **Strict templates.** A gesture needs *every* finger clearly in the
  required state. Half-curled, claw, fist, "L" shapes and four fingers
  give `UNKNOWN`. Extra checks:
  - an open palm must be spread;
  - a thumbs up must point roughly upward (thumbs-down is rejected) with
    a compact fist;
  - a palm turned edge-on (> 72°) is rejected, except for thumbs up.
- **Unreliable tracking = no hand.** A hand partly out of frame, low
  tracking confidence, or an implausible skeleton is treated as
  `NO_HAND`. Stability resets and nothing can fire.
- **Stability.** A gesture must be seen in 5 consecutive frames *and*
  held 0.5 s. It fires once, then re-arms only after release (0.4 s) or
  after the hand has been gone for 1 s. There is a 0.8 s cooldown
  between gestures.

## Testing

```powershell
python -m unittest discover -s tests -t .
```

112 automated tests, no camera needed:

- geometry invariance: translation, scale, in-plane rotations
  0/±30/±45/±60/±90/180°, 3D tilts;
- left/right-hand consistency and degenerate landmarks;
- every gesture at many orientations, and rejection of non-gestures;
- the same transforms applied to real recorded hand landmarks;
- stabilizer timing and hand-loss handling;
- the complete select → confirm → stop workflow for all three features,
  plus cancel, switch, timeout, repeated gestures and hand loss;
- the event schema and transports.

Webcam smoke test: run `python mavis_gesture.py`, then for each feature
show the finger count, a thumbs up, and an open palm. Check the panel
and the stdout events.

## Known limitations

- **Not 360°-robust.** Upside-down, edge-on or strongly foreshortened
  hands degrade MediaPipe itself. Natural tilts and rotations work.
- **Thumbs up shown knuckles-first** (fingertips hidden) is the least
  reliable pose.
- **`three_fingers` with the pinky clearly sticking out** reads as four
  fingers (`UNKNOWN`). Hold the pinky down with the thumb.
- **`open_palm` needs spread fingers.** A flat hand with the fingers
  together is not accepted.
- **Occlusion.** Heavy occlusion that MediaPipe "fills in" plausibly
  can't always be detected.
- **One hand at a time.**
- **Limited tuning data.** Thresholds were tuned on one person's hands
  and one webcam.
- **Speed.** About 13–22 FPS on the ARM laptop used for development,
  which runs x86 Python under emulation. Native x86 is faster.

## Repository notes

- The controller is `mavis_gesture.py` plus `hand_tracker.py`,
  `hand_geometry.py`, `gesture_classifier.py`, `stabilizer.py`,
  `gesture_mapping.py`, `feature_controller.py` and `events.py`.
- An optional KNN calibration mode (`--calibrate`, `--mode hybrid`)
  exists but is not needed.
- The original desktop-control tray app from `ZakLeshiy/gesture-control`
  is still included as legacy code: `tray_app.py`, `main.py`,
  `controller.py`, `actions.py`, `settings_gui.py`, `gesture_engine.py`,
  `start_tray.vbs`, `config/gestures.json`.
  - It executes OS actions (keys, mouse), so don't run it next to MAVIS.
  - It needs `pip install -r requirements-tray.txt`.
  - It is not used by the controller.
