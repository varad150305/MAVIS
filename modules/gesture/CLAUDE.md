# CLAUDE.md — MAVIS Gesture Controller

Developer and integration guide. The README is the short user guide; this
file explains how everything works and how to plug it into MAVIS.

## A. Project overview

### Purpose

A standalone, webcam-based controller that lets a user pick a MAVIS feature
with hand gestures:

- select a feature;
- confirm it;
- later stop it and return to selection.

It **does not start or stop anything itself**. It emits structured
*request* events (JSON Lines on stdout by default, or Python callbacks)
for a host application. No MAVIS code is imported or executed. The
project was adapted from `ZakLeshiy/gesture-control`. The original
tray/desktop-control app is still in the repo as legacy code (section C).

### Gesture mappings (config/gesture_mapping.json)

| Gesture id | Hand shape | Role | Feature id | Label |
|---|---|---|---|---|
| `one_finger` | index up, others folded | select | `ocr` | OCR |
| `two_fingers` | index + middle up | select | `translation` | Live Translation |
| `three_fingers` | index + middle + ring up, pinky folded | select | `price_comparison` | Price Comparison |
| `thumbs_up` | fist, thumb up | confirm | — | — |
| `open_palm` | all five fingers spread | back to selection (cancel / stop) | — | — |

### Architecture (three separated layers)

```
             A. GESTURE RECOGNITION                 B. FEATURE CONTROL         C. EVENT INTERFACE
frame ─► HandTracker ─► GestureClassifier ─► GestureStabilizer ─► FeatureController ─► EventBus ─┬─► JsonlWriter (stdout)
         hand_tracker   gesture_classifier   stabilizer            feature_controller   events   ├─► QueueSink
         (MediaPipe)    + hand_geometry      (debounce, holds,     (state machine)               ├─► stderr_logger
                        (per-frame label)     once-per-showing)                                  └─► any callable (MAVIS)
```

`mavis_gesture.GesturePipeline` wires A → B → C for frames supplied by the
caller. `mavis_gesture.run()` adds the camera, the preview window and the
default JSONL transport.

- **A** knows gestures, not features.
- **B** knows features and states, not cameras or output formats.
- **C** knows the wire format and transports, not gestures.

### State machine (feature_controller.py)

```
                 1/2/3 fingers                     thumbs up
  SELECTING_FEATURE ───────────► AWAITING_CONFIRMATION ───────────► FEATURE_ACTIVE
        ▲   ▲                        │  ▲  │ 1/2/3 fingers: replace     │
        │   │  open palm / timeout   │  └──┘ pending selection          │
        │   └────────────────────────┘                                  │
        │                        open palm (stop requested)             │
        └───────────────────────────────────────────────────────────────┘
```

| State | Input | Effect | Events emitted (in order) |
|---|---|---|---|
| SELECTING_FEATURE | 1/2/3 fingers | pending = feature → AWAITING | `feature.selection_pending` |
| SELECTING_FEATURE | thumbs up / open palm | ignored (UI notice) | — |
| AWAITING_CONFIRMATION | other 1/2/3 fingers | replace pending | `feature.selection_pending` (with `previous_feature`) |
| AWAITING_CONFIRMATION | same selection again | refresh the timeout | — |
| AWAITING_CONFIRMATION | thumbs up | active = pending → ACTIVE | `feature.start_requested` |
| AWAITING_CONFIRMATION | open palm | cancel → SELECTING | `feature.selection_cancelled` (reason `open_palm`), `navigation.return_to_selection` (reason `cancelled`) |
| AWAITING_CONFIRMATION | `confirmation_timeout_sec` elapsed | cancel → SELECTING | `feature.selection_cancelled` (reason `timeout`), `navigation.return_to_selection` (reason `timeout`) |
| FEATURE_ACTIVE | open palm | stop → SELECTING | `feature.stop_requested` (reason `open_palm`), `navigation.return_to_selection` (reason `stopped`) |
| FEATURE_ACTIVE | 1/2/3 fingers / thumbs up | ignored: only one active feature | — |
| FEATURE_ACTIVE | application exit | — | `feature.stop_requested` (reason `shutdown`) |
| any | no hand / unreliable hand | **nothing** | — |

Notes:

- The controller only receives *stabilized gesture events*: one per
  continuous presentation of a gesture, after the hold time. Holding a
  gesture never repeats an event.
- Hand loss never reaches the controller, so it cannot cancel, confirm
  or stop anything.
- The pending selection survives hand loss; only the timeout or a
  gesture ends it.
- An active feature has no timeout.

## B. Setup

Python **3.12** is required. MediaPipe has no wheels for 3.13+, and the
code uses `mp.solutions`, which was removed in mediapipe 0.10.31, hence
the pin `mediapipe==0.10.21`.

```powershell
# from the repository root, Windows PowerShell
py -3.12 -m venv .venv                     # or: uv venv --python 3.12 .venv
.venv\Scripts\Activate.ps1                 # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt            # mediapipe, opencv-contrib-python, numpy

python mavis_gesture.py                          # webcam demo (preview + JSONL on stdout)
python mavis_gesture.py --jsonl                  # stdout = JSON events only, preview still shown
python mavis_gesture.py --jsonl --no-preview     # headless, JSONL only
python -m unittest discover -s tests -t .        # automated tests (no camera needed)
```

Without activating the venv, use `.venv\Scripts\python.exe` instead of
`python`.

CLI options:

| Option | Meaning |
|---|---|
| `--config` | recognition settings (default `config/mavis_gestures.json`) |
| `--mapping` | mapping and timings (default `config/gesture_mapping.json`) |
| `--mode rules\|hybrid\|knn\|auto` | classifier mode |
| `--source world\|image` | landmark source |
| `--calibrate` | optional KNN sample recording; not needed |

Preview keys: `Q` / `Esc` quit. On exit, a session summary is printed to
stderr.

Development machine notes:

- Snapdragon X (ARM64) runs x86-64 Python under emulation:
  - MediaPipe takes ~40 ms per frame;
  - the loop runs at 13–22 FPS.
- `pip install opencv-python` must not be added. MediaPipe already
  depends on `opencv-contrib-python`, and two packages providing `cv2`
  conflict.

## C. Files and responsibilities

| File | Layer | Responsibility / key names |
|---|---|---|
| `hand_tracker.py` | A, hand tracking | `HandTracker.process_detailed(frame_bgr) -> list[HandDetection]` (image + world landmarks, handedness); `draw()`; `close()`. MediaPipe Hands wrapper. |
| `hand_geometry.py` | A, geometry | palm-local frame `build_palm_frame()`, `finger_metrics()` (0..1 extension per finger from joint angles), `finger_spread_deg()`, `hand_orientation()`, `landmark_sanity()`, `DegenerateHand`. Rotation, scale and translation invariant; left/right consistent. |
| `gesture_classifier.py` | A, classification | `GestureClassifier.classify(det) -> Classification` (`.gesture` ∈ `GESTURES` ∪ {`UNKNOWN`, `NO_HAND`}, `.confidence`, `.reason`, `.analysis`). `TEMPLATES` = finger states per gesture; `ClassifierSettings`; optional `KnnModel`. |
| `stabilizer.py` | A, stabilization | `GestureStabilizer.update(gesture, confidence, now, hand_present) -> fired gesture or None`. Handles stable frames, per-gesture hold time, cooldown, release-to-rearm and hand-loss reset. `StabilitySettings`. |
| `gesture_mapping.py` | B, config | `load_mapping() -> GestureMapping` (`selection`, `confirm_gesture`, `back_gesture`, `timings`, `feature_for()`, `label()`, `hold_seconds()`). The only code that reads `config/gesture_mapping.json`. |
| `feature_controller.py` | B, state machine | `FeatureController(mapping, emit)`: `on_gesture(g, now)`, `tick(now)`, `shutdown(now)`, `state`, `pending_feature`, `active_feature`. `State` enum. **Feature start**: `on_gesture` → `START_REQUESTED` branch. **Feature stop**: `on_gesture` → FEATURE_ACTIVE / back-gesture branch, plus `shutdown()`. |
| `events.py` | C, events | `Event`, `EVENT_FIELDS`, event-name constants, **`to_dict()` (the only serializer)**, `to_json()`, `EventBus`, `JsonlWriter`, `QueueSink`, `stderr_logger`. |
| `mavis_gesture.py` | wiring, app | `GesturePipeline` (`process_frame`, `process_detections`, `close`, `.bus`, `.controller`, `.stabilizer`, `.classifier`, `.tracker`), `run()` (camera loop + preview), `FrameGrabber` (single camera reader thread), `open_camera()`, `draw_recognition()`, `draw_state_panel()`, `build_classifier()`, `calibrate()`, `main()`. |
| `config/gesture_mapping.json` | config | gestures → features, labels, hints, interaction timings |
| `config/mavis_gestures.json` | config | camera, MediaPipe, classifier thresholds, stabilizer settings, optional KNN |
| `tests/` | tests | `test_geometry.py` (invariance, degenerate input), `test_classifier.py` (all gestures × rotations/tilts/hands, negatives, NO_HAND, pipeline hand loss), `test_real_landmarks.py` (real recorded landmarks, fixture `fixtures/real_calibration_samples.json`), `test_stabilizer.py`, `test_feature_controller.py` (state machine unit + end-to-end workflows), `test_events.py` (schema, transports), `test_legacy_engine.py`; helpers `synthetic_hand.py` (kinematic hand model), `helpers.py` |
| `requirements.txt` / `requirements-tray.txt` | setup | controller deps / + legacy tray deps |

**Legacy (original repository, not used by the controller):**

- files: `tray_app.py`, `main.py`, `controller.py`, `actions.py`,
  `settings_gui.py`, `gesture_engine.py`, `start_tray.vbs`,
  `config/gestures.json`;
- it is a desktop-control tray app that **executes OS actions** via
  pyautogui. Don't run it next to MAVIS;
- `gesture_engine.finger_states()` now delegates to `hand_geometry`;
- `test_legacy_engine.py` covers it;
- it can be deleted together with that test and `requirements-tray.txt`
  if it's no longer wanted.

Recognition details (in short):

- Landmarks go into a palm frame: origin at the wrist, y = wrist→middle
  MCP, x = across the palm, z = palm normal with a chirality sign so
  both hands match, unit = palm length.
- Each finger gets a score from total MCP+PIP+DIP flexion plus
  normalized tip reach. The thumb is scored mainly by clearance from the
  folded fingers and palm.
- Quality gates: palm edge-on > 72° (except thumbs-up), hand too small,
  thumbs-up not pointing up (thumbs-down rejected), open palm not spread.
- Unreliable landmarks (partly out of frame, low handedness score,
  degenerate or implausible skeleton) produce **NO_HAND**.

The README section "How recognition works" summarizes this. Exact
thresholds are in `ClassifierSettings` and `ExtensionThresholds`, which
document the real-data ranges they came from.

## D. Output interface

### Transport

Default: **JSON Lines on stdout**, one compact object per line,
flushed immediately.

- A background thread (`JsonlWriter`) writes them, so a slow reader
  never stalls recognition.
- With `--jsonl`, nothing else is written to stdout: `sys.stdout` is
  redirected to stderr. All diagnostics and MediaPipe logs go to stderr.

### Schema (schema_version 1)

Common fields on every event:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | `1`; bump on incompatible changes |
| `source` | str | always `"mavis_gesture"` |
| `event` | str | event type (below) |
| `gesture` | str \| null | gesture that caused the transition; `null` for timeout / shutdown |
| `state` | str | controller state **after** the transition |
| `timestamp` | float | Unix time in seconds (ms precision) when the transition happened |

Event-specific fields (always present, may be `null`):

| `event` | Extra fields | Meaning |
|---|---|---|
| `feature.selection_pending` | `feature`, `label`, `previous_feature` | a feature was selected and awaits confirmation; `previous_feature` is set when it replaced another pending one |
| `feature.selection_cancelled` | `feature`, `reason` (`open_palm` \| `timeout`) | the pending selection was dropped; nothing was ever started |
| `feature.start_requested` | `feature`, `label` | **start this feature now** |
| `feature.stop_requested` | `feature`, `reason` (`open_palm` \| `shutdown`) | **stop and release this feature now** |
| `navigation.return_to_selection` | `previous_feature`, `reason` (`stopped` \| `cancelled` \| `timeout`) | show the feature-selection UI again |

Feature ids: `ocr`, `translation`, `price_comparison`.

### Example session (all five gestures)

```json
{"schema_version":1,"source":"mavis_gesture","event":"feature.selection_pending","gesture":"one_finger","state":"AWAITING_CONFIRMATION","feature":"ocr","label":"OCR","previous_feature":null,"timestamp":1793372212.075}
{"schema_version":1,"source":"mavis_gesture","event":"feature.selection_pending","gesture":"two_fingers","state":"AWAITING_CONFIRMATION","feature":"translation","label":"Live Translation","previous_feature":"ocr","timestamp":1793372213.410}
{"schema_version":1,"source":"mavis_gesture","event":"feature.selection_pending","gesture":"three_fingers","state":"AWAITING_CONFIRMATION","feature":"price_comparison","label":"Price Comparison","previous_feature":"translation","timestamp":1793372214.902}
{"schema_version":1,"source":"mavis_gesture","event":"feature.start_requested","gesture":"thumbs_up","state":"FEATURE_ACTIVE","feature":"price_comparison","label":"Price Comparison","timestamp":1793372216.330}
{"schema_version":1,"source":"mavis_gesture","event":"feature.stop_requested","gesture":"open_palm","state":"SELECTING_FEATURE","feature":"price_comparison","reason":"open_palm","timestamp":1793372230.118}
{"schema_version":1,"source":"mavis_gesture","event":"navigation.return_to_selection","gesture":"open_palm","state":"SELECTING_FEATURE","previous_feature":"price_comparison","reason":"stopped","timestamp":1793372230.118}
```

Cancel and timeout:

```json
{"schema_version":1,"source":"mavis_gesture","event":"feature.selection_cancelled","gesture":"open_palm","state":"SELECTING_FEATURE","feature":"ocr","reason":"open_palm","timestamp":1793372301.552}
{"schema_version":1,"source":"mavis_gesture","event":"navigation.return_to_selection","gesture":"open_palm","state":"SELECTING_FEATURE","previous_feature":"ocr","reason":"cancelled","timestamp":1793372301.552}
{"schema_version":1,"source":"mavis_gesture","event":"feature.selection_cancelled","gesture":null,"state":"SELECTING_FEATURE","feature":"ocr","reason":"timeout","timestamp":1793372220.089}
{"schema_version":1,"source":"mavis_gesture","event":"navigation.return_to_selection","gesture":null,"state":"SELECTING_FEATURE","previous_feature":"ocr","reason":"timeout","timestamp":1793372220.089}
```

### Ordering guarantees

- Events are emitted in transition order with non-decreasing timestamps.
- `feature.stop_requested` always comes before its
  `navigation.return_to_selection`, and `feature.selection_cancelled`
  before its return. Both events of a pair share the same timestamp.
- Every `start_requested` is eventually followed by exactly one
  `stop_requested` for the same feature: by open palm, or with reason
  `shutdown` on a normal exit (`Q`, Ctrl+C, SIGTERM, Ctrl+Break). A
  hard kill or crash cannot emit it, so consumers should also stop
  features when the process dies or stdout closes.
- At most one feature is active, and no duplicate start/stop is emitted.

### Error and no-hand behaviour

- No hand, a hand partly out of frame, low tracking confidence, or a
  degenerate skeleton count as `NO_HAND`. That resets the stabilizer
  and produces **no events** (no palm, no stop, no navigation).
- Ambiguous poses are `UNKNOWN`: no event.
- Camera failure at startup exits with a message on stderr. A camera
  that stops delivering frames is reported on stderr and the process
  keeps waiting.
- A failing event subscriber is logged and isolated (`EventBus.publish`).

### Parsing (consumer side)

```python
import json, subprocess, sys

proc = subprocess.Popen([sys.executable, "mavis_gesture.py", "--jsonl", "--no-preview"],
                        cwd="path/to/gesture-control", stdout=subprocess.PIPE, text=True)
for line in proc.stdout:
    e = json.loads(line)
    if e.get("source") != "mavis_gesture" or e.get("schema_version") != 1:
        continue                         # ignore unknown producers / versions
    if e["event"] == "feature.start_requested":
        start(e["feature"])
    elif e["event"] == "feature.stop_requested":
        stop(e["feature"])
    elif e["event"] == "feature.selection_pending":
        show_pending(e["feature"], e["label"])
    elif e["event"] == "navigation.return_to_selection":
        show_selection_menu()
# stdout closed -> recognizer exited: stop any running feature here
```

Consumers should ignore unknown event types and unknown fields, so
additive changes stay compatible.

## E. MAVIS integration guide

> The MAVIS repository has **not** been inspected. Everything below marked
> `MAVIS:` must be identified in that repository first. Do not assume names.

### Where things live (gesture side)

| Question | Answer |
|---|---|
| 1. Gesture → feature mapping | `config/gesture_mapping.json`, loaded by `gesture_mapping.load_mapping()` |
| 2. State machine | `feature_controller.py`, class `FeatureController` |
| 3. Feature-start requests | `FeatureController.on_gesture()`, branch `elif gesture == m.confirm_gesture` → `ev.START_REQUESTED` |
| 4. Feature-stop requests | `FeatureController.on_gesture()`, FEATURE_ACTIVE / back-gesture branch → `ev.STOP_REQUESTED`; also `FeatureController.shutdown()` |
| 5. Where to connect MAVIS | `GesturePipeline.bus.subscribe(callback)` (`mavis_gesture.py`); for the stdout transport, `run()` subscribes `ev.JsonlWriter` |

### Option 1: separate process, read stdout (least coupling)

Run `mavis_gesture.py --jsonl --no-preview` as a child process and use
the parsing loop from section D. Nothing in this repo changes.

Caveat: both processes can't open the webcam at once. Use this option
only if MAVIS doesn't need the camera while gestures are active, or
use Option 2.

### Option 2: in-process, MAVIS owns the camera (recommended)

Use `GesturePipeline` directly and feed it MAVIS's frames. No second
`cv2.VideoCapture` is opened.

```python
# MAVIS side — module/function names below are placeholders
import queue, threading, time, cv2
import events as ev
from gesture_mapping import load_mapping
from mavis_gesture import GesturePipeline, load_settings

cfg = load_settings()                       # gesture-control/config/mavis_gestures.json
pipe = GesturePipeline(cfg, load_mapping())
inbox = ev.QueueSink()                      # events -> queue, consumed on MAVIS's thread
pipe.bus.subscribe(inbox)

latest = {"frame": None}                    # MAVIS camera loop stores its newest frame here
lock = threading.Lock()

def on_camera_frame(frame_bgr):             # MAVIS: call from the existing capture loop
    with lock:
        latest["frame"] = frame_bgr         # just a handoff, never blocks the camera

def gesture_worker(stop_flag):
    while not stop_flag.is_set():
        with lock:
            frame, latest["frame"] = latest["frame"], None
        if frame is None:
            time.sleep(0.005); continue
        if cfg["camera"]["mirror"]:
            frame = cv2.flip(frame, 1)      # handedness assumes a selfie view
        pipe.process_frame(frame)           # ~40 ms MediaPipe; keep off the UI thread
    pipe.close()                            # emits stop_requested if a feature is active
```

Rules for this setup:

- `GesturePipeline` is **not thread-safe**. Call `process_frame()` from
  one thread only.
- Subscribers run on that worker thread, so hand events to MAVIS's
  thread via `QueueSink` or similar.
- If MAVIS frames are already mirrored, don't flip them again.
- Frames must be BGR `uint8` (OpenCV convention). `process_frame`
  converts to RGB internally.
- If a MAVIS feature itself needs frames (OCR, translation), give it
  the same frames from MAVIS's capture loop. Neither side should open
  its own capture.

### Mapping events to MAVIS modules

```python
# MAVIS side. Every MAVIS:* name is a placeholder to be found in the MAVIS repo.
FEATURE_HANDLERS = {
    "ocr":              (MAVIS_ocr_start,         MAVIS_ocr_stop),          # MAVIS: OCR module entry/exit
    "translation":      (MAVIS_translation_start, MAVIS_translation_stop),  # MAVIS: live translation module
    "price_comparison": (MAVIS_price_start,       MAVIS_price_stop),        # MAVIS: price comparison module
}
active = None

def handle(e: ev.Event):                    # run on MAVIS's main/UI thread
    global active
    if e.name == ev.START_REQUESTED:
        if active is not None:              # defensive; controller never does this
            FEATURE_HANDLERS[active][1]()
        FEATURE_HANDLERS[e.get("feature")][0]()
        active = e.get("feature")
    elif e.name == ev.STOP_REQUESTED:
        if active == e.get("feature"):
            FEATURE_HANDLERS[active][1]()   # must stop workers and release resources
            active = None
    elif e.name == ev.RETURN_TO_SELECTION:
        MAVIS_show_feature_menu()           # MAVIS: whatever shows the selection UI
    elif e.name == ev.SELECTION_PENDING:
        MAVIS_show_pending(e.get("label"))  # optional UI hint

while running:                              # MAVIS main loop
    try:
        handle(inbox.queue.get_nowait())
    except queue.Empty:
        pass
```

What to look for in the MAVIS repository:

- for each of OCR, live translation and price comparison:
  - the function or class that starts it (`MAVIS:*_start`);
  - one that stops it and releases its resources (`MAVIS:*_stop`);
  - whether it runs a background thread or process;
- whether MAVIS already has a camera loop (to hook `on_camera_frame`);
- the UI entry point for the feature menu.

### Stopping, background workers and resources

On `feature.stop_requested`:

- the MAVIS stop function must signal its worker threads (e.g. a
  `threading.Event`), `join()` them with a timeout, close windows and
  files, and release models or network sessions;
- it must **not** release the shared camera if the gesture pipeline
  still uses it;
- stop must be idempotent: the controller sends exactly one stop per
  start, but a consumer may also stop on process exit.

On MAVIS shutdown:

- stop the gesture worker thread, then call `pipe.close()`. It returns
  and publishes a `stop_requested` (reason `shutdown`) if needed;
- handle that event before tearing down feature modules.

Don't run long work inside a subscriber: it blocks recognition. Queue
it instead.

### Changing the transport

| Want | Do |
|---|---|
| Python callback | `pipe.bus.subscribe(fn)`; `fn(event: events.Event)` gets the object; use `events.to_dict(event)` for a dict |
| Queue | `pipe.bus.subscribe(events.QueueSink(my_queue))` |
| Keep stdout too | `pipe.bus.subscribe(events.JsonlWriter(sys.stdout))` (call `.close()` on exit) |
| Socket / IPC / log file | write a callable `def send(e): sock.sendall((events.to_json(e) + "\n").encode())` and subscribe it |
| Remove stdout from the app | in `mavis_gesture.run()`, drop `bus.subscribe(writer)` |

The classifier, stabilizer and state machine never change for a new
transport.

## F. Extension guide

- **Change a gesture → feature mapping or label.** Edit
  `selection_gestures` in `config/gesture_mapping.json`. Feature ids
  must be unique; `GestureMapping` validates this.
- **Swap confirm / back gestures.** Edit `confirm_gesture` /
  `back_gesture`. A gesture may have only one role.
- **Change timings.** Comfortable defaults: holds 0.5 s, cooldown 0.8 s.
  - interaction timings (`config/gesture_mapping.json` → `timings`):
    - `selection_hold_sec`, `confirm_hold_sec`, `back_hold_sec`: how
      long a gesture must be held;
    - `confirmation_timeout_sec`: when a pending selection expires.
  - recognition stability (`config/mavis_gestures.json` → `stability`):
    - `stable_frames`: minimum consecutive frames;
    - `min_confidence`;
    - `cooldown_sec`: minimum gap between gesture events;
    - `release_sec`: neutral time before the same gesture can fire again;
    - `rearm_after_loss_sec`: hand-absence time before the same gesture
      can fire again.
- **Add a new gesture**, e.g. `four_fingers`:
  1. Add the name to `GESTURES` and a finger template to `TEMPLATES` in
     `gesture_classifier.py`. Use E/F/T per finger; see
     `finger_state()`.
  2. Add any extra check in `gesture_checks()` if needed.
  3. Add a synthetic pose in `tests/synthetic_hand.py` (`pose()`), plus
     tests in `test_classifier.py`. Make sure existing negatives such as
     `"four"` are updated if they become positives.
  4. Map it in `config/gesture_mapping.json`, either as a selection
     gesture or as a new role. A new role also needs a branch in
     `FeatureController.on_gesture()` and probably a new event type.
- **Change event names or payload fields.**
  1. Edit the constants and `EVENT_FIELDS` in `events.py`.
  2. Edit the fields passed in `FeatureController._event(...)` calls.
  3. The wire format is produced only by `events.to_dict()`.
  4. Bump `SCHEMA_VERSION` for incompatible changes, and update
     `tests/test_events.py`, `tests/test_feature_controller.py` and
     section D.
- **Add a transport.** Any callable taking an `events.Event`. Subscribe
  it on the `EventBus` (see "Changing the transport").

## Verification checklist for future changes

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -t .      # must stay green
.venv\Scripts\python.exe mavis_gesture.py --jsonl --no-preview   # stdout must contain only JSON lines
.venv\Scripts\python.exe mavis_gesture.py                        # manual: select -> thumbs up -> palm
```
