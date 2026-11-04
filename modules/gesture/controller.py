"""
controller.py — the main recognition loop, extracted from main.py.

The same loop is used by both the console mode (main.py) and the tray
application (tray_app.py). External control goes through threading.Event —
the standard way to communicate with a background thread:

  stop_event    — set -> the loop exits
  pause_event   — set -> the camera is released, recognition sleeps
  preview_event — set -> the video window is shown (Q in the window = stop)
"""

from __future__ import annotations

import threading
import time

import cv2

from actions import ActionExecutor
from gesture_engine import GestureEngine
from hand_tracker import HandTracker

WINDOW = "Gesture Control  [Q = quit]"


def run_loop(cfg: dict,
             stop_event: threading.Event | None = None,
             pause_event: threading.Event | None = None,
             preview_event: threading.Event | None = None) -> None:
    stop_event = stop_event or threading.Event()
    pause_event = pause_event or threading.Event()
    if preview_event is None:            # window visible by default
        preview_event = threading.Event()
        preview_event.set()

    s = cfg["settings"]
    cam_idx = s.get("camera_index", 0)
    stable_frames = s.get("stable_frames", 4)
    default_hold = s.get("hold_sec", 0.6)
    cooldown = s.get("cooldown_sec", 0.8)
    smooth = s.get("cursor_smoothing", 0.35)
    mirror = s.get("mirror", True)

    tracker = HandTracker()
    engine = GestureEngine(cfg["gestures"], knn_threshold=s.get("knn_threshold", 0.18))
    executor = ActionExecutor()

    cap = None
    window_visible = False

    # --- recognition state ---
    raw_name, raw_count = None, 0
    stable: str | None = None
    stable_since = 0.0
    fired_this_hold = False
    last_hold_repeat = 0.0
    last_fired = 0.0

    active_toggles: set[str] = set()
    toggle_repeat_at: dict[str, float] = {}

    overlay_text, overlay_until = "", 0.0
    sx, sy = 0.5, 0.5

    def fire(name: str, suffix: str = "") -> None:
        nonlocal overlay_text, overlay_until, last_fired
        do = cfg["gestures"][name].get("do", {"action": "none"})
        feedback = executor.execute(do)
        last_fired = time.time()
        overlay_text = f"{name}{suffix}: {feedback}"
        overlay_until = last_fired + 1.2
        print(f"[{time.strftime('%H:%M:%S')}] {overlay_text}")

    def reset_recognition() -> None:
        nonlocal raw_name, raw_count, stable, fired_this_hold
        raw_name, raw_count, stable, fired_this_hold = None, 0, None, False

    try:
        while not stop_event.is_set():
            # --- pause: release the camera so other apps can use it
            if pause_event.is_set():
                if cap is not None:
                    cap.release()
                    cap = None
                if window_visible:
                    cv2.destroyAllWindows()
                    window_visible = False
                reset_recognition()
                time.sleep(0.2)
                continue

            # --- (re)open the camera
            if cap is None:
                cap = cv2.VideoCapture(cam_idx)
                if not cap.isOpened():
                    print(f"[!] Camera {cam_idx} unavailable, retrying in 2 s...")
                    cap = None
                    time.sleep(2)
                    continue

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            if mirror:
                frame = cv2.flip(frame, 1)

            hands = tracker.process(frame)
            now = time.time()

            # --- cursor mode
            if hands and executor.mouse_mode:
                ix, iy = float(hands[0][8][0]), float(hands[0][8][1])
                sx = sx + smooth * (ix - sx)
                sy = sy + smooth * (iy - sy)
                executor.move_cursor(sx, sy)

            # --- raw recognition + debounce
            frame_size = (frame.shape[1], frame.shape[0])
            name = engine.recognize(hands[0], frame_size) if hands else None
            if name == raw_name:
                raw_count += 1
            else:
                raw_name, raw_count = name, 1

            # --- stable gesture change (rising edge)
            if raw_count >= stable_frames and raw_name != stable:
                stable = raw_name
                stable_since = now
                fired_this_hold = False
                last_hold_repeat = 0.0

                if stable is not None:
                    g = cfg["gestures"][stable]
                    if g.get("mode", "hold") == "toggle" and now - last_fired >= cooldown:
                        if stable in active_toggles:
                            active_toggles.discard(stable)
                            toggle_repeat_at.pop(stable, None)
                            fire(stable, " [OFF]")
                        else:
                            active_toggles.add(stable)
                            toggle_repeat_at[stable] = now
                            fire(stable, " [ON]")

            # --- hold gestures
            hold_progress = None
            if stable is not None:
                g = cfg["gestures"][stable]
                if g.get("mode", "hold") == "hold":
                    hold_sec = g.get("hold_sec", default_hold)
                    repeat = g.get("repeat_sec")
                    held = now - stable_since
                    if held >= hold_sec:
                        if not fired_this_hold and now - last_fired >= cooldown:
                            fire(stable)
                            fired_this_hold = True
                            last_hold_repeat = now
                        elif (fired_this_hold and repeat
                              and now - last_hold_repeat >= repeat):
                            fire(stable)
                            last_hold_repeat = now
                    else:
                        hold_progress = int(held / hold_sec * 100)

            # --- repeat active toggle gestures
            for tname in list(active_toggles):
                rep = cfg["gestures"].get(tname, {}).get("repeat_sec")
                if rep and now - toggle_repeat_at.get(tname, 0) >= rep:
                    fire(tname, " [rep]")
                    toggle_repeat_at[tname] = now

            # --- preview window (Feedback)
            if preview_event.is_set():
                tracker.draw(frame)
                status = f"mouse: {'ON' if executor.mouse_mode else 'off'}"
                if active_toggles:
                    status += "   active: " + ", ".join(sorted(active_toggles))
                cv2.putText(frame, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if hold_progress is not None:
                    cv2.putText(frame, f"{stable} {hold_progress}%", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                if now < overlay_until:
                    cv2.putText(frame, overlay_text, (10, frame.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                cv2.imshow(WINDOW, frame)
                window_visible = True
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
            elif window_visible:
                cv2.destroyAllWindows()
                window_visible = False
    finally:
        if cap is not None:
            cap.release()
        tracker.close()
        cv2.destroyAllWindows()
