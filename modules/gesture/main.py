"""
main.py — entry point for console mode. Wires the modules together:
camera -> Vision -> Intent -> Action.

Gesture trigger modes (the "mode" field in config/gestures.json):

  hold   — the gesture must be held for hold_sec seconds, then the command
           runs once; to repeat, release the gesture and show it again.
           With repeat_sec set, the command repeats while the gesture is
           held (e.g. continuous scrolling).

  toggle — the first read of the gesture turns the command ON, the next
           read turns it OFF. The hand can leave the frame in between.
           With repeat_sec set, the command repeats the whole time the
           toggle is active (even with no hand in the frame).

Launch modes:
  python main.py                          — normal control mode
  python main.py --list                   — show all gestures and actions
  python main.py --record rock            — record a new gesture "rock"
  python main.py --bind rock hotkey ctrl w      — bind an action to a gesture
  python main.py --mode rock toggle             — trigger mode
  python main.py --mode rock hold 1.2           — hold with a 1.2 s delay
  python main.py --mode rock hold 0.6 0.4       — hold + repeat every 0.4 s
  python main.py --delete rock            — delete a gesture

Hotkeys in the window:
  Q     — quit
  SPACE (in record mode) — capture a gesture sample
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from controller import run_loop
from gesture_engine import landmarks_to_features
from hand_tracker import HandTracker

CONFIG_PATH = Path(__file__).parent / "config" / "gestures.json"


# ---------------------------------------------------------------- config I/O

def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[ok] Config saved: {CONFIG_PATH}")


# ---------------------------------------------------------------- CLI commands

def cmd_list(cfg: dict) -> None:
    print(f"{'GESTURE':<16}{'TYPE':<10}{'MODE':<14}ACTION")
    print("-" * 62)
    for name, g in cfg["gestures"].items():
        do = g.get("do", {})
        desc = do.get("action", "none")
        if "keys" in do:
            desc += " " + "+".join(do["keys"])
        if "cmd" in do:
            desc += " " + do["cmd"]
        if "amount" in do:
            desc += f" {do['amount']}"
        mode = g.get("mode", "hold")
        if g.get("repeat_sec"):
            mode += f"+rep{g['repeat_sec']}"
        print(f"{name:<16}{g['type']:<10}{mode:<14}{desc}")


def cmd_bind(cfg: dict, args: list[str]) -> None:
    if len(args) < 2:
        sys.exit("Usage: --bind <gesture> <action> [params...]")
    name, kind, *params = args
    if name not in cfg["gestures"]:
        sys.exit(f"Gesture '{name}' not found. Record it first: --record {name}")

    do: dict = {"action": kind}
    if kind in ("hotkey", "press"):
        do["keys"] = params
    elif kind == "command":
        do["cmd"] = " ".join(params)
    elif kind == "scroll":
        do["amount"] = int(params[0]) if params else -300
    elif kind == "click":
        do["button"] = params[0] if params else "left"
    elif kind not in ("mouse_mode", "none"):
        sys.exit(f"Unknown action: {kind}")

    cfg["gestures"][name]["do"] = do
    save_config(cfg)


def cmd_mode(cfg: dict, args: list[str]) -> None:
    """--mode GESTURE hold|toggle [hold_sec] [repeat_sec]"""
    if len(args) < 2 or args[1] not in ("hold", "toggle"):
        sys.exit("Usage: --mode <gesture> hold|toggle [hold_sec] [repeat_sec]")
    name, mode, *params = args
    if name not in cfg["gestures"]:
        sys.exit(f"Gesture '{name}' not found.")

    g = cfg["gestures"][name]
    g["mode"] = mode
    g.pop("hold_sec", None)
    g.pop("repeat_sec", None)
    if mode == "hold" and len(params) >= 1:
        g["hold_sec"] = float(params[0])
    if len(params) >= 2:
        g["repeat_sec"] = float(params[1])
    elif mode == "toggle" and len(params) >= 1:
        g["repeat_sec"] = float(params[0])
    save_config(cfg)


def cmd_delete(cfg: dict, name: str) -> None:
    if cfg["gestures"].pop(name, None) is None:
        sys.exit(f"Gesture '{name}' not found.")
    save_config(cfg)


def cmd_record(cfg: dict, name: str, samples_needed: int = 12) -> None:
    """Record a custom gesture from the camera.

    Video is shown; the user holds the gesture and presses SPACE to capture
    a sample (a feature vector). After 12 samples (slightly vary the angle
    and distance between shots) the gesture is saved to the config.
    """
    cam_idx = cfg["settings"].get("camera_index", 0)
    mirror = cfg["settings"].get("mirror", True)

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        sys.exit(f"Camera {cam_idx} is unavailable.")
    tracker = HandTracker()
    samples: list[list[float]] = []

    print(f"Recording gesture '{name}': hold the gesture, SPACE — capture sample, Q — cancel.")
    try:
        while len(samples) < samples_needed:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            hands = tracker.process(frame)
            tracker.draw(frame)
            cv2.putText(frame, f"'{name}': {len(samples)}/{samples_needed}  [SPACE=snap  Q=quit]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Record gesture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Cancelled.")
                return
            if key == ord(" ") and hands:
                frame_size = (frame.shape[1], frame.shape[0])
                samples.append(landmarks_to_features(hands[0], frame_size).tolist())
                print(f"  sample {len(samples)}/{samples_needed}")
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    if len(samples) < samples_needed:
        print("Not enough samples, gesture not saved.")
        return

    cfg["gestures"][name] = {"type": "recorded", "samples": samples,
                             "mode": "hold", "do": {"action": "none"}}
    save_config(cfg)
    print("Now bind an action and a mode, for example:")
    print(f"  python main.py --bind {name} hotkey ctrl shift esc")
    print(f"  python main.py --mode {name} toggle")


# ---------------------------------------------------------------- main loop

def run(cfg: dict) -> None:
    print("Control started. Q in the window — quit. Emergency stop: mouse to the top-left corner.")
    run_loop(cfg)


# ---------------------------------------------------------------- entrypoint

def main() -> None:
    parser = argparse.ArgumentParser(description="Control your computer with hand gestures")
    parser.add_argument("--list", action="store_true", help="show gestures")
    parser.add_argument("--record", metavar="NAME", help="record a new gesture")
    parser.add_argument("--bind", nargs="+", metavar="ARG",
                        help="bind an action: --bind GESTURE ACTION [PARAMS]")
    parser.add_argument("--mode", nargs="+", metavar="ARG",
                        help="trigger mode: --mode GESTURE hold|toggle [hold_sec] [repeat_sec]")
    parser.add_argument("--delete", metavar="NAME", help="delete a gesture")
    args = parser.parse_args()

    cfg = load_config()

    if args.list:
        cmd_list(cfg)
    elif args.record:
        cmd_record(cfg, args.record)
    elif args.bind:
        cmd_bind(cfg, args.bind)
    elif args.mode:
        cmd_mode(cfg, args.mode)
    elif args.delete:
        cmd_delete(cfg, args.delete)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
