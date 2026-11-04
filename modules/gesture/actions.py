"""
actions.py — Action layer.

Executes the action bound to a gesture in config.json.

Supported action types:
  {"action": "hotkey",  "keys": ["ctrl", "c"]}      — key combination
  {"action": "press",   "keys": ["volumeup"]}       — single key presses
  {"action": "scroll",  "amount": -400}             — scrolling (negative = down)
  {"action": "command", "cmd": "notepad"}           — launch a program
  {"action": "mouse_mode"}                          — toggle cursor control
  {"action": "click",   "button": "left"}           — mouse click
  {"action": "none"}                                — do nothing (placeholder)

pyautogui key names: playpause, volumeup, volumedown, volumemute,
space, enter, esc, left/right/up/down, f1..f24, win, alt, tab, etc.
"""

from __future__ import annotations

import shlex
import subprocess

import pyautogui

pyautogui.FAILSAFE = True   # jerk the mouse to the top-left corner = emergency stop
pyautogui.PAUSE = 0.0


class ActionExecutor:
    def __init__(self):
        self.mouse_mode = False
        self.screen_w, self.screen_h = pyautogui.size()

    def execute(self, action_cfg: dict) -> str:
        """Executes the action, returns text for the overlay (Feedback)."""
        kind = action_cfg.get("action", "none")

        if kind == "hotkey":
            keys = action_cfg.get("keys", [])
            pyautogui.hotkey(*keys)
            return "+".join(keys)

        if kind == "press":
            keys = action_cfg.get("keys", [])
            for k in keys:
                pyautogui.press(k)
            return ",".join(keys)

        if kind == "scroll":
            amount = int(action_cfg.get("amount", -300))
            pyautogui.scroll(amount)
            return f"scroll {amount}"

        if kind == "command":
            cmd = action_cfg.get("cmd", "")
            if cmd:
                # Popen does not block the main loop
                subprocess.Popen(shlex.split(cmd))
            return f"run: {cmd}"

        if kind == "mouse_mode":
            self.mouse_mode = not self.mouse_mode
            return f"mouse mode: {'ON' if self.mouse_mode else 'OFF'}"

        if kind == "click":
            pyautogui.click(button=action_cfg.get("button", "left"))
            return f"{action_cfg.get('button', 'left')} click"

        return ""

    def move_cursor(self, norm_x: float, norm_y: float,
                    margin: float = 0.15) -> None:
        """Moves the cursor using the normalized finger position (0..1).

        margin — a "dead zone" along the frame edges: you can reach the
        screen edges without stretching your hand to the very border of
        the camera view. The position is linearly stretched from
        [margin, 1-margin] into [0, 1].
        """
        span = 1.0 - 2 * margin
        x = min(max((norm_x - margin) / span, 0.0), 1.0)
        y = min(max((norm_y - margin) / span, 0.0), 1.0)
        pyautogui.moveTo(int(x * self.screen_w), int(y * self.screen_h))
