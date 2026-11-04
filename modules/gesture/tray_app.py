"""
tray_app.py — minimalist system tray application.

A hand icon in the system tray (green — running, gray — paused).
Right click — menu:

  Settings           — gesture and command configuration window
  Pause              — stop recognition and release the camera
  Camera window      — show/hide the video preview
  Open config        — gestures.json in the default editor
  Reload config      — restart the loop with new settings
  Quit

Recognition runs in a background thread (threading.Thread), the tray menu
runs in the main one. They communicate through threading.Event —
thread-safe flags: the loop-owning thread simply checks them on every
iteration.

Launch without a console window (Windows):
  .venv\\Scripts\\pythonw.exe tray_app.py
or double-click start_tray.vbs.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pystray
from PIL import Image, ImageDraw
from pystray import MenuItem as item

from controller import run_loop
from main import CONFIG_PATH, load_config

# --- background thread control flags ---
stop_event = threading.Event()      # quit
pause_event = threading.Event()     # pause recognition
preview_event = threading.Event()   # camera window (off by default)

worker: threading.Thread | None = None
restart_requested = False


# ---------------------------------------------------------------- icon

def make_icon(active: bool) -> Image.Image:
    """Draw a simple palm: green — active, gray — paused."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = (46, 204, 113, 255) if active else (150, 150, 150, 255)

    d.rounded_rectangle([14, 28, 50, 58], radius=10, fill=color)      # palm
    finger_tops = [14, 8, 6, 10]                                      # varied lengths
    for x, top in zip((15, 24, 33, 42), finger_tops):
        d.rounded_rectangle([x, top, x + 7, 34], radius=3, fill=color)
    d.rounded_rectangle([48, 30, 60, 40], radius=4, fill=color)       # thumb
    return img


# ---------------------------------------------------------------- worker thread

def start_worker() -> None:
    global worker
    cfg = load_config()
    worker = threading.Thread(
        target=run_loop,
        args=(cfg, stop_event, pause_event, preview_event),
        daemon=True,
    )
    worker.start()


def watch_worker(icon: pystray.Icon) -> None:
    """If the loop exits on its own (Q in the window) — close the tray too."""
    global restart_requested
    while True:
        worker.join()
        if restart_requested:
            restart_requested = False
            stop_event.clear()
            start_worker()
            continue
        icon.stop()
        break


# ---------------------------------------------------------------- menu items

def toggle_pause(icon: pystray.Icon, _item) -> None:
    if pause_event.is_set():
        pause_event.clear()
    else:
        pause_event.set()
    icon.icon = make_icon(active=not pause_event.is_set())


def toggle_preview(icon: pystray.Icon, _item) -> None:
    if preview_event.is_set():
        preview_event.clear()
    else:
        preview_event.set()


def open_settings(_icon, _item) -> None:
    """Settings window — a separate process, to keep the GUI event loops apart."""
    gui = Path(__file__).parent / "settings_gui.py"
    subprocess.Popen([sys.executable, str(gui)])


def watch_config(_icon) -> None:
    """Auto-reload: config mtime changed — restart the recognition loop."""
    global restart_requested
    last = CONFIG_PATH.stat().st_mtime
    while True:
        time.sleep(2)
        try:
            m = CONFIG_PATH.stat().st_mtime
        except OSError:
            continue
        if m != last:
            last = m
            print("[tray] Config changed — restarting recognition")
            restart_requested = True
            stop_event.set()


def open_config(_icon, _item) -> None:
    path = str(CONFIG_PATH)
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def restart_worker(_icon, _item) -> None:
    """Reload the config: stop the loop gracefully, watch_worker starts a new one."""
    global restart_requested
    restart_requested = True
    stop_event.set()


def quit_app(icon: pystray.Icon, _item) -> None:
    stop_event.set()
    if worker is not None:
        worker.join(timeout=3)
    icon.stop()


# ---------------------------------------------------------------- launch

def main() -> None:
    start_worker()

    menu = pystray.Menu(
        item("Settings", open_settings),
        item("Pause", toggle_pause, checked=lambda _i: pause_event.is_set()),
        item("Camera window", toggle_preview, checked=lambda _i: preview_event.is_set()),
        pystray.Menu.SEPARATOR,
        item("Open config", open_config),
        item("Reload config", restart_worker),
        pystray.Menu.SEPARATOR,
        item("Quit", quit_app),
    )
    icon = pystray.Icon("gesture-control", make_icon(True), "Gesture Control", menu)

    threading.Thread(target=watch_worker, args=(icon,), daemon=True).start()
    threading.Thread(target=watch_config, args=(icon,), daemon=True).start()
    icon.run()   # blocks the main thread until icon.stop()


if __name__ == "__main__":
    main()
