"""Simulated camera for trying the product without a phone (always labelled "Simulated camera" in the UI).

Plays the rendered validation pages (llm_validation/images) one after another, each held steady for a few seconds with
a little sensor noise, then keeps showing the last page until the user presses Stop. Same interface as
mjpeg_stream.MJPEGStreamReader (connect / read / release), paced in real time.
"""
import time
from pathlib import Path

import cv2
import numpy as np

IMAGES = Path(__file__).resolve().parent.parent / "llm_validation" / "images"
DEFAULT_PAGES = ("invoice_clean", "table_clean", "exam_degraded", "minutes_degraded")


class DemoStream:
    def __init__(self, fps=15.0, hold_s=5.5, pages=DEFAULT_PAGES, size=(1280, 720)):
        self.url = "demo://simulated-camera"
        self.fps, self.hold_s, self.size = fps, hold_s, size
        self.pages = [p for p in pages if (IMAGES / f"{p}.png").exists()]
        self._variants = []
        self._idx = 0
        self._t0 = None
        self._rng = np.random.default_rng(7)

    def connect(self):
        if not self.pages:
            return False
        for name in self.pages:
            img = cv2.imread(str(IMAGES / f"{name}.png"))
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
            self._variants.append([np.clip(img.astype(np.float32) + self._rng.normal(0, 1.2, img.shape), 0, 255)
                                   .astype(np.uint8) for _ in range(6)])
        time.sleep(0.6)                      # a connection is not instantaneous; lets the UI show its state
        return True

    def read(self):
        now = time.perf_counter()
        self._t0 = self._t0 or now
        due = self._t0 + self._idx / self.fps
        if due > now:
            time.sleep(due - now)
        per_page = int(self.hold_s * self.fps)
        page = min(self._idx // per_page, len(self._variants) - 1)
        frame = self._variants[page][self._idx % 6]
        self._idx += 1
        return True, frame.copy()

    def release(self):
        self._variants = []
