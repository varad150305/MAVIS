"""Camera reader that stays at the live edge of the stream.

cv2.VideoCapture over HTTP buffers frames. When the reader is starved for a moment (the CPU is busy reading a page), the
phone keeps sending, the buffer fills, and every later read() returns an OLD frame: the picture runs behind and stays behind
for as long as the reader is not faster than the camera. Measured with timestamped frames, a 4 s stall left frames up to
1.8 s old, and 0.4 s old for another 8 s afterwards; with this reader the age stayed at the stall length itself and fell
back to 0.08 s at once.

read() grabs a frame, then keeps grabbing while grab() returns immediately (a buffered, stale frame); the first grab that has
to wait for the network means the newest frame has been reached. Only that frame is decoded to an image.
"""
import time

import cv2

DRAIN_MAX_GRABS = 150
DRAIN_WAIT_S = 0.015          # a grab that takes longer than this was waiting for the camera, not reading a buffer
BEHIND_AFTER_S = 0.04         # the caller was away longer than this since the last frame (more than a frame at 30 fps)


class LiveStream:
    def __init__(self, url):
        self.url = url
        self.stream = None
        self.connected = False
        self.frame_count = 0
        self.stale_dropped = 0          # buffered frames skipped to catch up, over the whole session
        self._last_done = None

    def connect(self):
        try:
            self.stream = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            self.connected = self.stream.isOpened()
            if self.connected:
                return True
        except Exception as exc:  # noqa: BLE001
            print(f"[STREAM] Connection error: {exc}")
        self.release()
        return False

    def read(self):
        if not self.connected or self.stream is None:
            return False, None
        try:
            now = time.perf_counter()
            behind = self._last_done is not None and now - self._last_done > BEHIND_AFTER_S
            if not self.stream.grab():
                self.connected = False
                return False, None
            dropped = 0
            # Only after the caller was away for more than a frame or so can frames be waiting. When it keeps up, read()
            # simply returns the next frame, and fps and latency are those of the camera.
            while behind and dropped < DRAIN_MAX_GRABS:
                started = time.perf_counter()
                if not self.stream.grab():
                    self.connected = False
                    return False, None
                if time.perf_counter() - started > DRAIN_WAIT_S:
                    break                    # that grab waited for a new frame: nothing older was left in the buffer
                dropped += 1                 # it returned at once, so the frame before it was stale
            ok, frame = self.stream.retrieve()
            self._last_done = time.perf_counter()
            if ok and frame is not None:
                self.frame_count += 1
                self.stale_dropped += dropped
                return True, frame
        except Exception as exc:  # noqa: BLE001
            print(f"[STREAM] Read error: {exc}")
        self.connected = False
        return False, None

    def release(self):
        if self.stream is not None:
            self.stream.release()
        self.stream = None
        self.connected = False
