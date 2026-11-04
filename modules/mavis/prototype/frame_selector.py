"""Stability-gated best-frame selection for the MAVIS capture loop.

Production port of the two best methods from the frame-selection benchmark
(frame_selection_benchmark/results_public_eval/final/FINAL_REPORT.md):

    "c3"  stability -> best   frame-difference stability signal
    "c5"  optical flow        identical structure, Farneback-flow signal

Both: a causal hysteresis segmenter marks the view STABLE after
MIN_STABLE_S of calm frames and MOVING again when the signal exceeds the high
threshold; the sharpest frame of the stable run (block-wise Laplacian
variance, Kumar 2013) is committed when the run ends, or every
MAX_SEGMENT_S while it lasts. A commit whose 32x32 fingerprint is within
DEDUP_THRESHOLD of the previous commit is dropped (same duplicate gate as the
B0 production loop).

The parameters below are the FROZEN pre-registered values from
frame_selection_benchmark/config/benchmark.yaml. Do not retune them on the two
recorded videos or on the public data (the benchmark rules this out); new
values must be pre-registered on new MAVIS recordings.

Differences from the benchmark classes (same decisions, cheaper bookkeeping):
  * only the running-best frame of the stable run is kept (not every frame),
    so memory is O(min_stable), not O(max_segment_s * fps) full frames;
    a tie keeps the first maximum, as Python max() did in the benchmark;
  * sharpness is computed lazily, only for frames that belong to a stable run;
  * the 32x32 fingerprint is computed only for the frame about to be committed;
  * motion/flow are normalised by the REAL time gap between processed frames
    (the live capture loop skips frames when it is slower than the camera), and
    the stable-run length uses a running fps estimate. With fixed_fps given
    (replay/parity mode) the gap is 1/fixed_fps and the run lengths are fixed,
    which reproduces the benchmark frame for frame.

The selector never touches OCR output or ground truth. It does not change
global OpenCV state; callers that care about CPU contention with the OCR
worker should call cv2.setNumThreads(1), as the benchmark timing runs did.
"""
import os
from collections import deque

import cv2
import numpy as np

SELECTOR_MODES = ("b0", "c3", "c5")
DEFAULT_FRAME_SELECTOR = "b0"
ENV_VAR = "MAVIS_FRAME_SELECTOR"

# --- frozen benchmark.yaml values ------------------------------------------
ANALYSIS_WIDTH = 640          # sharpness is measured on a 640 px wide grey image
MOTION_WIDTH = 160            # frame-difference signal on a 160 px wide grey image
FLOW_WIDTH = 320              # Farneback flow on a 320 px wide grey image
REFERENCE_FPS = 25.0          # signals are rescaled to a 25-fps equivalent
MIN_STABLE_S = 0.25
MAX_SEGMENT_S = 4.0
DEDUP_THRESHOLD = 5.0         # == PrototypeApp.DUPLICATE_DISTANCE_THRESHOLD
THRESHOLDS = {                # (high, low), fps-normalised
    "c3": (14.3, 8.8),        # mean abs frame difference, pooled p75 / median
    "c5": (4.4, 2.1),         # median flow px/frame @320 px, pooled p75 / median
}
FARNEBACK = dict(pyr_scale=0.5, levels=2, winsize=13, iterations=2, poly_n=5, poly_sigma=1.1)
BLOCK_ROWS, BLOCK_COLS = 2, 3

_MIN_DT_S, _MAX_DT_S = 1e-3, 1.0
_FPS_EMA_ALPHA = 0.1


def resolve_mode(explicit=None):
    """CLI value > environment > DEFAULT_FRAME_SELECTOR. Returns a lower-case mode."""
    mode = (explicit or os.environ.get(ENV_VAR) or DEFAULT_FRAME_SELECTOR).strip().lower()
    if mode not in SELECTOR_MODES:
        raise ValueError(f"unknown frame selector {mode!r}; choose one of {SELECTOR_MODES}")
    return mode


def build_selector(mode, fixed_fps=None):
    """Return a selector for 'c3'/'c5', or None for 'b0' (the cooldown loop in main.py)."""
    mode = resolve_mode(mode)
    return None if mode == "b0" else StabilitySelector(mode, fixed_fps=fixed_fps)


def _grey(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _downscale(img, width):
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def blockwise_sharpness(grey_640):
    """Mean of the Laplacian variance over a 2x3 grid (Kumar 2013 block pooling)."""
    lap = cv2.Laplacian(grey_640, cv2.CV_32F, ksize=1)
    h, w = lap.shape
    vals = [float(lap[r * h // BLOCK_ROWS:(r + 1) * h // BLOCK_ROWS,
                      c * w // BLOCK_COLS:(c + 1) * w // BLOCK_COLS].var())
            for r in range(BLOCK_ROWS) for c in range(BLOCK_COLS)]
    return float(np.mean(vals))


def fingerprint(frame_bgr):
    """32x32 grey INTER_AREA - identical to PrototypeApp._fingerprint."""
    return cv2.resize(_grey(frame_bgr), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


class StabilitySelector:
    """Feed every frame the capture loop sees to process(); it returns commits.

    A commit is a dict: frame (ndarray to OCR), frame_index (position of that
    frame in the sequence fed to process(), 0-based), decision_index (the
    frame at which the decision was made), sharpness, fingerprint, reason
    ('stable_end' | 'max_segment' | 'flush').
    """

    def __init__(self, mode="c3", fixed_fps=None):
        if mode not in THRESHOLDS:
            raise ValueError(f"mode must be one of {tuple(THRESHOLDS)}, got {mode!r}")
        self.mode = mode
        self.high, self.low = THRESHOLDS[mode]
        self.fixed_fps = fixed_fps
        self._fps = float(fixed_fps) if fixed_fps else None
        self._last_t = None
        self._prev_small = None       # c3: previous 160 px grey frame
        self._prev_flow = None        # c5: previous 320 px grey frame
        self._idx = -1
        self._stable = False
        self._calm = 0
        self._pre = deque()           # last (min_stable-1) calm frames before a run starts
        self._run_len = 0
        self._best = None             # running best of the current stable run
        self._last_fp = None
        self.stats = {"frames": 0, "stable_runs": 0, "commits": 0,
                      "dup_suppressed": 0, "max_segment_commits": 0}

    # -- run-length parameters (frames), from the (fixed or estimated) fps ---
    @property
    def fps(self):
        return self._fps if self._fps else REFERENCE_FPS

    @property
    def stable(self):
        return self._stable

    def _min_stable(self):
        return max(1, int(round(MIN_STABLE_S * self.fps)))

    def _max_len(self):
        return max(1, int(round(MAX_SEGMENT_S * self.fps)))

    # -- signal -----------------------------------------------------------------
    def _dt_and_update_fps(self, t):
        if self.fixed_fps:
            return 1.0 / self.fixed_fps
        if t is None:
            raise ValueError("process() needs a timestamp unless fixed_fps was given")
        if self._last_t is None:
            self._last_t = t
            return 1.0 / REFERENCE_FPS
        dt = min(max(t - self._last_t, _MIN_DT_S), _MAX_DT_S)
        self._last_t = t
        inst = 1.0 / dt
        self._fps = inst if self._fps is None else (1 - _FPS_EMA_ALPHA) * self._fps + _FPS_EMA_ALPHA * inst
        return dt

    def _signal(self, g, dt):
        k = REFERENCE_FPS * dt        # == reference_fps / fps in the benchmark
        if self.mode == "c3":
            small = cv2.GaussianBlur(_downscale(g, MOTION_WIDTH), (3, 3), 0.5)
            sig = 0.0 if self._prev_small is None else k * float(
                np.mean(cv2.absdiff(small, self._prev_small)))
            self._prev_small = small
            return sig
        small = _downscale(g, FLOW_WIDTH)
        if self._prev_flow is None:
            sig = 0.0
        else:
            fl = cv2.calcOpticalFlowFarneback(self._prev_flow, small, None, flags=0, **FARNEBACK)
            sig = k * float(np.median(cv2.magnitude(fl[..., 0], fl[..., 1]))) * (320.0 / FLOW_WIDTH)
        self._prev_flow = small
        return sig

    def _sharp(self, frame):
        return blockwise_sharpness(_downscale(_grey(frame), ANALYSIS_WIDTH))

    # -- hysteresis segmenter (Wigington 2024), same events as the benchmark ----
    def _update_segmenter(self, signal):
        if self._stable:
            if signal > self.high:
                self._stable, self._calm = False, 0
                return "stable_end"
            return None
        self._calm = self._calm + 1 if signal < self.low else 0
        if self._calm >= self._min_stable():
            self._stable = True
            return "stable_start"
        return None

    def _offer(self, idx, frame, sharp):
        self._run_len += 1
        if self._best is None or sharp > self._best["sharpness"]:   # strict: first max wins
            self._best = {"idx": idx, "frame": frame, "sharpness": sharp}

    def _finalize(self, decision_idx, reason):
        best, self._best, self._run_len = self._best, None, 0
        if best is None:
            return []
        fp = fingerprint(best["frame"])
        if self._last_fp is not None and float(np.mean(np.abs(fp - self._last_fp))) <= DEDUP_THRESHOLD:
            self.stats["dup_suppressed"] += 1
            return []
        self._last_fp = fp
        self.stats["commits"] += 1
        if reason == "max_segment":
            self.stats["max_segment_commits"] += 1
        return [{"frame": best["frame"], "frame_index": best["idx"], "decision_index": decision_idx,
                 "sharpness": best["sharpness"], "fingerprint": fp, "reason": reason}]

    # -- public API ---------------------------------------------------------------
    def process(self, frame, t=None):
        """Consume the next frame (BGR ndarray, not modified). t = capture time in s."""
        self._idx += 1
        idx = self._idx
        self.stats["frames"] += 1
        dt = self._dt_and_update_fps(t)
        g = _grey(frame)
        ev = self._update_segmenter(self._signal(g, dt))
        if ev == "stable_end":
            out = self._finalize(idx, "stable_end")
            self._pre.clear()
            return out
        if not self._stable:
            self._pre.append((idx, frame))
            while len(self._pre) > self._min_stable() - 1:
                self._pre.popleft()
            return []
        if ev == "stable_start":
            self.stats["stable_runs"] += 1
            self._run_len, self._best = 0, None
            for pidx, pframe in self._pre:      # the calm frames that established stability
                self._offer(pidx, pframe, self._sharp(pframe))
            self._pre.clear()
        self._offer(idx, frame, blockwise_sharpness(_downscale(g, ANALYSIS_WIDTH)))
        if self._run_len >= self._max_len():
            out = self._finalize(idx, "max_segment")
            return out
        return []

    def flush(self):
        """Commit the pending stable run (call once when scanning ends)."""
        if self._run_len > 0:
            return self._finalize(self._idx, "flush")
        return []
