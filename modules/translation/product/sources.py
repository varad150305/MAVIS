"""Audio sources handed to the unmodified main.run_m2()/run_m1() loops.

Both loops call `read()` in a tight `while True` and treat `None` as "no
frame yet - continue"; they stop only on KeyboardInterrupt, after which
they drain outstanding ASR work and finalize. The sources below keep that
contract and add the two things a product session needs:

  * `read()` waits briefly when no frame is ready instead of returning
    immediately - with the raw AudioStream the loop spins and keeps one CPU
    core busy even in silence (measured: ~0.93 cores idle);
  * a clean stop (stop event or duration limit) is delivered as the same
    KeyboardInterrupt the loops already handle, so every stop path goes
    through run_m2()'s existing drain/finalization code.

Phase 6 additions to LiveMicSource: a microphone stall (no frames at all,
which ordinary silence never causes - the callback keeps delivering quiet
frames) is reported once via `on_audio_health`, and so is its recovery;
stream stop()/close() failures (e.g. the device was unplugged) are recorded
in `cleanup_errors` instead of raised, because run_m2() calls them first in
its `finally` - an exception there would skip the ASR drain and finalization.
"""

from __future__ import annotations

import threading
import time
import traceback
import wave
from pathlib import Path

import numpy as np

from audio.audio_stream import AudioFrame, AudioStream


class InputMonitor:
    """Per-session input level statistics, optionally saving the audio.

    Every frame handed to run_m2 passes through `add()`. The levels answer
    "did the microphone capture usable speech?" after the fact (peak, RMS
    distribution, clipping); `save_path` additionally writes the exact
    16 kHz mono audio the pipeline consumed, so a live session can be
    replayed with `--wav` and compared across configurations.
    """

    def __init__(self, sample_rate: int = 16000, save_path: str | Path | None = None) -> None:
        self.sample_rate = sample_rate
        self.save_path = Path(save_path) if save_path is not None else None
        self._wav = None
        if self.save_path is not None:
            self._wav = wave.open(str(self.save_path), "wb")
            self._wav.setnchannels(1)
            self._wav.setsampwidth(2)
            self._wav.setframerate(sample_rate)
        self.frame_rms: list[float] = []
        self.peak = 0.0
        self.clipped_samples = 0
        self.samples = 0

    def add(self, samples: np.ndarray) -> None:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        self.frame_rms.append(float(np.sqrt(np.mean(x * x))))
        self.peak = max(self.peak, float(np.max(np.abs(x))))
        self.clipped_samples += int(np.count_nonzero(np.abs(x) >= 0.999))
        self.samples += x.size
        if self._wav is not None:
            self._wav.writeframes((np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None

    def summary(self) -> dict:
        def dbfs(v: float) -> float | None:
            return round(20.0 * float(np.log10(v)), 1) if v > 0 else None

        rms = sorted(self.frame_rms)
        pick = (lambda q: rms[min(len(rms) - 1, int(q * (len(rms) - 1)))]) if rms else (lambda q: 0.0)
        return {"audio_s": self.samples / self.sample_rate, "peak_dbfs": dbfs(self.peak),
                "frame_rms_dbfs": {"p10": dbfs(pick(0.10)), "p50": dbfs(pick(0.50)), "p90": dbfs(pick(0.90)),
                                   "p99": dbfs(pick(0.99))},
                "clipped_fraction": self.clipped_samples / self.samples if self.samples else 0.0,
                "saved_wav": str(self.save_path) if self.save_path is not None else None}


class _StopMixin:
    #: optional InputMonitor fed with every frame returned by read()
    monitor: InputMonitor | None = None

    def _check_stop(self, frames_read: int, frame_s: float) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            self.stop_reason = self.stop_reason or "stop requested"
            raise KeyboardInterrupt
        if self.max_duration_s is not None and frames_read * frame_s >= self.max_duration_s:
            self.stop_reason = self.stop_reason or f"duration limit {self.max_duration_s:g} s reached"
            raise KeyboardInterrupt


class LiveMicSource(_StopMixin):
    """Microphone frames from the existing AudioStream, without busy-waiting."""

    def __init__(
        self,
        audio_stream: AudioStream,
        stop_event: threading.Event | None = None,
        max_duration_s: float | None = None,
        idle_wait_s: float = 0.005,
        stall_after_s: float | None = 2.0,
        time_fn=time.perf_counter,
    ) -> None:
        self.audio_stream = audio_stream
        self.stop_event = stop_event
        self.max_duration_s = max_duration_s
        self.idle_wait_s = idle_wait_s
        self.frame_s = audio_stream.chunk_duration_ms / 1000.0
        self.frames_read = 0
        self.stop_reason: str | None = None
        #: perf_counter time at which frame 0 started (set on the first frame).
        self.t0: float | None = None
        self.started_at: float | None = None
        #: how long each frame waited between capture and consumption (s).
        self.consume_lag_s: list[float] = []
        #: called once capture has started (models are loaded by then).
        self.on_start = None
        #: no frame for this long while capturing = stall (None disables)
        self.stall_after_s = stall_after_s
        self._time = time_fn
        self._last_frame_at: float | None = None   # time_fn() of the last frame (or of start)
        self._stall: dict | None = None             # the open stall, if any
        #: every stall: {start_s, end_s (None while open), duration_s, after_frame}
        self.stalls: list[dict] = []
        #: callback(record) for audio_stall / audio_resumed / cleanup_error records
        self.on_audio_health = None
        #: stream stop()/close() failures: {op, error, traceback}
        self.cleanup_errors: list[dict] = []

    @property
    def dropped_frames(self) -> int:
        return self.audio_stream.dropped_frames

    def start(self) -> None:
        self.audio_stream.start()
        self.started_at = time.perf_counter()
        self._last_frame_at = self._time()
        if self.on_start is not None:
            self.on_start()

    def stop(self) -> None:
        self._safe("stop", self.audio_stream.stop)

    def close(self) -> None:
        self._safe("close", self.audio_stream.close)

    def _safe(self, op: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - recorded, see module docstring
            err = {"type": "cleanup_error", "op": f"audio_stream.{op}", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(), "t_s": self.clock()}
            self.cleanup_errors.append(err)
            self._emit(err)

    def _emit(self, record: dict) -> None:
        if self.on_audio_health is not None:
            try:
                self.on_audio_health(dict(record))
            except Exception:  # noqa: BLE001 - reporting must not break capture
                pass

    def _check_stall(self, now: float) -> None:
        if self.stall_after_s is None or self._stall is not None or self._last_frame_at is None:
            return
        gap = now - self._last_frame_at
        if gap >= self.stall_after_s:
            self._stall = {"start_s": self.clock() - gap, "end_s": None, "duration_s": None,
                           "after_frame": self.frames_read, "_t": self._last_frame_at}
            self.stalls.append(self._stall)
            self._emit({"type": "audio_stall", "t_s": self.clock(), "no_frames_for_s": round(gap, 3),
                        "threshold_s": self.stall_after_s, "frames_read": self.frames_read})

    def _end_stall(self, now: float) -> None:
        st, self._stall = self._stall, None
        st["duration_s"] = round(now - st.pop("_t"), 3)
        st["end_s"] = self.clock()
        self._emit({"type": "audio_resumed", "t_s": st["end_s"], "stalled_for_s": st["duration_s"],
                    "frames_read": self.frames_read})

    def stall_report(self) -> list[dict]:
        """Stalls for the session summary; one still open has end_s None."""

        return [{k: v for k, v in st.items() if not k.startswith("_")} for st in self.stalls]

    def clock(self) -> float:
        """Seconds on the session's audio clock (0 = start of frame 0)."""

        origin = self.t0 if self.t0 is not None else self.started_at
        return time.perf_counter() - origin if origin is not None else 0.0

    def read(self) -> AudioFrame | None:
        self._check_stop(self.frames_read, self.frame_s)
        frame = self.audio_stream.read()
        now = self._time()
        if frame is None:
            self._check_stall(now)
            time.sleep(self.idle_wait_s)
            return None
        if self._stall is not None:
            self._end_stall(now)
        self._last_frame_at = now
        if self.t0 is None:
            # capture_timestamp is taken when the callback fires, i.e. at the
            # END of the frame's audio.
            self.t0 = frame.capture_timestamp - frame.duration_ms / 1000.0
        self.frames_read += 1
        self.consume_lag_s.append(time.perf_counter() - frame.capture_timestamp)
        if self.monitor is not None:
            self.monitor.add(frame.samples)
        return frame


class PacedWavSource(_StopMixin):
    """Replays a 16 kHz mono WAV at 1x real time, frame by frame.

    Used for controlled, repeatable product tests of the same live path the
    microphone uses. Frames are released when their audio would have been
    fully captured (like a microphone callback); end of file is delivered as
    KeyboardInterrupt so run_m2() drains and finalizes exactly as on Ctrl+C.
    """

    def __init__(
        self,
        path: str | Path,
        chunk_duration_ms: int = 100,
        stop_event: threading.Event | None = None,
        max_duration_s: float | None = None,
        idle_wait_s: float = 0.005,
    ) -> None:
        self.path = Path(path)
        self.stop_event = stop_event
        self.max_duration_s = max_duration_s
        self.idle_wait_s = idle_wait_s
        self.stop_reason: str | None = None
        self.samples, self.sample_rate = read_wav_mono_16k(self.path)
        self.frame_samples = int(round(self.sample_rate * chunk_duration_ms / 1000.0))
        self.frame_s = self.frame_samples / self.sample_rate
        self.total_frames = int(np.ceil(len(self.samples) / self.frame_samples)) if len(self.samples) else 0
        self.frames_read = 0
        self.dropped_frames = 0
        self.t0: float | None = None
        self.consume_lag_s: list[float] = []
        self.on_start = None

    @property
    def audio_duration_s(self) -> float:
        return len(self.samples) / self.sample_rate

    def start(self) -> None:
        self.t0 = time.perf_counter()
        if self.on_start is not None:
            self.on_start()

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

    def clock(self) -> float:
        return time.perf_counter() - self.t0 if self.t0 is not None else 0.0

    def read(self) -> AudioFrame | None:
        if self.frames_read >= self.total_frames:
            self.stop_reason = self.stop_reason or "end of file"
            raise KeyboardInterrupt
        self._check_stop(self.frames_read, self.frame_s)
        k = self.frames_read
        due = self.t0 + (k + 1) * self.frame_s
        now = time.perf_counter()
        if now < due:
            time.sleep(min(due - now, self.idle_wait_s))
            return None
        chunk = self.samples[k * self.frame_samples:(k + 1) * self.frame_samples]
        if len(chunk) < self.frame_samples:
            chunk = np.concatenate([chunk, np.zeros(self.frame_samples - len(chunk), dtype=np.float32)])
        self.frames_read += 1
        self.consume_lag_s.append(now - due)
        if self.monitor is not None:
            self.monitor.add(chunk)
        return AudioFrame(
            samples=chunk,
            capture_timestamp=due,
            duration_ms=self.frame_s * 1000.0,
            sequence_number=k,
            wall_timestamp_ns=time.time_ns(),
            sample_rate=self.sample_rate,
            channels=1,
        )


def read_wav_mono_16k(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV as float32 mono at 16 kHz (multi-channel is averaged).

    Other sample rates are rejected rather than silently resampled.
    """

    with wave.open(str(path), "rb") as wav:
        channels, width, rate, count = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        raw = wav.readframes(count)
    if rate != 16000:
        raise ValueError(f"{path}: expected 16000 Hz audio, got {rate} Hz")
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"{path}: unsupported sample width {width} bytes")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), rate


def check_input_device(device: int | str | None, sample_rate: int = 16000, channels: int = 1) -> str:
    """Validate the input device BEFORE any model is loaded.

    Returns the device's name; raises RuntimeError with an actionable
    message otherwise (run_m2 would only discover this after ~20 s of model
    loading, and then exit with a generic "Runtime error").
    """

    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise RuntimeError("sounddevice is not installed") from exc
    try:
        info = sd.query_devices(device, kind="input")
        sd.check_input_settings(device=device, samplerate=sample_rate, channels=channels, dtype="float32")
    except Exception as exc:
        raise RuntimeError(
            f"Input device {device!r} is not usable at {sample_rate} Hz / {channels} ch: {exc}. "
            "Run with --list-devices to see the available inputs."
        ) from exc
    return str(info["name"])
