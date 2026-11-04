"""Process-level runtime setup and self-measurement for the product launcher.

Nothing here changes a system setting: offline model loading and UTF-8
console output are per-process environment choices, and the EcoQoS opt-out
is the same per-process request evaluation/mt_baseline.py already makes.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from typing import Any, Callable


def prefer_offline_models() -> None:
    """Load cached Hugging Face models without network checks.

    All models the product uses are already in the local cache; main.py does
    not set this, so its startup otherwise depends on hub reachability.
    An explicit environment value set by the user is respected.
    """

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def utf8_console(stream=None):
    """Make the console stream able to print Devanagari.

    A real Windows console already writes Unicode; a redirected stdout
    defaults to the ANSI code page, where printing Hindi raises
    UnicodeEncodeError. Returns the (reconfigured) stream.
    """

    stream = stream if stream is not None else sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    return stream


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def _kernel32():
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    return k32


def memory_mb() -> dict[str, float] | None:
    """Current and peak working set of this process (Windows; None elsewhere)."""

    if sys.platform != "win32":
        return None
    try:
        k32 = _kernel32()
        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return {"working_set_mb": counters.WorkingSetSize / 2**20, "peak_working_set_mb": counters.PeakWorkingSetSize / 2**20}
    except Exception:
        return None


def disable_power_throttling() -> bool:
    """Opt this process out of Windows EcoQoS execution-speed throttling,
    which otherwise deschedules a process whose window is not in focus."""

    if sys.platform != "win32":
        return False
    try:
        k32 = _kernel32()
        state = _PowerThrottlingState(1, 0x1, 0x0)
        return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state)))
    except Exception:
        return False


def keep_system_awake(enable: bool = True) -> bool:
    """Ask Windows not to sleep while a session runs (display may still turn off).

    A diagnostics session on 2026-10-28 was suspended mid-run (decode lag
    jumped to ~360 s). SetThreadExecutionState lasts until it is cleared
    with enable=False or the calling thread exits.
    """

    if sys.platform != "win32":
        return False
    es_continuous, es_system_required = 0x80000000, 0x00000001
    try:
        flags = es_continuous | (es_system_required if enable else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags)))
    except Exception:
        return False


def on_ac_power() -> bool | None:
    """True on AC, False on battery, None if unknown (Windows only).

    Recorded with every session because CPU speed on this hardware depends
    strongly on the power source (see the research run logs)."""

    if sys.platform != "win32":
        return None

    class _SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    status = _SPS()
    try:
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return None
    except Exception:
        return None
    return {0: False, 1: True}.get(status.ACLineStatus)


class ResourceSampler(threading.Thread):
    """Samples this process's CPU use (all threads) and working set once per
    interval. CPU is reported as cores busy (1.0 = one full logical core)."""

    def __init__(self, interval_s: float = 1.0,
                 on_sample: Callable[[dict[str, float]], None] | None = None) -> None:
        super().__init__(daemon=True, name="product-resource-sampler")
        self.interval_s = interval_s
        #: called with each sample (product.session publishes it); errors ignored
        self.on_sample = on_sample
        self.samples: list[dict[str, float]] = []
        self._stop_event = threading.Event()
        self._t0_wall = time.perf_counter()
        self._t0_cpu = time.process_time()

    def run(self) -> None:
        prev_wall, prev_cpu = time.perf_counter(), time.process_time()
        while not self._stop_event.wait(self.interval_s):
            wall, cpu = time.perf_counter(), time.process_time()
            memory = memory_mb() or {}
            sample = {
                "t_s": wall - self._t0_wall,
                "cores_busy": (cpu - prev_cpu) / max(wall - prev_wall, 1e-9),
                "working_set_mb": memory.get("working_set_mb", float("nan")),
            }
            self.samples.append(sample)
            if self.on_sample is not None:
                try:
                    self.on_sample(sample)
                except Exception:
                    pass
            prev_wall, prev_cpu = wall, cpu

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        self.join(timeout=5)
        wall = time.perf_counter() - self._t0_wall
        cpu = time.process_time() - self._t0_cpu
        busy = sorted(s["cores_busy"] for s in self.samples)
        memory = memory_mb() or {}
        return {
            "wall_s": wall,
            "process_cpu_s": cpu,
            "mean_cores_busy": cpu / wall if wall > 0 else None,
            "p95_cores_busy": busy[int(0.95 * (len(busy) - 1))] if busy else None,
            "max_cores_busy": busy[-1] if busy else None,
            "logical_cpus": os.cpu_count(),
            "peak_working_set_mb": memory.get("peak_working_set_mb"),
            "samples": len(self.samples),
        }
