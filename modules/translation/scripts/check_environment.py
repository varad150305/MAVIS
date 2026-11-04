"""Environment and isolation check for the handoff package.

    .venv\\Scripts\\python.exe -B scripts\\check_environment.py

Checks: Python 3.10 x64; required packages and versions; CPU torch; tkinter;
every project module (product, main, asr, audio, vad, streaming,
translation, utils, evaluation) resolves INSIDE this package; nothing on
sys.path points at the research repository; the Hugging Face cache is
package-local; the host integration flag is off; output root is inside
the package. Exit code 0 = all good.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPECTED = {"torch": "2.7.1", "torchaudio": "2.7.1", "numpy": "2.2.6", "PyYAML": "6.0.2", "sounddevice": "0.5.1",
            "silero-vad": "6.2.1", "faster-whisper": "1.1.1", "ctranslate2": "4.8.1", "transformers": "4.52.4",
            "tokenizers": "0.21.4", "huggingface-hub": "0.36.2", "sentencepiece": "0.2.0",
            "opencv-python": "4.11.0"}
PROJECT = ["product", "product.session", "product.session_worker", "product.live_translate", "product.live_pipeline",
           "product.final_mt", "product.ui.app", "main", "asr.whisper_engine", "audio.audio_stream", "vad.vad",
           "streaming.streaming_buffer", "streaming.prefix_agreement", "translation.translator",
           "translation.scheduling", "utils.config", "evaluation.asr_evaluation"]


def main() -> int:
    bad: list[str] = []
    print(f"Python {platform.python_version()} {platform.machine()} ({sys.executable})")
    if sys.version_info[:2] != (3, 10):
        bad.append("Python 3.10 expected (tested 3.10.11)")
    for name, want in EXPECTED.items():
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            bad.append(f"{name} not installed")
            continue
        if not have.startswith(want):
            bad.append(f"{name} {have} (expected {want})")
    try:
        import torch

        if torch.version.cuda is not None:
            print(f"note: torch has CUDA {torch.version.cuda}; the product runs on CPU only")
    except Exception as exc:  # noqa: BLE001
        bad.append(f"torch import failed: {exc}")
    try:
        import tkinter  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        bad.append(f"tkinter unavailable (desktop UI): {exc}")
    for mod in PROJECT:
        try:
            m = importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"import {mod} failed: {type(exc).__name__}: {exc}")
            continue
        f = Path(m.__file__).resolve()
        if ROOT not in f.parents:
            bad.append(f"{mod} resolved OUTSIDE the package: {f}")
    leaks = [p for p in sys.path if "live_translation_final" in p]
    if leaks:
        bad.append(f"sys.path points at the research repository: {leaks}")
    import os

    hf = Path(os.environ.get("HF_HOME", "")).resolve()
    if ROOT not in hf.parents and not os.environ.get("MAVIS_HF_HOME"):
        bad.append(f"HF_HOME is not package-local: {hf}")
    from product.session import feature_enabled
    from product.live_translate import DEFAULT_CONFIG
    from utils.config import load_config

    if feature_enabled():
        bad.append("host integration flag product.integration.live_translation_enabled is ON (must stay off)")
    out_root = (ROOT / load_config(DEFAULT_CONFIG).get("product")["output_root"]).resolve()
    if ROOT not in out_root.parents:
        bad.append(f"output_root outside the package: {out_root}")
    print(f"HF_HOME = {hf}")
    print(f"session output root = {out_root}")
    for b in bad:
        print("PROBLEM: " + b)
    print("environment OK" if not bad else f"{len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
