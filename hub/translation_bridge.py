"""Runs INSIDE Yogita's translation package environment (her Python 3.10 .venv, cwd = her package folder).

MAVIS's hub (Python 3.11, different numpy/OpenCV) starts this script with her interpreter, so her module runs exactly as
she ships it. It uses only her documented host API (INTEGRATION.md):

    create_host_session(SessionOptions(config=...), on_event=...)  ->  session.start() / session.stop()

The feature flag product.integration.live_translation_enabled is OFF in her config. As INTEGRATION.md says, the host
enables it "in a copy of the config passed as SessionOptions(config=...)": this script writes that copy to MAVIS's own
data folder and never edits her file.

Protocol with the hub: one JSON event per line on stdout; commands on stdin ("start", "stop", "quit").
"""
import json
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, os.getcwd())

import yaml  # noqa: E402  (her environment)

from product.live_translate import DEFAULT_CONFIG  # noqa: E402
from product.session import SessionOptions, create_host_session  # noqa: E402

OUT_LOCK = threading.Lock()


def emit(obj):
    with OUT_LOCK:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def host_config(data_dir: Path) -> str:
    cfg = yaml.safe_load(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))
    cfg.setdefault("product", {}).setdefault("integration", {})["live_translation_enabled"] = True
    out = data_dir / "translation_host_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return str(out)


def pick_microphone():
    """Which microphone her worker opens (passed as SessionOptions(device=...)).

    MAVIS_TRANSLATION_MIC set (e.g. "DroidCam"): the first input whose name contains it and accepts her format
    (16 kHz mono) - Windows lists each microphone once per audio API, so a bare name would be ambiguous.
    Not set: the laptop's microphone, i.e. the system default - unless that default is the phone's DroidCam
    microphone, in which case the first other microphone (MAVIS keeps the phone for the OCR module)."""
    try:
        import sounddevice as sd
    except ImportError:
        return None, "system default microphone"

    def usable(idx):
        try:
            sd.check_input_settings(device=idx, samplerate=16000, channels=1, dtype="float32")
            return True
        except Exception:  # noqa: BLE001
            return False

    devices = list(enumerate(sd.query_devices()))
    want = (os.environ.get("MAVIS_TRANSLATION_MIC") or "").strip()
    if want:
        for idx, info in devices:
            if info["max_input_channels"] > 0 and want.lower() in info["name"].lower() and usable(idx):
                return idx, info["name"]
        return None, f"no input device matching {want!r}; using the system default microphone"
    try:
        default = sd.query_devices(kind="input")["name"]
    except Exception:  # noqa: BLE001
        return None, "system default microphone"
    if "droidcam" not in default.lower():
        return None, default
    for idx, info in devices:
        if info["max_input_channels"] > 0 and "droidcam" not in info["name"].lower() and usable(idx):
            return idx, info["name"]
    return None, default


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "runtime_data"
    config = host_config(data_dir)
    session = None
    emit({"type": "bridge", "state": "ready"})
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "start":
            if session is not None and session.state.value not in ("stopped", "failed", "idle"):
                emit({"type": "bridge", "state": "busy"})
                continue
            try:
                device, mic = pick_microphone()
                emit({"type": "bridge", "state": "microphone", "device": device, "name": mic})
                session = create_host_session(
                    SessionOptions(config=config, session_name="mavis", device=device), on_event=lambda e: emit(e))
                session.start()
                emit({"type": "bridge", "state": "started", "session_id": session.session_id})
            except Exception as exc:  # noqa: BLE001
                emit({"type": "bridge", "state": "error", "error": f"{type(exc).__name__}: {exc}"})
        elif cmd == "stop":
            if session is not None:
                threading.Thread(target=lambda s=session: (s.stop(), emit({"type": "bridge", "state": "stopped"})),
                                 daemon=True).start()
        elif cmd == "quit":
            break
    if session is not None:
        try:
            session.stop(timeout_s=30)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
