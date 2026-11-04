"""Self-contained smoke tests of the handoff package (no research data needed).

    .venv\\Scripts\\python.exe -m pytest tests_smoke -q            # fast checks (~20 s)
    .venv\\Scripts\\python.exe -m pytest tests_smoke -q -m models  # + end-to-end replay with the real models (~1-2 min)

Everything they write goes under <package>/runtime_data/.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROV = json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
RUNTIME = ROOT / "runtime_data"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_project_modules_resolve_inside_the_package():
    r = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "check_environment.py")], cwd=ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "environment OK" in r.stdout


def test_copied_files_match_their_provenance():
    """Frozen engine files are byte-identical to the research runtime; product
    files match what was packaged (verbatim or the documented transform)."""

    for e in PROV["files"]:
        assert _sha(ROOT / e["dest_path"]) == e["dest_sha256"], e["dest_path"]
        if e["group"] == "frozen_engine":
            assert e["copy"] == "verbatim" and e["dest_sha256"] == e["source_sha256"], e["dest_path"]


#: files allowed to contain the strings, and why
ALLOWED = {
    "SOURCE_PROVENANCE.json": "documents where the files came from",
    "evaluation/mt3c_parameters.json": "frozen, SHA-pinned parameters; its 'python_executable' field is provenance "
                                       "metadata of the research run and is never read by any code",
    "scripts/check_environment.py": "contains the search string it checks sys.path for",
}


def test_no_runtime_reference_to_the_research_repository_or_user_paths():
    offenders = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in (".py", ".yaml", ".ps1", ".txt", ".ini", ".json"):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith((".venv/", "_handoff_audit/", "models/", "runtime_data/", "tests_smoke/")) or rel in ALLOWED:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "live_translation_final" in text or "Yogita" in text or "C:\\Users\\" in text:
            offenders.append(rel)
    assert offenders == []


def test_configuration_is_package_local_and_the_host_flag_is_off():
    sys.path.insert(0, str(ROOT))
    from product.live_translate import DEFAULT_CONFIG
    from product.session import feature_enabled
    from utils.config import load_config

    cfg = load_config(DEFAULT_CONFIG)
    p = cfg.get("product")
    assert feature_enabled() is False and p["integration"]["live_translation_enabled"] is False
    assert (ROOT / p["output_root"]).resolve().is_relative_to(ROOT)
    assert (ROOT / p["final_mt"]["model_dir"]).resolve().is_relative_to(ROOT / "models")
    assert cfg.get("asr", "model_size") == "tiny" and cfg.get("streaming_buffer", "min_new_audio_ms") == 2000
    assert p["policy"] == "mt3c" and _sha(ROOT / p["mt3c_parameters"]) == p["mt3c_parameters_sha256"]


def test_ui_opens_without_starting_a_session_or_capture():
    code = textwrap.dedent("""
        import sys, tkinter as tk
        from product.ui.app import TranslatorApp
        root = tk.Tk(); root.withdraw()
        app = TranslatorApp(root); root.update()
        assert app.session is None
        assert app.state_lbl.cget("text") == "IDLE"
        assert "sounddevice" not in sys.modules and "torch" not in sys.modules
        root.destroy(); print("UI-OK")
    """)
    r = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    if "TclError" in r.stderr and "display" in r.stderr:
        pytest.skip("no display")
    assert r.returncode == 0 and "UI-OK" in r.stdout, r.stderr[-2000:]


FAKE = textwrap.dedent('''
    import os, sys
    def run_session(argv, console, *, event_sink, stop_event, ready_callback):
        import product
        console.write("Session folder: fake\\n")
        ready_callback(0.1, {"on_ac_power": None})
        assert stop_event.wait(30)
        return 0, {"stop_reason": stop_event.reason, "product_file": product.__file__, "cwd": os.getcwd(),
                   "hf_home": os.environ.get("HF_HOME"), "sys_path": sys.path[:3], "argv": argv}
''')


def test_session_worker_runs_from_the_package_even_when_the_host_runs_elsewhere(tmp_path, monkeypatch):
    """The real worker process (product.session_worker) imports `product` from
    this package and runs with the package as cwd, whatever the host's cwd."""

    sys.path.insert(0, str(ROOT))
    from product import session as S

    (tmp_path / "fake_rs.py").write_text(FAKE, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("MAVIS_SESSION_WORKER_RUNNER", "fake_rs:run_session")
    monkeypatch.chdir(tmp_path)                                        # host somewhere else
    s = S.TranslationSession(S.SessionOptions(wav="relative.wav"))
    s.start()
    t0 = time.time()
    while s.state is not S.SessionState.READY and time.time() - t0 < 60:
        time.sleep(0.05)
    assert s.stop(timeout_s=30) and s.state is S.SessionState.STOPPED, s.error
    assert Path(s.summary["product_file"]).resolve().is_relative_to(ROOT)
    assert Path(s.summary["cwd"]).resolve() == ROOT
    assert Path(s.summary["hf_home"]).resolve().is_relative_to(ROOT / "models")
    wav = s.summary["argv"][s.summary["argv"].index("--wav") + 1]
    assert Path(wav) == tmp_path / "relative.wav"                     # host-relative path kept its meaning


@pytest.mark.models
def test_end_to_end_replay_through_the_session_api():
    """Real models, real worker: a Windows-TTS reading of the first demo
    sentences, replayed at 1x. Output goes to runtime_data/smoke_sessions/."""

    sys.path.insert(0, str(ROOT))
    from product import session as S
    from product.ui_model import TranscriptModel

    wav = RUNTIME / "demo" / "smoke_tts.wav"
    if not wav.exists():
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                            str(ROOT / "scripts" / "make_demo_wav.ps1"), "-Out", str(wav), "-Lines", "4"],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            pytest.skip(f"could not synthesize demo audio: {r.stderr[-300:]}")
    out = RUNTIME / "smoke_sessions"
    s = S.TranslationSession(S.SessionOptions(wav=str(wav), overrides=[f"product.output_root={out.as_posix()}"]))
    s.start()
    events = list(s.iter_events(timeout_s=0.5))
    assert s.state is S.SessionState.STOPPED, s.error
    m = TranscriptModel(s.session_id)
    for e in events:
        m.apply(e)
    assert m.problems == [] and m.errors == []
    closed = [g for g in m.ordered_segments() if g.closed]
    assert closed and all(g.hindi_final for g in closed)
    english = " ".join(" ".join(g.english) for g in closed)
    assert "morning" in english and "technology" in english, english
    folder = ROOT / s.summary["session_dir"] if not Path(s.summary["session_dir"]).is_absolute() \
        else Path(s.summary["session_dir"])
    assert folder.resolve().is_relative_to(ROOT)
    assert {"session_start.json", "lifecycle.jsonl", "session_summary.json", "transcript.jsonl"} <= \
        {p.name for p in folder.iterdir()}
    assert s.summary["session_id"] == s.session_id and s.summary["cleanup"]["ok"] is True
