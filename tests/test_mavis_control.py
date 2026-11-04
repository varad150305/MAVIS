"""Gesture-controlled module switching, and the live translation bridge (with a stand-in translation package)."""
import asyncio
import os
import sys
import textwrap
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

SUITE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUITE))

from hub.mavis_controller import READY_SECONDS, ModuleController  # noqa: E402


def run_flow(steps):
    c, out, t = ModuleController(), [], 100.0
    for fired, dt in steps:
        t += dt
        out += [e["event"] + (":" + e["module"]["id"] if "module" in e else "") for e in c.update(fired, t)]
    return c, out


def test_thumbs_up_then_fingers_switch_modules_and_open_palm_turns_off():
    c, out = run_flow([("thumbs_up", 0), ("one_finger", 1), ("open_palm", 5),
                       ("thumbs_up", 1), ("two_fingers", 1), ("open_palm", 5),
                       ("thumbs_up", 1), ("three_fingers", 1), ("open_palm", 5)])
    assert out == ["ready", "module_on:ocr", "module_off:ocr", "ready", "module_on:translation", "module_off:translation",
                   "ready", "module_on:drishti", "module_off:drishti"]
    assert c.state == "idle"


def test_fingers_without_ready_do_nothing_and_ready_times_out():
    c, out = run_flow([("one_finger", 0), ("two_fingers", 1), ("thumbs_up", 1), (None, READY_SECONDS + 0.1),
                       ("one_finger", 0.1)])
    assert out == ["ready", "ready_timeout"]
    assert c.state == "idle"


def test_while_active_only_open_palm_acts():
    c, out = run_flow([("thumbs_up", 0), ("three_fingers", 1), ("one_finger", 1), ("thumbs_up", 1), ("two_fingers", 1)])
    assert out == ["ready", "module_on:drishti"] and c.state == "active" and c.module["id"] == "drishti"
    c2, out2 = run_flow([("thumbs_up", 0), ("open_palm", 1)])
    assert out2 == ["ready", "ready_cancelled"] and c2.state == "idle"


# ---------------------------------------------------------------- translation bridge with a stand-in package

FAKE_SESSION = textwrap.dedent('''
    """STAND-IN for Yogita's product.session (test only): same API and event contract, no models."""
    import threading, time, uuid
    from enum import Enum
    from dataclasses import dataclass, field
    class SessionState(str, Enum):
        IDLE="idle"; STARTING="starting"; READY="ready"; STOPPING="stopping"; STOPPED="stopped"; FAILED="failed"
    class FeatureDisabled(RuntimeError): pass
    @dataclass
    class SessionOptions:
        config: str = None
        session_name: str = None
        overrides: list = field(default_factory=list)
    def feature_enabled(path):
        import yaml
        return bool(yaml.safe_load(open(path, encoding="utf-8"))["product"]["integration"]["live_translation_enabled"])
    def create_host_session(options, on_event=None):
        if not feature_enabled(options.config):
            raise FeatureDisabled("flag off")
        return S(on_event)
    class S:
        def __init__(self, cb):
            self.cb, self.session_id, self.seq, self.state = cb, uuid.uuid4().hex[:12], 0, SessionState.IDLE
        def pub(self, t, **kw):
            self.seq += 1
            self.cb(dict(schema="mavis.live_translation.v1", session_id=self.session_id, seq=self.seq, type=t, **kw))
        def start(self):
            self.state = SessionState.STARTING
            def run():
                self.pub("session_state", state="starting")
                self.state = SessionState.READY
                self.pub("session_state", state="ready", ready_in_s=0.1)
                self.pub("en_committed", segment_id=1, utterance_id=1, new_text="good morning", stable_text="good morning")
                self.pub("hi_draft", segment_id=1, target_text="सुबह अच्छी")
                self.pub("segment_closed", segment_id=1, reason="pause")
                self.pub("hi_final", segment_id=1, target_text="सुप्रभात", kind="final_mt")
            threading.Thread(target=run, daemon=True).start()
        def stop(self, timeout_s=120):
            self.pub("session_state", state="stopping")
            self.state = SessionState.STOPPED
            self.pub("session_state", state="stopped", stop_reason="host")
            return True
''')


def make_fake_package(root: Path) -> Path:
    pkg = root / "translation"
    (pkg / "product" / "configs").mkdir(parents=True)
    (pkg / "product" / "__init__.py").write_text("")
    (pkg / "product" / "session.py").write_text(FAKE_SESSION, encoding="utf-8")
    (pkg / "product" / "live_translate.py").write_text(
        "from pathlib import Path\nDEFAULT_CONFIG = Path(__file__).parent / 'configs' / 'live_m2_mt3c.yaml'\n")
    (pkg / "product" / "configs" / "live_m2_mt3c.yaml").write_text(
        "product:\n  integration:\n    live_translation_enabled: false\n")
    (pkg / "models" / "nllb-200-distilled-600M-ct2-int8").mkdir(parents=True)
    (pkg / "models" / "nllb-200-distilled-600M-ct2-int8" / "model.bin").write_bytes(b"x")
    (pkg / "models" / "huggingface" / "hub").mkdir(parents=True)
    (pkg / ".venv" / "bin").mkdir(parents=True)
    os.symlink(sys.executable, pkg / ".venv" / "bin" / "python")
    return pkg


def test_translation_events_reach_the_page_and_her_config_is_untouched(tmp_path, monkeypatch):
    pkg = make_fake_package(tmp_path)
    original_cfg = (pkg / "product" / "configs" / "live_m2_mt3c.yaml").read_text()
    monkeypatch.setenv("MAVIS_TRANSLATION_DIR", str(pkg))
    from hub import paths, translation_service
    monkeypatch.setattr(translation_service, "DATA", tmp_path / "data")
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")

    async def main():
        app = web.Application()
        translation_service.add_routes(app)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            ws = await client.ws_connect("/hub/translation/ws")
            hello = await ws.receive_json()
            assert hello["status"]["ok"] is True, hello
            await ws.send_json({"cmd": "start"})
            got = []
            for _ in range(30):
                e = await asyncio.wait_for(ws.receive_json(), 20)
                got.append(e)
                if e.get("type") == "hi_final":
                    break
            await ws.send_json({"cmd": "stop"})
            for _ in range(10):
                e = await asyncio.wait_for(ws.receive_json(), 20)
                got.append(e)
                if e.get("type") == "session_state" and e.get("state") == "stopped":
                    break
            await ws.close()
            return got
        finally:
            await app["translation"].shutdown()
            await client.close()

    got = asyncio.run(main())
    types = [(e["type"], e.get("state")) for e in got if e["type"] != "bridge"]
    assert ("session_state", "ready") in types
    assert [e["target_text"] for e in got if e["type"] == "hi_final"] == ["सुप्रभात"]
    assert types[-1] == ("session_state", "stopped")
    assert (tmp_path / "data" / "translation_host_config.yaml").read_text(encoding="utf-8").count("live_translation_enabled: true") == 1
    assert (pkg / "product" / "configs" / "live_m2_mt3c.yaml").read_text() == original_cfg     # her file unchanged


def test_status_explains_what_is_missing(tmp_path, monkeypatch):
    from hub import translation_service
    monkeypatch.setenv("MAVIS_TRANSLATION_DIR", str(tmp_path / "nothing"))
    assert translation_service.TranslationManager().status()["ok"] is False
    pkg = tmp_path / "pkg"
    (pkg / ".venv" / "bin").mkdir(parents=True)
    os.symlink(sys.executable, pkg / ".venv" / "bin" / "python")
    monkeypatch.setenv("MAVIS_TRANSLATION_DIR", str(pkg))
    st = translation_service.TranslationManager().status()
    assert st["ok"] is False and "models" in st["message"].lower()
