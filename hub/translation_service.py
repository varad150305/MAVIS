"""Live translation mode: Yogita's English -> Hindi speech translation module, run as she ships it.

Her package needs its own Python 3.10 environment (numpy 2, a newer OpenCV, CPU torch) that cannot share MAVIS's
environment, so the hub never imports it. Instead it starts hub/translation_bridge.py with *her* interpreter
(<package>/.venv/Scripts/python.exe) in her package folder; the bridge uses her documented host API and prints her
`mavis.live_translation.v1` events as JSON lines, which the hub forwards to the browser page /translate/.

The microphone is the default input device of the computer running MAVIS (her worker opens it, as designed).
Where her package lives: MAVIS_TRANSLATION_DIR, else modules/translation.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from aiohttp import WSMsgType, web

from .paths import DATA, MODULES

BRIDGE = Path(__file__).with_name("translation_bridge.py")


def package_dir() -> Path:
    return Path(os.environ.get("MAVIS_TRANSLATION_DIR") or (MODULES / "translation"))


def venv_python(pkg: Path) -> Path | None:
    for p in (pkg / ".venv" / "Scripts" / "python.exe", pkg / ".venv" / "bin" / "python"):
        if p.exists():
            return p
    return None


def models_present(pkg: Path) -> bool:
    return (pkg / "models" / "nllb-200-distilled-600M-ct2-int8" / "model.bin").exists() and \
        (pkg / "models" / "huggingface" / "hub").exists()


class TranslationManager:
    def __init__(self):
        self.proc: asyncio.subprocess.Process | None = None
        self.clients: set[web.WebSocketResponse] = set()
        self.reader: asyncio.Task | None = None
        self.last_state = "off"
        self.history: list[dict] = []        # events of the current session, replayed to late joiners

    def status(self) -> dict:
        pkg = package_dir()
        py = venv_python(pkg)
        ok = pkg.exists() and py is not None and models_present(pkg)
        if not pkg.exists():
            msg = "Translation module not found (modules/translation)."
        elif py is None:
            msg = "Run the translation module's setup (its own Python 3.10 environment)."
        elif not models_present(pkg):
            msg = "Translation models missing (models/ folder)."
        else:
            msg = "Ready"
        return {"ok": ok, "message": msg, "state": self.last_state, "folder": str(pkg)}

    async def _ensure_bridge(self):
        if self.proc and self.proc.returncode is None:
            return
        pkg = package_dir()
        py = venv_python(pkg)
        if py is None:
            raise RuntimeError(self.status()["message"])
        flags = 0x08000000 if sys.platform == "win32" else 0      # CREATE_NO_WINDOW
        self.proc = await asyncio.create_subprocess_exec(
            str(py), "-u", str(BRIDGE), str(DATA), cwd=str(pkg),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, creationflags=flags, limit=2**22)
        self.reader = asyncio.create_task(self._read())

    async def _read(self):
        assert self.proc and self.proc.stdout
        async for raw in self.proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except ValueError:
                evt = {"type": "log", "text": line[:500]}
            if evt.get("type") == "session_state":
                self.last_state = evt.get("state", self.last_state)
                if evt.get("state") == "starting":
                    self.history = []
            if evt.get("type") not in ("log", "asr_decode", "resources", "utterance_timing"):
                self.history.append(evt)
                self.history = self.history[-400:]
            await self._broadcast(evt)
        await self._broadcast({"type": "bridge", "state": "exited"})
        self.last_state = "off"

    async def _broadcast(self, evt):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(evt)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def command(self, cmd: str):
        if cmd == "start":
            await self._ensure_bridge()
        if self.proc and self.proc.returncode is None and self.proc.stdin:
            self.proc.stdin.write((cmd + "\n").encode())
            await self.proc.stdin.drain()

    async def shutdown(self):
        if self.proc and self.proc.returncode is None:
            try:
                await self.command("quit")
                await asyncio.wait_for(self.proc.wait(), 40)
            except Exception:  # noqa: BLE001
                self.proc.kill()


async def translation_ws(request: web.Request):
    mgr: TranslationManager = request.app["translation"]
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    mgr.clients.add(ws)
    await ws.send_json({"type": "hello", "status": mgr.status(), "history": mgr.history})
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    cmd = json.loads(msg.data).get("cmd")
                except ValueError:
                    continue
                if cmd in ("start", "stop"):
                    try:
                        await mgr.command(cmd)
                    except Exception as exc:  # noqa: BLE001
                        await ws.send_json({"type": "bridge", "state": "error", "error": str(exc)})
            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break
    finally:
        mgr.clients.discard(ws)
        if not mgr.clients and mgr.last_state in ("starting", "ready"):
            await mgr.command("stop")          # page closed / module turned off: stop the session
    return ws


def add_routes(app: web.Application):
    app["translation"] = TranslationManager()
    app.router.add_get("/hub/translation/ws", translation_ws)

    async def _cleanup(app):
        await app["translation"].shutdown()
    app.on_cleanup.append(_cleanup)
