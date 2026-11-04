"""MAVIS OCR mode: the original MAVIS web app, mounted inside the hub at /mavis/.

What the hub adds around MAVIS (none of it edits MAVIS's files):

1. Mounting. MAVIS's own aiohttp application (webapp.server.make_app) is mounted unchanged under /mavis/. Its page and
   script use root paths (/api/..., /static/...); the hub serves rewritten copies of index.html and app.js that point to
   /mavis/api/... and /mavis/static/... instead. The files on disk are not touched.

2. Phone access. MAVIS refuses any request not addressed to 127.0.0.1 (its `local_only` guard). The hub has already
   checked the access key (see security.py) and the request's origin, so for /mavis/ requests it presents the request to
   MAVIS as addressed to this computer.

3. "Use this phone's camera". MAVIS reads pages from a DroidCam MJPEG stream (http://<ip>:<port>/video). The hub can be
   that stream: the phone's browser sends its camera frames over a WebSocket, and a small loopback-only server
   (127.0.0.1:RELAY_PORT/video) serves them as MJPEG, exactly the format DroidCam serves. MAVIS's own LiveStream reads it
   like any DroidCam, so the scanning, OCR and question-answering code is the original code path. No DroidCam app needed.
"""
from __future__ import annotations

import asyncio
import re
import time

from aiohttp import WSMsgType, web

from .paths import MAVIS, add_import_paths

add_import_paths()

from webapp.core import Busy, Refused  # noqa: E402  (MAVIS)
from webapp.live_stream import LiveStream  # noqa: E402
from webapp.server import SELECTORS, make_app  # noqa: E402

MAVIS_STATIC = MAVIS / "webapp" / "static"
PREFIX = "/mavis"
_PATHS = re.compile(r"""(["'`(])/(api|static)/""")


def rewrite(text: str) -> str:
    return _PATHS.sub(lambda m: f"{m.group(1)}{PREFIX}/{m.group(2)}/", text)


# ---------------------------------------------------------------- camera relay (phone -> MJPEG "DroidCam")

class CameraRelay:
    def __init__(self):
        self.jpeg: bytes | None = None
        self.seq = 0
        self.updated = 0.0
        self.publishers = 0
        self._cond = asyncio.Condition()

    async def publish(self, jpeg: bytes):
        async with self._cond:
            self.jpeg, self.seq, self.updated = jpeg, self.seq + 1, time.time()
            self._cond.notify_all()

    async def next(self, after: int, timeout=5.0):
        async with self._cond:
            try:
                await asyncio.wait_for(self._cond.wait_for(lambda: self.seq > after), timeout)
            except asyncio.TimeoutError:
                return after, None
            return self.seq, self.jpeg

    def live(self) -> bool:
        return self.publishers > 0 and time.time() - self.updated < 3.0


async def relay_video(request: web.Request):
    """MJPEG in DroidCam's format, for MAVIS's cv2.VideoCapture. Served on 127.0.0.1 only."""
    relay: CameraRelay = request.app["relay"]
    resp = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                                       "Cache-Control": "no-store", "Connection": "close"})
    await resp.prepare(request)
    seq = 0
    try:
        while True:
            seq, jpeg = await relay.next(seq)
            if jpeg is None:
                if not relay.live():
                    break                     # the phone stopped sending: end the stream (MAVIS sees the camera drop)
                continue
            await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                             + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return resp


def make_relay_app(relay: CameraRelay) -> web.Application:
    app = web.Application()
    app["relay"] = relay  # noqa
    app.router.add_get("/video", relay_video)
    return app


# ---------------------------------------------------------------- mounting

def mount(app: web.Application, relay: CameraRelay, relay_port: int, loopback_host: str):
    mavis = make_app()                       # MAVIS's own application, unchanged
    state = mavis["state"]
    app["mavis_state"] = state

    async def index(_request):
        html = rewrite((MAVIS_STATIC / "index.html").read_text(encoding="utf-8"))
        html = html.replace("</body>", '<link rel="stylesheet" href="/hub/static/addons.css">\n'
                                       '<script src="/hub/static/mavis-addon.js"></script>\n</body>')
        return web.Response(text=html, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def app_js(_request):
        js = rewrite((MAVIS_STATIC / "app.js").read_text(encoding="utf-8"))
        return web.Response(text=js, content_type="application/javascript", headers={"Cache-Control": "no-store"})

    async def to_slash(_request):
        raise web.HTTPFound(PREFIX + "/")

    async def camera_ws(request):
        """The phone's browser pushes JPEG frames here while it is used as MAVIS's camera."""
        ws = web.WebSocketResponse(max_msg_size=4_000_000, heartbeat=20)
        await ws.prepare(request)
        relay.publishers += 1
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY and msg.data[:2] == b"\xff\xd8":
                    await relay.publish(msg.data)
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            relay.publishers -= 1
        return ws

    async def start_phone_scan(request):
        """Starts a MAVIS scan reading from the relay. Same checks as MAVIS's own /api/scan/start."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        selector = body.get("selector", "b0")
        if selector not in SELECTORS:
            return web.json_response({"error": {"message": "Unknown capture mode."}}, status=400)
        try:
            interval = float(body.get("interval_s", 5))
        except (TypeError, ValueError):
            return web.json_response({"error": {"message": "The capture interval must be a number."}}, status=400)
        if selector == "interval" and not 2 <= interval <= 30:
            return web.json_response({"error": {"message": "Choose a capture interval between 2 and 30 seconds."}},
                                     status=400)
        if not relay.live():
            return web.json_response({"error": {"message": "No camera picture is arriving from this device yet."}},
                                     status=409)
        stream = LiveStream(f"http://127.0.0.1:{relay_port}/video")
        source = {"kind": "droidcam", "label": "\u00b7 this device's camera"}
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, state.start_scan, stream, source, selector, interval)
        except (Busy, Refused) as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=409)
        return web.json_response({"ok": True})

    # specific routes first: they win over the mounted sub-application
    app.router.add_get(PREFIX, to_slash)
    app.router.add_get(PREFIX + "/", index)
    app.router.add_get(PREFIX + "/static/app.js", app_js)
    app.router.add_get("/hub/camera/ws", camera_ws)
    app.router.add_post("/hub/mavis/phone-scan", start_phone_scan)
    app.add_subapp(PREFIX + "/", mavis)

    @web.middleware
    async def present_as_local(request, handler):
        if request.path.startswith(PREFIX + "/"):
            headers = request.headers.copy()
            headers["Host"] = loopback_host
            if "Origin" in headers:
                headers["Origin"] = "http://" + loopback_host
            request = request.clone(headers=headers)
        return await handler(request)

    return present_as_local


def status(state) -> dict:
    try:
        snap = state.snapshot()
    except Exception:  # noqa: BLE001
        return {}
    return {"reader": snap.get("reader"), "llm": snap.get("llm")}
