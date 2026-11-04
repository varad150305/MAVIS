"""MAVIS hub: one server, three modes.

  /            launcher (pick a mode)
  /drishti/    Drishti   - AI eyes: describe scenes, read text, money, colours, prices, translation ... (browser app)
  /mavis/      MAVIS OCR - scan pages, read them with OCR, then ask questions about the document
  /gesture/    Gesture   - choose a feature with hand gestures; output shown on the device you are holding

Listens on:
  http://127.0.0.1:<port>          for this computer (a secure context, so the camera works)
  https://<this computer's IP>:<https_port>   for a phone / tablet on the same Wi-Fi (access key required)
"""
from __future__ import annotations

import asyncio
import io
import os
import socket
import sys

from aiohttp import web

from . import drishti_api, offline, security, translation_service
from .paths import DRISHTI, DRISHTI_WEB, HUB_STATIC

NO_STORE = {"Cache-Control": "no-store"}
DRISHTI_HEADERS = {"Permissions-Policy": "camera=*, microphone=*, geolocation=*",
                   "Cross-Origin-Opener-Policy": "same-origin-allow-popups"}


def page(request, name, status=200):
    """A hub page, with its font addresses pointed at the offline mirror (so pages look the same without internet)."""
    html = (HUB_STATIC / name).read_text(encoding="utf-8")
    return web.Response(text=offline.rewrite(html, offline.origin_of(request)), content_type="text/html",
                        status=status, headers=NO_STORE)


def redirect(to):
    async def handler(_request):
        raise web.HTTPFound(to)
    return handler


class Hub:
    def __init__(self, port=8000, https_port=8443, relay_port=8766, lan=True):
        self.port, self.https_port, self.relay_port, self.lan = port, https_port, relay_port, lan
        self.key = security.load_access_key()
        self.addresses = security.lan_addresses() if lan else []
        self.errors: dict[str, str] = {}
        self.notes: list[str] = []
        self.gesture_ok = False
        self.mavis_state = None

    # ------------------------------------------------------------ links
    def phone_links(self) -> list[str]:
        if not self.lan:
            return []
        return [f"https://{a}:{self.https_port}/?key={self.key}" for a in self.addresses]

    # ------------------------------------------------------------ app
    def build(self) -> web.Application:
        hub = self

        @web.middleware
        async def guard(request, handler):
            # 1. a link with ?key= : remember the key in a cookie, then continue without it in the address bar
            qkey = request.query.get("key")
            if qkey is not None and request.method == "GET":
                if qkey == hub.key:
                    q = request.rel_url.query.copy()
                    q.pop("key")
                    resp = web.HTTPFound(request.rel_url.with_query(q))
                    resp.set_cookie(security.COOKIE, hub.key, max_age=60 * 60 * 24 * 60, httponly=True,
                                    samesite="Lax", secure=request.secure)
                    raise resp
            # 2. no access: a friendly page instead of the app
            if not security.has_access(request, hub.key):
                if request.path.endswith("/ws") or request.path.startswith(("/api/", "/mavis/api/", "/hub/")):
                    return web.json_response({"error": {"message": "Open the link shown on the computer first."}},
                                             status=403)
                return page(request, "locked.html", status=403)
            # 3. state-changing requests must come from this hub's own pages
            if request.method not in ("GET", "HEAD", "OPTIONS") and not security.same_origin(request):
                return web.Response(status=403, text="Cross-site request refused.")
            return await handler(request)

        app = web.Application(middlewares=[guard], client_max_size=8 * 1024 ** 2)

        # ---- launcher and shared assets
        async def launcher(request):
            return page(request, "index.html")

        async def gesture_page(request):
            return page(request, "gesture.html")

        async def offline_page(request):
            return page(request, "offline.html")

        async def control_page(request):
            return page(request, "control.html")

        async def translate_page(request):
            return page(request, "translate.html")

        async def status(request):
            local = security.is_local(request)
            mavis = {}
            if hub.mavis_state is not None:
                from .mavis_bridge import status as mstatus
                mavis = mstatus(hub.mavis_state)
            return web.json_response({
                "modes": {
                    "drishti": {"ok": (DRISHTI_WEB / "index.html").exists(), **drishti_api.status()},
                    "mavis": {"ok": hub.mavis_state is not None, "error": hub.errors.get("mavis"), **mavis},
                    "gesture": {"ok": hub.gesture_ok, "error": hub.errors.get("gesture")},
                    "translation": request.app["translation"].status(),
                },
                "this_computer": local,
                "secure": request.secure or local,
                "phone_links": hub.phone_links() if local else [],
            }, headers=NO_STORE)

        async def qr(request):
            if not security.is_local(request):
                raise web.HTTPForbidden()
            links = hub.phone_links()
            n = int(request.query.get("i", "0") or 0)
            if not links or n >= len(links):
                raise web.HTTPNotFound()
            import segno
            buf = io.BytesIO()
            segno.make(links[n], error="m").save(buf, kind="svg", scale=6, border=2, dark="#1d1a15", light="#ffffff")
            return web.Response(body=buf.getvalue(), content_type="image/svg+xml", headers=NO_STORE)

        async def cert(_r):
            f = security.cert_path()
            if not f.exists():
                raise web.HTTPNotFound()
            return web.FileResponse(f, headers={"Content-Disposition": 'attachment; filename="mavis.crt"'})

        app.router.add_get("/", launcher)
        app.router.add_get("/gesture", redirect("/gesture/"))
        app.router.add_get("/gesture/", gesture_page)
        app.router.add_get("/control", redirect("/control/"))
        app.router.add_get("/control/", control_page)
        app.router.add_get("/translate", redirect("/translate/"))
        app.router.add_get("/translate/", translate_page)
        translation_service.add_routes(app)
        app.router.add_get("/offline", redirect("/offline/"))
        app.router.add_get("/offline/", offline_page)
        app.router.add_get("/hub/status", status)
        app.router.add_get("/hub/qr.svg", qr)
        app.router.add_get("/hub/cert.crt", cert)
        app.router.add_static("/hub/static/", HUB_STATIC, show_index=False)

        # ---- Drishti (static web app + its API routes)
        async def drishti_index(request):
            html = (DRISHTI_WEB / "index.html").read_text(encoding="utf-8")
            # Drishti's internet addresses (models, libraries, fonts) -> this computer's offline mirror
            html = offline.rewrite(html, offline.origin_of(request))
            html = html.replace("</body>", '<link rel="stylesheet" href="/hub/static/addons.css">\n'
                                           '<script src="/hub/static/drishti-addon.js"></script>\n</body>', 1)
            return web.Response(text=html, content_type="text/html", headers={**NO_STORE, **DRISHTI_HEADERS})

        app.router.add_get("/drishti", redirect("/drishti/"))
        app.router.add_get("/drishti/", drishti_index)
        app.router.add_get("/drishti/index.html", drishti_index)
        app.router.add_static("/drishti/about/", DRISHTI / "website", show_index=False)
        app.router.add_static("/drishti/files/", DRISHTI_WEB, show_index=False)
        drishti_api.add_routes(app)
        offline.add_routes(app, hub)

        # ---- Gesture
        try:
            from . import gesture_service
            gesture_service.add_routes(app)
            hub.gesture_ok = True
        except Exception as exc:  # noqa: BLE001
            hub.errors["gesture"] = f"{type(exc).__name__}: {exc}"
            print(f"[hub] gesture mode unavailable: {hub.errors['gesture']}", file=sys.stderr)

        # ---- MAVIS
        self.relay = None
        # No OCR key at all: let MAVIS read pages with its own local engine (its documented MAVIS_READER=local setting)
        # instead of refusing to scan. This is also what makes OCR mode usable with no internet and no keys.
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("OCRSPACE_API_KEY")) \
                and not os.environ.get("MAVIS_READER"):
            os.environ["MAVIS_READER"] = "local"
            self.notes.append("OCR mode: no OCR key in keys.env, so pages are read on this computer (local engine).")
        try:
            from . import mavis_bridge
            self.relay = mavis_bridge.CameraRelay()
            mw = mavis_bridge.mount(app, self.relay, self.relay_port, f"127.0.0.1:{self.port}")
            app.middlewares.append(mw)
            hub.mavis_state = app["mavis_state"]
        except Exception as exc:  # noqa: BLE001
            hub.errors["mavis"] = f"{type(exc).__name__}: {exc}"
            print(f"[hub] MAVIS mode unavailable: {hub.errors['mavis']}", file=sys.stderr)

        @web.middleware
        async def headers(request, handler):
            resp = await handler(request)
            resp.headers.setdefault("X-Content-Type-Options", "nosniff")
            resp.headers.setdefault("Referrer-Policy", "same-origin")
            if "Permissions-Policy" not in resp.headers:
                resp.headers["Permissions-Policy"] = "camera=(self), microphone=(self), geolocation=(self)"
            return resp

        app.middlewares.append(headers)
        return app

    # ------------------------------------------------------------ run
    async def start(self):
        app = self.build()
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        sites = [web.TCPSite(runner, "127.0.0.1", self.port)]
        ssl_ctx = None
        if self.lan:
            ssl_ctx = security.ssl_context(self.addresses)
            if ssl_ctx is None:
                print("[hub] 'cryptography' is not installed: phone access (HTTPS) is off.", file=sys.stderr)
                self.lan = False
            else:
                sites.append(web.TCPSite(runner, "0.0.0.0", self.https_port, ssl_context=ssl_ctx))
        for s in sites:
            await s.start()
        if self.relay is not None:
            from .mavis_bridge import make_relay_app
            rrunner = web.AppRunner(make_relay_app(self.relay), access_log=None)
            await rrunner.setup()
            await web.TCPSite(rrunner, "127.0.0.1", self.relay_port).start()
        return runner


def port_free(port, host="127.0.0.1") -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def serve(hub: Hub, open_browser=True):
    async def main():
        await hub.start()
        print("\n  MAVIS is running\n")
        print(f"  On this computer:  http://127.0.0.1:{hub.port}/")
        for link in hub.phone_links():
            print(f"  On your phone:     {link}")
        if hub.lan and hub.phone_links():
            print("                     (same Wi-Fi; or scan the QR code on the computer's launcher page)")
        print(f"  Offline readiness: http://127.0.0.1:{hub.port}/offline/")
        for note in hub.notes:
            print(f"\n  {note}")
        for name, err in hub.errors.items():
            print(f"  ! {name} mode is unavailable: {err}")
        print("\n  Ctrl+C to stop.\n")
        if open_browser:
            import webbrowser
            asyncio.get_running_loop().call_later(1.0, webbrowser.open, f"http://127.0.0.1:{hub.port}/")
        while True:
            await asyncio.sleep(3600)

    if sys.platform == "win32":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
