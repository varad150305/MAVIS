"""aiohttp server: static UI, JSON API, server-sent events and the live MJPEG feed. Local use only (127.0.0.1)."""
import asyncio
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "prototype", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from webapp.core import Busy, Refused, WebApp  # noqa: E402
from webapp.demo import DemoStream  # noqa: E402
from webapp.live_stream import LiveStream  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
SELECTORS = ("b0", "c3", "interval")     # b0 = smart default, c3 = experimental, interval = one page every N seconds
EVENT_TICK_S = 0.5


def _json(data, status=200):
    return web.json_response(data, status=status, dumps=lambda d: json.dumps(d, ensure_ascii=False))


def _error(message, status=400, code="error"):
    return _json({"error": {"code": code, "message": message}}, status=status)


@web.middleware
async def local_only(request, handler):
    """The server shows a camera feed and document text: accept only requests addressed to this machine."""
    host = request.headers.get("Host", "").rsplit(":", 1)[0].strip("[]").lower()
    if host not in ("127.0.0.1", "localhost", "::1"):
        return web.Response(status=403, text="MAVIS only answers on localhost.")
    if request.method not in ("GET", "HEAD"):
        origin = request.headers.get("Origin")
        if origin and re.match(r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$", origin) is None:
            return web.Response(status=403, text="Cross-site request refused.")
    response = await handler(request)
    response.headers.setdefault("Cache-Control", "no-store")
    return response


@web.middleware
async def refusals(request, handler):
    try:
        return await handler(request)
    except Busy as exc:
        return _error(str(exc), 409, "busy")
    except Refused as exc:
        return _error(str(exc), 409, "refused")


async def read_body(request):
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def make_app(state=None):
    state = state or WebApp()
    pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="web-io")
    app = web.Application(middlewares=[local_only, refusals])
    app["state"] = state
    routes = web.RouteTableDef()

    async def run(fn, *args):
        return await asyncio.get_running_loop().run_in_executor(pool, fn, *args)

    @routes.get("/")
    async def index(_request):
        return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @routes.get("/api/state")
    async def get_state(_request):
        return _json(state.snapshot())

    @routes.get("/api/events")
    async def events(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-store",
                                           "X-Accel-Buffering": "no"})
        await resp.prepare(request)
        last, last_write = None, 0.0
        try:
            while True:
                snap = state.snapshot()
                body = {k: v for k, v in snap.items() if k != "now"}
                encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
                now = time.time()
                if encoded != last:
                    payload = json.dumps({**snap}, ensure_ascii=False, separators=(",", ":"))
                    await resp.write(f"data: {payload}\n\n".encode("utf-8"))
                    last, last_write = encoded, now
                elif now - last_write > 4:
                    await resp.write(b": keepalive\n\n")
                    last_write = now
                await asyncio.sleep(EVENT_TICK_S)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    # ---------------------------------------------------------------- scanning
    @routes.post("/api/scan/start")
    async def scan_start(request):
        body = await read_body(request)
        kind = body.get("kind")
        selector = body.get("selector", "b0")
        if selector not in SELECTORS:
            return _error("Unknown capture mode.")
        if kind == "demo":
            stream, source = DemoStream(), {"kind": "demo", "label": "Simulated camera"}
        elif kind == "droidcam":
            ip, port = str(body.get("ip", "")).strip(), str(body.get("port", "")).strip()
            if not ip:
                return _error("Enter the DroidCam IP address.")
            if not port.isdigit() or not 1 <= int(port) <= 65535:
                return _error("The port must be a number from 1 to 65535.")
            if not re.fullmatch(r"[A-Za-z0-9.\-]+", ip):
                return _error("That does not look like an IP address or host name.")
            stream, source = LiveStream(f"http://{ip}:{port}/video"), {"kind": "droidcam", "label": f"{ip}:{port}"}
        else:
            return _error("Choose a camera source.")
        try:
            interval = float(body.get("interval_s", 5))
        except (TypeError, ValueError):
            return _error("The capture interval must be a number of seconds.")
        if selector == "interval" and not 2 <= interval <= 30:
            return _error("Choose a capture interval between 2 and 30 seconds.")
        await run(state.start_scan, stream, source, selector, interval)
        return _json({"ok": True})

    @routes.post("/api/scan/stop")
    async def scan_stop(_request):
        state.stop_scan()
        return _json({"ok": True})

    @routes.post("/api/scan/cancel")
    async def scan_cancel(_request):
        state.cancel_scan()
        return _json({"ok": True})

    @routes.post("/api/scan/dismiss")
    async def scan_dismiss(_request):
        state.dismiss_scan()
        return _json({"ok": True})

    @routes.get("/api/scan/video")
    async def scan_video(request):
        scan = state.scan
        if scan is None:
            return web.Response(status=404)
        resp = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                                           "Cache-Control": "no-store"})
        await resp.prepare(request)
        seq = -1
        try:
            while scan.phase in ("connecting", "scanning"):
                seq, jpeg = await run(scan.video_jpeg, seq)
                if jpeg is None:
                    await asyncio.sleep(0.03)
                    continue
                await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                scan.note_video_frame()
                await asyncio.sleep(0.02)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    @routes.get("/api/scan/pages/{n}/image")
    async def scan_page_image(request):
        scan = state.scan
        # the URL carries the scan id (?s=...) so a browser can never show a photo cached from an earlier scan
        if scan is None or request.query.get("s") not in (None, scan.id):
            return web.Response(status=404)
        jpeg = scan.page_image(int(request.match_info["n"])) if request.match_info["n"].isdigit() else None
        if not jpeg:
            return web.Response(status=404)
        return web.Response(body=jpeg, content_type="image/jpeg")

    # ---------------------------------------------------------------- documents
    @routes.get("/api/documents")
    async def documents(_request):
        return _json({"documents": await run(state.library.list)})

    @routes.get("/api/documents/{doc_id}")
    async def document(request):
        doc = state.library.get(request.match_info["doc_id"])
        if doc is None:
            return _error("Document not found.", 404, "not_found")
        return _json({**doc.detail(), "prepare": state.prepare_state(doc.id)})

    @routes.patch("/api/documents/{doc_id}")
    async def rename(request):
        doc = state.library.get(request.match_info["doc_id"])
        if doc is None:
            return _error("Document not found.", 404, "not_found")
        body = await read_body(request)
        doc.rename(str(body.get("title", "")))
        state._docs_version += 1
        return _json({"ok": True, "title": doc.title})

    @routes.delete("/api/documents/{doc_id}")
    async def delete_document(request):
        await run(state.delete_document, request.match_info["doc_id"])
        return _json({"ok": True})

    @routes.post("/api/documents/{doc_id}/pages/{n}/reread")
    async def reread_page(request):
        if not request.match_info["n"].isdigit():
            return _error("Unknown page.", 404, "not_found")
        result = await run(state.reread_page, request.match_info["doc_id"], int(request.match_info["n"]))
        return _json(result)

    @routes.get("/api/documents/{doc_id}/pages/{n}/image")
    async def document_page_image(request):
        doc = state.library.get(request.match_info["doc_id"])
        path = doc.page_image_path(request.match_info["n"]) if doc is not None and request.match_info["n"].isdigit() else None
        if path is None:
            return web.Response(status=404)
        return web.FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})

    @routes.get("/api/documents/{doc_id}/export/{kind}")
    async def document_export(request):
        doc = state.library.get(request.match_info["doc_id"])
        kind = request.match_info["kind"]
        if doc is None:
            return _error("Document not found.", 404, "not_found")
        if kind == "text":
            return web.Response(text=doc.text_export(), content_type="text/plain", charset="utf-8",
                                headers={"Content-Disposition": f'attachment; filename="{doc.id}.txt"'})
        path = doc.export_path(kind)
        if path is None or not path.exists():
            return _error("That export does not exist.", 404, "not_found")
        return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="{doc.id}_{path.name}"'})

    @routes.post("/api/documents/{doc_id}/prepare")
    async def prepare(request):
        status = await run(state.prepare, request.match_info["doc_id"], True)
        return _json({"prepare": status})

    @routes.post("/api/documents/{doc_id}/ask")
    async def ask(request):
        body = await read_body(request)
        turn = await run(state.ask, request.match_info["doc_id"], body.get("mode", "answer"), body.get("question", ""))
        return _json({"turn": turn})

    @routes.post("/api/reader/check")
    async def reader_check(_request):
        return _json(await run(state.check_reader))

    @routes.post("/api/llm/check")
    async def llm_check(_request):
        return _json(await run(state.check_llm))

    app.add_routes(routes)
    app.router.add_static("/static/", STATIC, show_index=False)
    return app


def serve(host="127.0.0.1", port=8765, state=None):
    web.run_app(make_app(state), host=host, port=port, print=lambda *_: None, access_log=None)
