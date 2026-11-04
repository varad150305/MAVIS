"""Offline support: a local mirror of everything Drishti downloads, plus an offline-readiness report for all modes.

WHY. Drishti's on-device ("local-only") models run in the browser: SmolVLM-500M (scene description, Transformers.js),
COCO-SSD (objects, TensorFlow.js), Tesseract.js (reading text), YAMNet (sound alerts) and Depth-Anything (distance).
But the browser fetches those models and their libraries from the internet (cdn.jsdelivr.net, huggingface.co,
storage.googleapis.com, tfhub.dev) the first time. Without internet, a phone that has not downloaded them yet cannot
use them, and every device has to download the same ~2 GB again.

HOW. Drishti's page is served with those internet addresses pointed at this computer: /hub/mirror/<host>/<path>.
  * Online: the hub fetches the file from the original address, saves it under offline-cache/ and serves it.
  * Offline: the hub serves the saved copy. Phones on the same Wi-Fi get it from this computer, not the internet.
Text files (scripts, JSON, CSS) are served with the same address rewrite, so the libraries' own downloads (model
weights, WASM engines, language data) also go through the mirror. Drishti's files on disk are unchanged.

The other two modes: Gesture is fully local already (MediaPipe's hand model ships inside the Python package).
MAVIS OCR has its own offline path: a local OCR engine (RapidOCR PP-OCRv5, downloaded once by MAVIS's own code) and a
local language model through Ollama. The readiness report below covers all three.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path

import aiohttp
from aiohttp import web

from .paths import MAVIS, SUITE

CACHE = Path(os.environ.get("MAVIS_OFFLINE_CACHE", SUITE / "offline-cache"))
PREFIX = "/hub/mirror/"
# Every host Drishti (or a library it loads) downloads code, models or fonts from.
HOSTS = ("cdn.jsdelivr.net", "huggingface.co", "storage.googleapis.com", "tfhub.dev", "www.kaggle.com",
         "tessdata.projectnaptha.com", "unpkg.com", "fonts.googleapis.com", "fonts.gstatic.com")
TEXT_TYPES = ("javascript", "json", "css", "html", "text/plain", "ecmascript")
OFFLINE_ONLY = os.environ.get("MAVIS_OFFLINE_ONLY") == "1"      # tests: never contact the internet
# tests only: {"cdn.jsdelivr.net": "http://127.0.0.1:19999"} sends a host's downloads to a local fake server
UPSTREAM = json.loads(os.environ.get("MAVIS_MIRROR_UPSTREAM", "{}"))
_HOST_RE = re.compile(r"(?:https?:)?//(" + "|".join(re.escape(h) for h in sorted(HOSTS, key=len, reverse=True)) + r")\b")

# Groups shown on the readiness page: (label, what it is for, cache prefixes that must exist)
DRISHTI_GROUPS = [
    ("Scene description", "SmolVLM-500M, Describe mode", ["huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct/"],
     "loadLocalVLM"),
    ("Object detection", "COCO-SSD, Objects / Navigate / Guardian",
     ["cdn.jsdelivr.net/npm/@tensorflow/tfjs@", "storage.googleapis.com/tfjs-models/savedmodel/ssd_mobilenet"],
     "loadObjectDetector"),
    ("Text reading", "Tesseract.js, Read Text / Currency", ["cdn.jsdelivr.net/npm/tesseract.js", "eng.traineddata"],
     "loadOcrWorker"),
    ("Sound alerts", "YAMNet", ["tfhub.dev/google/tfjs-model/yamnet"], "loadSoundClassifier"),
    ("Distance", "Depth-Anything small", ["huggingface.co/Xenova/depth-anything-small-hf/"], "loadDepthModel"),
]


# ---------------------------------------------------------------- rewriting

def rewrite(text: str, origin: str) -> str:
    """https://cdn.jsdelivr.net/x -> <origin>/hub/mirror/cdn.jsdelivr.net/x (only for the mirrored hosts)."""
    return _HOST_RE.sub(lambda m: f"{origin}{PREFIX}{m.group(1)}", text)


def origin_of(request: web.Request) -> str:
    return f"{request.scheme}://{request.host}"


# ---------------------------------------------------------------- cache layout

def cache_key(host: str, path: str, query: str) -> str:
    key = f"{host}/{path}"
    if key.endswith("/") or not path:
        key += "__index"
    if query:
        key += "__q" + hashlib.sha1(query.encode()).hexdigest()[:12]
    return key


FILE_SUFFIX = "~file"     # a saved file never shares its name with a folder: jsDelivr serves both ".../pkg@3" and
                          # ".../pkg@3/dist/x.wasm", which on disk would need ".../pkg@3" to be a file AND a folder


def _files(key: str):
    body = CACHE / (key + FILE_SUFFIX)
    return body, body.with_name(body.name + ".meta.json")


def cached(key: str):
    body, meta = _files(key)
    if meta.exists() and (body.exists() or "redirect" in json.loads(meta.read_text("utf-8"))):
        return json.loads(meta.read_text("utf-8"))
    return None


def _is_text(ctype: str) -> bool:
    return any(t in (ctype or "") for t in TEXT_TYPES)


# ---------------------------------------------------------------- the mirror route

class Mirror:
    def __init__(self):
        self.locks: dict[str, asyncio.Lock] = {}
        self.session: aiohttp.ClientSession | None = None
        self.stats = {"served_from_cache": 0, "downloaded": 0, "missing_offline": 0}

    async def client(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=10,
                                                                                sock_read=120))
        return self.session

    async def handle(self, request: web.Request):
        host = request.match_info["host"]
        path = request.match_info["path"]
        if host not in HOSTS:
            raise web.HTTPNotFound()
        query = request.query_string
        key = cache_key(host, path, query)
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:                       # one download per file, even when several tabs ask at once
            meta = cached(key)
            if meta is None:
                if OFFLINE_ONLY:
                    return self._missing(host, path)
                try:
                    meta = await self._download(host, path, query, key)
                except _Upstream as exc:
                    return web.Response(status=exc.status, text=exc.text)
                except Exception:  # noqa: BLE001  (no internet, DNS failure, timeout)
                    return self._missing(host, path)
            else:
                self.stats["served_from_cache"] += 1
        return self._serve(request, key, meta)

    def _missing(self, host, path):
        self.stats["missing_offline"] += 1
        return web.json_response({"error": "not_downloaded",
                                  "message": f"{host}/{path} has not been downloaded yet and there is no internet. "
                                             "Open MAVIS once with internet (or use 'Prepare for offline') first."},
                                 status=504)

    async def _download(self, host, path, query, key):
        url = f"{UPSTREAM.get(host, 'https://' + host)}/{path}" + (f"?{query}" if query else "")
        s = await self.client()
        async with s.get(url, allow_redirects=True) as r:
            if r.status != 200:
                raise _Upstream(r.status, (await r.text())[:300])
            final = r.url
            for public, base in sorted(UPSTREAM.items(), key=lambda kv: kv[0] != host):   # tests: a local fake
                if str(final).startswith(base):                                            # server stands in
                    final = final.__class__("https://" + public + str(final)[len(base):])
            ctype = r.headers.get("Content-Type", "application/octet-stream")
            # A redirect to another mirrored address (e.g. a version range resolving to an exact version): keep it a
            # redirect, so relative imports inside the file resolve exactly as they do on the original site.
            if final.host in HOSTS and (final.host, final.path.lstrip("/"), final.query_string) != (host, path, query):
                fkey = cache_key(final.host, final.path.lstrip("/"), final.query_string)
                await self._store(fkey, r, ctype, str(final))
                target = PREFIX + final.host + "/" + final.path.lstrip("/") + (
                    f"?{final.query_string}" if final.query_string else "")
                return self._write_meta(key, {"redirect": target, "url": url})
            return await self._store(key, r, ctype, url)

    async def _store(self, key, resp, ctype, url):
        body, _ = _files(key)
        body.parent.mkdir(parents=True, exist_ok=True)
        tmp = body.with_name(body.name + ".part")
        size = 0
        with open(tmp, "wb") as f:
            async for chunk in resp.content.iter_chunked(1 << 20):
                f.write(chunk)
                size += len(chunk)
        tmp.replace(body)
        self.stats["downloaded"] += 1
        return self._write_meta(key, {"content_type": ctype, "size": size, "url": url, "saved_at": time.time()})

    @staticmethod
    def _write_meta(key, meta):
        _, m = _files(key)
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(json.dumps(meta), "utf-8")
        return meta

    @staticmethod
    def _serve(request, key, meta):
        if "redirect" in meta:
            raise web.HTTPFound(meta["redirect"])
        body, _ = _files(key)
        ctype = meta.get("content_type", "application/octet-stream")
        headers = {"Cache-Control": "public, max-age=86400", "X-MAVIS-Offline-Cache": "hit"}
        if _is_text(ctype):
            text = body.read_bytes().decode("utf-8", "replace")
            return web.Response(body=rewrite(text, origin_of(request)).encode("utf-8"),
                                headers={**headers, "Content-Type": ctype})
        return web.FileResponse(body, headers={**headers, "Content-Type": ctype})


class _Upstream(Exception):
    def __init__(self, status, text):
        super().__init__(text)
        self.status, self.text = status, text


# ---------------------------------------------------------------- readiness report

def _cache_index():
    """{key: size} of every saved file."""
    out = {}
    if CACHE.exists():
        for m in CACHE.rglob("*.meta.json"):
            try:
                meta = json.loads(m.read_text("utf-8"))
            except ValueError:
                continue
            key = str(m.relative_to(CACHE)).replace(os.sep, "/")[: -len(FILE_SUFFIX + ".meta.json")]
            out[key] = meta.get("size", 0)
    return out


def mavis_ocr_models():
    """MAVIS's local OCR models (its offline fallback), in MAVIS's own cache folder."""
    d = MAVIS / "prototype" / "models"
    files = [p for p in d.rglob("*.onnx")] if d.exists() else []
    return {"ready": len(files) >= 2, "files": len(files), "bytes": sum(p.stat().st_size for p in files)}


_net = {"t": 0.0, "online": None}


async def internet() -> bool:
    if OFFLINE_ONLY:
        return False
    if time.time() - _net["t"] < 30 and _net["online"] is not None:
        return _net["online"]
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4)) as s:
            async with s.head("https://cdn.jsdelivr.net/") as r:
                ok = r.status < 500
    except Exception:  # noqa: BLE001
        ok = False
    _net.update(t=time.time(), online=ok)
    return ok


def drishti_report():
    idx = _cache_index()
    groups = []
    for label, purpose, prefixes, loader in DRISHTI_GROUPS:
        hits = {k: v for k, v in idx.items() if any(p in k for p in prefixes)}
        ready = all(any(p in k for k in idx) for p in prefixes)
        groups.append({"label": label, "purpose": purpose, "ready": ready, "files": len(hits),
                       "bytes": sum(hits.values()), "loader": loader})
    return {"groups": groups, "files": len(idx), "bytes": sum(idx.values()),
            "ready": all(g["ready"] for g in groups)}


def add_routes(app: web.Application, hub):
    mirror = Mirror()
    app["mirror"] = mirror  # noqa

    async def mirror_route(request):
        return await mirror.handle(request)

    async def npm_root(request):
        # jsDelivr's ES-module builds import other packages by root-relative paths ("/npm/x@1/+esm"); inside the
        # mirror those land on this origin, so send them to the mirror too.
        q = ("?" + request.query_string) if request.query_string else ""
        raise web.HTTPFound(PREFIX + "cdn.jsdelivr.net/npm/" + request.match_info["path"] + q)

    async def status(_request):
        llm = {}
        if hub.mavis_state is not None:
            try:
                llm = dict(hub.mavis_state.snapshot().get("llm") or {})
            except Exception:  # noqa: BLE001
                llm = {}
        return web.json_response({
            "internet": await internet(),
            "cache_folder": str(CACHE),
            "drishti": drishti_report(),
            "mavis": {"local_ocr": mavis_ocr_models(),
                      "llm": {"status": llm.get("status"), "model": llm.get("model"), "message": llm.get("message")}},
            "gesture": {"ready": hub.gesture_ok, "note": "MediaPipe's hand model is part of the installed package"},
            "mirror": mirror.stats,
        }, headers={"Cache-Control": "no-store"})

    async def prepare_mavis_ocr(_request):
        """Downloads MAVIS's local OCR models with MAVIS's own code (as its SETUP.md describes), while online."""
        if mavis_ocr_models()["ready"]:
            return web.json_response({"ok": True, "already": True})
        if not await internet():
            return web.json_response({"ok": False, "message": "No internet: the models can only be downloaded online."},
                                     status=409)

        def build():
            from ocr_engine import PrototypeOCREngine   # MAVIS's prototype/ocr_engine.py (on the import path)
            PrototypeOCREngine._init_reader()

        try:
            await asyncio.get_running_loop().run_in_executor(None, build)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "message": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"ok": True, **mavis_ocr_models()})

    app.router.add_route("GET", PREFIX + "{host}/{path:.*}", mirror_route)
    app.router.add_route("HEAD", PREFIX + "{host}/{path:.*}", mirror_route)
    app.router.add_get("/npm/{path:.*}", npm_root)
    app.router.add_get("/hub/offline/status", status)
    app.router.add_post("/hub/offline/mavis-ocr", prepare_mavis_ocr)
