"""Offline mirror tests: a local fake server stands in for the internet (cdn.jsdelivr.net, huggingface.co)."""
import asyncio
import json
import re
import sys
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

SUITE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUITE))

from hub import offline  # noqa: E402
from hub.paths import DRISHTI_WEB  # noqa: E402

WASM = bytes(range(256)) * 40
JS = ('import "./dep.js"; const w = "https://cdn.jsdelivr.net/npm/lib@1.0.0/dist/core.wasm"; '
      'const m = "https://huggingface.co/org/model/resolve/main/model.onnx"; const f = "//fonts.gstatic.com/x.woff2";')


def fake_internet():
    app = web.Application()
    hits = {"n": 0}

    async def js(_r):
        hits["n"] += 1
        return web.Response(text=JS, content_type="application/javascript")

    async def wasm(_r):
        return web.Response(body=WASM, content_type="application/wasm")

    async def range_(_r):
        raise web.HTTPFound("/npm/lib@1.0.0/dist/lib.min.js")

    async def onnx(_r):
        return web.Response(body=b"ONNX" * 1000, content_type="application/octet-stream")

    app.router.add_get("/npm/lib@1.0.0/dist/lib.min.js", js)
    app.router.add_get("/npm/lib@1.0.0/dist/core.wasm", wasm)
    app.router.add_get("/npm/lib@1", range_)
    app.router.add_get("/org/model/resolve/main/model.onnx", onnx)
    return app, hits


def run(coro_fn, tmp_path, offline_only=False):
    async def main():
        up_app, hits = fake_internet()
        up = TestServer(up_app)
        await up.start_server()
        base = str(up.make_url("")).rstrip("/")
        offline.UPSTREAM.clear()
        offline.UPSTREAM.update({"cdn.jsdelivr.net": base, "huggingface.co": base})
        offline.CACHE = tmp_path / "cache"
        offline.OFFLINE_ONLY = offline_only

        class FakeHub:
            mavis_state = None
            gesture_ok = True

        app = web.Application()
        offline.add_routes(app, FakeHub())
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            return await coro_fn(client, hits, up)
        finally:
            await client.close()
            await up.close()
            m = app.get("mirror")
            if m and m.session:
                await m.session.close()
            offline.OFFLINE_ONLY = False
            offline.UPSTREAM.clear()
    return asyncio.run(main())


def test_text_is_downloaded_saved_and_rewritten(tmp_path):
    async def go(c, hits, _up):
        r1 = await c.get("/hub/mirror/cdn.jsdelivr.net/npm/lib@1.0.0/dist/lib.min.js")
        t1 = await r1.text()
        r2 = await c.get("/hub/mirror/cdn.jsdelivr.net/npm/lib@1.0.0/dist/lib.min.js")
        return r1.status, t1, r2.headers.get("X-MAVIS-Offline-Cache"), hits["n"], str(c.make_url(""))
    s1, text, hit, n, origin = run(go, tmp_path)
    origin = origin.rstrip("/")
    assert s1 == 200 and hit == "hit" and n == 1                     # second request did not touch the internet
    assert f"{origin}/hub/mirror/cdn.jsdelivr.net/npm/lib@1.0.0/dist/core.wasm" in text
    assert f"{origin}/hub/mirror/huggingface.co/org/model/resolve/main/model.onnx" in text
    assert f"{origin}/hub/mirror/fonts.gstatic.com/x.woff2" in text
    assert 'import "./dep.js"' in text                                # relative imports untouched
    assert not re.search(r"(?<!mirror/)https://cdn\.jsdelivr\.net", text)
    offline.CACHE = tmp_path / "cache"
    raw = offline._files("cdn.jsdelivr.net/npm/lib@1.0.0/dist/lib.min.js")[0].read_text()
    assert raw == JS                                                   # saved exactly as downloaded


def test_binary_is_byte_identical_and_served_offline(tmp_path):
    async def online(c, *_):
        r = await c.get("/hub/mirror/cdn.jsdelivr.net/npm/lib@1.0.0/dist/core.wasm")
        return r.status, await r.read(), r.headers.get("Content-Type")
    status, body, ctype = run(online, tmp_path)
    assert status == 200 and body == WASM and ctype == "application/wasm"

    async def offline_(c, *_):
        r = await c.get("/hub/mirror/cdn.jsdelivr.net/npm/lib@1.0.0/dist/core.wasm")
        miss = await c.get("/hub/mirror/cdn.jsdelivr.net/npm/never-downloaded.js")
        return r.status, await r.read(), miss.status, await miss.json()
    status, body, miss_status, miss = run(offline_, tmp_path, offline_only=True)
    assert status == 200 and body == WASM                               # no internet: served from this computer
    assert miss_status == 504 and miss["error"] == "not_downloaded"


def test_version_range_redirect_is_kept_and_replayed_offline(tmp_path):
    async def go(c, *_):
        r = await c.get("/hub/mirror/cdn.jsdelivr.net/npm/lib@1", allow_redirects=False)
        return r.status, r.headers.get("Location")
    status, loc = run(go, tmp_path)
    assert status == 302 and loc == "/hub/mirror/cdn.jsdelivr.net/npm/lib@1.0.0/dist/lib.min.js"
    status, loc = run(go, tmp_path, offline_only=True)
    assert status == 302 and loc == "/hub/mirror/cdn.jsdelivr.net/npm/lib@1.0.0/dist/lib.min.js"


def test_upstream_errors_pass_through_and_are_not_saved(tmp_path):
    async def go(c, *_):
        r = await c.get("/hub/mirror/cdn.jsdelivr.net/npm/does-not-exist.js")
        bad_host = await c.get("/hub/mirror/evil.example/x.js")
        return r.status, bad_host.status
    status, bad = run(go, tmp_path)
    assert status == 404 and bad == 404
    offline.CACHE = tmp_path / "cache"
    assert not offline._files("cdn.jsdelivr.net/npm/does-not-exist.js")[0].exists()


def test_root_relative_npm_paths_go_to_the_mirror(tmp_path):
    async def go(c, *_):
        r = await c.get("/npm/lib@1.0.0/+esm?x=1", allow_redirects=False)
        return r.status, r.headers.get("Location")
    status, loc = run(go, tmp_path)
    assert status == 302 and loc == "/hub/mirror/cdn.jsdelivr.net/npm/lib@1.0.0/+esm?x=1"


def test_readiness_report_counts_saved_models(tmp_path):
    async def go(c, *_):
        await c.get("/hub/mirror/huggingface.co/org/model/resolve/main/model.onnx")
        return await (await c.get("/hub/offline/status")).json()
    st = run(go, tmp_path, offline_only=False)
    assert st["gesture"]["ready"] is True
    assert st["drishti"]["files"] == 1 and st["drishti"]["bytes"] == 4000
    assert [g["label"] for g in st["drishti"]["groups"]] == [g[0] for g in offline.DRISHTI_GROUPS]


def test_drishti_page_downloads_point_to_this_computer():
    html = (DRISHTI_WEB / "index.html").read_text(encoding="utf-8")
    out = offline.rewrite(html, "https://192.168.1.5:8443")
    for host in ("cdn.jsdelivr.net", "tfhub.dev", "fonts.googleapis.com"):
        assert f"https://{host}" in html                               # the original uses the internet...
        assert not re.search(rf"(?<!/hub/mirror/){re.escape(host)}", out.replace("https://192.168.1.5:8443", ""))
    assert "https://192.168.1.5:8443/hub/mirror/cdn.jsdelivr.net/npm/@huggingface/transformers@3" in out
    assert "api.groq.com" in out                                       # ...cloud APIs are left alone


def test_a_file_and_a_folder_with_the_same_name_can_both_be_saved(tmp_path):
    """jsDelivr serves ".../lib@1.0.0/dist/lib.min.js" and ".../lib@1" (file) next to ".../lib@1/..." paths."""
    async def go(c, *_):
        a = await c.get("/hub/mirror/huggingface.co/org/model/resolve/main/model.onnx")
        b = await c.get("/hub/mirror/huggingface.co/org/model")            # 404 upstream, but must not crash
        return a.status, b.status
    a, b = run(go, tmp_path)
    assert a == 200 and b in (404, 405)
