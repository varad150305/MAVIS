"""Tests for the hub (the new code that joins the three projects). The projects' own tests stay in modules/*/tests.

    python -m pytest tests -q          (from the suite folder, with the suite's environment)
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

SUITE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUITE))

from hub.paths import GESTURE, MAVIS, add_import_paths  # noqa: E402

add_import_paths()

from hub.gesture_service import GestureSession  # noqa: E402
from hub.mavis_bridge import rewrite  # noqa: E402
from hand_tracker import HandDetection  # noqa: E402  (gesture project)
from hand_geometry import image_to_metric  # noqa: E402

FIXTURE = GESTURE / "tests" / "fixtures" / "real_calibration_samples.json"
SAMPLES = json.loads(FIXTURE.read_text(encoding="utf-8"))["samples"]


def _det(s):
    w, h = s["frame_size"]
    img = image_to_metric(np.asarray(s["image_landmarks"]), w, h) / np.array([w, h, w])
    return HandDetection(img, np.asarray(s["world_landmarks"], float), s["handedness_label"], 0.99, (w, h))


def _sample(gesture):
    return next(s for s in SAMPLES if s["gesture"] == gesture and s.get("rule_prediction") == gesture)


def _hold(session, det, t0, seconds, fps=20):
    """Feeds the same detection for `seconds`, like a held hand; returns (events, last payload, end time)."""
    events, payload, t = [], None, t0
    for _ in range(int(seconds * fps)):
        r = session.pipe.process_detections([det] if det is not None else [], t)
        payload = session.describe(r, [det] if det is not None else [], t, (480, 640, 3))
        events += payload["events"]
        t += 1.0 / fps
    return events, payload, t


@pytest.fixture(scope="module")
def session():
    s = GestureSession()
    yield s
    s.close()


def test_full_workflow_reaches_the_page_as_data(session):
    """Real recorded hands: 1 finger -> thumbs up -> open palm, as the page would receive it."""
    t = 1000.0
    ev1, p, t = _hold(session, _det(_sample("one_finger")), t, 1.2)
    assert [e["event"] for e in ev1] == ["feature.selection_pending"]
    assert ev1[0]["feature"] == "ocr"
    assert p["controller"]["state"] == "AWAITING_CONFIRMATION"
    assert p["controller"]["pending_label"] == "OCR"
    assert p["recognition"]["gesture"] == "one_finger" and p["recognition"]["confirmed"]
    assert len(p["hand"]["points"]) == 21 and len(p["recognition"]["fingers"]) == 5

    _, _, t = _hold(session, None, t, 1.2)                       # hand out of view: nothing happens
    assert session.pipe.controller.state.value == "AWAITING_CONFIRMATION"

    ev2, p, t = _hold(session, _det(_sample("thumbs_up")), t, 1.2)
    assert [e["event"] for e in ev2] == ["feature.start_requested"]
    assert p["controller"]["active_feature"] == "ocr"

    _, _, t = _hold(session, None, t, 1.2)
    ev3, p, t = _hold(session, _det(_sample("open_palm")), t, 1.2)
    assert [e["event"] for e in ev3] == ["feature.stop_requested", "navigation.return_to_selection"]
    assert p["controller"]["state"] == "SELECTING_FEATURE"
    json.dumps(p)                                                # the payload is plain JSON


def test_jpeg_frames_and_classic_view(session):
    frame = np.full((480, 640, 3), 90, np.uint8)
    jpeg = cv2.imencode(".jpg", frame)[1].tobytes()
    payload, image = session.process(jpeg, classic=True)
    assert payload["type"] == "frame" and payload["hand"] is None
    view = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    assert view.shape[0] == 480 and view.shape[1] > 640              # camera + original state panel side by side
    payload, image = session.process(b"not a jpeg", classic=False)
    assert payload["type"] == "error" and image is None


def test_menu_matches_the_mapping_file(session):
    m = session.menu()
    mapping = json.loads((GESTURE / "config" / "gesture_mapping.json").read_text(encoding="utf-8"))
    assert [x["gesture"] for x in m["selection"]] == list(mapping["selection_gestures"])
    assert m["confirm_gesture"] == mapping["confirm_gesture"]


def test_mavis_paths_are_rewritten_everywhere():
    js = (MAVIS / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
    html = (MAVIS / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    out_js, out_html = rewrite(js), rewrite(html)
    for text in (out_js, out_html):
        for quote in "\"'`(":
            assert quote + "/api/" not in text and quote + "/static/" not in text
    quoted = sum(js.count(q + "/api/") for q in "\"'`(")           # real URLs (comments are left as they are)
    assert out_js.count("/mavis/api/") == quoted > 20
    assert "EventSource('/mavis/api/events')" in out_js


def test_relay_feeds_mavis_livestream():
    """Frames pushed like a phone does arrive in MAVIS's own LiveStream reader at full size."""
    import threading

    from aiohttp import web
    from hub.mavis_bridge import CameraRelay, make_relay_app
    from webapp.live_stream import LiveStream

    port = 18766
    result = {}

    async def main():
        relay = CameraRelay()
        runner = web.AppRunner(make_relay_app(relay))
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        relay.publishers = 1
        frame = cv2.imencode(".jpg", np.full((1080, 1920, 3), 200, np.uint8))[1].tobytes()
        for _ in range(5):
            await relay.publish(frame)
            await asyncio.sleep(0.05)

        def read():
            s = LiveStream(f"http://127.0.0.1:{port}/video")
            result["connected"] = s.connect()
            shapes = []
            for _ in range(8):
                ok, f = s.read()
                if ok:
                    shapes.append(f.shape)
            result["shapes"] = shapes
            s.release()

        th = threading.Thread(target=read)
        th.start()
        t0 = time.time()
        while th.is_alive() and time.time() - t0 < 20:
            await relay.publish(frame)
            await asyncio.sleep(0.05)
        th.join(5)
        await runner.cleanup()

    asyncio.run(main())
    assert result["connected"]
    assert len(result["shapes"]) == 8 and set(result["shapes"]) == {(1080, 1920, 3)}


# ---------------------------------------------------------------- Drishti API routes and the access guard

def _client_run(coro_fn, env=None):
    """Runs coro_fn(client) against the full hub app (all three modes mounted)."""
    import os
    from aiohttp.test_utils import TestClient, TestServer
    from hub.server import Hub

    old = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    try:
        async def main():
            hub = Hub(port=18000, lan=False)
            client = TestClient(TestServer(hub.build()))
            await client.start_server()
            try:
                return await coro_fn(client, hub)
            finally:
                await client.close()
        return asyncio.run(main())
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_drishti_proxy_routes_match_the_vercel_functions():
    async def go(c, _hub):
        out = {}
        out["get"] = await (await c.get("/api/groq")).json()
        r = await c.post("/api/groq", json={"model": "x"})
        out["post_no_key"] = (r.status, await r.json())
        r = await c.post("/api/cloudflare", json={})
        out["cf_no_creds"] = (r.status, await r.json())
        r = await c.post("/api/cloudflare", json={}, headers={"X-CF-Account": "a", "X-CF-Token": "t"})
        out["cf_no_model"] = (r.status, await r.json())
        statuses = [(await c.post("/api/sarvam", json={})).status for _ in range(41)]
        out["sarvam_limit"] = statuses[-1]
        out["options"] = (await c.options("/api/groq")).headers.get("Access-Control-Allow-Origin")
        return out
    out = _client_run(go, {"GROQ_API_KEY": "", "SARVAM_API_KEY": ""})
    assert out["get"] == {"ok": True, "configured": False}
    assert out["post_no_key"] == (500, {"error": "proxy_not_configured"})
    assert out["cf_no_creds"] == (400, {"error": "missing_credentials"})
    assert out["cf_no_model"] == (400, {"error": "missing_model"})
    assert out["sarvam_limit"] == 429                       # same 40 requests/minute limit as api/sarvam.js
    assert out["options"] == "*"


def test_every_mode_is_served_and_pages_point_to_their_own_paths():
    async def go(c, _hub):
        res = {}
        for path in ("/", "/gesture/", "/drishti/", "/mavis/", "/mavis/static/app.js", "/mavis/static/styles.css",
                     "/hub/static/gesture.js", "/hub/status", "/mavis/api/state", "/drishti/about/index.html"):
            r = await c.get(path)
            res[path] = (r.status, await r.text())
        res["redirect"] = (await c.get("/drishti", allow_redirects=False)).status
        return res
    res = _client_run(go)
    for path, value in res.items():
        if path != "redirect":
            assert value[0] == 200, path
    assert res["redirect"] == 302
    assert "drishti-addon.js" in res["/drishti/"][1] and "setMode" in res["/drishti/"][1]
    assert "mavis-addon.js" in res["/mavis/"][1] and "/mavis/static/app.js" in res["/mavis/"][1]
    status = json.loads(res["/hub/status"][1])
    assert all(status["modes"][m]["ok"] for m in ("drishti", "mavis", "gesture"))


def test_access_key_rules():
    from types import SimpleNamespace

    from hub import security
    key = "k" * 12
    phone = SimpleNamespace(remote="192.168.1.40", cookies={}, headers={})
    assert not security.has_access(phone, key)
    phone.cookies = {security.COOKIE: "wrong"}
    assert not security.has_access(phone, key)
    phone.cookies = {security.COOKIE: key}
    assert security.has_access(phone, key)
    assert security.has_access(SimpleNamespace(remote="127.0.0.1", cookies={}, headers={}), key)
    same = SimpleNamespace(headers={"Origin": "https://192.168.1.5:8443", "Host": "192.168.1.5:8443"})
    other = SimpleNamespace(headers={"Origin": "https://evil.example", "Host": "192.168.1.5:8443"})
    assert security.same_origin(same) and not security.same_origin(other)
