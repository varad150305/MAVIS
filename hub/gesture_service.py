"""Gesture mode: the original gesture controller, fed with frames from the browser's camera.

THE FIX. mavis_gesture.py opens the camera of the computer it runs on and draws its output with cv2.imshow(), i.e. in a
window on that computer. That is why the output always "went back to the system" instead of appearing on the phone.

Here the camera belongs to whatever device the page is open on (phone, tablet or laptop). The page sends JPEG frames over
a WebSocket; each frame goes through the *unchanged* gesture code, exactly as the project's own integration guide
recommends (CLAUDE.md, "Option 2: in-process, MAVIS owns the camera"):

    HandTracker -> GestureClassifier -> GestureStabilizer -> FeatureController -> EventBus

and the result (hand skeleton, gesture, confidence, hold progress, state machine, events) goes back to the same page,
which draws it on that device's screen. Nothing is shown on the computer. Optionally the page can also ask for the
"classic" picture: the original draw_recognition() + draw_state_panel() rendering, sent back as an image.
"""
from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from aiohttp import WSMsgType, web

from .paths import add_import_paths

add_import_paths()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import events as ev  # noqa: E402  (gesture project)
from gesture_classifier import NO_HAND, UNKNOWN, finger_state  # noqa: E402
from gesture_mapping import load_mapping  # noqa: E402
from hand_geometry import FINGER_NAMES  # noqa: E402
from mavis_gesture import (GesturePipeline, draw_recognition, draw_state_panel,  # noqa: E402
                           load_settings)

from .mavis_controller import ModuleController  # noqa: E402

MAX_FRAME_BYTES = 2_000_000


class GestureSession:
    """One browser tab = one pipeline (GesturePipeline is not thread-safe, so each runs on its own thread)."""

    def __init__(self):
        self.cfg = load_settings()
        self.mapping = load_mapping()
        self.mirror = self.cfg.get("camera", {}).get("mirror", True)
        self.pipe = GesturePipeline(self.cfg, self.mapping)
        self.fps = 0.0
        self._last = None
        self.counts = {"frames": 0, "events": 0}
        self.modules = ModuleController()      # MAVIS flow: thumbs up -> 1/2/3 fingers -> open palm

    # ---- one frame
    def process(self, jpeg: bytes, classic: bool):
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"type": "error", "message": "Could not decode the camera frame."}, None
        if self.mirror:
            frame = cv2.flip(frame, 1)       # same as mavis_gesture.run(): handedness assumes a selfie view
        now = time.time()
        dets = self.pipe.tracker.process_detailed(frame)
        r = self.pipe.process_detections(dets, now)
        if self._last is not None and now > self._last:
            inst = 1.0 / (now - self._last)
            self.fps = 0.9 * self.fps + 0.1 * inst if self.fps else inst
        self._last = now
        self.counts["frames"] += 1
        self.counts["events"] += len(r.events)
        payload = self.describe(r, dets, now, frame.shape)
        payload["mavis_events"] = self.modules.update(r.gesture_event, now)
        payload["mavis"] = self.modules.describe(now)
        image = None
        if classic:
            view = frame.copy()
            self.pipe.tracker.draw(view)
            draw_recognition(view, r.classification, self.pipe.stabilizer, now, self.fps,
                             self.pipe.classifier, self.mapping)
            panel = draw_state_panel(view.shape[0], self.pipe.controller, self.pipe.stabilizer, now)
            ok, buf = cv2.imencode(".jpg", np.hstack([view, panel]), [cv2.IMWRITE_JPEG_QUALITY, 75])
            image = buf.tobytes() if ok else None
        return payload, image

    def describe(self, r, dets, now, shape) -> dict:
        stab, ctl, m = self.pipe.stabilizer, self.pipe.controller, self.mapping
        res = r.classification
        hand = None
        if dets:
            pts = dets[0].image
            hand = {"points": [[round(float(x), 4), round(float(y), 4)] for x, y, _ in pts]}
        rec = None
        if res is not None:
            a = res.analysis
            g = res.gesture
            fingers = []
            if a.metrics:
                for name in FINGER_NAMES:
                    sc = float(a.metrics[name].score)
                    fingers.append({"name": name, "score": round(sc, 3),
                                    "state": finger_state(name, sc, self.pipe.classifier.settings)})
            rec = {
                "gesture": g,
                "is_gesture": g not in (UNKNOWN, NO_HAND),
                "no_hand": g == NO_HAND,
                "meaning": m.meaning(g) if g not in (UNKNOWN, NO_HAND) else "",
                "confidence": round(float(res.confidence), 3),
                "reason": res.reason,
                "hand": a.hand,
                "orientation": a.orientation.describe() if a.orientation else None,
                "rejected": bool(a.reject_reason),
                "fingers": fingers,
                "confirmed": stab.stable == g and g != UNKNOWN,
            }
        last = ctl.last_event
        return {
            "type": "frame",
            "t": now,
            "fps": round(self.fps, 1),
            "size": [int(shape[1]), int(shape[0])],
            "mirrored_input": self.mirror,
            "hand": hand,
            "recognition": rec,
            "stability": {
                "candidate": stab.candidate,
                "progress": round(float(stab.progress_at(now)), 3),
                "armed": bool(stab.armed),
                "cooldown": round(float(stab.cooldown_left(now)), 2),
                "fired": r.gesture_event,
            },
            "controller": {
                "state": ctl.state.value,
                "pending_feature": ctl.pending_feature,
                "pending_label": m.label(ctl.pending_feature) if ctl.pending_feature else None,
                "active_feature": ctl.active_feature,
                "active_label": m.label(ctl.active_feature) if ctl.active_feature else None,
                "confirm_left": round(float(ctl.confirmation_time_left(now)), 1)
                if ctl.pending_feature else None,
                "confirm_timeout": m.timings.confirmation_timeout_sec,
                "notice": ctl.notice[0] if ctl.notice and now - ctl.notice[1] < 2.0 else None,
                "last_event": ev.to_dict(last) if last is not None else None,
            },
            "events": [ev.to_dict(e) for e in r.events],
        }

    def menu(self) -> dict:
        m = self.mapping
        return {"type": "hello",
                "selection": [{"gesture": g, "feature": s.feature, "label": s.label, "hint": s.hint}
                              for g, s in m.selection.items()],
                "confirm_gesture": m.confirm_gesture, "back_gesture": m.back_gesture,
                "timings": asdict(m.timings),
                "classifier_mode": self.pipe.classifier.mode}

    def close(self):
        events = self.pipe.close()            # stop_requested for a still-active feature (reason "shutdown")
        return [ev.to_dict(e) for e in events]


async def gesture_ws(request: web.Request):
    ws = web.WebSocketResponse(max_msg_size=MAX_FRAME_BYTES, heartbeat=20)
    await ws.prepare(request)
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gesture")
    try:
        session = await loop.run_in_executor(pool, GestureSession)
    except Exception as exc:  # noqa: BLE001
        await ws.send_json({"type": "error", "message": f"Gesture engine could not start: {exc}"})
        await ws.close()
        pool.shutdown(wait=False)
        return ws
    await ws.send_json(session.menu())
    classic = False
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                payload, image = await loop.run_in_executor(pool, session.process, msg.data, classic)
                await ws.send_str(json.dumps(payload, separators=(",", ":")))
                if image is not None:
                    await ws.send_bytes(image)
            elif msg.type == WSMsgType.TEXT:
                try:
                    cmd = json.loads(msg.data)
                except ValueError:
                    continue
                if cmd.get("cmd") == "classic":
                    classic = bool(cmd.get("on"))
            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break
    finally:
        try:
            final = await loop.run_in_executor(pool, session.close)
            if final and not ws.closed:
                await ws.send_json({"type": "closing", "events": final})
        except Exception:  # noqa: BLE001
            pass
        pool.shutdown(wait=False)
    return ws


def add_routes(app: web.Application):
    app.router.add_get("/hub/gesture/ws", gesture_ws)
