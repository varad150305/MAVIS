"""The camera picture as the transcript background, with Netflix-style
subtitles (handoff UI changes).

A generated video file stands in for the webcam (MAVIS_UI_CAMERA=<file>), so
no camera hardware is opened. Each check runs the real Tk UI in a child
process (window made fully transparent)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _make_video(path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (320, 240))
    for i in range(30):                                  # dark moving pattern (worst case for dark text)
        f = np.zeros((240, 320, 3), dtype=np.uint8)
        f[:, :, 0] = (np.arange(320) * 2 + i * 8) % 256 // 3
        f[:, :, 2] = 40
        out.write(f)
    out.release()


CHILD = textwrap.dedent(r'''
    import sys, time, tkinter as tk
    from product import session as S
    from product.ui.app import TranslatorApp, VEIL_ALPHA, SUB_BOTTOM, SUB_WIDTH
    from product.ui_model import TranscriptModel

    seq = 0
    def apply(t, **f):
        global seq
        seq += 1
        app.model.apply({"schema": S.SCHEMA, "session_id": "A", "seq": seq, "type": t, **f})
        app._render_transcript()

    def visible():
        c = app.canvas
        return [c.itemcget(i, "text") for i in c.find_withtag("content") if c.type(i) == "text"]

    def wait(cond, s=8):
        t0 = time.time()
        while time.time() - t0 < s and not cond():
            root.update(); time.sleep(0.02)

    root = tk.Tk(); root.geometry("1100x700+0+0"); root.attributes("-alpha", 0.0)
    app = TranslatorApp(root)
    app.model = TranscriptModel("A")
    mode = sys.argv[1]
    wait(lambda: app._video_on)
    if mode == "nocamera":
        apply("en_committed", segment_id=1, utterance_id=1, new_text="good morning", stable_text="good morning")
        assert not app._video_on and app.text.winfo_manager() == "pack", "fallback: plain text view"
        assert "good morning" in app.text.get("1.0", "end")
        print("FALLBACK-OK", app.camera.error if app.camera else "camera off"); root.destroy(); sys.exit(0)

    assert app._video_on, f"no video: {app.camera.error}"
    assert app.canvas.winfo_manager() == "pack" and app.text.winfo_manager() == ""
    c = app.canvas

    # segment 1: draft, then final
    apply("en_committed", segment_id=1, utterance_id=1, new_text="good morning everyone", stable_text="good morning everyone")
    assert visible() == ["SEGMENT 1   · listening…", "good morning everyone", "translating…"], visible()
    apply("hi_draft", segment_id=1, invocation_id=1, target_text="DRAFT-1")
    assert visible() == ["SEGMENT 1   · listening…", "good morning everyone", " DRAFT – may change ", "DRAFT-1"]
    apply("segment_closed", segment_id=1, reason="sentence end", utterance_ids=[1])
    apply("hi_final", segment_id=1, target_text="सुप्रभात सभी को", kind="final_mt")
    assert visible() == ["SEGMENT 1", "good morning everyone", " FINAL ", "सुप्रभात सभी को"], visible()
    # styling comes from the text-widget tags (existing format and hierarchy)
    by_text = {c.itemcget(i, "text"): i for i in c.find_withtag("content") if c.type(i) == "text"}
    for txt, tag in [("सुप्रभात सभी को", "hi_final"), (" FINAL ", "badge_final"), ("good morning everyone", "en"),
                     ("SEGMENT 1", "seg")]:
        assert c.itemcget(by_text[txt], "font") == str(app.text.tag_cget(tag, "font")), txt
        assert c.itemcget(by_text[txt], "fill") == str(app.text.tag_cget(tag, "foreground")), txt
    # bottom-centre, one veil behind the block
    w, h = c.winfo_width(), c.winfo_height()
    x0, y0, x1, y1 = c.bbox("content")
    assert abs(y1 - h * (1 - SUB_BOTTOM)) <= 2, (y1, h)
    assert abs((x0 + x1) / 2 - w / 2) <= 3, (x0, x1, w)
    assert len(app._veil) == 1

    # provisional English while the previous segment is closed -> it starts the next subtitle alone
    apply("en_provisional", utterance_id=2, text="today we")
    assert visible() == ["provisional - may change  ", "today we"], visible()
    # segment 2 replaces segment 1; English and Hindi stay with segment 2
    apply("en_committed", segment_id=2, utterance_id=2, new_text="today we talk", stable_text="today we talk")
    assert visible() == ["SEGMENT 2   · listening…", "today we talk", "translating…"], visible()
    assert not any("good morning" in t or "सुप्रभात" in t for t in visible())
    # provisional update inside the open segment: shown under its English, updated in place
    apply("en_provisional", utterance_id=3, text="about the")
    assert visible()[:4] == ["SEGMENT 2   · listening…", "today we talk", "provisional - may change  ", "about the"]
    apply("en_provisional", utterance_id=3, text="about the weather")
    assert visible().count("provisional - may change  ") == 1 and "about the weather" in visible()
    apply("hi_draft", segment_id=2, invocation_id=1, target_text="आज हम बात करते हैं")
    apply("en_committed", segment_id=2, utterance_id=3, new_text="about the weather",
          stable_text="today we talk about the weather")
    assert visible() == ["SEGMENT 2   · listening…", "today we talk about the weather", " DRAFT – may change ",
                         "आज हम बात करते हैं"], visible()
    apply("segment_closed", segment_id=2, reason="sentence end", utterance_ids=[2, 3])
    apply("hi_final", segment_id=2, target_text="आज हम मौसम के बारे में बात करते हैं", kind="final_mt")
    assert visible() == ["SEGMENT 2", "today we talk about the weather", " FINAL ", "आज हम मौसम के बारे में बात करते हैं"]
    # a late draft for segment 1 / a Hindi event for an old segment never shows
    apply("hi_draft_late", segment_id=1, invocation_id=2, target_text="LATE")
    assert "LATE" not in visible()

    # long subtitle: wraps, keeps the newest words, stays within a few lines
    long_en = " ".join(f"word{i}" for i in range(80))
    long_hi = " ".join(["बहुत लंबा हिंदी वाक्य"] * 30)
    apply("en_committed", segment_id=3, utterance_id=4, new_text=long_en, stable_text=long_en)
    apply("hi_draft", segment_id=3, invocation_id=1, target_text=long_hi)
    by_text = {c.itemcget(i, "text"): i for i in c.find_withtag("content") if c.type(i) == "text"}
    en = next(t for t in by_text if "word79" in t)
    hi = next(t for t in by_text if "हिंदी" in t)
    assert en.startswith("… ") and en.endswith("word79") and hi.startswith("… ")
    from tkinter import font as tkfont
    for t, tag, n in [(en, "en", 2), (hi, "hi_draft", 3)]:
        bx = c.bbox(by_text[t])
        ls = tkfont.Font(font=app.text.tag_cget(tag, "font")).metrics("linespace")
        assert bx[3] - bx[1] <= n * ls + 2 and bx[3] - bx[1] > ls, (tag, bx)      # wrapped, capped
        assert bx[2] - bx[0] <= w * SUB_WIDTH + 2
    x0, y0, x1, y1 = c.bbox("content")
    assert (y1 - y0) < 0.5 * h, (y0, y1, h)                                       # does not grow indefinitely

    # internal history and the plain view keep every segment
    full = app.text.get("1.0", "end")
    assert "good morning everyone" in full and "today we talk about the weather" in full and "word79" in full

    # the video keeps updating; the veil makes the text block readable
    f0, p0 = app.camera.frames, app._photo
    wait(lambda: app.camera.frames > f0 + 3 and app._photo is not p0, 2)
    assert app.camera.frames > f0 and app._photo is not p0
    vx0, vy0, vx1, vy1 = app._veil[0]
    px = app._photo.get(int(vx0) + 2, int(vy0) + 2)
    assert min(px) >= int(255 * VEIL_ALPHA) - 1, px

    # camera gone -> back to the plain text view with the full transcript
    app.camera.close()
    wait(lambda: not app._video_on, 6)
    assert not app._video_on and app.text.winfo_manager() == "pack"
    print("VIDEO-OK"); root.destroy()
''')


def _run(mode: str, camera: str):
    env = {**os.environ, "MAVIS_UI_CAMERA": camera}
    r = subprocess.run([sys.executable, "-B", "-c", CHILD, mode], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=120, encoding="utf-8")
    if "TclError" in r.stderr and "display" in r.stderr:
        pytest.skip("no display")
    return r


def test_netflix_style_subtitles_over_the_camera_picture(tmp_path):
    video = tmp_path / "speaker.avi"
    _make_video(video)
    r = _run("video", str(video))
    assert r.returncode == 0 and "VIDEO-OK" in r.stdout, r.stdout + r.stderr[-3000:]


def test_without_a_camera_the_plain_text_view_is_unchanged(tmp_path):
    r = _run("nocamera", str(tmp_path / "missing_camera.avi"))
    assert r.returncode == 0 and "FALLBACK-OK" in r.stdout, r.stdout + r.stderr[-3000:]
    r = _run("nocamera", "off")
    assert r.returncode == 0 and "FALLBACK-OK" in r.stdout, r.stdout + r.stderr[-3000:]
