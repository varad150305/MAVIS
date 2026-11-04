"""Tests for prototype/frame_selector.py.

Unit tests use synthetic frames. The parity tests replay the two recorded
videos through the production selector in fixed-fps mode and require the
commits to equal the frozen benchmark results frame for frame
(frame_selection_benchmark/results/raw/<video>/{stability_best,optical_flow}.json).
They are skipped when the recordings are not on disk.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "prototype"))

from frame_selector import (  # noqa: E402
    DEDUP_THRESHOLD, SELECTOR_MODES, StabilitySelector, build_selector, fingerprint, resolve_mode,
)

BENCH = REPO / "frame_selection_benchmark"


def textured(seed=0, shape=(360, 640)):
    """Blocky random pattern: survives the 160 px / 320 px analysis downscales
    (white noise would average to flat grey) and has strong edges to blur."""
    rng = np.random.default_rng(seed)
    cells = (rng.random((shape[0] // 8, shape[1] // 8)) * 255).astype(np.uint8)
    g = cv2.resize(cells, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def blur(img, k):
    return cv2.GaussianBlur(img, (k, k), 0) if k > 1 else img


# --- configuration ------------------------------------------------------------

def test_modes_and_builder(monkeypatch):
    monkeypatch.delenv("MAVIS_FRAME_SELECTOR", raising=False)
    assert set(SELECTOR_MODES) == {"b0", "c3", "c5"}
    assert build_selector("b0") is None
    assert build_selector("c3").mode == "c3"
    assert build_selector("C5").mode == "c5"
    with pytest.raises(ValueError):
        resolve_mode("c9")
    assert resolve_mode(None) == "b0"          # B0 is the default; C3/C5 are opt-in
    monkeypatch.setenv("MAVIS_FRAME_SELECTOR", "c3")
    assert resolve_mode(None) == "c3"
    assert resolve_mode("b0") == "b0"          # explicit beats environment


def test_needs_timestamp_without_fixed_fps():
    with pytest.raises(ValueError):
        StabilitySelector("c3").process(textured())


# --- behaviour ------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["c3", "c5"])
def test_commits_sharpest_frame_of_steady_run_at_stable_end(mode):
    fps = 25
    sharp, soft = textured(1), blur(textured(1), 5)
    sel = StabilitySelector(mode, fixed_fps=fps)
    out = []
    # a steady view whose frames alternate between soft and one sharp frame
    for i in range(40):
        out += sel.process(sharp if i == 17 else soft)
    assert out == [] and sel.stable                      # nothing committed while still steady
    # the view slides away (global motion, as in a page being moved): the run
    # ends and the sharpest frame is committed
    for i in range(4):
        out += sel.process(np.roll(sharp, 12 * (i + 1), axis=1))
    assert len(out) == 1
    assert out[0]["frame_index"] == 17 and out[0]["reason"] == "stable_end"
    assert out[0]["decision_index"] > 39
    assert out[0]["frame"] is sharp


def test_characterisation_hard_cut_is_seen_by_frame_difference_not_by_flow():
    """Measured on synthetic frames: an instantaneous swap to unrelated content
    gives a frame-difference signal of ~65-70 (threshold 14.3) but a median
    Farneback flow of only ~3.6 px (threshold 4.4). C5 therefore only ends a
    stable run when there is coherent motion (a page being turned or moved);
    C3 also ends it on a cut. Documented here, not asserted as desirable."""
    base = textured(1)
    for mode, ends_run in (("c3", True), ("c5", False)):
        sel = StabilitySelector(mode, fixed_fps=25)
        for _ in range(30):
            sel.process(base)
        assert sel.stable
        out = []
        for i in range(4):
            out += sel.process(textured(50 + i))
        assert (len(out) == 1) is ends_run, mode


@pytest.mark.parametrize("mode", ["c3", "c5"])
def test_max_segment_bounds_the_wait_and_dup_gate_suppresses_repeats(mode):
    fps = 25
    frame = textured(2)
    sel = StabilitySelector(mode, fixed_fps=fps)
    commits = []
    for _ in range(int(4.0 * fps) * 3 + 10):             # ~12 s of an unchanging view
        commits += sel.process(frame)
    assert len(commits) == 1 and commits[0]["reason"] == "max_segment"
    assert commits[0]["decision_index"] < int(4.0 * fps) + 5
    assert sel.stats["dup_suppressed"] >= 2               # later max-segment runs are identical to the commit
    assert sel.flush() == []                              # a duplicate run is not committed on flush either


@pytest.mark.parametrize("mode", ["c3", "c5"])
def test_flush_commits_pending_run_once(mode):
    sel = StabilitySelector(mode, fixed_fps=25)
    frame = textured(3)
    for _ in range(30):
        assert sel.process(frame) == []
    out = sel.flush()
    assert len(out) == 1 and out[0]["reason"] == "flush"
    assert sel.flush() == []


@pytest.mark.parametrize("mode", ["c3", "c5"])
def test_constantly_changing_view_never_commits(mode):
    sel = StabilitySelector(mode, fixed_fps=25)
    out = []
    for i in range(60):
        out += sel.process(textured(i))
    assert out == [] and sel.flush() == []


def test_fingerprint_matches_production_definition():
    f = textured(4, (720, 1280))
    ref = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    assert np.array_equal(fingerprint(f), ref)
    assert DEDUP_THRESHOLD == 5.0


def test_live_timestamp_mode_matches_fixed_fps_mode_on_uniform_timing():
    fps = 20.0
    frames = [textured(7)] * 40 + [textured(8 + i) for i in range(3)]
    a, b = StabilitySelector("c3", fixed_fps=fps), StabilitySelector("c3")
    out_a, out_b = [], []
    for i, f in enumerate(frames):
        out_a += a.process(f)
        out_b += b.process(f, i / fps)
    assert [c["frame_index"] for c in out_a] == [c["frame_index"] for c in out_b]
    assert len(out_a) == 1


# --- parity with the frozen benchmark ---------------------------------------------

def _load_videos():
    import yaml
    reg = yaml.safe_load(open(BENCH / "config" / "videos.yaml", encoding="utf-8"))["videos"]
    return {v["id"]: v for v in reg}


def _frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                return
            yield fr
    finally:
        cap.release()


@pytest.mark.parametrize("vid", ["r1_steady_doc", "r2_multi_doc"])
@pytest.mark.parametrize("mode,bench_id", [("c3", "stability_best"), ("c5", "optical_flow")])
def test_parity_with_benchmark_commits(vid, mode, bench_id):
    if not (BENCH / "config" / "videos.yaml").exists():
        pytest.skip("the frozen frame-selection benchmark is not part of this repository")
    videos = _load_videos()
    path = REPO / videos[vid]["path"]
    raw_path = BENCH / "results" / "raw" / vid / f"{bench_id}.json"
    manifest = json.load(open(BENCH / "data" / "videos" / f"{vid}_manifest.json", encoding="utf-8"))
    if not path.exists() or not raw_path.exists():
        pytest.skip("recording or frozen benchmark commits not available")
    cv2.setNumThreads(1)                                  # the benchmark replay ran single-threaded
    expected = [(c["decision_frame"], c["frames"][0]) for c in json.load(open(raw_path))["commits"]]
    sel = StabilitySelector(mode, fixed_fps=manifest["effective_fps"])
    got = []
    for fr in _frames(path):
        got += sel.process(fr)
    got += sel.flush()
    assert [(c["decision_index"], c["frame_index"]) for c in got] == expected
