"""Phase 5: product session API, event contract, transcript model and
performance panel (standalone product; no MAVIS host or OCR exists yet)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from product import session as S
from product.metrics_panel import LiveMetrics
from product.ui_model import TranscriptModel

REPO = Path(__file__).resolve().parents[2]


# -- a fake pipeline with the run_session contract ---------------------------------


def fake_runner(script, *, block_until_stop=False, raise_exc=None, exit_code=0):
    """run_session(argv, console, event_sink, stop_event, ready_callback) stand-in
    that emits `script` (display records) and returns a summary."""

    def run(argv, console, *, event_sink, stop_event, ready_callback):
        ready_callback(1.25, {"on_ac_power": True})
        for rec in script:
            event_sink(dict(rec))
        if raise_exc is not None:
            raise raise_exc
        if block_until_stop:
            assert stop_event.wait(10)
        return exit_code, {"stop_reason": "stop requested" if stop_event.is_set() else "end", "argv": argv,
                           **({"error": "boom"} if exit_code else {})}
    return run


SCRIPT = [
    {"type": "utterance_timing", "utterance_id": 1, "t_s": 1.0, "onset_s": 1.0},
    {"type": "en_provisional", "utterance_id": 1, "t_s": 3.2, "text": "good morning every", "render_s": 3.2},
    {"type": "utterance_timing", "utterance_id": 1, "t_s": 3.2, "first_text_hypothesis_s": 3.2},
    {"type": "en_update", "segment_id": 1, "utterance_id": 1, "t_s": 5.2, "new_text": "good morning",
     "stable_text": "good morning", "utterance_final": False, "render_s": 5.3},
    {"type": "hi_draft", "segment_id": 1, "invocation_id": 1, "source_text": "good morning", "source_word_count": 2,
     "target_text": "DRAFT-1", "completed_s": 6.0, "render_s": 6.0, "inference_ms": 300.0},
    {"type": "utterance_timing", "utterance_id": 1, "t_s": 6.5, "last_word_end_s": 6.4, "vad_end_s": 6.6},
    {"type": "en_update", "segment_id": 1, "utterance_id": 1, "t_s": 7.0, "new_text": "everyone",
     "stable_text": "good morning everyone", "utterance_final": True, "render_s": 7.0},
    {"type": "segment_closed", "segment_id": 1, "t_s": 7.2, "reason": "sentence end", "words": 3,
     "utterance_ids": [1], "final_source": "good morning everyone", "render_s": 7.2},
    {"type": "hi_final", "segment_id": 1, "t_s": 8.0, "target_text": "FINAL-1", "kind": "final_mt", "render_s": 8.0},
    {"type": "hi_draft_after_final", "segment_id": 1, "invocation_id": 2, "source_text": "good morning everyone",
     "source_word_count": 3, "target_text": "LATE-DRAFT", "completed_s": 8.3},
    {"type": "asr_decode", "utterance_id": 1, "window_id": 1, "is_final": False, "success": True,
     "inference_ms": 480.0, "rtf": 0.24, "queue_wait_ms": 0.5, "completion_lag_s": 0.5, "t_s": 3.2},
    {"type": "resources", "t_s": 4.0, "cores_busy": 2.1, "working_set_mb": 1600.0, "dropped_frames": 0},
]


def run_session(script, **kw):
    s = S.TranslationSession(S.SessionOptions(wav="x.wav"), runner=fake_runner(script, **kw))
    s.start()
    return s


# -- lifecycle ------------------------------------------------------------------


def test_session_lifecycle_and_ordered_events():
    s = run_session(SCRIPT)
    events = list(s.iter_events(timeout_s=0.05))
    assert s.state is S.SessionState.STOPPED and not s.running
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert {e["session_id"] for e in events} == {s.session_id} and all(e["schema"] == S.SCHEMA for e in events)
    states = [e["state"] for e in events if e["type"] == "session_state"]
    assert states == ["starting", "ready", "stopped"]
    types = [e["type"] for e in events]
    assert types.index("en_committed") < types.index("hi_draft") < types.index("segment_closed") \
        < types.index("hi_final") < types.index("session_summary")
    assert "hi_draft_late" in types and "en_update" not in types       # transcript names are mapped


def test_stop_before_and_after_ready_and_repeated_stop():
    s = run_session(SCRIPT, block_until_stop=True)
    time.sleep(0.05)
    assert s.running
    assert s.stop(timeout_s=5) and s.stop(timeout_s=5)                  # idempotent
    assert s.state is S.SessionState.STOPPED and s.summary["stop_reason"] == "stop requested"
    states = [e["state"] for e in s.poll_events() if e["type"] == "session_state"]
    assert states[-2:] == ["stopping", "stopped"]
    fresh = S.TranslationSession(S.SessionOptions())
    assert fresh.stop() and fresh.state is S.SessionState.STOPPED       # never started


def test_one_session_per_process_and_lock_released_after_stop_and_failure():
    s1 = run_session([], block_until_stop=True)
    with pytest.raises(S.SessionBusy):
        S.TranslationSession(S.SessionOptions(), runner=fake_runner([])).start()
    s1.stop(timeout_s=5)
    s2 = run_session([], raise_exc=RuntimeError("model missing"))
    list(s2.iter_events(timeout_s=0.05))
    assert s2.state is S.SessionState.FAILED and "model missing" in s2.error
    s3 = run_session([])                                                # lock was released by the failure
    list(s3.iter_events(timeout_s=0.05))
    assert s3.state is S.SessionState.STOPPED
    with pytest.raises(RuntimeError):
        s3.start()                                                      # a session object starts once


def test_pipeline_exception_and_nonzero_exit_become_failed_events():
    s = run_session(SCRIPT[:3], raise_exc=ValueError("bad frame"))
    events = list(s.iter_events(timeout_s=0.05))
    err = [e for e in events if e["type"] == "error"]
    assert err and "ValueError: bad frame" in err[0]["message"] and "traceback" in err[0]
    assert events[-1]["type"] == "session_state" and events[-1]["state"] == "failed"
    s = run_session([], exit_code=2)
    events = list(s.iter_events(timeout_s=0.05))
    assert s.state is S.SessionState.FAILED and events[-1]["error"] == "boom"


def test_host_callback_errors_do_not_break_the_session():
    seen = []

    def cb(e):
        seen.append(e["type"])
        raise RuntimeError("host bug")

    s = S.TranslationSession(S.SessionOptions(), on_event=cb, runner=fake_runner(SCRIPT))
    s.start()
    list(s.iter_events(timeout_s=0.05))
    assert s.state is S.SessionState.STOPPED and "hi_final" in seen


def test_options_map_to_cli_arguments():
    o = S.SessionOptions(wav="a.wav", device=3, save_audio=True, show_provisional_english=True, session_name="ui",
                         overrides=["asr.model_size=base"], duration_s=5)
    argv = o.argv()
    assert argv[:2] == ["--wav", "a.wav"] and "--save-audio" in argv
    assert "product.display_provisional_en=true" in argv and "asr.model_size=base" in argv
    assert "product.display_provisional_en=true" not in S.SessionOptions().argv()   # provisional off by default


# -- feature flag ------------------------------------------------------------------


def test_host_feature_flag_defaults_off(tmp_path):
    assert S.feature_enabled() is False
    with pytest.raises(S.FeatureDisabled):
        S.create_host_session()
    cfg = (REPO / "product" / "configs" / "live_m2_mt3c.yaml").read_text(encoding="utf-8")
    on = tmp_path / "on.yaml"
    on.write_text(cfg.replace("live_translation_enabled: false", "live_translation_enabled: true"), encoding="utf-8")
    assert S.feature_enabled(str(on)) is True
    assert isinstance(S.create_host_session(S.SessionOptions(config=str(on))), S.TranslationSession)


# -- transcript model ----------------------------------------------------------------


def _events(script, session_id="A"):
    return [{"schema": S.SCHEMA, "session_id": session_id, "seq": i + 1, "t_s": r.get("t_s"),
             "type": S.RECORD_TYPES[r["type"]], **{k: v for k, v in r.items() if k not in ("type", "t_s")}}
            for i, r in enumerate(script)]


def test_model_separates_provisional_committed_draft_and_final():
    m = TranscriptModel("A")
    ev = _events(SCRIPT)
    for e in ev[:3]:
        m.apply(e)
    assert m.provisional == {1: "good morning every"} and m.segments == {}
    m.apply(ev[3])                                               # committed words clear the provisional line
    assert m.provisional == {} and m.segments[1].english == ["good", "morning"]
    m.apply(ev[4])
    assert m.segments[1].hindi_state == "draft" and m.segments[1].hindi_draft == "DRAFT-1"
    for e in ev[5:]:
        m.apply(e)
    seg = m.segments[1]
    assert seg.hindi_final == "FINAL-1" and seg.hindi_state == "final" and seg.closed
    assert m.ignored_late_drafts == 1 and seg.english == ["good", "morning", "everyone"]


def test_model_never_overwrites_a_final_and_flags_unknown_segments():
    m = TranscriptModel("A")
    for e in _events(SCRIPT):
        m.apply(e)
    stray = {"schema": S.SCHEMA, "session_id": "A", "seq": 100, "type": "hi_draft", "segment_id": 1,
             "target_text": "STRAY"}
    second = {"schema": S.SCHEMA, "session_id": "A", "seq": 101, "type": "hi_final", "segment_id": 1,
              "target_text": "OTHER"}
    orphan = {"schema": S.SCHEMA, "session_id": "A", "seq": 102, "type": "hi_final", "segment_id": 9,
              "target_text": "X"}
    assert not m.apply(stray) and not m.apply(second) and not m.apply(orphan)
    assert m.segments[1].hindi_final == "FINAL-1" and 9 not in m.segments
    assert any("unknown segment 9" in p for p in m.problems) and any("second final" in p for p in m.problems)


def test_model_drops_stale_and_duplicate_events():
    m = TranscriptModel("NEW")
    old = _events(SCRIPT, session_id="OLD")
    assert not any(m.apply(e) for e in old) and m.ignored_stale == len(old)
    new = _events(SCRIPT, session_id="NEW")
    m.apply(new[3])
    assert not m.apply(new[3]) and not m.apply(new[0])               # repeated / out-of-order seq
    assert m.segments[1].english == ["good", "morning"]


def test_segments_stay_associated_across_interleaved_events():
    recs = []
    for sid, uid in ((1, 1), (2, 2)):
        recs.append({"type": "en_update", "segment_id": sid, "utterance_id": uid, "new_text": f"s{sid}",
                     "stable_text": f"s{sid}", "utterance_final": True, "t_s": float(sid)})
    recs += [{"type": "hi_draft", "segment_id": 2, "invocation_id": 1, "target_text": "D2"},
             {"type": "segment_closed", "segment_id": 1, "utterance_ids": [1]},
             {"type": "hi_final", "segment_id": 1, "target_text": "F1", "kind": "final_mt"},
             {"type": "segment_closed", "segment_id": 2, "utterance_ids": [2]},
             {"type": "hi_final", "segment_id": 2, "target_text": "F2", "kind": "final_mt"}]
    m = TranscriptModel("A")
    for e in _events(recs):
        m.apply(e)
    assert [(s.segment_id, s.english, s.hindi_final) for s in m.ordered_segments()] == \
        [(1, ["s1"], "F1"), (2, ["s2"], "F2")]


# -- performance panel ---------------------------------------------------------------


def _panel(script, provisional=False):
    lm = LiveMetrics("A")
    lm.apply({"session_id": "A", "type": "session_state", "state": "starting",
              "info": {"show_provisional_english": provisional}})
    lm.apply({"session_id": "A", "type": "session_state", "state": "ready", "ready_in_s": 8.1,
              "info": {"on_ac_power": True}})
    for e in _events(script):
        lm.apply(e)
    return lm, {r.key: r for r in lm.rows()}


def test_panel_measures_from_speech_onset_with_sample_counts():
    lm, rows = _panel(SCRIPT)
    assert rows["ready"].value == "8.1 s"
    assert rows["first_en"].value.startswith("median 4.30 s") and "n=1" in rows["first_en"].value  # 5.3 - 1.0
    assert rows["first_en_event"].value.startswith("median 4.20 s")                                 # 5.2 - 1.0
    assert "p90/p95 need more samples" in rows["first_en"].value and rows["first_en"].state == "partial"
    assert rows["first_draft"].value.startswith("median 5.00 s")                                    # 6.0 - 1.0
    assert rows["final"].value.startswith("median 1.60 s")                                          # 8.0 - 6.4
    assert rows["close_final"].value.startswith("median 0.80 s")
    assert "480 ms" in rows["asr"].value and "RTF median 0.24" in rows["asr"].value
    assert rows["health"].value == "AC / 0 / 0 / 0" and "not" in rows["health"].definition
    assert "first_prov" not in rows                                  # provisional off: row absent, not 0


def test_panel_shows_collecting_instead_of_zero_and_no_live_wer():
    lm, rows = _panel([])
    for key in ("first_en", "first_draft", "final", "asr", "cpu"):
        assert rows[key].value == "collecting..." and rows[key].state == "collecting"
    assert rows["wer"].value == "not available live" and rows["wer"].state == "n/a"
    assert "accuracy" not in " ".join(r.label.lower() for r in rows.values())
    assert "stable_token_ratio" not in " ".join(r.definition for r in rows.values())


def test_panel_provisional_row_is_labelled_and_reference_wer_is_offline(tmp_path):
    lm, rows = _panel(SCRIPT, provisional=True)
    assert "may be wrong" in rows["first_prov"].label and rows["first_prov"].value.startswith("median 2.20 s")
    report = tmp_path / "score.json"
    report.write_text('{"recording": "mic_20261029", "limitation": "not annotated", "results": '
                      '[{"kind": "session", "wer": 0.084, "S": 9, "D": 1, "I": 3, "ref_words": 154}]}',
                      encoding="utf-8")
    lm.load_reference_score(report)
    row = {r.key: r for r in lm.rows()}["wer"]
    assert "offline" in row.label and row.value.startswith("8.4%") and "not independently annotated" in row.definition


def test_panel_uses_the_session_summary_after_stop():
    lm, _ = _panel(SCRIPT)
    lm.apply({"session_id": "A", "type": "session_summary", "summary": {"user_latency": {
        "onset_to_first_committed_en_render_s": {"n": 25, "missing": 0, "p50": 4.0, "p90": 4.5, "p95": 4.6,
                                                 "max": 5.0, "insufficient_n_for": None}}}})
    row = {r.key: r for r in lm.rows()}["first_en"]
    assert row.value.startswith("median 4.00 s, p90 4.50 s") and row.state == "ok"


def test_panel_and_summary_share_one_latency_implementation():
    from product import live_pipeline as lp

    lm, _ = _panel(SCRIPT)
    assert lm.latency()["definitions"] is lp.USER_LATENCY_DEFINITIONS


# -- the real pipeline through the session API ------------------------------------


def test_real_pipeline_through_session_api_on_a_saved_recording(tmp_path):
    """Short 1x replay of a saved recording through run_session: structured
    events only, one final per closed segment, clean stop, Phase 4.6 settings."""

    wav = REPO / "product_outputs" / "diagnostics" / "inputs" / "mic_20261028b.wav"
    if not wav.exists():
        pytest.skip("saved recording not present")
    opts = S.SessionOptions(wav=str(wav), duration_s=18, overrides=[f"product.output_root={tmp_path.as_posix()}"])
    s = S.TranslationSession(opts)
    try:
        s.start()
    except S.SessionBusy:
        pytest.skip("another session is running")
    events = list(s.iter_events(timeout_s=0.5))
    if s.state is S.SessionState.FAILED and "Whisper" in (s.error or ""):
        pytest.skip(f"models unavailable: {s.error}")
    assert s.state is S.SessionState.STOPPED, s.error
    m = TranscriptModel(s.session_id)
    for e in events:
        m.apply(e)
    assert m.problems == [] and m.errors == []
    closed = [seg for seg in m.ordered_segments() if seg.closed]
    assert closed and all(seg.hindi_final for seg in closed)
    assert any("good morning" in " ".join(seg.english) for seg in closed)
    summary = s.summary
    assert summary["prompt_min_words"] == 3 and summary["display_provisional_en"] is False
    assert summary["user_latency"]["onset_to_first_committed_en_s"]["n"] >= 1
    types = {e["type"] for e in events}
    assert {"en_committed", "hi_final", "asr_decode", "utterance_timing", "resources"} <= types
    assert "en_provisional" not in types


# -- the Tk view (hidden window) ---------------------------------------------------


def test_ui_renders_draft_final_provisional_and_panel_labels():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    root.withdraw()
    try:
        from product.ui.app import TranslatorApp

        app = TranslatorApp(root)
        app.model, app.metrics = TranscriptModel("A"), LiveMetrics("A")
        app.metrics.apply({"session_id": "A", "type": "session_state", "state": "starting",
                           "info": {"show_provisional_english": True}})
        for e in _events(SCRIPT[:5]):                     # provisional, committed, first draft
            app.model.apply(e)
            app.metrics.apply(e)
        app.model.apply({"schema": S.SCHEMA, "session_id": "A", "seq": 50, "type": "en_provisional",
                         "utterance_id": 2, "text": "and wel"})
        app._render_transcript()
        text = app.text.get("1.0", "end")
        assert "good morning" in text and "DRAFT – may change \nDRAFT-1" in text      # phase 9 labels
        assert "provisional - may change  and wel" in text and "FINAL" not in text
        for e in _events(SCRIPT)[5:]:
            app.model.apply({**e, "seq": e["seq"] + 100})
        app._render_transcript()
        text = app.text.get("1.0", "end")
        assert " FINAL \nFINAL-1" in text and "DRAFT-1" not in text and "LATE-DRAFT" not in text
        assert "may change" not in text.split("FINAL-1")[0].split("good morning")[1]   # a final is never a draft
        app._render_panel()
        labels = [w.cget("text") for w in app.panel.winfo_children()]
        assert "Speech -> first English (printed)" in labels and "Reference WER" in labels
        assert "not available live" in labels
    finally:
        root.destroy()
