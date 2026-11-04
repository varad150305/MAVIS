"""Product-only regression tests for the 2026-10-28 live-quality work.

Covers the commit-boundary repair (against the REAL, unmodified
PrefixAgreementTracker), Whisper prompt carry-over, config overrides,
input-level monitoring and the hypothesis record. Model-free and fast.
"""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from product import live_pipeline as lp


def hyp(uid, window_id, words, audio_start=0.0, audio_end=None, is_final=False, generation=None):
    """ASRHypothesis with absolute word spans [(word, abs_start, abs_end)]."""

    from asr.whisper_engine import WordTiming
    from streaming.streaming_asr import ASRHypothesis

    end = audio_end if audio_end is not None else (words[-1][2] if words else audio_start + 1.0)
    timings = tuple(WordTiming(word=w, start=a - audio_start, end=b - audio_start) for w, a, b in words)
    return ASRHypothesis(
        hypothesis_id=window_id, utterance_id=uid, window_id=window_id, generation=generation or window_id,
        submitted_at=0.0, started_at=0.0, completed_at=0.0, queue_wait_ms=0.0, segment_to_transcript_ms=0.0,
        window_created_at=end, audio_start_time=audio_start, audio_end_time=end,
        window_duration_ms=(end - audio_start) * 1000.0, context_duration_ms=0.0,
        new_audio_duration_ms=(end - audio_start) * 1000.0, is_final=is_final, success=True, error=None,
        text=" ".join(w for w, _, _ in words), inference_time_ms=1.0, rtf=0.1, words=timings)


def spans(text, start, width=0.3):
    return [(w, start + i * width, start + (i + 1) * width) for i, w in enumerate(text.split())]


def run_tracker(hypotheses, repair=None):
    from streaming.prefix_agreement import PrefixAgreementTracker

    tracker = PrefixAgreementTracker(agreement_depth=2)
    for h in hypotheses:
        if repair is not None:
            h, _ = repair.apply(tracker, h)
        tracker.process(h)
    return tracker


def boundary_scenario(final_words):
    """Live session 2026-10-27, utterance 33: 'it will be easier for' was
    confirmed by two decodes; the buffer was trimmed at the end of 'for'
    (1.5 s); the final decode of the trimmed audio started 'people' a few
    tens of ms before that boundary."""

    first = spans("it will be easier for", 0.0)                          # for ends at 1.5
    return [hyp(33, 1, first, audio_end=2.0),
            hyp(33, 2, first + spans("people to", 1.55), audio_end=4.0),
            hyp(33, 3, final_words, audio_start=1.4, audio_end=4.3, is_final=True)]


def test_frozen_tracker_drops_a_word_that_starts_just_before_the_boundary():
    """Reproduces audit D7 on the unmodified tracker (documents the defect)."""

    tracker = run_tracker(boundary_scenario(spans("people to communicate", 1.46)))
    assert tracker.committed_text_for(33) == "it will be easier for to communicate"


def test_boundary_repair_keeps_that_word():
    tracker = run_tracker(boundary_scenario(spans("people to communicate", 1.46)), lp.BoundaryRepair())
    assert tracker.committed_text_for(33) == "it will be easier for people to communicate"


def test_boundary_repair_removes_a_repeated_committed_word_instead_of_duplicating_it():
    final = [("for", 1.25, 1.5)] + spans("people to communicate", 1.5)
    plain = run_tracker(boundary_scenario(final))
    repaired = run_tracker(boundary_scenario(final), lp.BoundaryRepair())
    assert repaired.committed_text_for(33) == "it will be easier for people to communicate"
    assert plain.committed_text_for(33) == "it will be easier for people to communicate"  # 'for' < boundary: dropped


def test_boundary_repair_does_not_touch_words_well_inside_confirmed_audio_or_first_decodes():
    from streaming.prefix_agreement import PrefixAgreementTracker

    tracker = PrefixAgreementTracker(agreement_depth=2)
    first = hyp(1, 1, spans("hello there", 0.0))
    same, info = lp.BoundaryRepair().apply(tracker, first)
    assert same is first and info is None  # nothing confirmed yet
    tracker.process(first)
    tracker.process(hyp(1, 2, spans("hello there friend", 0.0)))  # confirms "hello there" (boundary 0.6)
    later = hyp(1, 3, spans("hello there friend", 0.0))
    out, info = lp.BoundaryRepair(tolerance_s=0.2).apply(tracker, later)
    assert out is later and info is None  # "there" starts 0.3 s before the boundary: outside tolerance


def test_boundary_repair_leaves_a_genuine_new_word_equal_to_old_text_far_from_the_boundary():
    from streaming.prefix_agreement import PrefixAgreementTracker

    tracker = PrefixAgreementTracker(agreement_depth=2)
    tracker.process(hyp(1, 1, spans("go", 0.0)))
    tracker.process(hyp(1, 2, spans("go", 0.0)))           # confirms "go", boundary 0.3
    h = hyp(1, 3, [("go", 2.0, 2.3)])                       # a new "go" 1.7 s later
    out, info = lp.BoundaryRepair().apply(tracker, h)
    assert out is h and info is None


# -- prompt carry-over -----------------------------------------------------


class _Controller:
    def __init__(self):
        self.updates = []

    def submit_update(self, **kw):
        self.updates.append(kw)


def test_prompt_carry_prepends_previous_utterances_words():
    from streaming.prefix_agreement import PrefixAgreementTracker

    hook = lp.StableUpdateHook(None, _Controller(), lambda: 0.0, 0.1, prompt_carry_words=3)
    tracker = hook.build_prefix_agreement_tracker(lambda c, l: PrefixAgreementTracker(agreement_depth=2))(None, None)
    tracker.process(hyp(1, 1, spans("students should develop practical", 0.0), is_final=True))
    assert tracker.confirmed_prompt_text(2) == "should develop practical"     # new utterance: carried context
    tracker.process(hyp(2, 2, spans("skills", 2.0), is_final=False))
    tracker.process(hyp(2, 3, spans("skills alongside", 2.0), is_final=False))
    assert tracker.confirmed_prompt_text(2) == "should develop practical skills"
    assert hook.hypotheses[0]["words"][0][0] == "students"                      # every hypothesis recorded


def test_prompt_unchanged_without_carry_option():
    from streaming.prefix_agreement import PrefixAgreementTracker

    hook = lp.StableUpdateHook(None, _Controller(), lambda: 0.0, 0.1)
    tracker = hook.build_prefix_agreement_tracker(lambda c, l: PrefixAgreementTracker(agreement_depth=2))(None, None)
    tracker.process(hyp(1, 1, spans("good morning", 0.0), is_final=True))
    assert tracker.confirmed_prompt_text(2) is None
    assert tracker.confirmed_prompt_text(1) == "good morning"


# -- launcher config overrides and input monitor ----------------------------


def _config():
    from utils.config import load_config

    return load_config(Path(__file__).resolve().parents[1] / "configs" / "live_m2_mt3c.yaml")


def test_set_overrides_change_only_the_session_copy_and_reject_unknown_keys():
    from product.live_translate import apply_overrides

    cfg = _config()
    applied = apply_overrides(cfg, ["asr.model_size=base", "asr.language=en", "product.boundary_repair={enabled: true}"])
    assert cfg.get("asr", "model_size") == "base" and cfg.get("asr", "language") == "en"
    assert cfg.get("product", "boundary_repair") == {"enabled": True}
    assert applied["asr.model_size"] == "base"
    assert _config().get("asr", "model_size") == "tiny"  # file untouched
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["asr.modelsize=base"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["no_equals_sign"])


def test_product_config_asr_settings_are_the_frozen_research_values():
    """The product default must stay the frozen M2 operating point unless a
    documented product decision changes it (see the diagnostics report)."""

    from utils.config import load_config

    product = _config()
    research = load_config(Path(__file__).resolve().parents[2] / "evaluation" / "eval_configs" / "m2_preroll_cpu8.yaml")
    for key in ("model_size", "device", "compute_type", "beam_size", "temperature", "word_timestamps", "cpu_threads"):
        assert product.get("asr", key) == research.get("asr", key), key
    for key in ("speech_start_threshold", "speech_end_threshold", "minimum_speech_duration_ms",
                "minimum_silence_duration_ms"):
        assert product.get("vad", key) == research.get("vad", key), key
    assert product.get("streaming_buffer") == research.get("streaming_buffer")
    assert product.get("translation", "revision") == "75d7f7c9232b2891c7d65fe4ef635616c72be867"


def test_input_monitor_reports_levels_and_saves_exact_audio(tmp_path):
    from product.sources import InputMonitor, read_wav_mono_16k

    path = tmp_path / "input.wav"
    mon = InputMonitor(16000, path)
    frames = [np.full(1600, 0.1, dtype=np.float32), np.zeros(1600, dtype=np.float32), np.full(1600, 1.0, np.float32)]
    for f in frames:
        mon.add(f)
    mon.close()
    s = mon.summary()
    assert s["audio_s"] == pytest.approx(0.3)
    assert s["peak_dbfs"] == pytest.approx(0.0, abs=0.01)
    assert s["clipped_fraction"] == pytest.approx(1 / 3)
    data, rate = read_wav_mono_16k(path)
    assert rate == 16000 and len(data) == 4800
    assert np.allclose(data[:1600], 0.1, atol=1e-4)


def test_input_monitor_without_saving_writes_nothing(tmp_path):
    from product.sources import InputMonitor

    mon = InputMonitor(16000, None)
    mon.add(np.zeros(1600, dtype=np.float32))
    mon.close()
    assert mon.summary()["saved_wav"] is None and mon.summary()["peak_dbfs"] is None
    assert list(tmp_path.iterdir()) == []


# -- repetition guard ------------------------------------------------------


def test_collapse_repeats_handles_loops_but_keeps_ordinary_repetition():
    assert lp.collapse_repeats("s s s s s s s s".split()) == ["s"]
    assert lp.collapse_repeats("thank you thank you thank you thank you so much".split()) == "thank you so much".split()
    assert lp.collapse_repeats("no no no".split()) == "no no no".split()          # 3 < min_run
    assert lp.collapse_repeats("very very good".split()) == "very very good".split()


def test_repetition_guard_drops_loop_words_across_commits_of_one_utterance():
    guard = lp.RepetitionGuard()
    assert guard.filter(2, "the s s".split()) == "the s s".split()
    assert guard.filter(2, "s s s s s s".split()) == ["s"]                        # 4th copy onward dropped
    assert guard.filter(2, "according to".split()) == "according to".split()
    assert guard.filter(3, "s".split()) == ["s"]                                 # per utterance
    assert guard.dropped == 5


def test_hook_applies_repetition_guard_to_emitted_text_only():
    from streaming.prefix_agreement import PrefixAgreementTracker

    ctl = _Controller()
    hook = lp.StableUpdateHook(None, ctl, lambda: 0.0, 0.1, repetition_guard=lp.RepetitionGuard())
    tracker = hook.build_prefix_agreement_tracker(lambda c, l: PrefixAgreementTracker(agreement_depth=2))(None, None)
    tracker.process(hyp(1, 1, spans("go s s s s s s s s home", 0.0), is_final=True))
    assert ctl.updates[-1]["new_text"] == "go s s s home"
    assert tracker.committed_text_for(1) == "go s s s s s s s s home"           # tracker unchanged


# -- final-quality MT: display / association -------------------------------


def _build_fq(draft_delay=0.0, final_delay=0.0, final_fail=False):
    import time as _time

    from product.tests.test_live_pipeline import Clock, FakeEngine

    class FinalEngine(FakeEngine):
        def translate_request(self, request):
            self.sources.append(request.source_text)
            _time.sleep(self.delay_s)
            if final_fail:
                raise RuntimeError("synthetic final MT failure")

            class _Resp:
                target_text = f"FINAL({request.source_text})"
                inference_time_ms = 1.0

            return _Resp()

    clock = Clock()
    out = io.StringIO()
    display = lp.Display(out, None, clock)
    engine, final_engine = FakeEngine(delay_s=draft_delay), FinalEngine(delay_s=final_delay)
    worker = lp.MTWorker(engine, clock, display.translation)
    final_worker = lp.MTWorker(final_engine, clock, display.final_translation)
    from translation.scheduling import EagerPolicy

    controller = lp.SegmentController(EagerPolicy, lp.AsyncSegmentTranslator(worker, clock), clock,
                                      lp.SegmentRules(close_after_words=3, idle_close_s=5, hard_max_words=60),
                                      display, final_worker)
    for t in (worker, final_worker, controller):
        t.start()
    return clock, display, worker, final_worker, controller, engine, final_engine, out


def _send(controller, clock, uid, text, final=False):
    t = clock()
    controller.submit_update(utterance_id=uid, new_text=text, utterance_final=final, t=t, onset_s=max(0.0, t - 0.5))


def _finish(clock, controller, *workers):
    controller.stop(clock())
    for w in workers:
        assert w.shutdown(timeout_s=10)
    assert controller.error is None


def test_final_quality_translation_is_the_segment_final_for_the_full_source():
    import time as _time

    clock, display, worker, fworker, controller, engine, fengine, out = _build_fq()
    _send(controller, clock, 1, "please wait a")
    _send(controller, clock, 1, "moment", final=True)          # closes: 4 words >= 3
    _send(controller, clock, 2, "thank you", final=True)
    _time.sleep(0.2)
    _finish(clock, controller, worker, fworker)
    assert fengine.sources == ["please wait a moment", "thank you"]  # one call per segment, complete source
    assert display.finals == {1: "FINAL(please wait a moment)", 2: "FINAL(thank you)"}
    assert display.final_kinds == {1: "final_mt", 2: "final_mt"}
    metrics = lp.translation_metrics(controller.segments, display)
    assert metrics["final_mt"]["calls"] == 2 and metrics["final_mt"]["errors"] == 0


def test_a_draft_finishing_after_the_final_is_never_shown_as_current():
    import time as _time

    clock, display, worker, fworker, controller, engine, fengine, out = _build_fq(draft_delay=0.3)
    _send(controller, clock, 1, "good morning everyone", final=True)
    _time.sleep(0.6)
    _finish(clock, controller, worker, fworker)
    lines = [l for l in out.getvalue().splitlines() if "HI #1" in l]
    assert lines[-1].endswith("=| FINAL(good morning everyone)")
    assert not any("~|" in l for l in lines[lines.index(lines[-1]):])
    assert all("~|" not in l for l in lines)  # the slow draft completed after the final: recorded, not shown


def test_final_quality_failure_falls_back_to_the_draft_translation():
    import time as _time

    clock, display, worker, fworker, controller, engine, fengine, out = _build_fq(final_fail=True)
    _send(controller, clock, 1, "it will be easier", final=True)
    _time.sleep(0.2)
    _finish(clock, controller, worker, fworker)
    assert display.finals[1] == "HI(it will be easier)"
    assert display.final_kinds[1] == "draft_fallback"
    assert "final-quality MT failed" in out.getvalue()


def test_finals_belong_to_their_own_segment_when_segments_close_quickly():
    import time as _time

    clock, display, worker, fworker, controller, engine, fengine, out = _build_fq(final_delay=0.05)
    sources = [f"sentence number {i} here" for i in range(5)]
    for i, s in enumerate(sources):
        _send(controller, clock, i + 1, s, final=True)
    _time.sleep(0.6)
    _finish(clock, controller, worker, fworker)
    assert display.finals == {i + 1: f"FINAL({s})" for i, s in enumerate(sources)}


# -- real models (skipped when not available locally) ----------------------


def _nllb_available():
    from product.final_mt import DEFAULT_MODEL_DIR

    return (DEFAULT_MODEL_DIR / "model.bin").exists()


@pytest.mark.skipif(not _nllb_available(), reason="converted NLLB model not present")
def test_nllb_final_translator_translates_clean_english_faithfully():
    from product import runtime
    from product.final_mt import NllbCt2Translator

    runtime.prefer_offline_models()
    mt = NllbCt2Translator(threads=4)
    out = mt.translate("artificial intelligence is helping people solve complex problems more efficiently")
    assert "कृत्रिम" in out and "समस्या" in out
    assert mt.translate("") == ""
    assert mt.translate("thank you") == mt.translate("thank you")  # deterministic


def test_draft_mt_is_deterministic_and_stateless_for_the_recorded_bible_source():
    """The 'Bible' draft of 2026-10-27 is opus-mt's own output for that
    (misrecognized) source - same input, same output, in any order."""

    from product import runtime
    from product.tools.mt_probe import load_engine

    runtime.prefer_offline_models()
    engine = load_engine(threads=4)
    src = "real time translation can make communication in the earth between people speak the friend languages"
    first = engine.translate(src).translated_text
    engine.translate("good morning today i want to discuss the importance")
    assert engine.translate(src).translated_text == first
    assert "बाइबल" in first
