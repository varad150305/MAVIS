"""Phase 4.6 gates (2026-10-29), product-only.

Gate A - finalization word loss, saved session 20261029-112452_m2_mic,
utterance 17: confirmed "my normal piece in a" (boundary 69.36 s); the
final window was trimmed to 68.3 s, the product prompt kept only "my", and
the final decode collapsed to "cell phone." with both words before the
boundary, so nothing more was committed and "quiet room using my usual
microphone" was lost. The saved WAV is 16-bit; the live decode saw float32
samples, so the window is re-decoded with a fixed sub-LSB perturbation
(seed 2) that reproduces "cell phone." verbatim.

Gate B - onset-based latency metrics in every session summary.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from product import live_pipeline as lp
from product.tests.test_quality_fixes import _Controller, hyp

REPO = Path(__file__).resolve().parents[2]
SESSION = REPO / "product_outputs" / "sessions" / "20261029-112452_m2_mic"
OFFSET = 2337560.7176506002  # capture clock - WAV seconds (product.tools.script_score.wav_offset)
W40_START, W40_FRAMES, BOUNDARY = 2337629.0173824, 40, 2337630.0776
# U17's two non-final decodes (absolute capture-clock word spans, from asr_hypotheses.jsonl)
W38 = [("my", 67.70, 68.12), ("normal", 68.12, 68.56), ("piece", 68.56, 68.98), ("in", 68.98, 69.26),
       ("a", 69.26, 69.36), ("quadro", 69.36, 69.68)]
W39 = [("My", 67.70, 68.12), ("normal", 68.12, 68.56), ("piece", 68.56, 68.96), ("in", 68.96, 69.26),
       ("a", 69.26, 69.36), ("quadrum", 69.36, 70.10), ("using", 70.10, 70.54), ("my", 70.54, 70.90),
       ("usual", 70.90, 71.26), ("micromancer.", 71.26, 71.68)]


class _Buffer:
    def __init__(self):
        self.state = None
        self.utterance_pre_roll = {}

    def trim_confirmed(self, t):
        pass


def _abs(words):
    return [(w, a + OFFSET, b + OFFSET) for w, a, b in words]


def _u17_hook(prompt_min_words):
    """Hook + real tracker after W38/W39 of utterance 17, buffer trimmed to the W40 window start."""

    from streaming.prefix_agreement import PrefixAgreementTracker

    hook = lp.StableUpdateHook(None, _Controller(), lambda: 0.0, 0.1, trim_keep_s=1.0,
                               boundary_repair=lp.BoundaryRepair(), prompt_min_words=prompt_min_words)
    buf = hook.build_streaming_buffer(lambda c, l: _Buffer())(None, None)
    tracker = hook.build_prefix_agreement_tracker(lambda c, l: PrefixAgreementTracker(agreement_depth=2))(None, None)
    tracker.process(hyp(17, 38, _abs(W38), audio_start=67.70 + OFFSET, audio_end=69.70 + OFFSET))
    tracker.process(hyp(17, 39, _abs(W39), audio_start=67.70 + OFFSET, audio_end=71.70 + OFFSET))
    assert tracker.committed_text_for(17) == "my normal piece in a"
    buf.state = SimpleNamespace(utterance_id=17, context_audio=[SimpleNamespace(capture_timestamp=W40_START)],
                                new_audio_since_last_asr=[])
    return hook, tracker


def test_trimmed_prompt_fragment_is_dropped_below_prompt_min_words():
    hook, tracker = _u17_hook(prompt_min_words=0)
    assert tracker.confirmed_prompt_text(17) == "my"          # phase-3 behaviour: a 1-word fragment
    hook, tracker = _u17_hook(prompt_min_words=3)
    assert tracker.confirmed_prompt_text(17) is None
    assert tracker.confirmed_prompt_text(17) is None             # requested every frame ...
    hook._record(tracker, hyp(17, 40, [], audio_start=W40_START, audio_end=W40_START + 4.0, is_final=True))
    assert hook.prompts_dropped == 1                             # ... counted once per decoded window


def test_prompt_min_words_keeps_long_prompts_and_untrimmed_prompts():
    from streaming.prefix_agreement import PrefixAgreementTracker

    hook = lp.StableUpdateHook(None, _Controller(), lambda: 0.0, 0.1, trim_keep_s=1.0, prompt_min_words=3)
    buf = hook.build_streaming_buffer(lambda c, l: _Buffer())(None, None)
    tracker = hook.build_prefix_agreement_tracker(lambda c, l: PrefixAgreementTracker(agreement_depth=2))(None, None)
    words = [(w, i * 0.4, i * 0.4 + 0.4) for i, w in enumerate("one two three four five six seven".split())]
    tracker.process(hyp(1, 1, words, audio_end=3.0))
    tracker.process(hyp(1, 2, words + [("eight", 2.8, 3.2)], audio_end=3.2))
    # window starts after the confirmed words: nothing audible, full prompt kept
    buf.state = SimpleNamespace(utterance_id=1, context_audio=[SimpleNamespace(capture_timestamp=10.0)],
                                new_audio_since_last_asr=[])
    assert tracker.confirmed_prompt_text(1) == tracker.committed_text_for(1)
    # window starts 1 s before the boundary: 3+ words remain -> the trimmed prompt is used
    boundary = tracker.confirmed_boundary_time(1)
    buf.state = SimpleNamespace(utterance_id=1, context_audio=[SimpleNamespace(capture_timestamp=boundary - 1.0)],
                                new_audio_since_last_asr=[])
    prompt = tracker.confirmed_prompt_text(1)
    assert prompt and len(prompt.split()) >= 3 and hook.prompts_dropped == 0


@pytest.fixture(scope="module")
def w40_audio_and_engine():
    if not (SESSION / "input.wav").exists():
        pytest.skip("saved session audio not present")
    import numpy as np

    from asr.whisper_engine import WhisperEngine
    from product import runtime
    from product.sources import read_wav_mono_16k

    runtime.prefer_offline_models()
    try:
        engine = WhisperEngine("tiny", "cpu", "int8", language="en", beam_size=1, temperature=0.0, cpu_threads=8,
                               vad_filter=False, word_timestamps=True)
    except Exception as exc:
        pytest.skip(f"Whisper tiny unavailable: {exc}")
    engine.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    x, _ = read_wav_mono_16k(SESSION / "input.wav")
    k0 = round((W40_START - OFFSET) / 0.1)
    a = x[k0 * 1600:(k0 + W40_FRAMES) * 1600]
    noise = (np.random.default_rng(2).random(a.shape).astype(np.float32) - 0.5) / 32768.0
    return (a + noise).astype(np.float32), engine


def _finalize_with_decode(prompt_min_words, audio, engine):
    """Run the product path for the final window: prompt from the hook,
    real decode, boundary repair, the unchanged tracker, finalization."""

    hook, tracker = _u17_hook(prompt_min_words)
    prompt = tracker.confirmed_prompt_text(17)
    r = engine.transcribe(audio, 16000, initial_prompt=prompt)
    words = [(w.word, W40_START + w.start, W40_START + w.end) for w in r.words]
    final = hyp(17, 40, words, audio_start=W40_START, audio_end=W40_START + W40_FRAMES * 0.1, is_final=True)
    tracker.process(final)
    tracker.finalize_utterance(17)
    return prompt, r.text.strip(), tracker.committed_text_for(17), hook


def test_saved_window_collapse_loses_words_with_the_phase3_prompt(w40_audio_and_engine):
    prompt, text, committed, hook = _finalize_with_decode(0, *w40_audio_and_engine)
    assert prompt == "my" and text == "cell phone."
    assert committed == "my normal piece in a"                     # the live loss, reproduced


def test_saved_window_keeps_the_spoken_tail_with_prompt_min_words(w40_audio_and_engine):
    prompt, text, committed, hook = _finalize_with_decode(3, *w40_audio_and_engine)
    assert prompt is None
    assert committed.endswith("quiet room using my usual microphone")
    assert committed.split().count("my") == 2                       # no duplicated words at the boundary


def test_final_decode_without_new_words_and_uncovered_audio_is_recorded():
    hook, tracker = _u17_hook(prompt_min_words=0)
    wrapped = tracker.process
    collapsed = hyp(17, 40, [("cell", W40_START, W40_START + 0.38), ("phone.", W40_START + 0.38, W40_START + 1.32)],
                    audio_start=W40_START, audio_end=W40_START + 4.0, is_final=True)
    wrapped(collapsed)
    assert len(hook.final_coverage_gaps) == 1
    gap = hook.final_coverage_gaps[0]
    assert gap["window_id"] == 40 and gap["uncovered_s"] == pytest.approx(2.68, abs=0.01)


# -- Gate B ------------------------------------------------------------------


def test_dist_n_keeps_missing_values_and_limits_percentiles_by_sample_size():
    d = lp._dist_n([1.0, None, 2.0, 3.0])
    assert d["n"] == 3 and d["missing"] == 1 and d["p50"] == 2.0
    assert d["p90"] is None and d["p95"] is None and d["insufficient_n_for"] == ["p90", "p95"]
    d = lp._dist_n([float(i) for i in range(20)])
    assert d["p90"] is not None and d["p95"] is not None and d["insufficient_n_for"] is None
    assert lp._dist_n([])["n"] == 0 and lp._dist_n([])["p50"] is None


def test_display_records_render_time_separately_from_event_time():
    t = [5.0]
    display = lp.Display(io.StringIO(), None, lambda: t[0])
    seg = lp.Segment(segment_id=1, opened_s=4.0, rate_origin_s=3.0)
    seg.words = ["hello"]
    t[0] = 4.25
    display.english(seg, ["hello"], 4.0, True, utterance_id=3)
    rec = display.records[-1]
    assert rec["t_s"] == 4.0 and rec["render_s"] == 4.25 and rec["utterance_id"] == 3


def test_provisional_english_is_display_only():
    out = io.StringIO()
    display = lp.Display(out, None, lambda: 1.0)
    display.provisional(7, "my normal piece in a quadro", 1.0)
    assert "(provisional, may change)" in out.getvalue()
    assert display.records[-1]["type"] == "en_provisional" and display.finals == {}


def test_user_latency_metrics_measure_from_vad_onset_not_commit():
    clock = [0.0]
    display = lp.Display(None, None, lambda: clock[0])
    hook = SimpleNamespace(on_provisional=None, utterance_times={
        1: {"onset_s": 2.0, "first_text_hypothesis_s": 4.2, "first_commit_s": 6.2, "last_word_end_s": 6.0,
            "vad_end_s": 6.3}})
    seg = lp.Segment(segment_id=1, opened_s=6.2, rate_origin_s=2.0)
    seg.words, seg.utterance_ids, seg.closed_s = ["a", "b", "c"], [1], 7.0
    clock[0] = 6.3
    display.english(seg, ["a", "b", "c"], 6.2, True, utterance_id=1)
    display._write("draft", {"type": "hi_draft", "segment_id": 1, "completed_s": 6.9})
    clock[0] = 8.0
    display._write("final", {"type": "hi_final", "segment_id": 1, "t_s": 7.9})
    m = lp.user_latency_metrics(hook, [seg], display)
    assert m["onset_to_first_tentative_en_s"]["p50"] == pytest.approx(2.2)
    assert m["onset_to_first_committed_en_s"]["p50"] == pytest.approx(4.2)
    assert m["onset_to_first_committed_en_render_s"]["p50"] == pytest.approx(4.3)
    assert m["onset_to_first_hindi_draft_render_s"]["p50"] == pytest.approx(4.3)
    assert m["last_word_to_final_hindi_render_s"]["p50"] == pytest.approx(2.0)
    assert m["close_to_final_hindi_render_s"]["p50"] == pytest.approx(1.0)
    assert m["onset_to_first_provisional_en_render_s"] is None    # option off: not reported as 0
    assert "COMMIT time" in m["definitions"]["not_speech_latency"]


def test_product_config_sets_phase46_options():
    import yaml

    cfg = yaml.safe_load((REPO / "product" / "configs" / "live_m2_mt3c.yaml").read_text(encoding="utf-8"))
    assert cfg["product"]["prompt_min_words"] == 3
    assert cfg["streaming_buffer"]["min_new_audio_ms"] == 2000
