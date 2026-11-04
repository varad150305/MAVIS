"""Product-only regression tests for the phase-3 fixes (saved session
product_outputs/sessions/20261028-194812_m2_mic):

* context retention at commit boundaries (`trim_keep_s`) and the matching
  Whisper prompt (confirmed words still audible are omitted);
* sentence-aware segment closing (`SegmentRules.sentence_gap_s`).

The model test re-decodes the exact live window that turned "helps people
solve" into "health people all" (skipped when the recording or the cached
Whisper tiny model is absent). Everything else is model-free.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from product import live_pipeline as lp
from product.tests.test_live_pipeline import build, finish, update
from product.tests.test_quality_fixes import _Controller, hyp, spans

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION = REPO_ROOT / "product_outputs" / "sessions" / "20261028-194812_m2_mic"
SESSION_OFFSET = 2281367.9875  # WAV seconds = session clock - offset (see product/tools/window_replay.py)


# -- context retention -----------------------------------------------------


class _Buffer:
    def __init__(self):
        self.trimmed_at = []
        self.state = None
        self.utterance_pre_roll = {}

    def trim_confirmed(self, t):
        self.trimmed_at.append(t)


def test_trim_keep_moves_the_trim_point_earlier_and_zero_keeps_research_behaviour():
    kept = lp.StableUpdateHook(None, _Controller(), lambda: 0.0, 0.1, trim_keep_s=1.0)
    buf = kept.build_streaming_buffer(lambda c, l: _Buffer())(None, None)
    buf.trim_confirmed(10.0)
    assert buf.trimmed_at == [9.0] and kept.trims == 1

    plain = lp.StableUpdateHook(None, _Controller(), lambda: 0.0, 0.1)
    buf = plain.build_streaming_buffer(lambda c, l: _Buffer())(None, None)
    buf.trim_confirmed(10.0)
    assert buf.trimmed_at == [10.0] and plain.trims == 0


def _hook_with_tracker(trim_keep_s):
    from streaming.prefix_agreement import PrefixAgreementTracker

    hook = lp.StableUpdateHook(None, _Controller(), lambda: 0.0, 0.1, trim_keep_s=trim_keep_s)
    buf = hook.build_streaming_buffer(lambda c, l: _Buffer())(None, None)
    tracker = hook.build_prefix_agreement_tracker(lambda c, l: PrefixAgreementTracker(agreement_depth=2))(None, None)
    return hook, buf, tracker


def _state(uid, first_frame_t):
    return SimpleNamespace(utterance_id=uid, context_audio=[SimpleNamespace(capture_timestamp=first_frame_t)],
                           new_audio_since_last_asr=[])


def test_prompt_omits_confirmed_words_still_audible_in_the_retained_audio():
    """Saved session, utterance 15: 'i am speaking at' confirmed. With 1 s of
    audio kept before the boundary, 'speaking at' is heard again; prompting
    with it made Whisper skip those words and stamp 'a normal' at the window
    start, where the tracker drops them."""

    hook, buf, tracker = _hook_with_tracker(1.0)
    words = spans("i am speaking at a normal", 0.0, width=0.5)            # "at" ends at 2.0
    tracker.process(hyp(15, 1, words[:4] + [("a", 2.0, 2.2)], audio_end=2.2))
    tracker.process(hyp(15, 2, words, audio_end=3.0))                      # agreement confirms the common prefix
    confirmed = tracker.committed_text_for(15).split()
    boundary = tracker.confirmed_boundary_time(15)
    assert confirmed[:4] == ["i", "am", "speaking", "at"]
    buf.state = _state(15, boundary - 1.0)                                 # retained audio starts 1 s early
    prompt = tracker.confirmed_prompt_text(15)
    audible = [w for w, a, b in words if b > boundary - 1.0 and a < boundary]
    assert prompt == " ".join(confirmed[:len(confirmed) - len(audible)])
    assert not prompt.endswith(audible[-1])


def test_prompt_is_unchanged_without_trim_keep():
    hook, buf, tracker = _hook_with_tracker(0.0)
    words = spans("i am speaking at", 0.0, width=0.5)
    tracker.process(hyp(15, 1, words, audio_end=2.2))
    tracker.process(hyp(15, 2, words, audio_end=2.5))
    buf.state = _state(15, 0.5)
    assert tracker.confirmed_prompt_text(15) == tracker.committed_text_for(15)


def test_sentence_end_detection_ignores_ellipses():
    hook = lp.StableUpdateHook(None, _Controller(), lambda: 0.0, 0.1)
    for text, expected in [("Thank you.", True), ("Is it?", True), ("I am speaking at...", False),
                           ("this test should not", False), ('He said "stop."', True)]:
        hook.hypotheses = [{"utterance_id": 1, "success": True, "text": text}]
        assert hook._ends_sentence(1) is expected, text


def test_saved_session_window_is_recovered_with_context_and_trimmed_prompt():
    """The exact live failure: window 8 of utterance 3 decoded the audio
    trimmed at the end of 'intelligence' and produced 'health people all
    complex problems'. Same audio with 1 s of context and the trimmed
    prompt: 'helps people (to) solve complex problems'."""

    from product.tools import window_replay as wr

    wav = SESSION / "input.wav"
    if not wav.exists():
        pytest.skip("saved session recording not present")
    try:
        from asr.whisper_engine import WhisperEngine

        from product import runtime

        runtime.prefer_offline_models()
        engine = WhisperEngine("tiny", "cpu", "int8", language="en", beam_size=1, temperature=0.0, cpu_threads=8,
                               vad_filter=False, word_timestamps=True)
    except Exception as exc:  # model not cached
        pytest.skip(f"Whisper tiny unavailable: {exc}")
    from product.sources import read_wav_mono_16k

    samples, sr = read_wav_mono_16k(wav)
    w = next(x for x in wr.windows(SESSION) if x["window_id"] == 8)

    def decode(keep, prompt):
        start = w["audio_start"] - keep
        a, b = int(round((start - SESSION_OFFSET) * sr)), int(round((w["audio_end"] - SESSION_OFFSET) * sr))
        r = engine.transcribe(samples[a:b], sr, initial_prompt=prompt)
        return wr.accept(r.words, start, w["boundary"], (w["prompt"] or "").split())

    assert decode(0.0, w["prompt"]).startswith("health people all complex")          # live defect reproduced
    fixed = decode(1.0, wr.prompt_before(w, w["audio_start"] - 1.0))
    assert fixed.startswith("helps people") and "solve complex problems" in fixed


# -- sentence-aware segment closing ----------------------------------------


def sentence_update(controller, clock, uid, text, sentence_end=True):
    t = clock()
    controller.submit_update(utterance_id=uid, new_text=text, utterance_final=True, t=t, onset_s=max(0.0, t - 0.5),
                             sentence_end=sentence_end)


def test_sentence_end_closes_after_a_short_pause_instead_of_the_idle_timeout():
    rules = lp.SegmentRules(close_after_words=20, idle_close_s=5.0, hard_max_words=60, sentence_gap_s=0.2)
    clock, display, worker, controller, engine, out = build(rules=rules)
    controller.submit_speech(active=True, t=clock())
    controller.submit_speech(active=False, t=clock())
    sentence_update(controller, clock, 1, "good morning today i want to discuss education")
    time.sleep(0.6)
    assert controller.active is None
    assert controller.segments[0].close_reason == "sentence end + 0.2 s pause"
    finish(clock, controller, worker)


def test_sentence_end_closes_at_once_when_the_pause_before_the_next_utterance_was_long():
    """Saved session: the next sentence often starts before the utterance's
    final commit arrives (decode ~1 s); the pause is then already known."""

    rules = lp.SegmentRules(close_after_words=20, idle_close_s=5.0, hard_max_words=60, sentence_gap_s=0.6)
    clock, display, worker, controller, engine, out = build(rules=rules)
    now = clock()
    controller.submit_speech(active=True, t=now - 3.0)
    controller.submit_speech(active=False, t=now - 1.0)
    controller.submit_speech(active=True, t=now - 0.1)          # 0.9 s pause
    sentence_update(controller, clock, 1, "artificial intelligence helps people solve complex problems")
    sentence_update(controller, clock, 2, "students should develop practical skills", sentence_end=False)
    time.sleep(0.1)
    finish(clock, controller, worker)
    assert [s.close_reason for s in controller.segments] == ["sentence end + 0.6 s pause", "session end"]
    assert display.finals[1] == "HI(artificial intelligence helps people solve complex problems)"


def test_a_hesitation_after_a_period_does_not_split_the_sentence():
    """Saved session: 'this test should not.' (Whisper's period at a 0.24 s
    hesitation) continued with 'take much time ...'."""

    rules = lp.SegmentRules(close_after_words=20, idle_close_s=5.0, hard_max_words=60, sentence_gap_s=0.6)
    clock, display, worker, controller, engine, out = build(rules=rules)
    now = clock()
    controller.submit_speech(active=True, t=now - 2.0)
    controller.submit_speech(active=False, t=now - 0.5)
    controller.submit_speech(active=True, t=now - 0.26)         # 0.24 s pause
    sentence_update(controller, clock, 9, "this test should not")
    sentence_update(controller, clock, 10, "take much time")
    time.sleep(0.1)
    finish(clock, controller, worker)
    assert len(controller.segments) == 1
    assert " ".join(controller.segments[0].words) == "this test should not take much time"


def test_short_sentences_and_non_sentence_ends_keep_the_old_rules():
    rules = lp.SegmentRules(close_after_words=20, idle_close_s=0.4, hard_max_words=60, sentence_gap_s=0.1,
                            sentence_min_words=3)
    clock, display, worker, controller, engine, out = build(rules=rules)
    controller.submit_speech(active=False, t=clock())
    sentence_update(controller, clock, 1, "hello")               # 1 word < sentence_min_words
    time.sleep(0.25)
    assert controller.active is not None, "a one-word sentence waits for the idle timeout"
    time.sleep(0.5)
    assert controller.segments[0].close_reason == "idle 0.4 s"
    finish(clock, controller, worker)


def test_sentence_rule_is_off_by_default_and_validated():
    assert lp.SegmentRules().sentence_gap_s is None
    with pytest.raises(ValueError):
        lp.SegmentRules(idle_close_s=1.0, sentence_gap_s=2.0)


def test_product_config_enables_the_phase3_fixes():
    import yaml

    cfg = yaml.safe_load((REPO_ROOT / "product" / "configs" / "live_m2_mt3c.yaml").read_text(encoding="utf-8"))
    product = cfg["product"]
    assert product["trim_keep_s"] == 1.0
    assert product["segment"]["sentence_gap_s"] == 0.6
    assert cfg["asr"]["model_size"] == "tiny"   # default model unchanged


def test_a_hesitation_disarms_the_sentence_pause_for_the_following_fragment():
    """Replay regression (p3b keep1_sentence): after 'this test should not.'
    + a 0.1 s hesitation, the next fragment's speech ended and the 0.6 s
    sentence pause fired before that fragment's own commit arrived (~1 s
    decode lag), splitting the sentence."""

    rules = lp.SegmentRules(close_after_words=20, idle_close_s=5.0, hard_max_words=60, sentence_gap_s=0.2)
    clock, display, worker, controller, engine, out = build(rules=rules)
    now = clock()
    controller.submit_speech(active=True, t=now - 2.0)
    controller.submit_speech(active=False, t=now - 0.6)
    controller.submit_speech(active=True, t=now - 0.5)          # 0.1 s hesitation
    sentence_update(controller, clock, 9, "this test should not")
    controller.submit_speech(active=False, t=clock())           # fragment ends; its commit is still decoding
    time.sleep(0.5)                                             # > sentence_gap_s
    assert controller.active is not None, "must not close before the fragment's words arrive"
    sentence_update(controller, clock, 10, "take much time")
    time.sleep(0.5)
    finish(clock, controller, worker)
    assert len(controller.segments) == 1
    assert controller.segments[0].close_reason == "sentence end + 0.2 s pause"


def test_unknown_pause_after_several_newer_utterances_keeps_the_segment_open():
    rules = lp.SegmentRules(close_after_words=20, idle_close_s=5.0, hard_max_words=60, sentence_gap_s=0.2)
    clock, display, worker, controller, engine, out = build(rules=rules)
    controller.submit_speech(active=False, t=clock())
    t = clock()
    controller.submit_update(utterance_id=11, new_text="while i speak", utterance_final=True, t=t, onset_s=t - 0.5,
                             sentence_end=True, newer_utterances=1)  # next fragment already started and ended
    time.sleep(0.5)
    assert controller.active is not None
    finish(clock, controller, worker)
