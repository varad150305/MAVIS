"""Phase 9: targeted regressions for the four streaming defects confirmed by
the Phase 8A audit (product_outputs/diagnostics/phase8/). Inputs mirror the
live sessions 20261029-181057_m2_mic_ui (battery) / -181611 (AC); the full
replays of those recordings are in product_outputs/diagnostics/phase9/."""

from __future__ import annotations

import time
from types import SimpleNamespace

import yaml

from product import live_pipeline as lp
from product.tests.test_live_pipeline import build, finish
from product.tests.test_phase3_fixes import sentence_update

BASE = 2361935.14   # capture clock of WAV 0 (battery session); times below are WAV seconds + BASE


def _hook(**kw):
    return lp.StableUpdateHook(None, None, time.perf_counter, 0.1, **kw)


def _hyp(uid, start, end, words):
    return {"utterance_id": uid, "success": True, "is_final": True, "audio_start": BASE + start,
            "audio_end": BASE + end, "words": [[w, BASE + a, BASE + b] for w, a, b in words], "text": ""}


# -- 1. word duplicated across a contiguous utterance cut -------------------------


def _people(cur_start=29.40, prev_last_end=29.38, cur_first_start=29.40, cur_words=("People", "who", "speak")):
    h = _hook(cross_utterance_dedup=True)
    h._committed = [(6, w) for w in "real time translation can make communication easier between people".split()]
    h.hypotheses = [
        _hyp(6, 25.10, 29.40, [(" easier", 27.62, 28.42), (" between", 28.42, 28.88), (" people.", 28.88,
                                                                                         prev_last_end)]),
        _hyp(7, cur_start, cur_start + 2.3, [(" " + cur_words[0], cur_first_start, cur_first_start + 0.62),
                                            (" " + cur_words[1], cur_first_start + 0.62, cur_first_start + 1.0)]),
    ]
    return h


def test_duplicate_word_at_a_contiguous_cut_is_dropped_once():
    h = _people()
    assert h._dedup_at_cut(7, "people who speak different") == "who speak different"
    assert h.cross_boundary_dedups == [{"utterance_id": 7, "dropped": ["people"], "cut_gap_s": 0.0}]


def test_repetition_separated_by_a_real_pause_is_kept():
    # "... between people. [0.6 s pause] People who ..." - windows are not contiguous
    h = _people(cur_start=30.00, cur_first_start=30.05)
    assert h._dedup_at_cut(7, "people who speak different") == "people who speak different"


def test_repetition_not_touching_the_cut_is_kept():
    h = _people(prev_last_end=29.0)                       # old word ended 0.4 s before the cut
    assert h._dedup_at_cut(7, "people who speak different") == "people who speak different"
    h = _people(cur_first_start=29.7)                     # new word starts 0.3 s after the cut
    assert h._dedup_at_cut(7, "people who speak different") == "people who speak different"


def test_only_an_utterances_first_commit_and_only_equal_words_are_affected():
    h = _people()
    assert h._dedup_at_cut(7, "persons who speak") == "persons who speak"
    h._committed.append((7, "who"))
    assert h._dedup_at_cut(7, "people people") == "people people"   # later commit of utt 7: untouched


def test_dedup_is_off_unless_configured():
    h = _people()
    h.cross_utterance_dedup = False
    out = []
    h.tracker = SimpleNamespace(commit_events_for=lambda uid: [SimpleNamespace(new_text="people who", is_final=False,
                                                                                commit_id=1)])
    h.controller = SimpleNamespace(submit_update=lambda **kw: out.append(kw["new_text"]))
    h._emit(7, 0)
    assert out == ["people who"]


# -- 2./3. trimmed or prompted final window -> whole utterance, no prompt -----------


class FakeBuffer:
    """StreamingBuffer stand-in with the research trim semantics."""

    def __init__(self, uid, start, end):
        frames = [SimpleNamespace(capture_timestamp=BASE + t / 10, duration_ms=100.0)
                  for t in range(int(start * 10), int(end * 10))]
        self.state = SimpleNamespace(utterance_id=uid, context_audio=frames, new_audio_since_last_asr=[])
        self.built = None

    def trim_confirmed(self, cut):
        self.state.context_audio = [f for f in self.state.context_audio
                                    if f.capture_timestamp + f.duration_ms / 1000.0 > cut]

    def try_build_final_window(self, initial_prompt=None):
        frames = [*self.state.context_audio, *self.state.new_audio_since_last_asr]
        self.built = (round(frames[0].capture_timestamp - BASE, 2), round(frames[-1].capture_timestamp - BASE + 0.1, 2),
                      initial_prompt)
        return self.built


def _buffer(hook, uid, start, end):
    fb = FakeBuffer(uid, start, end)
    return hook.build_streaming_buffer(lambda config, logger: fb)(None, None)


def test_short_trimmed_final_window_is_the_whole_utterance_without_prompt():
    """AC utterance 17 (72.2-78.3 s): live final window 75.0-78.3 with prompt
    'i am speaking at my normal pace' -> 'using my use of microphone'."""

    h = _hook(trim_keep_s=1.0, final_window_full_max_s=8.0)
    buf = _buffer(h, 17, 72.2, 78.3)
    buf.trim_confirmed(BASE + 76.0)                       # commit boundary 76.0 -> trimmed at 75.0
    assert buf.state.context_audio[0].capture_timestamp - BASE >= 74.9
    assert buf.try_build_final_window(initial_prompt="i am speaking at my normal pace") == (72.2, 78.3, None)
    assert h.final_window_restores == [{"utterance_id": 17, "window_s": 6.1,
                                        "prompt_dropped": "i am speaking at my normal pace"}]


def test_long_utterance_keeps_the_trimmed_final_window_and_its_prompt():
    h = _hook(trim_keep_s=1.0, final_window_full_max_s=8.0)
    buf = _buffer(h, 3, 10.0, 20.0)                       # 10 s utterance > 8 s
    buf.trim_confirmed(BASE + 16.0)
    assert buf.try_build_final_window(initial_prompt="earlier words") == (15.0, 20.0, "earlier words")
    assert h.final_window_restores == []


def test_untrimmed_final_window_and_other_utterances_are_unchanged():
    h = _hook(trim_keep_s=1.0, final_window_full_max_s=8.0)
    buf = _buffer(h, 5, 18.3, 20.4)                       # never trimmed
    assert buf.try_build_final_window(initial_prompt=None) == (18.3, 20.4, None)
    buf.trim_confirmed(BASE + 19.9)                       # trimmed audio of utterance 5 ...
    buf.state.utterance_id = 6                            # ... is never restored into utterance 6
    buf.try_build_final_window(initial_prompt="x")
    assert buf.built[2] == "x" and h.final_window_restores == [] and 5 not in h._trimmed_audio


def test_final_window_restore_is_off_unless_configured():
    h = _hook(trim_keep_s=1.0)
    buf = _buffer(h, 17, 72.2, 78.3)
    buf.trim_confirmed(BASE + 76.0)
    assert buf.try_build_final_window(initial_prompt="p")[2] == "p"


# -- 4. segment boundaries ------------------------------------------------------------


def test_long_segment_does_not_close_at_a_vad_split_inside_a_sentence():
    """AC: '... recognize the boundary.' (>= 20 words, utterance end) was
    closed at once; the speaker went straight on with 'without cutting off
    the previous sentence'."""

    rules = lp.SegmentRules(close_after_words=20, idle_close_s=5.0, hard_max_words=60, sentence_gap_s=0.6,
                            long_close_needs_pause=True)
    clock, display, worker, controller, engine, out = build(rules=rules)
    now = clock()
    controller.submit_speech(active=True, t=now - 6.0)
    controller.submit_speech(active=False, t=now - 0.40)
    controller.submit_speech(active=True, t=now - 0.35)        # 0.05 s split: no real pause
    sentence_update(controller, clock, 15, "when i speak continuously the system should not lose words at the end of "
                                           "the sentence when i speak between sentences the system should recognize "
                                           "the boundary")
    time.sleep(0.1)
    assert controller.active is not None                       # kept open: the speaker went on at once
    controller.submit_speech(active=False, t=clock())          # utterance 16 ends ...
    sentence_update(controller, clock, 16, "without cutting off the previous sentence")   # ... its commit follows
    time.sleep(0.9)
    assert len(controller.segments) == 1
    seg = controller.segments[0]
    assert seg.close_reason == "sentence end + 0.6 s pause"
    assert " ".join(seg.words).endswith("recognize the boundary without cutting off the previous sentence")
    finish(clock, controller, worker)
    assert len(display.finals) == 1


def test_long_segment_still_closes_after_a_real_pause():
    rules = lp.SegmentRules(close_after_words=20, idle_close_s=5.0, hard_max_words=60, sentence_gap_s=0.6,
                            long_close_needs_pause=True)
    clock, display, worker, controller, engine, out = build(rules=rules)
    controller.submit_speech(active=True, t=clock())
    controller.submit_speech(active=False, t=clock())
    sentence_update(controller, clock, 1, " ".join(["word"] * 21), sentence_end=False)   # no period either
    time.sleep(0.9)
    assert controller.active is None
    assert controller.segments[0].close_reason == "utterance end at >= 20 words + 0.6 s pause"
    finish(clock, controller, worker)


def test_long_segment_closes_at_once_without_the_option():
    rules = lp.SegmentRules(close_after_words=20, idle_close_s=5.0, hard_max_words=60, sentence_gap_s=0.6)
    clock, display, worker, controller, engine, out = build(rules=rules)
    sentence_update(controller, clock, 1, " ".join(["word"] * 21))
    time.sleep(0.1)
    assert controller.segments[0].close_reason == "utterance end at >= 20 words"
    finish(clock, controller, worker)


def test_four_word_sentence_joins_the_continuation():
    """Battery: 'please wait a moment.' (~0.5 s reading pause) closed alone;
    'while i explain the next part of the presentation' became a fragment."""

    rules = lp.SegmentRules(close_after_words=20, idle_close_s=5.0, hard_max_words=60, sentence_gap_s=0.6,
                            sentence_min_words=5)
    clock, display, worker, controller, engine, out = build(rules=rules)
    controller.submit_speech(active=True, t=clock())
    controller.submit_speech(active=False, t=clock())
    sentence_update(controller, clock, 11, "please wait a moment")
    time.sleep(0.8)                                            # > sentence_gap_s: 4 words do not close
    assert controller.active is not None
    sentence_update(controller, clock, 12, "while i explain the next part of the presentation")
    time.sleep(0.8)
    assert len(controller.segments) == 1 and controller.active is None
    assert " ".join(controller.segments[0].words) == ("please wait a moment while i explain the next part of the "
                                                      "presentation")
    finish(clock, controller, worker)


def test_product_config_enables_the_phase9_fixes_and_keeps_the_model_and_window():
    from pathlib import Path

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "live_m2_mt3c.yaml")
                         .read_text(encoding="utf-8"))
    p = cfg["product"]
    assert p["final_window_full_max_s"] == 8.0 and p["cross_utterance_dedup"] is True
    assert p["segment"]["sentence_min_words"] == 5 and p["segment"]["long_close_needs_pause"] is True
    assert cfg["asr"]["model_size"] == "tiny" and cfg["streaming_buffer"]["min_new_audio_ms"] == 2000
    assert p["integration"]["live_translation_enabled"] is False


def test_long_segment_closes_without_a_pause_once_it_reaches_the_limit():
    rules = lp.SegmentRules(close_after_words=20, idle_close_s=5.0, hard_max_words=60, sentence_gap_s=0.6,
                            long_close_needs_pause=True, long_close_max_words=32)
    clock, display, worker, controller, engine, out = build(rules=rules)
    now = clock()
    controller.submit_speech(active=True, t=now - 9.0)
    controller.submit_speech(active=False, t=now - 0.40)
    controller.submit_speech(active=True, t=now - 0.30)        # short reading pause
    sentence_update(controller, clock, 1, " ".join(["first"] * 25))
    time.sleep(0.1)
    assert controller.active is not None                       # 25 words: waits for a real pause
    sentence_update(controller, clock, 2, " ".join(["second"] * 8))
    time.sleep(0.1)
    assert controller.active is None                           # 33 >= 32: closed at the utterance end
    assert controller.segments[0].close_reason == "utterance end at >= 20 words"
    finish(clock, controller, worker)
