"""Viewer-facing latency definitions and computation (phase 4.6), moved out
of product.live_pipeline in phase 7 unchanged, so that a UI/host process can
compute the live panel without importing the pipeline (torch, models).
product.live_pipeline re-exports every name here."""

from __future__ import annotations

from typing import Any


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q / 100.0 * (len(ordered) - 1))))]


def _dist_n(values: list[float | None]) -> dict[str, Any]:
    """Distribution with explicit missing values and sample-size limits:
    p90 needs n >= 10 and p95 n >= 20, otherwise they are null (and named
    in `insufficient_n_for`), never extrapolated."""

    present = [v for v in values if v is not None]
    n = len(present)
    out = {"n": n, "missing": len(values) - n, "mean": sum(present) / n if n else None,
           "p50": _pct(present, 50), "p90": _pct(present, 90) if n >= 10 else None,
           "p95": _pct(present, 95) if n >= 20 else None, "max": max(present) if present else None}
    out["insufficient_n_for"] = [q for q, need in (("p90", 10), ("p95", 20)) if 0 < n < need] or None
    return out


USER_LATENCY_DEFINITIONS = {
    "time_base": "session clock seconds = seconds of input audio since frame 0 (WAV seconds of input.wav)",
    "speech_onset": "VAD speech onset of the utterance: onset frame index x frame duration (StreamingBuffer "
                    "utterance_pre_roll onset_sequence_number); for a segment, the onset of its first utterance",
    "onset_to_first_tentative_en_s": "per utterance: first successful non-empty Whisper hypothesis processed by the "
                                     "pipeline - onset (event time; on screen only with display_provisional_en)",
    "onset_to_first_provisional_en_render_s": "per utterance: first provisional English line printed - onset "
                                              "(only with display_provisional_en)",
    "onset_to_first_committed_en_s": "per utterance: first non-empty LocalAgreement commit - onset (event time)",
    "onset_to_first_committed_en_render_s": "per utterance: first EN line carrying its words printed - onset",
    "onset_to_first_hindi_draft_render_s": "per segment: first Hindi draft line printed - segment onset (null when "
                                           "the segment had no draft before its final)",
    "onset_to_final_hindi_render_s": "per segment: final Hindi line printed - segment onset",
    "last_word_to_final_hindi_render_s": "per segment: final Hindi line printed - end of the last word of the "
                                         "final decode of its last utterance (Whisper word timestamp, approx.)",
    "vad_end_to_final_hindi_render_s": "per segment: final Hindi line printed - end of the audio of the final window",
    "close_to_final_hindi_render_s": "per segment: final Hindi line printed - segment close",
    "commit_to_en_render_s": "per EN line: printed - commit event time (controller queue + console write)",
    "not_speech_latency": "translation.word_to_hindi_s and segment_first_output_s start at COMMIT time, not at "
                          "speech; they exclude the ASR and agreement wait",
}


def latency_core(records: list[dict[str, Any]], times: dict[int, dict[str, Any]], segments: list[dict[str, Any]],
                 provisional_enabled: bool) -> dict[str, Any]:
    """Latencies from display records (transcript record types, with
    render_s), per-utterance timing (onset_s, first_text_hypothesis_s,
    first_commit_s, last_word_end_s, vad_end_s) and segments
    ({segment_id, utterance_ids, closed_s}). Used for the session summary
    and, on partial data, for the live performance panel (product.metrics_panel)."""

    recs = records
    first_render: dict[str, dict[int, float]] = {"en": {}, "prov": {}}
    for r in recs:
        if r["type"] == "en_update" and r.get("utterance_id") is not None and r["new_text"]:
            first_render["en"].setdefault(r["utterance_id"], r["render_s"])
        elif r["type"] == "en_provisional":
            first_render["prov"].setdefault(r["utterance_id"], r["render_s"])

    def since(u, value):
        onset = times.get(u, {}).get("onset_s")
        return value - onset if value is not None and onset is not None else None

    committed = [u for u, t in times.items() if t.get("first_commit_s") is not None]
    per_utt = {u: {"onset_s": times[u].get("onset_s"),
                   "tentative": since(u, times[u].get("first_text_hypothesis_s")),
                   "provisional_render": since(u, first_render["prov"].get(u)),
                   "committed": since(u, times[u].get("first_commit_s")),
                   "committed_render": since(u, first_render["en"].get(u))} for u in committed}
    per_seg = []
    for seg in segments:
        mine = [r for r in recs if r.get("segment_id") == seg["segment_id"]]
        drafts = [r["render_s"] for r in mine if r["type"] == "hi_draft"]
        final = next((r["render_s"] for r in mine if r["type"] == "hi_final"), None)
        first_u = seg["utterance_ids"][0] if seg["utterance_ids"] else None
        last_u = seg["utterance_ids"][-1] if seg["utterance_ids"] else None
        onset = times.get(first_u, {}).get("onset_s")
        last_word = times.get(last_u, {}).get("last_word_end_s")
        vad_end = times.get(last_u, {}).get("vad_end_s")

        def minus(a, b):
            return a - b if a is not None and b is not None else None
        first_draft = min(drafts) if drafts else None
        per_seg.append({"segment_id": seg["segment_id"], "onset_s": onset, "last_word_end_s": last_word,
                        "first_draft": minus(first_draft, onset) if first_draft is not None and
                        (final is None or first_draft <= final) else None,
                        "final": minus(final, onset), "last_word": minus(final, last_word),
                        "vad_end": minus(final, vad_end), "close": minus(final, seg.get("closed_s")),
                        "draft_before_last_word": bool(first_draft is not None and last_word is not None
                                                       and first_draft <= last_word)})
    en_lag = [r["render_s"] - r["t_s"] for r in recs if r["type"] == "en_update"]
    return {
        "definitions": USER_LATENCY_DEFINITIONS,
        "utterances_with_committed_words": len(per_utt),
        "segments": len(per_seg),
        "onset_to_first_tentative_en_s": _dist_n([v["tentative"] for v in per_utt.values()]),
        "onset_to_first_provisional_en_render_s": _dist_n([v["provisional_render"] for v in per_utt.values()])
        if provisional_enabled else None,
        "onset_to_first_committed_en_s": _dist_n([v["committed"] for v in per_utt.values()]),
        "onset_to_first_committed_en_render_s": _dist_n([v["committed_render"] for v in per_utt.values()]),
        "onset_to_first_hindi_draft_render_s": _dist_n([s["first_draft"] for s in per_seg]),
        "onset_to_final_hindi_render_s": _dist_n([s["final"] for s in per_seg]),
        "last_word_to_final_hindi_render_s": _dist_n([s["last_word"] for s in per_seg]),
        "vad_end_to_final_hindi_render_s": _dist_n([s["vad_end"] for s in per_seg]),
        "close_to_final_hindi_render_s": _dist_n([s["close"] for s in per_seg]),
        "segments_with_hindi_draft_before_last_word": sum(s["draft_before_last_word"] for s in per_seg),
        "commit_to_en_render_s": _dist_n(en_lag),
        "per_utterance": {str(u): v for u, v in per_utt.items()},
        "per_segment": per_seg,
    }
