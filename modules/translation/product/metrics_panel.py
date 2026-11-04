"""Performance-panel model: live metrics from session events (UI-independent).

Uses the same latency definitions and code as session_summary.json
`user_latency` (product.live_pipeline.latency_core / USER_LATENCY_DEFINITIONS):
every latency is measured from the VAD speech onset, never from commit time.
Values that are not available yet are shown as "collecting" (no samples) or
with their sample count and the reason a percentile is missing - never 0.

What the panel deliberately does NOT show:
  * a live WER or "accuracy" (no reference transcript exists live); a WER is
    shown only when a script_score report of a scored scripted recording is
    loaded, labelled offline/reference-based;
  * zero MT errors as translation quality (it counts failed MT calls only);
  * stable_token_ratio or any agreement statistic as a probability of
    correctness.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from product.session import RECORD_TYPES

#: event type -> transcript record type (inverse of product.session.RECORD_TYPES)
_RECORD_NAME = {v: k for k, v in RECORD_TYPES.items()}


@dataclass
class PanelRow:
    key: str
    label: str
    value: str
    definition: str
    state: str          # "ok" | "collecting" | "partial" | "n/a"


def _fmt_dist(d: dict[str, Any] | None, unit: str = " s", scale: float = 1.0) -> tuple[str, str]:
    if not d or not d.get("n"):
        return "collecting...", "collecting"
    parts = [f"median {d['p50'] * scale:.2f}{unit}"]
    if d.get("p90") is not None:
        parts.append(f"p90 {d['p90'] * scale:.2f}{unit}")
    parts.append(f"max {d['max'] * scale:.2f}{unit}")
    note = f"n={d['n']}"
    if d.get("missing"):
        note += f", {d['missing']} missing"
    if d.get("insufficient_n_for"):
        note += f"; {'/'.join(d['insufficient_n_for'])} need more samples"
    return f"{', '.join(parts)}  ({note})", "ok" if not d.get("insufficient_n_for") else "partial"


class LiveMetrics:
    """Accumulates one session's events; `rows()` gives the panel."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id
        self.records: list[dict] = []          # display records (transcript names)
        self.timing: dict[int, dict] = {}      # utterance_id -> latest timing
        self.segments: dict[int, dict] = {}    # segment_id -> {utterance_ids, closed_s}
        self.decodes: list[dict] = []
        self.resources: list[dict] = []
        self.ready_in_s: float | None = None
        self.info: dict[str, Any] = {}
        self.provisional_enabled = False
        self.mt_errors = 0
        self.summary: dict[str, Any] | None = None
        self.reference_score: dict[str, Any] | None = None

    def apply(self, e: dict[str, Any]) -> None:
        if self.session_id is None:
            self.session_id = e.get("session_id")
        if e.get("session_id") != self.session_id:
            return
        t = e.get("type")
        if t == "session_state":
            if e.get("state") == "starting":
                self.provisional_enabled = bool((e.get("info") or {}).get("show_provisional_english"))
            if e.get("ready_in_s") is not None:
                self.ready_in_s = e["ready_in_s"]
            if e.get("info"):
                self.info.update(e["info"])
        elif t in ("en_committed", "en_provisional", "hi_draft", "hi_final"):
            self.records.append({**e, "type": _RECORD_NAME[t]})
            if t == "en_committed":
                seg = self.segments.setdefault(e["segment_id"], {"utterance_ids": [], "closed_s": None})
                uid = e.get("utterance_id")
                if uid is not None and uid not in seg["utterance_ids"]:
                    seg["utterance_ids"].append(uid)
                if uid is not None and e.get("new_text") and e.get("t_s") is not None:
                    # en_committed.t_s is the commit event time (same value the
                    # hook publishes as first_commit_s; whichever arrives first)
                    self.timing.setdefault(uid, {}).setdefault("first_commit_s", e["t_s"])
        elif t == "segment_closed":
            seg = self.segments.setdefault(e["segment_id"], {"utterance_ids": [], "closed_s": None})
            seg["closed_s"] = e.get("t_s")
            for u in e.get("utterance_ids") or []:
                if u not in seg["utterance_ids"]:
                    seg["utterance_ids"].append(u)
        elif t == "utterance_timing":
            cur = self.timing.setdefault(e["utterance_id"], {})
            cur.update({k: v for k, v in e.items() if k.endswith("_s") and v is not None and k != "t_s"})
        elif t == "asr_decode":
            self.decodes.append(e)
        elif t == "resources":
            self.resources.append(e)
        elif t in ("mt_error", "final_mt_error"):
            self.mt_errors += 1
        elif t == "session_summary":
            self.summary = e.get("summary")

    def load_reference_score(self, path: str | Path) -> dict[str, Any]:
        """A product.tools.script_score --out report (offline, reference-based)."""

        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        session_rows = [r for r in doc.get("results", []) if r.get("kind") == "session"] or doc.get("results", [])
        if not session_rows:
            raise ValueError("no scored session in this report")
        r = session_rows[0]
        self.reference_score = {"wer": r["wer"], "S": r["S"], "D": r["D"], "I": r["I"], "ref_words": r["ref_words"],
                                "recording": doc.get("recording"), "target": r.get("target"),
                                "limitation": doc.get("limitation")}
        return self.reference_score

    # ------------------------------------------------------------------ panel
    def latency(self) -> dict[str, Any]:
        if self.summary and self.summary.get("user_latency"):
            return self.summary["user_latency"]           # authoritative after stop
        from product.latency import latency_core

        segs = [{"segment_id": k, **v} for k, v in sorted(self.segments.items())]
        return latency_core(self.records, self.timing, segs, self.provisional_enabled)

    def rows(self) -> list[PanelRow]:
        from product.latency import USER_LATENCY_DEFINITIONS as D

        lat = self.latency()
        rows: list[PanelRow] = []

        def latency_row(key, label, field_, definition):
            value, state = _fmt_dist(lat.get(field_))
            rows.append(PanelRow(key, label, value, definition, state))

        ready = self.ready_in_s if self.ready_in_s is not None else (self.summary or {}).get("startup_to_listening_s")
        rows.append(PanelRow("ready", "Ready in", f"{ready:.1f} s" if ready is not None else "starting...",
                             "launch of the pipeline -> audio capture started (models loaded)",
                             "ok" if ready is not None else "collecting"))
        latency_row("first_en", "Speech -> first English (printed)", "onset_to_first_committed_en_render_s",
                    D["onset_to_first_committed_en_render_s"] + " [render time]")
        latency_row("first_en_event", "  same, pipeline event time", "onset_to_first_committed_en_s",
                    D["onset_to_first_committed_en_s"] + " [event time: before queueing and printing]")
        if self.provisional_enabled:
            latency_row("first_prov", "Speech -> provisional English (may be wrong)",
                        "onset_to_first_provisional_en_render_s", D["onset_to_first_provisional_en_render_s"])
        latency_row("first_draft", "Speech -> first Hindi draft (printed)", "onset_to_first_hindi_draft_render_s",
                    D["onset_to_first_hindi_draft_render_s"])
        latency_row("final", "Last word -> final Hindi (printed)", "last_word_to_final_hindi_render_s",
                    D["last_word_to_final_hindi_render_s"])
        latency_row("close_final", "Segment close -> final Hindi", "close_to_final_hindi_render_s",
                    D["close_to_final_hindi_render_s"])

        ok = [d for d in self.decodes if d.get("success")]
        if ok:
            inf = statistics.median(d["inference_ms"] for d in ok if d.get("inference_ms") is not None)
            rtf = sorted(d["rtf"] for d in ok if d.get("rtf") is not None)
            value = (f"median {inf:.0f} ms, RTF median {statistics.median(rtf):.2f} max {rtf[-1]:.2f}  "
                     f"(n={len(ok)} decodes)")
            rows.append(PanelRow("asr", "ASR decode time / RTF", value,
                                 "Whisper inference per decode; RTF = inference / audio window (< 1 = faster than "
                                 "real time)", "ok"))
        else:
            rows.append(PanelRow("asr", "ASR decode time / RTF", "collecting...", "Whisper inference per decode",
                                 "collecting"))

        res = self.resources
        if res:
            cores = [r["cores_busy"] for r in res if r.get("cores_busy") is not None]
            mem = [r["working_set_mb"] for r in res if r.get("working_set_mb") is not None]
            value = (f"CPU now {cores[-1]:.1f} cores (mean {sum(cores) / len(cores):.1f}), "
                     f"memory now {mem[-1]:.0f} MB (peak {max(mem):.0f} MB)") if cores and mem else "collecting..."
            rows.append(PanelRow("cpu", "CPU / memory (this process)", value,
                                 "process CPU time per wall second (1.0 = one logical core) and working set; other "
                                 "applications are not included", "ok" if cores and mem else "collecting"))
        else:
            rows.append(PanelRow("cpu", "CPU / memory (this process)", "collecting...", "sampled once per second",
                                 "collecting"))

        power = self.info.get("on_ac_power")
        failed = sum(1 for d in self.decodes if not d.get("success"))
        dropped = res[-1].get("dropped_frames") if res else None
        rows.append(PanelRow("health", "Power / dropped frames / failed decodes / MT errors",
                             f"{'AC' if power else 'battery' if power is not None else 'unknown'} / "
                             f"{dropped if dropped is not None else 'collecting'} / {failed} / {self.mt_errors}",
                             "MT errors = translation calls that failed; it says nothing about translation quality",
                             "ok"))
        if self.reference_score:
            s = self.reference_score
            rows.append(PanelRow("wer", "Reference WER (offline, scripted read)",
                                 f"{s['wer']:.1%}  (S{s['S']} D{s['D']} I{s['I']} / {s['ref_words']} words, "
                                 f"{s['recording']})",
                                 "scored offline against the reading script with product.tools.script_score; spoken "
                                 "wording not independently annotated", "ok"))
        else:
            rows.append(PanelRow("wer", "Reference WER", "not available live",
                                 "WER needs a reference transcript: score a scripted recording offline "
                                 "(product.tools.script_score) and load the report", "n/a"))
        return rows
