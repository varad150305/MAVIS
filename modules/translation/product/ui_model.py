"""UI-independent transcript state built from session events.

The standalone UI (product/ui) renders this model; a future MAVIS host can
reuse it. Rules (each covered by product/tests/test_phase5_session.py):

  * events of another session_id, or with a seq not above the last applied
    one, are ignored (no stale or duplicated events);
  * English: committed words accumulate per segment; a provisional line is
    kept per utterance and cleared once that utterance's committed words
    arrive (or its segment closes) - provisional text is never shown as
    committed and never translated;
  * Hindi: a draft replaces the previous draft of its segment; the first
    final fixes the segment's Hindi - later drafts (hi_draft_late or a
    stray hi_draft) and repeated finals never overwrite it;
  * a draft or final for a segment the model has not seen any English for
    is recorded as an association problem, not attached elsewhere;
  * audio_stall sets `audio_warning` (cleared by audio_resumed), so a host
    never shows a stalled microphone as plain "ready"; cleanup_error is an
    error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SegmentView:
    segment_id: int
    english: list[str] = field(default_factory=list)
    utterance_ids: list[int] = field(default_factory=list)
    hindi_draft: str | None = None
    hindi_final: str | None = None
    final_kind: str | None = None
    closed: bool = False
    close_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def hindi_state(self) -> str:
        if self.hindi_final is not None:
            return "final"
        return "draft" if self.hindi_draft is not None else "waiting"


class TranscriptModel:
    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id
        self.last_seq = 0
        self.segments: dict[int, SegmentView] = {}
        self.provisional: dict[int, str] = {}       # utterance_id -> text
        self.state = "idle"
        self.ready_in_s: float | None = None
        self.problems: list[str] = []
        self.errors: list[str] = []
        self.ignored_stale = 0
        self.ignored_late_drafts = 0
        self.audio_warning: str | None = None        # set while the microphone delivers no frames
        self.audio_stalls = 0
        #: set when audio resumes after a stall (shown briefly by the UI)
        self.audio_notice: str | None = None
        self.version = 0                             # bumps on every visible change

    def reset(self, session_id: str) -> None:
        self.__init__(session_id)

    def apply(self, event: dict[str, Any]) -> bool:
        """Apply one event; returns True when something visible changed."""

        if self.session_id is None:
            self.session_id = event.get("session_id")
        if event.get("session_id") != self.session_id or event.get("seq", 0) <= self.last_seq:
            self.ignored_stale += 1
            return False
        self.last_seq = event["seq"]
        handler = getattr(self, "_on_" + event.get("type", ""), None)
        changed = bool(handler(event)) if handler is not None else False
        if changed:
            self.version += 1
        return changed

    # ------------------------------------------------------------------ handlers
    def _seg(self, segment_id: int) -> SegmentView:
        if segment_id not in self.segments:
            self.segments[segment_id] = SegmentView(segment_id)
        return self.segments[segment_id]

    def _on_session_state(self, e):
        self.state = e["state"]
        if e["state"] in ("stopped", "failed"):
            self.audio_warning = None
        if e.get("ready_in_s") is not None:
            self.ready_in_s = e["ready_in_s"]
        if e.get("error"):
            self.errors.append(e["error"])
        if e.get("cleanup_ok") is False:
            self.errors.append("session stopped, but audio cleanup failed (see session_summary.json cleanup)")
        return True

    def _on_en_provisional(self, e):
        self.provisional[e["utterance_id"]] = e["text"]
        return True

    def _on_en_committed(self, e):
        seg = self._seg(e["segment_id"])
        seg.english = e["stable_text"].split() if e.get("stable_text") is not None else seg.english + e["new_text"].split()
        uid = e.get("utterance_id")
        if uid is not None:
            if uid not in seg.utterance_ids:
                seg.utterance_ids.append(uid)
            self.provisional.pop(uid, None)
        return True

    def _on_hi_draft(self, e):
        seg = self.segments.get(e["segment_id"])
        if seg is None:
            self.problems.append(f"draft for unknown segment {e['segment_id']}")
            return False
        if seg.hindi_final is not None:
            self.ignored_late_drafts += 1
            return False
        seg.hindi_draft = e["target_text"]
        return True

    def _on_hi_draft_late(self, e):
        self.ignored_late_drafts += 1
        return False

    def _on_hi_final(self, e):
        seg = self.segments.get(e["segment_id"])
        if seg is None:
            self.problems.append(f"final for unknown segment {e['segment_id']}")
            return False
        if seg.hindi_final is not None:
            self.problems.append(f"second final for segment {e['segment_id']} ignored")
            return False
        seg.hindi_final, seg.final_kind = e["target_text"], e.get("kind")
        return True

    def _on_segment_closed(self, e):
        seg = self._seg(e["segment_id"])
        seg.closed, seg.close_reason = True, e.get("reason")
        for uid in e.get("utterance_ids") or []:
            self.provisional.pop(uid, None)
        return True

    def _on_mt_error(self, e):
        seg = self.segments.get(e.get("segment_id"))
        msg = f"MT error (segment {e.get('segment_id')}): {e.get('error')}"
        if seg is not None:
            seg.errors.append(msg)
        self.errors.append(msg)
        return True

    _on_final_mt_error = _on_mt_error

    def _on_audio_stall(self, e):
        self.audio_stalls += 1
        self.audio_notice = None
        self.audio_warning = (f"No audio from the microphone for {e.get('no_frames_for_s', 0):.0f} s - check that it is "
                              f"connected (the session is still running; Stop ends it)")
        return True

    def _on_audio_resumed(self, e):
        self.audio_warning = None
        self.audio_notice = f"Microphone audio resumed (no audio for {e.get('stalled_for_s', 0):.1f} s)"
        return True

    def _on_cleanup_error(self, e):
        self.errors.append(f"audio cleanup failed ({e.get('op')}): {e.get('error')}")
        return True

    def _on_error(self, e):
        self.errors.append(f"{e.get('where', 'pipeline')}: {e.get('message')}")
        return True

    # ------------------------------------------------------------------ views
    def ordered_segments(self) -> list[SegmentView]:
        return [self.segments[k] for k in sorted(self.segments)]
