"""Standalone desktop UI for live English -> Hindi translation (Tkinter, stdlib only).

    .venv\\Scripts\\python.exe -m product.ui

A thin view over product.session.TranslationSession (pipeline, one worker
process per session), product.ui_model (transcript state) and
product.metrics_panel (performance panel). It polls the session's event
queue on the Tk thread every 100 ms; it never parses console text and never
touches the pipeline's internals.

Display conventions (phase 9):
  status pill  top right: IDLE / STARTING (loading models) / LISTENING /
               NO AUDIO (red) / STOPPING / STOPPED / FAILED (red)
  banner       full-width bar under the toolbar: red for a microphone stall
               or an error, green for a few seconds when audio resumes
  English      committed words, dark grey; a provisional line (opt-in) in
               light grey italics, labelled "provisional - may change"
  Hindi        "DRAFT - may change" (amber badge, grey italics) until the
               segment's single final arrives; "FINAL" (green badge, large
               bold) afterwards - a final is never replaced
  video        (handoff) the speaker's camera is the background of the
               transcript area, with Netflix-style subtitles: only the active
               segment (its English, provisional line if enabled, and its
               Hindi draft/final, with the text widget's own tag fonts and
               colours) is drawn bottom-centre over the live picture and is
               replaced by the next segment; long text keeps its newest words
               within 2 (English) / 3 (Hindi) lines; one translucent white
               veil behind the block for readability. Camera: MAVIS_UI_CAMERA = device index
               (default 0), a video file, or "off". Without a live picture
               the transcript is the plain text widget exactly as before.
               The camera is read in this UI process only; the translation
               worker is not involved.
"""

from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

from product.metrics_panel import LiveMetrics
from product.session import SessionBusy, SessionOptions, SessionState, TranslationSession
from product.ui_model import TranscriptModel

HINDI_FAMILY = "Nirmala UI"
UI_FAMILY = "Segoe UI"
POLL_MS = 100
NOTICE_S = 6.0                       # how long the green "audio resumed" banner stays

BG = "#f4f5f7"
CARD = "#ffffff"
INK = "#1f2328"
MUTED = "#6b7280"
FINAL_INK = "#0b3d91"
DRAFT_INK = "#7a7a7a"
RED, RED_BG = "#ffffff", "#c62828"
GREEN, GREEN_BG = "#ffffff", "#2e7d32"
#: status pill: text, background
PILL = {
    "idle": ("IDLE", "#9aa0a6"),
    "starting": ("STARTING - loading models", "#b26a00"),
    "ready": ("LISTENING", "#2e7d32"),
    "no_audio": ("NO AUDIO", RED_BG),
    "stopping": ("STOPPING - finishing translations", "#b26a00"),
    "stopped": ("STOPPED", "#5f6368"),
    "failed": ("FAILED", RED_BG),
}

VIDEO_FPS = 15
VIDEO_STALE_S = 2.0                  # no camera picture for this long -> plain text view
VEIL_ALPHA = 0.72                    # white veil behind each text line drawn over the video
VEIL_PAD = 4
# Netflix-style subtitle over the video (only the active segment is shown)
SUB_WIDTH = 0.82                     # subtitle block at most this share of the video width
SUB_BOTTOM = 0.05                    # gap below the block, share of the video height
SUB_MAX_LINES = {"en": 2, "prov": 2, "hi_final": 3, "hi_draft": 3}


def camera_source() -> str | None:
    """MAVIS_UI_CAMERA: device index (default "0"), a video file path, or off/none/empty."""

    value = os.environ.get("MAVIS_UI_CAMERA", "0").strip()
    return None if value.lower() in ("", "off", "none", "false", "no") else value


class CameraFeed:
    """Reads the speaker's camera on a background thread (OpenCV). `latest()`
    returns the newest BGR frame, or None while there is no live picture
    (no camera, OpenCV missing, device busy or unplugged)."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.error: str | None = None
        self.frames = 0
        self._frame = None
        self._t = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="ui-camera").start()

    def _run(self) -> None:
        try:
            import cv2

            cv2.setNumThreads(1)
            is_device = self.source.isdigit()
            cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW) if is_device else cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.error = f"camera {self.source!r} not available"
                return
            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        if not is_device and cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                            continue                      # a video file loops
                        self.error = "camera delivered no picture"
                        time.sleep(0.2)
                        continue
                    with self._lock:
                        self._frame, self._t = frame, time.monotonic()
                        self.frames += 1
                    if not is_device:
                        time.sleep(1.0 / VIDEO_FPS)       # a device paces itself
            finally:
                cap.release()
        except Exception as exc:  # noqa: BLE001 - no camera = plain text view, never a crash
            self.error = f"{type(exc).__name__}: {exc}"

    def latest(self):
        with self._lock:
            if self._frame is None or time.monotonic() - self._t > VIDEO_STALE_S:
                return None
            return self._frame

    def close(self) -> None:
        self._stop.set()


class TranslatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.session: TranslationSession | None = None
        self.model = TranscriptModel()
        self.metrics = LiveMetrics()
        self._rendered_version = -1
        self._quit_when_done = False
        self._stopping = False
        self._notice_until = 0.0
        self._notice_text: str | None = None
        self._panel_key = None
        self._video_on = False
        self._photo = None
        self._veil: list[tuple[int, int, int, int]] = []     # canvas boxes of drawn text lines
        source = camera_source()
        self.camera = CameraFeed(source) if source else None
        root.title("Live English → Hindi translation (MAVIS product, standalone)")
        root.geometry("1280x800")
        root.minsize(980, 620)
        root.configure(background=BG)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._style()
        self._build()
        self.root.after(POLL_MS, self._poll)
        if self.camera is not None:
            self.root.after(int(1000 / VIDEO_FPS), self._video_tick)

    # ------------------------------------------------------------------ layout
    def _style(self) -> None:
        st = ttk.Style(self.root)
        st.configure("TFrame", background=BG)
        st.configure("Card.TFrame", background=CARD)
        st.configure("TLabel", background=BG, foreground=INK, font=(UI_FAMILY, 10))
        st.configure("Card.TLabel", background=CARD, foreground=INK, font=(UI_FAMILY, 10))
        st.configure("Title.TLabel", font=(UI_FAMILY, 14, "bold"))
        st.configure("PanelHead.TLabel", background=CARD, font=(UI_FAMILY, 12, "bold"))
        st.configure("RowLabel.TLabel", background=CARD, foreground=INK, font=(UI_FAMILY, 9, "bold"))
        st.configure("RowValue.TLabel", background=CARD, foreground=INK, font=(UI_FAMILY, 10))
        st.configure("Note.TLabel", background=CARD, foreground=MUTED, font=(UI_FAMILY, 8))
        st.configure("TCheckbutton", background=BG)
        st.configure("TRadiobutton", background=BG)
        st.configure("Primary.TButton", font=(UI_FAMILY, 10, "bold"), padding=(14, 4))

    def _build(self) -> None:
        # row 1: title, Start/Stop, status pill (always visible, never clipped)
        top = ttk.Frame(self.root, padding=(14, 10, 14, 4))
        top.pack(fill="x")
        ttk.Label(top, text="English → Hindi live translation", style="Title.TLabel").pack(side="left")
        self.state_lbl = tk.Label(top, text=PILL["idle"][0], bg=PILL["idle"][1], fg="#ffffff",
                                  font=(UI_FAMILY, 10, "bold"), padx=12, pady=4)
        self.state_lbl.pack(side="right")
        self.stop_btn = ttk.Button(top, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="right", padx=(0, 10))
        self.start_btn = ttk.Button(top, text="Start", command=self.start, style="Primary.TButton")
        self.start_btn.pack(side="right", padx=(0, 6))

        # row 2: source options
        bar = ttk.Frame(self.root, padding=(14, 0, 14, 8))
        bar.pack(fill="x")
        ttk.Label(bar, text="Source:").pack(side="left")
        self.source = tk.StringVar(value="microphone")
        ttk.Radiobutton(bar, text="Microphone", variable=self.source, value="microphone").pack(side="left", padx=4)
        ttk.Label(bar, text="device").pack(side="left", padx=(6, 2))
        self.device = tk.StringVar()
        ttk.Entry(bar, textvariable=self.device, width=6).pack(side="left")
        ttk.Radiobutton(bar, text="WAV file", variable=self.source, value="wav").pack(side="left", padx=(14, 4))
        self.wav_path = tk.StringVar()
        ttk.Entry(bar, textvariable=self.wav_path, width=30).pack(side="left", padx=2)
        ttk.Button(bar, text="Browse...", command=self._browse).pack(side="left")
        self.save_audio = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Save audio", variable=self.save_audio).pack(side="left", padx=(14, 4))
        self.provisional = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Show provisional English (may be wrong)",
                        variable=self.provisional).pack(side="left", padx=4)

        # banner: stall / error (red) or recovery (green); hidden when empty
        self.banner = tk.Label(self.root, text="", anchor="w", font=(UI_FAMILY, 11, "bold"), padx=14, pady=6)
        self.status = self.banner           # (name kept for hosts/tests reading the status text)

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self._body = body
        left = ttk.Frame(body, style="Card.TFrame")
        body.add(left, weight=3)
        self.text = tk.Text(left, wrap="word", state="disabled", padx=18, pady=12, relief="flat", borderwidth=0,
                            background=CARD, foreground=INK, font=(UI_FAMILY, 11), spacing1=1, spacing3=1,
                            cursor="arrow")
        scroll = ttk.Scrollbar(left, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)
        self._scroll = scroll
        # video view of the same transcript (shown instead of the text widget while a camera picture exists)
        self.canvas = tk.Canvas(left, background=CARD, highlightthickness=0, borderwidth=0, cursor="arrow")
        self._video_item = self.canvas.create_image(0, 0, anchor="nw", tags=("video",))
        self.canvas.bind("<Configure>", lambda e: self._video_on and self._render_canvas())
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))
        t = self.text
        t.tag_configure("seg", foreground=MUTED, font=(UI_FAMILY, 8, "bold"), spacing1=14)
        t.tag_configure("en", foreground="#374151", font=(UI_FAMILY, 11), spacing1=3, lmargin1=2, lmargin2=2)
        t.tag_configure("prov", foreground="#9ca3af", font=(UI_FAMILY, 11, "italic"))
        t.tag_configure("badge_final", foreground="#ffffff", background="#2e7d32", font=(UI_FAMILY, 8, "bold"))
        t.tag_configure("badge_draft", foreground="#5c3c00", background="#ffe08a", font=(UI_FAMILY, 8, "bold"))
        t.tag_configure("badge_wait", foreground=MUTED, font=(UI_FAMILY, 9, "italic"))
        t.tag_configure("hi_final", foreground=FINAL_INK, font=(HINDI_FAMILY, 15, "bold"), spacing1=4, spacing3=6,
                        lmargin1=2, lmargin2=2)
        t.tag_configure("hi_draft", foreground=DRAFT_INK, font=(HINDI_FAMILY, 13, "italic"), spacing1=4, spacing3=6,
                        lmargin1=2, lmargin2=2)
        t.tag_configure("label", foreground=MUTED, font=(UI_FAMILY, 8))
        t.tag_configure("err", foreground="#b00020", font=(UI_FAMILY, 9))
        t.tag_configure("rule", foreground="#e5e7eb", font=(UI_FAMILY, 4))
        t.tag_configure("empty", foreground=MUTED, font=(UI_FAMILY, 11, "italic"), spacing1=24)

        right = ttk.Frame(body, style="Card.TFrame", padding=(16, 12))
        body.add(right, weight=1)
        ttk.Label(right, text="Performance (this session)", style="PanelHead.TLabel").pack(anchor="w")
        ttk.Label(right, text="Measured from the detected start of speech. 'collecting' = not enough samples yet.",
                  style="Note.TLabel", wraplength=360, justify="left").pack(anchor="w", pady=(2, 8))
        self.panel = ttk.Frame(right, style="Card.TFrame")
        self.panel.pack(fill="x")
        ttk.Separator(right).pack(fill="x", pady=10)
        ttk.Label(right, style="Note.TLabel", wraplength=360, justify="left", text=(
            "Printed = when the line appeared; event time = when the pipeline produced it.\n"
            "No live accuracy is shown: WER needs a reference transcript (offline, scripted recordings only).\n"
            "MT errors count failed translation calls only - zero errors does not mean the translation "
            "is correct.")).pack(anchor="w")
        ttk.Button(right, text="Load reference WER report...", command=self._load_score).pack(anchor="w", pady=(10, 0))
        self._render_transcript()

    def _browse(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("WAV audio", "*.wav")])
        if path:
            self.wav_path.set(path)
            self.source.set("wav")

    def _load_score(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("script_score report", "*.json")])
        if not path:
            return
        try:
            self.metrics.load_reference_score(path)
        except Exception as exc:
            messagebox.showerror("Reference WER", f"Could not read the report: {exc}")
        self._render_panel()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.session is not None and self.session.running:
            return
        wav = self.wav_path.get().strip() if self.source.get() == "wav" else None
        if self.source.get() == "wav" and not wav:
            messagebox.showwarning("Source", "Choose a WAV file or select Microphone.")
            return
        options = SessionOptions(wav=wav, device=self.device.get().strip() or None, save_audio=self.save_audio.get(),
                                 show_provisional_english=self.provisional.get(), session_name="ui")
        session = TranslationSession(options)
        try:
            session.start()
        except SessionBusy as exc:
            messagebox.showerror("Busy", str(exc))
            return
        self.session = session
        self.model = TranscriptModel(session.session_id)     # new session: old events can never apply
        self.metrics = LiveMetrics(session.session_id)
        self._rendered_version = -1
        self._stopping = False
        self._notice_until = 0.0
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

    def stop(self) -> None:
        if self.session is None:
            return
        self.stop_btn.configure(state="disabled")
        self._stopping = True
        self._set_pill("stopping")
        # stop() blocks while ASR/MT drain; run it off the Tk thread
        session = self.session

        def stop_or_terminate() -> None:
            if not session.stop(timeout_s=120):
                session.terminate()       # never stay "stopping" forever; the session ends as failed
        threading.Thread(target=stop_or_terminate, daemon=True, name="ui-stop").start()

    def on_close(self) -> None:
        if self.session is not None and self.session.running:
            if not messagebox.askokcancel("Quit", "Stop the running session and quit?"):
                return
            # drain off the Tk thread; _poll closes the window once the session has ended
            self._stopping = True
            self._set_pill("stopping")
            self._quit_when_done = True
            threading.Thread(target=self._stop_for_quit, daemon=True, name="ui-quit").start()
            return
        self._destroy()

    def _destroy(self) -> None:
        if self.camera is not None:
            self.camera.close()
        self.root.destroy()

    def _stop_for_quit(self) -> None:
        session = self.session
        if not session.stop(timeout_s=60):
            session.terminate()           # worker did not finish draining in time (phase 7)
            session.stop(timeout_s=15)

    # ------------------------------------------------------------------ rendering
    def _set_pill(self, key: str) -> None:
        text, bg = PILL[key]
        self.state_lbl.configure(text=text, bg=bg)

    def _set_banner(self, text: str | None, kind: str = "error") -> None:
        if not text:
            if self.banner.winfo_manager():
                self.banner.pack_forget()
            self.banner.configure(text="")
            return
        fg, bg = (GREEN, GREEN_BG) if kind == "ok" else (RED, RED_BG)
        self.banner.configure(text=text, fg=fg, bg=bg)
        if not self.banner.winfo_manager():
            self.banner.pack(fill="x", padx=14, pady=(0, 8), before=self._body)

    def _poll(self) -> None:
        if self.session is not None:
            for event in self.session.poll_events():
                was_stalled = self.model.audio_warning is not None
                self.model.apply(event)
                self.metrics.apply(event)
                if was_stalled and self.model.audio_warning is None and self.model.audio_notice:
                    self._notice_until = time.monotonic() + NOTICE_S
            state = self.session.state
            stalled = state is SessionState.READY and self.model.audio_warning is not None
            if stalled:
                self._set_pill("no_audio")
            elif self._stopping and state in (SessionState.READY, SessionState.STARTING):
                self._set_pill("stopping")
            else:
                self._set_pill(state.value if state.value in PILL else "idle")
            if state in (SessionState.STOPPED, SessionState.FAILED) and not self.session.running:
                self.start_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                self._stopping = False
            errors = self.model.errors + self.model.problems
            if self.model.audio_warning:
                self._set_banner(self.model.audio_warning, "error")
            elif errors:
                self._set_banner(errors[-1], "error")
            elif self.model.audio_notice and time.monotonic() < self._notice_until:
                self._set_banner(self.model.audio_notice, "ok")
            else:
                self._set_banner(None)
            if self.model.version != self._rendered_version:
                self._render_transcript()
                self._render_panel()
                self._rendered_version = self.model.version
            elif self.metrics.resources:
                self._render_panel()
            if self._quit_when_done and not self.session.running:
                self._destroy()
                return
        self.root.after(POLL_MS, self._poll)

    def _transcript_runs(self) -> list[tuple[str, str]]:
        """The transcript as (text, tag) runs - one definition for both views."""

        runs: list[tuple[str, str]] = []
        segments = self.model.ordered_segments()
        if not segments and not self.model.provisional:
            runs.append(("Press Start and speak English - the transcript and its Hindi translation "
                         "appear here.\n", "empty"))
        for seg in segments:
            runs += self._segment_runs(seg)
        for uid, text in sorted(self.model.provisional.items()):
            runs += self._provisional_runs(text)
        return runs

    @staticmethod
    def _segment_runs(seg, provisional: str | None = None) -> list[tuple[str, str]]:
        runs = [(f"SEGMENT {seg.segment_id}" + ("" if seg.closed else "   · listening…") + "\n", "seg"),
                (" ".join(seg.english) + "\n", "en")]
        if provisional:
            runs += TranslatorApp._provisional_runs(provisional)
        if seg.hindi_final is not None:
            runs.append((" FINAL ", "badge_final"))
            runs.append(("\n" + seg.hindi_final + "\n", "hi_final"))
        elif seg.hindi_draft is not None:
            runs.append((" DRAFT – may change ", "badge_draft"))
            runs.append(("\n" + seg.hindi_draft + "\n", "hi_draft"))
        else:
            runs.append(("translating…\n", "badge_wait"))
        for err in seg.errors:
            runs.append((err + "\n", "err"))
        return runs

    @staticmethod
    def _provisional_runs(text: str) -> list[tuple[str, str]]:
        return [("provisional - may change  ", "label"), (text + "\n", "prov")]

    def _subtitle_runs(self) -> list[tuple[str, str]]:
        """Netflix-style: only the active segment (English + its Hindi). A
        provisional line belongs to the open segment; once that segment has
        closed, a provisional line starts the next subtitle on its own."""

        segments = self.model.ordered_segments()
        prov = self.model.provisional
        latest_prov = prov[max(prov)] if prov else None
        seg = segments[-1] if segments else None
        if seg is None or (seg.closed and latest_prov):
            return self._provisional_runs(latest_prov) if latest_prov else []
        return self._segment_runs(seg, None if seg.closed else latest_prov)

    def _render_transcript(self) -> None:
        t = self.text
        first = self._rendered_version < 0
        at_end = first or t.yview()[1] > 0.98
        t.configure(state="normal")
        t.delete("1.0", "end")
        for text, tag in self._transcript_runs():
            t.insert("end", text, tag)
        t.configure(state="disabled")
        if at_end:
            t.see("end")
        if self._video_on:
            self._render_canvas(follow=first or None)

    # ------------------------------------------------------------------ video background
    def _video_tick(self) -> None:
        try:
            frame = self.camera.latest()
            if (frame is not None) != self._video_on:
                self._show_video(frame is not None)
            if frame is not None:
                self._draw_frame(frame)
        except tk.TclError:
            return                        # window closed
        self.root.after(int(1000 / VIDEO_FPS), self._video_tick)

    def _show_video(self, on: bool) -> None:
        """Swap the transcript view: canvas over the camera picture, or the plain text widget."""

        self._video_on = on
        if on:
            self.text.pack_forget()
            self.canvas.pack(fill="both", expand=True)
            self.canvas.configure(yscrollcommand=self._scroll.set)
            self._scroll.configure(command=self.canvas.yview)
            self.root.update_idletasks()
            self._render_canvas(follow=True)
        else:
            self.canvas.pack_forget()
            self.text.pack(fill="both", expand=True)
            self._scroll.configure(command=self.text.yview)
            self.text.configure(yscrollcommand=self._scroll.set)
            self.text.see("end")

    def _tag(self, tag: str, option: str, default=""):
        value = self.text.tag_cget(tag, option)
        return value if value not in ("", None) else default

    def _render_canvas(self, follow: bool | None = None) -> None:
        """Netflix-style subtitle over the video: only the active segment's runs
        (`_subtitle_runs`), with the text widget's own tag styles, centred near
        the bottom. Long text keeps its newest words within SUB_MAX_LINES (older
        words drop off the front as "…"), so the block never grows past a few
        lines; one translucent veil sits behind the whole block."""

        c = self.canvas
        c.delete("content")
        self._veil = []
        base_font = self.text.cget("font")
        w, h = max(c.winfo_width(), 200), max(c.winfo_height(), 100)
        maxw = int(min(w * SUB_WIDTH, w - 40))
        # split into display lines like the text widget: a line's newline character
        # belongs to the run that contains it (it sets the line height and spacing3)
        lines: list[list[tuple[str, str]]] = [[]]
        ends: list[str | None] = [None]
        for text, tag in self._subtitle_runs():
            for i, part in enumerate(text.split("\n")):
                if i:
                    ends[-1] = tag
                    lines.append([])
                    ends.append(None)
                if part:
                    lines[-1].append((part, tag))
        fonts: dict[str, tkfont.Font] = {}

        def font(tag: str) -> tkfont.Font:
            spec = str(self._tag(tag, "font", base_font))
            if spec not in fonts:
                fonts[spec] = tkfont.Font(root=self.root, font=spec)
            return fonts[spec]

        y = 0
        for line, end in zip(lines, ends):
            if not line:
                continue
            first, last = line[0][1], end or line[-1][1]
            y += int(self._tag(first, "spacing1", self.text.cget("spacing1")))
            used = [font(t) for _, t in line] + ([font(end)] if end else [])
            ascent = max(f.metrics("ascent") for f in used)          # common baseline, as in the text widget
            height = max(f.metrics("linespace") for f in used)
            x = 0
            bottom = y + height
            ids = []
            for part, tag in line:
                f = font(tag)
                item = c.create_text(x, y + ascent - f.metrics("ascent"), text=part, anchor="nw", justify="center",
                                     font=self._tag(tag, "font", base_font),
                                     fill=self._tag(tag, "foreground", self.text.cget("foreground")),
                                     width=max(40, maxw - x), tags=("content",))
                self._fit_lines(item, f, SUB_MAX_LINES.get(tag))
                box = c.bbox(item)
                top, low = min(y, box[1]), max(y + height, box[3])
                ids.append(item)
                bg = self._tag(tag, "background", None)
                if bg:                                               # tag background spans the line height
                    rect = c.create_rectangle(box[0], top, box[2], low, fill=bg, outline=bg, tags=("content",))
                    c.tag_lower(rect, item)
                    ids.append(rect)
                x, bottom = box[2], max(bottom, low)
            for i in ids:                                            # centre the line horizontally
                c.move(i, (w - x) / 2, 0)
            y = bottom + int(self._tag(last, "spacing3", self.text.cget("spacing3")))
        block = c.bbox("content")
        if block:                                                    # bottom-centre, like film subtitles
            c.move("content", 0, h * (1 - SUB_BOTTOM) - block[3])
            x0, y0, x1, y1 = c.bbox("content")
            self._veil = [(x0 - 3 * VEIL_PAD, y0 - 2 * VEIL_PAD, x1 + 3 * VEIL_PAD, y1 + 2 * VEIL_PAD)]
        c.configure(scrollregion=(0, 0, w, h))
        c.yview_moveto(0)
        c.tag_lower("video")

    def _fit_lines(self, item: int, font: tkfont.Font, max_lines: int | None) -> None:
        """Keep a wrapped subtitle line within max_lines: drop the oldest words (shown as "…")."""

        if not max_lines:
            return
        c = self.canvas
        limit = max_lines * font.metrics("linespace") + 2
        words = c.itemcget(item, "text").split()
        while len(words) > 1:
            x0, y0, x1, y1 = c.bbox(item)
            if y1 - y0 <= limit:
                return
            words = words[1:]
            c.itemconfigure(item, text="… " + " ".join(words))

    def _draw_frame(self, frame) -> None:
        """Camera picture scaled to cover the canvas, with the white veil behind the text lines."""

        import cv2

        c = self.canvas
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10 or h < 10:
            return
        fh, fw = frame.shape[:2]
        scale = max(w / fw, h / fh)
        img = cv2.resize(frame, (max(w, int(fw * scale + 0.5)), max(h, int(fh * scale + 0.5))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        oy, ox = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
        img = img[oy:oy + h, ox:ox + w].copy()
        vx, vy = int(c.canvasx(0)), int(c.canvasy(0))
        mask = None
        for x0, y0, x1, y1 in self._veil:
            x0, x1, y0, y1 = max(0, x0 - vx), min(w, x1 - vx), max(0, y0 - vy), min(h, y1 - vy)
            if x0 < x1 and y0 < y1:
                if mask is None:
                    import numpy as np

                    mask = np.zeros((h, w), dtype=bool)
                mask[y0:y1, x0:x1] = True
        if mask is not None:
            region = img[mask].astype("float32")
            img[mask] = (region * (1.0 - VEIL_ALPHA) + 255.0 * VEIL_ALPHA).astype("uint8")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._photo = tk.PhotoImage(data=b"P6 %d %d 255 " % (w, h) + rgb.tobytes(), format="PPM")
        c.itemconfigure(self._video_item, image=self._photo)
        c.coords(self._video_item, vx, vy)
        c.tag_lower("video")

    def _render_panel(self) -> None:
        rows = self.metrics.rows()
        key = tuple((r.label, r.value, r.state) for r in rows)
        if key == self._panel_key:
            return                        # unchanged: avoid flicker
        self._panel_key = key
        for child in self.panel.winfo_children():
            child.destroy()
        for i, row in enumerate(rows):
            ttk.Label(self.panel, text=row.label, style="RowLabel.TLabel").grid(row=2 * i, column=0, sticky="w",
                                                                                pady=(6 if i else 0, 0))
            colour = {"collecting": "#888888", "n/a": "#888888", "partial": "#7a5c00"}.get(row.state, INK)
            ttk.Label(self.panel, text=row.value, style="RowValue.TLabel", foreground=colour, wraplength=360,
                      justify="left").grid(row=2 * i + 1, column=0, sticky="w", padx=(10, 0))


def main() -> int:
    root = tk.Tk()
    TranslatorApp(root)
    root.mainloop()
    return 0
