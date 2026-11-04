# CLAUDE.md — MAVIS translation module (demo handoff package)

Read this before changing anything. This folder is a **self-contained copy** of the standalone English → Hindi live speech translation product. It exists so that this module can be demonstrated and integrated into MAVIS by a later integration phase.

---

## A. Purpose and boundaries

**What it does.** Live English speech, from a microphone or a 16 kHz WAV replayed at 1×, is turned into:
- English text: words confirmed by agreement between successive decodes;
- Hindi drafts while the sentence is still being spoken;
- one **final Hindi** translation per segment, never replaced once shown.

It runs on CPU only (Windows, Python 3.10). It comes with a desktop UI (Tkinter), a console CLI, and a session API for a future host.

**What it does not do.**
- **No OCR**, and no other MAVIS module. **The complete MAVIS host is external** and does not exist in this folder.
- No Hindi → English, no other language pairs, no speaker diarization, no live accuracy or WER.
- No network service: the product is a desktop program plus a Python API.

**What this folder is.** A demo and integration package, **not** a replacement for the research repository the module came from. Research datasets, evaluation outputs, benchmark results, manuscripts and earlier session recordings were deliberately **not** copied.

**Strict rules for agents working here:**
1. **Never edit the frozen engine files.** These are `main.py`, `asr/`, `audio/`, `vad/`, `streaming/`, `translation/`, `utils/` and `evaluation/`. They are byte-identical copies of the frozen research runtime: `SOURCE_PROVENANCE.json` lists each with its SHA-256, and `tests_smoke` fails if one changes. Product behaviour is changed only in `product/` and its config.
2. Do not edit or "re-generate" `evaluation/mt3c_parameters.json`. Its SHA-256 is checked at every start.
3. Do not touch the original research repository: this package must never import from it or write to it.
4. Keep `product.integration.live_translation_enabled: false` until the MAVIS integration phase explicitly decides otherwise.
5. Do not claim live-microphone acceptance from replay or import tests (see `DEMO_READINESS.md`).

---

## B. Pipeline architecture (verified against the code)

```
microphone (sounddevice, 16 kHz mono float32, 100 ms frames)        product/sources.py LiveMicSource
  | or a WAV replayed at 1x                                         product/sources.py PacedWavSource
  v
main.run_m2()  -- frozen M2 streaming ASR, driven by the product    main.py (+ product/live_pipeline.py hooks)
  |  Silero VAD (speech start 0.6 / end 0.3, 100 ms min speech/silence)   vad/vad.py
  |  StreamingBuffer: a window per >= 2000 ms new audio, 8000 ms ceiling, 300 ms pre-roll   streaming/streaming_buffer.py
  |  Whisper tiny int8, faster-whisper, language en, beam 1           asr/whisper_engine.py, asr/worker.py (thread)
  |  PrefixAgreementTracker (LocalAgreement-2): stable English commits   streaming/prefix_agreement.py
  v
StableUpdateHook (product layer, wraps the frozen objects without editing them)   product/live_pipeline.py
  |  boundary repair, repetition guard, trim_keep_s context, prompt rules,
  |  whole-utterance final window, cross-utterance dedup
  v
SegmentController (thread): groups commits into translation segments        product/live_pipeline.py
  |  MT3-C scheduler (frozen AdaptiveDeadlinePolicy, evaluation/mt3c_parameters.json)  translation/scheduling.py
  |      -> opus-mt-en-hi DRAFT calls (MTWorker thread)               translation/translator.py
  |  segment closes -> NLLB-200 600M int8 FINAL call (2nd MTWorker)   product/final_mt.py
  v
Display: console lines + transcript.jsonl + structured records        product/live_pipeline.py Display
  v
run_session() -> (exit code, summary); session folder files            product/live_translate.py
  v
session_worker process: JSON lines on its stdout                       product/session_worker.py
  v
TranslationSession (host process): events "mavis.live_translation.v1" product/session.py
  v
TranscriptModel + LiveMetrics -> Tk UI                                 product/ui_model.py, metrics_panel.py, ui/app.py
```

**Process model.**
- **Every session runs in its own worker process** (`python -m product.session_worker`), started by `TranslationSession`. Models load in that process and its memory (about 1.6 GB peak) returns to the OS when the session ends. The UI process stays at about 35 MB and never imports torch.
- The worker starts with this package as its working directory and `PYTHONPATH`, so imports always resolve here.
- A stop request goes to the worker's stdin. The worker drains ASR and MT, closes the last segment, writes all files, sends its result and exits.
- If the host disappears, the worker's stdin closes and the worker stops and finalizes on its own.
- `SessionOptions(isolation="thread")` runs in-process instead. That mode exists for tests and debugging; it leaks about 1.3 GB of native memory per session.

**Threads inside a session:** main (VAD + run loop), ASR worker, segment controller, MT draft worker, NLLB final worker, resource sampler.

**Model assets** (bundled under `models/`; `product/__init__.py` points `HF_HOME` there and sets offline mode):

| model | role | location |
|---|---|---|
| Systran/faster-whisper-tiny @ d90ca5fe | ASR | `models/huggingface/hub/models--Systran--faster-whisper-tiny` |
| Helsinki-NLP/opus-mt-en-hi @ 75d7f7c9 | Hindi drafts | `models/huggingface/hub/models--Helsinki-NLP--opus-mt-en-hi` |
| facebook/nllb-200-distilled-600M @ f8d333a0 | tokenizer for finals | `models/huggingface/hub/models--facebook--nllb-200-distilled-600M` (tokenizer files only) |
| NLLB-200 distilled 600M, CTranslate2 int8 | final Hindi | `models/nllb-200-distilled-600M-ct2-int8` |
| Silero VAD | VAD | inside the `silero-vad==6.2.1` pip package |

---

## C. Setup and run (PowerShell, from this folder)

```powershell
cd "<path>\mavis_translation_demo_handoff"

py -3.10 --version                     # need Python 3.10.x 64-bit with tcl/tk (tested 3.10.11)
.\setup_demo.ps1                       # creates .\.venv, installs CPU torch + requirements, checks, verifies models
.\setup_demo.ps1 -Python "C:\Path\To\python.exe"   # if py/python on PATH is not 3.10
.\setup_models.ps1                     # verify bundled models (size + SHA-256)
.\setup_models.ps1 -Load               # ...and load each model once
.\.venv\Scripts\python.exe -B scripts\check_environment.py   # imports resolve in this package, flag off, paths local

.\.venv\Scripts\python.exe -m product.ui        # DESKTOP UI (default demo); same as .\run_demo.ps1

.\run_cli_demo.ps1 --list-devices      # microphones (* = system default)
.\run_cli_demo.ps1                     # console translator, default mic; Ctrl+C = graceful stop
.\run_cli_demo.ps1 --device 1 --save-audio
.\scripts\make_demo_wav.ps1            # Windows-TTS reading of the demo script -> runtime_data\demo\demo_script_tts.wav
.\run_cli_demo.ps1 --wav runtime_data\demo\demo_script_tts.wav --duration 40

.\.venv\Scripts\python.exe -m pytest tests_smoke -q            # fast smoke tests
.\.venv\Scripts\python.exe -m pytest tests_smoke -q -m models  # + end-to-end replay with real models
.\.venv\Scripts\python.exe -m pytest -q                        # smoke + carried-over product tests
```

**CLI options** (`python -m product.live_translate --help`):
- `--wav FILE`: replay a 16 kHz WAV at 1×.
- `--device IDX|NAME`: choose the microphone.
- `--duration S`: stop after S seconds of audio.
- `--save-audio`: write `input.wav` in the session folder.
- `--session-name NAME`: add a suffix to the session folder name.
- `--set KEY=VALUE`: override a config value for this session, e.g. `--set asr.model_size=base`.
- `--policy mt3c|eager|fixed`, `--fixed-k N`: draft MT scheduling.
- `--list-devices`, `--config FILE`, `--session-id ID`.
- `--system m1` is **not packaged**: the M1 baseline config was not copied.

**Stopping safely:**
- UI: press **Stop**. The status pill shows STOPPING until the drain finishes; closing the window also stops and drains.
- CLI: press Ctrl+C once.
- API: `session.stop()`.

A stop always drains ASR and MT, closes the last segment and writes every file. Do not kill the process if you want the files.

**Session output:** `runtime_data\sessions\<YYYYMMDD-HHMMSS>_m2_<mic|wav>[_name]\`. See §E for the files.

**Missing models:**
- startup error, or `setup_models.ps1` reports MISSING: restore the `models\` folder from the handoff copy;
- or run `.\setup_models.ps1 -Download`: needs internet, about 3.5 GB transient download, NLLB conversion; this fallback was not exercised during packaging.

**No microphone:**
- `--list-devices` shows none, or the session fails at start with "Input device … is not usable": the device check runs before any model loads;
- check Windows sound settings and privacy (microphone access for desktop apps), then pick the device with `--device`, or the UI's "device" field.

---

## D. Configuration (`product/configs/live_m2_mt3c.yaml`; values as shipped)

| area | key | shipped value | notes |
|---|---|---|---|
| audio | `audio.device` | `null` (system default) | UI "device" field / `--device` |
| audio | `audio.sample_rate`, `chunk_duration_ms` | 16000, 100 | do not change (VAD and Whisper expect 16 kHz) |
| audio saving | (option) | off in CLI; **on by default in the UI** ("Save audio") | `--save-audio`; writes `input.wav` in the session folder |
| VAD | `vad.speech_start_threshold` / `speech_end_threshold` | 0.6 / 0.3 | min speech/silence 100 ms |
| ASR | `asr.model_size`, `compute_type`, `language`, `beam_size`, `cpu_threads` | tiny, int8, en, 1, 8 | `--set asr.model_size=base` is more accurate but needs that model downloaded and AC power |
| streaming | `streaming_buffer.min_new_audio_ms`, `max_buffer_ms`, `pre_roll_ms` | 2000, 8000, 300 | 2000 ms kept after A/B tests (1000 ms: more errors) |
| agreement | `prefix_agreement.agreement_depth` | 2 | LocalAgreement-2 |
| draft MT | `translation.model`, `revision` | Helsinki-NLP/opus-mt-en-hi @ 75d7f7c9 | en → hi, greedy |
| scheduler | `product.policy`, `mt3c_parameters(_sha256)` | mt3c, `evaluation/mt3c_parameters.json` (SHA checked) | `eager` / `fixed` exist for comparison only |
| final MT | `product.final_mt.enabled`, `model_dir`, `threads` | true, `models/nllb-200-distilled-600M-ct2-int8`, 4 | NLLB-200 600M int8 |
| segmentation | `product.segment.*` | close_after_words 20, idle_close_s 2.5, hard_max_words 60, sentence_gap_s 0.6, sentence_min_words 5, long_close_needs_pause true, long_close_max_words 32 | see §F for the trade-off |
| ASR context | `product.trim_keep_s`, `prompt_min_words` | 1.0, 3 | |
| final window | `product.final_window_full_max_s` | 8.0 | final window = whole utterance, no prompt, if ≤ 8 s |
| dedup | `product.cross_utterance_dedup` | true | drops a word duplicated across a contiguous VAD cut |
| guards | `boundary_repair.enabled`, `repetition_guard.enabled`, `vad_reset_after_silence_s` | true, true, 2.0 | |
| camera background | env `MAVIS_UI_CAMERA` (not in the YAML) | `0` = first camera | the desktop UI shows the speaker's camera behind the transcript with Netflix-style subtitles: only the active segment (English + its Hindi draft/final, same text styling) bottom-centre, replaced by the next segment, capped at 2/3 lines, white readability veil; the full history stays in the model and in the plain view; a device index, a video file path, or `off`. No live picture means the plain white transcript view. Read in the UI process only (OpenCV); the translation worker never touches the camera |
| display | `product.display_provisional_en` | false | UI checkbox "Show provisional English" (labelled, may be wrong, never translated) |
| stall | `product.audio_stall_s` | 2.0 | no audio frames for 2 s → NO AUDIO warning (`null` = off) |
| output | `product.output_root` | `runtime_data/sessions` | relative to this folder; an absolute path is allowed |
| host flag | `product.integration.live_translation_enabled` | **false** | `create_host_session()` refuses while false; the standalone UI/CLI ignore it |

Per-session overrides: `--set` (CLI) or `SessionOptions(overrides=[...])`. Only existing keys can be overridden, except `product.*`.

---

## E. Session API and event contract (`product/session.py`, `INTEGRATION.md`)

```python
from product.session import SessionOptions, TranslationSession, create_host_session, feature_enabled

session = create_host_session(SessionOptions(device=None, save_audio=True), on_event=None)  # FeatureDisabled while the flag is off
# standalone (no flag check): TranslationSession(SessionOptions(...))
session.start()                 # returns at once; worker loads models (~8 s AC, ~14-17 s battery)
for e in session.poll_events(): # non-blocking; call from the host's UI loop (e.g. every 100 ms)
    ...                         # or session.iter_events() (blocking), or on_event=callback (worker-reader thread)
session.stop(timeout_s=120)     # drain + finalize; True when the worker exited; idempotent; OK during loading
session.terminate()             # last resort if stop() timed out (session ends 'failed')
session.summary                 # = session_summary.json
```

`SessionOptions` fields:
- `wav`, `device`, `save_audio`, `show_provisional_english`;
- `session_name`, `config`, `overrides`, `duration_s`;
- `isolation` (`"process"` default).

**Rules:**
- One session at a time per host: a second `start()` raises `SessionBusy`.
- A `TranslationSession` object starts once.

**Events** (`schema: "mavis.live_translation.v1"`). Every event carries:
- `session_id` (12 hex characters). **Drop events of any other session.**
- `seq`, strictly increasing 1, 2, 3, …
- `type`;
- `t_s`: session audio seconds;
- `render_s`, on display events.

| type | meaning |
|---|---|
| `session_state` | `starting` → `ready` (`ready_in_s`, `info.on_ac_power`) → `stopping` → `stopped` (`stop_reason`, `cleanup_ok`) / `failed` (`error`). Never `ready` after `stopping`. |
| `en_committed` | `segment_id`, `utterance_id`, `new_text`, `stable_text` (the segment's whole confirmed English) |
| `en_provisional` | opt-in unconfirmed hypothesis; never translated |
| `hi_draft` | provisional Hindi for `segment_id`; replaced by later drafts |
| `hi_final` | exactly one per segment, after `segment_closed`; never replaced. `kind`: `final_mt` (NLLB) or `draft`/`draft_fallback` |
| `hi_draft_late` | a draft that finished after the final: **do not show** |
| `segment_closed` | `reason`, `utterance_ids`, `final_source` |
| `mt_error`, `final_mt_error` | a translation call failed (not a quality signal) |
| `asr_decode`, `utterance_timing`, `resources` | metrics (`resources` about 1/s: CPU cores, working set, dropped frames) |
| `audio_stall` / `audio_resumed` | the microphone delivered **no frames** for 2 s (not silence) / frames again |
| `cleanup_error` | audio stream stop/close failed (e.g. device unplugged); the summary has `cleanup.ok=false` |
| `session_summary` | the full summary (authoritative metrics, `user_latency`) |
| `error` | exception; `where="worker"` when the worker crashed or exited early (message names the session folder) |

`product.ui_model.TranscriptModel` applies these rules (stale and foreign events, finals immutable, late drafts ignored, stall warning). `product.metrics_panel.LiveMetrics` computes the panel with the same latency code as the summary.

**Session folder files:**
- **written immediately:** `session_start.json` (session_id, environment incl. power, argv, effective config) and `lifecycle.jsonl` (created, loading, stop_requested, ready, audio_stall, cleanup_error, pipeline_finished, finalized/failed);
- **written incrementally:** `transcript.jsonl`, `m2.log`, `m2_events.jsonl`, `m2_console.txt`;
- **written at the end:** `decisions.jsonl`, `segments.json`, `asr_hypotheses.jsonl`, `resource_samples.json`, `session_summary.json`;
- **optional:** `input.wav`.

A folder whose `lifecycle.jsonl` lacks `finalized`/`failed` belongs to a session that is still running or was killed.

**Microphone ownership (integration).** A session opens the input device itself (sounddevice/PortAudio) in its worker. A host that also captures audio, or runs OCR or other modules that use the microphone, must coordinate. Either give this module exclusive use of the device during a session, or implement audio routing in a later phase; routing does not exist yet. Do not let several modules open the microphone independently.

**Resources.** One worker peaks at about **1.6 GB** working set and uses about 2–2.6 CPU cores on average (p95 4–6 of 8) on the development laptop. **Concurrent load with OCR has not been measured.**

**Enabling for a host** (integration phase only, not now): set `product.integration.live_translation_enabled: true` in a copy of the config and pass it as `SessionOptions(config=...)`, or edit the product config. Then use `create_host_session`.

---

## F. Known limitations (Phase 8A/9 evidence; recorded audio unless stated)

- **Whisper tiny misrecognizes words**, e.g. "the library of the office" for "reliable software", "student" for "students", "the phrase", "cutting of". A larger model would fix some of these but is slower; the default stays tiny.
- **Hindi finals can contain grammatical slips or omissions**:
  - gender agreement, e.g. "पिछली वाक्य", "एक नई कार्य";
  - "the system" dropped from a first clause;
  - occasional odd word choices.

  There is no Hindi reference or native-speaker review. **MT errors = 0 only means no translation call failed.**
- **Spontaneous speech is clearly less reliable** than the scripted read used for acceptance.
- **Not instantaneous** (replays on AC):
  - first confirmed English about 3.8–4.3 s after speech onset;
  - first Hindi draft about 4.9–5.6 s;
  - final Hindi about 2.3–2.6 s after the last word (median);
  - the longest segment reaches its final about 17–20 s after its first word.

  Battery is slower than AC.
- **Segment trade-offs:**
  - sentences under 5 words wait for the 2.5 s idle close or join the next sentence;
  - segments are capped by length (32 words), so reading pauses shorter than 0.6 s can still join two or three sentences.
- **The UI's metrics are timing and resource measurements**: they say nothing about correctness. There is no live WER.
- **Concurrency with OCR or other MAVIS modules is not established.**
- **Camera background:** verified with generated video files, not with a physical webcam. It costs some UI-process CPU (15 fps redraw). The camera is opened when the UI starts and released when it closes; set `MAVIS_UI_CAMERA=off` to disable it.
- **Tk Devanagari rendering:** Tk occasionally draws a vowel sign with a dotted circle (e.g. "स◌े").
- **No live-microphone run** has been done with the final Phase 9 build. Its fixes were verified by replaying recordings.

---

## G. Troubleshooting and safety

| symptom | what to do |
|---|---|
| `setup_demo.ps1`: Python 3.10 not found | install Python 3.10 x64 with tcl/tk, or `-Python <path>` |
| "no tkinter" | reinstall Python with "tcl/tk and IDLE" |
| first start slow | models load per session: about 8 s on AC, 14–17 s on battery (status pill: STARTING) |
| `OSError`/`LocalEntryNotFoundError` mentioning a model | `.\setup_models.ps1` (shows MISSING/CHANGED files); restore `models\` |
| "Input device ... is not usable" | wrong or absent microphone; `.\run_cli_demo.ps1 --list-devices`, then set the device |
| red **NO AUDIO** pill and banner | the microphone stopped delivering frames (unplugged, muted at driver level, taken by another app). The session keeps running: reconnect it (a green "audio resumed" banner follows), or press Stop. Silence does **not** cause this. |
| session ends `failed`, "translation worker exited…" | the worker crashed or was killed; read the named session folder's `lifecycle.jsonl` and `m2.log`; models and memory are the usual causes |
| Stop takes long | it drains outstanding ASR/MT (seconds); after 120 s the UI terminates the worker (session `failed`) |
| where are my results? | `runtime_data\sessions\<timestamp>_m2_...\` (`segments.json` = English + final Hindi per segment) |

**Product defect or missing integration?**
- These are product defects, and belong in this module: wrong or missing words in the transcript; a final replaced; events out of order; files missing after a clean stop.
- These are missing integration, for the MAVIS phase: no OCR; no shared microphone; nothing happens through `create_host_session` (flag off); no MAVIS shell.

**Safety:**
- Run everything from this folder.
- Never point `product.output_root` at the original research repository.
- Never run research evaluations from here; they were not copied on purpose.

Further reading: `README.md` (quick start), `ARCHITECTURE.md` (module map), `INTEGRATION.md` (host contract), `DEMO_READINESS.md` (acceptance evidence), `SOURCE_PROVENANCE.json` (what was copied, and its hashes).
