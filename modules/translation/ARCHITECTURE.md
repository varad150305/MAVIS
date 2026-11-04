# Architecture

## Module map

| path | layer | role |
|---|---|---|
| `product/ui/app.py`, `product/ui/__main__.py` | view | Tkinter desktop UI (`python -m product.ui`). Polls session events every 100 ms. Status pill, stall/error banner, DRAFT/FINAL badges, performance panel. `CameraFeed` (OpenCV thread, UI process only) puts the speaker's camera behind the transcript (canvas view, Netflix-style subtitle of the active segment); without a picture it shows the plain text widget |
| `product/ui_model.py` | view model | `TranscriptModel`: applies events (drops stale or foreign `session_id`, keeps finals immutable, ignores late drafts, tracks stall warnings) |
| `product/metrics_panel.py`, `product/latency.py` | view model | `LiveMetrics`: performance panel; same latency code as the session summary (torch-free) |
| `product/session.py` | **host API** | `TranslationSession`, `SessionOptions`, `create_host_session`, `ProcessRunner` (one worker process per session), event schema `mavis.live_translation.v1` |
| `product/session_worker.py` | worker entry | `python -m product.session_worker <args>`: runs `run_session`, streams JSON lines, reads `stop` from stdin |
| `product/live_translate.py` | session runner / CLI | `run_session()`: config, session folder, lifecycle evidence, audio source, model loading, summary. Also the CLI (`python -m product.live_translate`) |
| `product/live_pipeline.py` | product pipeline | `StableUpdateHook` (wraps the frozen VAD, buffer and tracker objects: boundary repair, context trim, prompt rules, final-window restore, cross-utterance dedup); `SegmentController` + MT3-C `LiveScheduler`; `MTWorker`; `Display` |
| `product/final_mt.py` | product MT | NLLB-200 600M CTranslate2 int8 final translator |
| `product/sources.py` | audio | `LiveMicSource` (microphone, stall detection, safe stop/close) and `PacedWavSource` (1× WAV replay) |
| `product/lifecycle.py`, `product/runtime.py` | support | durable lifecycle files and atomic JSON; offline-model and Windows helpers (power, keep-awake, memory) |
| `product/configs/live_m2_mt3c.yaml` | config | the only configuration file |
| `product/tools/convert_ct2.py` | setup tool | converts NLLB to CTranslate2 (used only by `setup_models.ps1 -Download`) |
| `main.py` | **frozen engine** | `run_m2()`: M2 streaming loop (VAD → StreamingBuffer → ASR worker → PrefixAgreementTracker, drain on stop) |
| `asr/` | frozen | `WhisperEngine` (faster-whisper), `ASRWorker` thread |
| `audio/audio_stream.py` | frozen | sounddevice input stream (100 ms frames) |
| `vad/vad.py` | frozen | Silero VAD state machine |
| `streaming/` | frozen | StreamingBuffer, prefix agreement (LocalAgreement-2), word alignment, back-pressure |
| `translation/` | frozen | opus-mt translator, MT3-C scheduling (`StreamingMTScheduler`, `AdaptiveDeadlinePolicy`), incremental helpers |
| `utils/` | frozen | config loader, event logger |
| `evaluation/asr_evaluation.py`, `evaluation/mt3c_parameters.json` | frozen | latency summary helpers imported by `main.run_m2`; frozen MT3-C parameters (SHA-checked) |

The product never edits a frozen object's code. It patches `main.build_*` factories during a session (`unittest.mock.patch`) and wraps methods on the objects they return.

## Processes and threads

```
Host / UI process (~35 MB, no torch)                    Worker process (one per session; ~1.6 GB peak)
---------------------------------------                 ---------------------------------------------------
TranslatorApp (Tk main loop, 100 ms poll)               product.session_worker
  TranslationSession                                      run_session()
    ProcessRunner ---- Popen(python -m ... ) -------->      main.run_m2() loop (VAD, buffer)    [main thread]
       stdin:  "stop\n"  --------------------------->        ASR worker (Whisper)                 [thread]
       stdout: JSON lines (rec / ready / console /  <---      SegmentController (MT3-C)            [thread]
               result / fatal)                               MTWorker opus-mt drafts              [thread]
       stderr: last lines kept for error messages            MTWorker NLLB finals                 [thread]
    _publish(): session_id, seq -> queue / on_event          ResourceSampler                      [thread]
```

## Event flow (one utterance)

1. Frames arrive every 100 ms. The VAD marks speech, and the StreamingBuffer builds a window once at least 2 s of new audio has arrived.
2. Whisper decodes the window. The agreement tracker commits the words two consecutive decodes agree on, giving an `en_committed` event.
3. The segment controller hands the stable text to MT3-C. MT3-C either waits or schedules an opus-mt draft, giving `hi_draft`.
4. At the utterance end, the final window is decoded (the whole utterance if it is ≤ 8 s), and the remaining words are committed.
5. The segment closes on a sentence end after a pause, on length, or after 2.5 s idle, giving `segment_closed`. NLLB translates the whole segment, giving `hi_final`.
6. On Stop, the worker drains everything, writes the session files, sends `session_summary`, and the state becomes `stopped`.

## Runtime dependencies

```
product.ui ──► product.session ──► (subprocess) product.session_worker ──► product.live_translate
                                                                              │
     product.live_pipeline ◄─────────────────────────────────────────────────┤
          │                                                                   ▼
          ├─ translation.scheduling / translator (transformers, torch)   main.run_m2
          ├─ product.final_mt (ctranslate2, transformers tokenizer)        ├─ vad.vad (silero-vad, torch)
          └─ streaming.prefix_agreement                                   ├─ streaming.* , audio.audio_stream (sounddevice)
                                                                          ├─ asr.* (faster-whisper → ctranslate2)
                                                                          └─ evaluation.asr_evaluation, utils.*
```

Third-party packages: `requirements-demo.txt`, plus CPU `torch`/`torchaudio` from `requirements-torch-cpu.txt`.
