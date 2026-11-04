# Demo readiness

## Conclusion (Phase 9, carried forward)

The module is **ready for a supervised standalone demonstration** of English → Hindi in the desktop UI:
- with a scripted or controlled read (`demo_assets/demo_script.txt`);
- preferably on AC power;
- after **one rehearsal with the physical microphone**.

It is not evidence of general accuracy, and it is not integrated into MAVIS.

## Two different kinds of evidence

| evidence | build | audio | what it shows |
|---|---|---|---|
| **Phase 8A live-microphone sessions** (battery and AC, 2026-10-29, physical microphone, desktop UI) | the build *before* the Phase 9 fixes | live microphone | WER against the script: 7.14 % (battery), 6.49 % (AC); about 5.8 % / 2.6 % after the reading deviations agreed by all offline models. One final per segment, no dropped frames, clean shutdown, worker isolation working. Four streaming defects found |
| **Phase 9 regression replays** | the build in this package | the two Phase 8A recordings replayed at 1× (on AC) | all four defects fixed ("people people", "paste", "use of microphone", fragment / joined segments). Displayed WER 5.84 % (battery recording), 5.19 % (AC recording). Latencies below |

**This package's build has not had a fresh live-microphone run.** Its four fixes were verified by replaying recorded audio. Rehearse with the microphone before the demo.

Phase 9 replay figures, final configuration:

| | first committed English (median, from speech onset) | first Hindi draft | final Hindi from last word (median / max) | final Hindi from first word (max) |
|---|---|---|---|---|
| battery recording | 4.31 s | 5.57 s | 2.34 / 3.10 s | 19.7 s |
| AC recording | 3.82 s | 4.87 s | 2.59 / 3.75 s | 16.9 s |

Live sessions (Phase 8A): startup to listening 13.7 s on battery and 8.4 s on AC; ASR inference median 906 ms (battery) vs 487 ms (AC); worker peak about 1.6 GB.

## Demo prerequisites

- `.\setup_demo.ps1` completed. `.\setup_models.ps1 -Load` OK.
- The microphone chosen, and checked with `.\run_cli_demo.ps1 --list-devices`.
- AC power if possible: battery is slower to start and to translate.
- A quiet room, and the presenter reading the demo script at a normal pace.
- Optional rehearsal without a microphone: `.\scripts\make_demo_wav.ps1`, then choose the WAV in the UI.

## Known limitations

- **Whisper tiny misrecognitions**, e.g. "the library of the office", "student", "the phrase", "cutting of".
- **Hindi grammar and omission slips:** gender agreement, a dropped subject. There is no native-speaker review and no Hindi reference.
- **Spontaneous speech** is clearly weaker than the scripted read.
- **Latency:**
  - English about 4 s after speech starts, Hindi drafts about 5 s;
  - finals about 2.3–2.6 s after the last word, longer for long segments;
  - sentences under 5 words close after a 2.5 s pause or join the next one.
- **The UI metrics measure time and resources, not correctness.** "0 MT errors" only means no call failed.
- **Not established:** concurrency with OCR or other MAVIS modules, and microphone sharing.
- **Tk rendering:** Tk occasionally draws a Devanagari vowel sign with a dotted circle.

## Package validation (packaging, 2026-10-29)

Results of the validation run in this folder are in `_handoff_audit/validation_report.md`. In summary:
- setup, `pip check`, the environment check and model verification all passed, and every model loads offline from `models/`;
- package tests: 120 passed, 3 skipped (historical recordings not packaged), 3 deselected (need research configs or diagnostic tools);
- a real end-to-end replay through the worker with a Windows text-to-speech reading gave word-perfect English and complete Hindi finals.

No physical-microphone session was run during packaging.
