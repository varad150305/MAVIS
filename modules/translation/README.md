# MAVIS translation module — demo handoff

A live **English speech → Hindi** translator for Windows, running on CPU. It has:
- a desktop UI;
- a console CLI;
- a session API for a future MAVIS host.

This folder is self-contained: its own code, its own models (`models/`, about 1.0 GB) and its own Python environment.

## Prerequisites

- Windows 10/11 x64, about 4 GB free RAM per running session, about 3 GB disk (models 1.0 GB + environment 1.7 GB + downloads cache).
- **Python 3.10.x 64-bit** with tcl/tk (tested 3.10.11).
- Internet **once**, to install packages. The models are bundled.
- A microphone, for live use. A 16 kHz WAV works without one.

## Quick start (PowerShell)

```powershell
cd "<path>\mavis_translation_demo_handoff"
.\setup_demo.ps1                          # venv + packages + checks + model verification
.\.venv\Scripts\python.exe -m product.ui  # desktop UI  (or .\run_demo.ps1)
```

In the UI:
1. Choose **Microphone** (optionally a device number) or a **WAV file**.
2. Press **Start**, then speak English.
3. Hindi appears per segment, first as a grey **DRAFT – may change**, then as the bold **FINAL**.
4. **Stop** ends the session cleanly.

If a camera is available, the transcript area shows the live camera picture with Netflix-style subtitles: only the current segment's English and Hindi, at the bottom centre, replaced as the next segment arrives. Choose another camera with `$env:MAVIS_UI_CAMERA="1"` before launching, or turn it off with `$env:MAVIS_UI_CAMERA="off"`. Without a camera the transcript area is plain white, as before.

Other commands:
- CLI: `.\run_cli_demo.ps1 --list-devices`, then `.\run_cli_demo.ps1 [--device N] [--save-audio]`.
- Rehearse without a microphone: `.\scripts\make_demo_wav.ps1`, then choose `runtime_data\demo\demo_script_tts.wav` in the UI.
- Smoke tests: `.\.venv\Scripts\python.exe -m pytest tests_smoke -q`.

Session results go to `runtime_data\sessions\`.

## Layout

```
CLAUDE.md              operational manual (start here if you are an agent)
ARCHITECTURE.md        modules, processes, data flow
INTEGRATION.md         host session API + event contract (MAVIS integration is a later phase)
DEMO_READINESS.md      what was tested, how, and the known limitations
LICENSES_AND_MODEL_NOTICES.md
SOURCE_PROVENANCE.json every copied file with source/destination SHA-256; every model file's hash
requirements-*.txt     torch-cpu / runtime / test / lock
setup_demo.ps1  setup_models.ps1  run_demo.ps1  run_cli_demo.ps1
product/               the product (UI, session API, worker, pipeline glue, config, product tests)
main.py asr/ audio/ vad/ streaming/ translation/ utils/ evaluation/
                       FROZEN research runtime used by the product (byte-identical; do not edit)
models/                bundled model assets (Hugging Face cache layout + NLLB CTranslate2 int8)
scripts/               check_environment.py, verify_models.py, make_demo_wav.ps1
tests_smoke/           package smoke tests
demo_assets/           the demo reading script (text)
runtime_data/sessions/ session output (empty at handoff)
_handoff_audit/        packaging audit: source manifests, git status, build script
```

Scope: no OCR, no MAVIS host, and the integration flag is **off**. See `CLAUDE.md`.
