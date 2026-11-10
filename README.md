# MAVIS — Multimodal Assistance for Vision, Interaction, and Speech

One laptop, one launcher, and **your hand chooses the help**:

| Gesture | Effect |
|---|---|
| 👍 Thumbs-up | Ready — choose within 8 s |
| ☝️ One finger | **OCR + LLM** — scan printed pages, ask questions, get page-cited answers with verified quotes |
| ✌️ Two fingers | **Live Translation** — English speech → Hindi subtitles over the live laptop camera |
| 🤟 Three fingers | **Drishti** — scene assistant for blind and visually impaired users (11 modes, voice in English + 8 Indian languages) |
| 🖐 Open palm | Turns the running module off |

Open **Gesture control** (`/control/`) from the launcher. The chosen module opens on the same screen and starts by
itself; a small gesture box stays in the corner. Each module can also be opened directly from the launcher.

| Module | Source project | Route |
|---|---|---|
| Gesture controller | `modules/gesture` (YogitaAher20/gesture_recognition) | `/control/` |
| OCR + LLM | `modules/mavis` (varad150305/mavis-ocr-llm) | `/mavis/` |
| Live Translation | `modules/translation` (YogitaAher20/mavis_translation_demo_handoff) | `/translate/` |
| Drishti | `modules/drishti` | `/drishti/` |

Drishti's own Read Text and Translate modes are hidden, because the OCR + LLM and Live Translation modules do those jobs.

**No original code was changed.** All 314 original files are checked by SHA-256 in `ORIGINALS.sha256`;
run `python -m hub.verify_originals` to check. Everything new is in `hub/`, `tests/`, `tools/` and the start scripts.

## Start it (Windows)

1. Install **Python 3.11** (tick "Add python.exe to PATH") and run **`setup.bat`** once.
2. For Live Translation, install **Python 3.10** and run **`setup_translation.bat`** once (downloads ~3.5 GB of models).
   If a model download stops part-way, run `tools\repair_translation_models.bat`.
3. For OCR + LLM answers, install [Ollama](https://ollama.com) and run `ollama pull gemma3:4b`.
4. Optional: copy `keys.env.example` to `keys.env` and add `GROQ_API_KEY` (Drishti cloud fallback) or `GEMINI_API_KEY`.
5. Double-click **`run.bat`** — it starts Ollama, the hub and the launcher at **http://127.0.0.1:8000**. That is the
   only step needed for a demo.

## Cameras and microphones

Every module uses the laptop's own camera and microphone (`hub/static/device-pref.js` skips DroidCam's virtual devices).
OCR + LLM offers two choices on its page, with no default: **Use this device's camera**, or the **DroidCam phone app
over Wi-Fi** (enter its IP address and port; no DroidCam Windows client needed).

## How it fits together

- `hub/` — Python 3.11 aiohttp server: launcher, gesture service + `mavis_controller.py` (idle → ready → active),
  OCR app mount and camera relay, Drishti API routes, translation bridge, one colour scheme (`mavis-theme.css`).
- Live Translation keeps its own Python 3.10 environment. The hub starts `hub/translation_bridge.py` with that
  environment's Python; the bridge uses the package's host API and relays events to `/translate/`.
- Drishti runs in the browser (COCO-SSD, SmolVLM-500M, Tesseract.js, YAMNet); cloud models are only a fallback.

## Tests

```bat
.venv\Scripts\python -m pytest tests -q                                                  & rem hub: 21 tests
cd modules\gesture && ..\..\.venv\Scripts\python -m unittest discover -s tests -t .     & rem 112 tests
cd modules\mavis && ..\..\.venv\Scripts\python -m pytest prototype/test_ocr_engine.py tests -q   & rem 149 passed, 5 skipped
```

The translation package's own suite (120 passed, 3 skipped) runs in its environment; see `modules/translation`.

## Documents

`docs/` holds the final project report (PDF and Word), its LaTeX source, and the presentation.

## Layout

```
hub/            the server that joins the modules (static/: launcher, gesture control, subtitle page, add-ons, theme)
modules/        the original projects, unchanged (see ORIGINALS.sha256)
tests/          hub and controller tests (e2e/: test-only stand-ins)
tools/          setup helpers (translation model repair)
docs/           report and presentation
setup.bat, setup_translation.bat, run.bat, run.sh, keys.env.example, requirements-lock.txt
```

Ports: 8000 (this computer), 8443 (HTTPS), 8766 (internal relay). Use `run.bat --no-lan` for this computer only.
