# MAVIS: camera-to-chat document assistant (OCR + local LLM module)

MAVIS turns a phone camera pointed at printed pages into a searchable, question-answering document. It is a **web application
that runs on your own computer**:

1. **Scan**: connect a DroidCam phone camera (or use the built-in simulated camera). Pages are captured automatically.
2. **Read**: every captured page is read by a **cloud OCR API** (Google Gemini by default, or OCR.space). If the computer is
   **offline**, a local OCR engine (RapidOCR / PP-OCRv5) reads the pages instead.
3. **Finalize**: when scanning stops, MAVIS waits for every queued page, cleans the text, and freezes the document.
4. **Ask**: only after the document is finalized, a **local** model (Gemma 3 4B through Ollama) answers questions and
   summarises, citing the page and quote each answer came from.

> Status: working standalone module, intended to be driven by a separate gesture-recognition controller. That integration
> **does not exist yet**; the interface a controller can use, and what is missing, is documented in
> [docs/INTEGRATION_CONTRACT.md](docs/INTEGRATION_CONTRACT.md).

## What leaves your computer
Page **photos** are sent to the OCR provider you configure (Google Gemini or OCR.space). Nothing is sent when the computer is
offline (local fallback). Questions and answers never leave the computer: the language model runs locally. Free tiers of
cloud providers may use submitted data to improve their products; read their terms. Details:
[docs/SECURITY_AND_DATA_FLOW.md](docs/SECURITY_AND_DATA_FLOW.md).

## Requirements (verified on Windows 11 with Python 3.11.6; nothing else has been tested)
- Python 3.11, Tk is **not** needed.
- [Ollama](https://ollama.com) with the `gemma3:4b` model (about 3.3 GB; the model runs on CPU here and needs about 5 GB of RAM).
- An API key for one OCR provider (Google AI Studio for Gemini, or a free OCR.space key).
- A phone with the DroidCam app on the same network, or use the simulated camera.

## Quick start
```bash
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --no-deps -r requirements-lock.txt
ollama pull gemma3:4b
setx GEMINI_API_KEY "your-key"        # then open a NEW terminal; never commit keys
.venv\Scripts\python -m webapp        # opens http://127.0.0.1:8765
```
Use OCR.space instead with `setx OCRSPACE_API_KEY "..."` and `setx MAVIS_OCR_PROVIDER ocrspace`. Full instructions, the list of
environment variables and troubleshooting: [docs/SETUP.md](docs/SETUP.md).

## Capture modes (chosen per scan on the home screen)
- **Smart** (default, "B0"): a page a few seconds after the view changes, skipping repeats of the same view.
- **Timed**: one page every N seconds (2 to 30).
- **Steady view** ("C3", opt-in, experimental): waits until the page is held steady and keeps the sharpest frame.

## Guarantees the code enforces
- OCR must finish and the document must be finalized before the language model runs; one model request at a time.
- Starting a new scan stops any running model request.
- Pages are never silently dropped: a page that cannot be read is listed with the reason and can be read again.
- The OCR text itself is never rewritten by the language model.

## Layout
| Path | Purpose |
|---|---|
| `webapp/` | the application: server, scan session, OCR providers, local model client, browser UI (`static/`) |
| `prototype/` | shared modules the web app uses: `llm_stage.py` (question answering), `frame_selector.py`, `ocr_engine.py` (local fallback) |
| `postprocess/` | text clean-up, confidence scoring, export (schema v2) |
| `tests/`, `prototype/test_ocr_engine.py` | automated tests |
| `llm_validation/images/` | six synthetic pages used by the simulated camera |
| `docs/` | setup, architecture, security, integration contract, third-party notices, history |

The directory name `prototype/` is historical.

## Tests
```bash
.venv\Scripts\python -m pip install --no-deps -r requirements-dev.txt
.venv\Scripts\python -m pytest prototype/test_ocr_engine.py tests -q
```
See [docs/SETUP.md](docs/SETUP.md#verification-status-read-this-first) for what has and has not been verified.

## Documentation
[SETUP](docs/SETUP.md) · [ARCHITECTURE](docs/ARCHITECTURE.md) · [SECURITY AND DATA FLOW](docs/SECURITY_AND_DATA_FLOW.md) ·
[CONTROLLER CONTRACT](docs/INTEGRATION_CONTRACT.md) · [THIRD-PARTY NOTICES](docs/THIRD_PARTY_NOTICES.md) ·
[history](docs/history/README.md) · [CLAUDE.md](CLAUDE.md)

## Licence
**Not yet chosen.** No licence file is included; until the owner adds one, all rights are reserved. Third-party components
and the model terms are listed in [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).
