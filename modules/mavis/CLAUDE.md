# CLAUDE.md: working on MAVIS

Read this before changing anything. It is written for an AI coding agent (and humans) integrating other modules with MAVIS.

## What this repository is
A local web app: phone camera -> pages -> cloud OCR (offline: local OCR) -> finalized document -> local Gemma Q&A.
Primary entry point: `python -m webapp` (http://127.0.0.1:8765). Architecture: `docs/ARCHITECTURE.md`.

## Commands (Windows, Python 3.11; run from the repository root)
```bash
.venv\Scripts\python -m webapp [--port 8765] [--no-browser]            # run
.venv\Scripts\python -m pytest prototype/test_ocr_engine.py tests -q    # tests (needs requirements-dev.txt)
PYTHONPATH=. .venv\Scripts\python tests/test_postprocessing.py          # standalone scripts (not collected by pytest)
PYTHONPATH=. .venv\Scripts\python tests/test_wordfreq_correction.py
```
Tests use fake cameras, fake OCR readers and mock HTTP servers. They never need a key, Ollama or a camera.

## Hard rules (do not break these)
1. **OCR before LLM.** The model must never run while a scan is connecting, capturing or finalizing, and a document is askable
   only after it is finalized with OCR idle. Enforced in `webapp/core.py` (`_llm_blocker`, `cancel_llm`, one `_llm_lock`) and
   `webapp/session.py` (`llm_blocker`). Any new module that uses the CPU heavily must be sequenced the same way.
2. **B0 is the default capture mode; C3 is opt-in.** Do not change the default or retune the selector constants
   (`prototype/frame_selector.py`, B0 constants in `webapp/session.py`: 3 s cooldown, fingerprint distance 5.0).
3. **Do not modify frozen modules' behaviour**: `prototype/frame_selector.py`, `prototype/llm_stage.py`,
   `prototype/ocr_engine.py`, `postprocess/*`. (`llm_stage.quote_match` has one documented speed-only change,
   verified by `tests/test_quote_match_equivalence.py`.) Add behaviour in `webapp/` instead.
4. **The language model never edits OCR text.** Answers are separate objects; `ocr_text_unmodified` stays true.
5. **Secrets**: API keys come from environment variables only. Never write a key to a file, a log line, a test fixture, a commit
   or an error message. (`tests/test_gemini_reader.py` and `tests/test_ocrspace_reader.py` assert keys never leak.)
6. **User data**: `exports/` holds scan photos, OCR text and conversations. It is git-ignored; never commit, upload or print it.
7. **Local only**: the server binds to 127.0.0.1 and rejects foreign Host/Origin headers. Do not bind it to a public address
   or remove those checks without adding authentication.
8. **Pages must not be silently lost.** A failed page stays visible with its reason (`failed`) and can be re-read.

## Where things live
- `webapp/core.py`: `WebApp`, the single application object (scan, library, reader status, model lock, questions).
- `webapp/session.py`: `ScanSession`, one scan (camera thread, capture thread, collector, shutdown/finalize).
- `webapp/gemini_reader.py`, `webapp/ocrspace_reader.py`: OCR providers; `RemoteFirstEngine` is the shared engine
  (parallelism, retries, rate-limit waiting, offline fallback). Add a provider by adding one reader class + `make_reader`.
- `webapp/workspace.py`: documents on disk (`exports/web/<scan-id>/`), history, delete, re-read.
- `webapp/server.py`: HTTP/SSE routes. `webapp/static/`: the whole UI (no build step, no external requests).
- `prototype/llm_stage.py`: `DocumentAssistant.ask()`, answer verification (quote and number checks).
- `postprocess/`: cleaning and the schema v2 export.

## Working style for this repo
- Keep changes small and tested; run the full test command before declaring anything done.
- Prefer new files in `webapp/` over editing shared modules; do not restructure directories.
- State what is verified (tested/run) versus assumed. Real provider calls (Gemini, OCR.space) are only covered by mocks in the
  test suite.
- Do not implement Whisper ASR, web scraping or gesture modules here. `docs/INTEGRATION_CONTRACT.md` describes the interface a
  controller can use today and lists the gaps; ask before adding routes or abstractions to close a gap.

## Known limits (see docs/SETUP.md for troubleshooting)
Windows 11 + Python 3.11 only; one scan at a time; answers have no multi-turn memory; Gemma on CPU takes about 30 s per
answer; OCR quality depends heavily on lighting and framing.
