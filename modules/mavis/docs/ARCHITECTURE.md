# MAVIS architecture

Everything below was read from the code and exercised by the test suite.

## 1. End-to-end flow
```
DroidCam / simulated camera
        │  MJPEG over HTTP (OpenCV FFmpeg)
        ▼
LiveStream ──► ScanSession.camera thread ──► latest frame
                                      │
                                      ▼
                       capture thread: frame selection
                       B0 (default) | timed | C3/C5 (opt-in)
                                      │  one frame per page + saved photo
                                      ▼
                       RemoteFirstEngine  (queue ≤ 8, parallel workers)
                         ├─ cloud OCR: GeminiReader | OcrSpaceReader   (normal path)
                         └─ local OCR: RapidOCR PP-OCRv5               (only while offline)
                                      │  text per page (watermark lines removed)
                                      ▼
                       collector thread: page records, activity log
                                      │  user presses Stop
                                      ▼
 finalize: stop capture → wait for every queued page → clean text → export → freeze snapshot
                                      │  finalized=True is set last
                                      ▼
 WebApp.prepare: load model + read the whole document into it (prefix cache)
                                      ▼
 WebApp.ask → DocumentAssistant.ask → Ollama (gemma3:4b, JSON schema) → verified answer → chat.json
```
Browser UI: `webapp/static/` talks to `webapp/server.py` over JSON routes, a server-sent-events stream (`/api/events`) and an
MJPEG preview (`/api/scan/video`).

## 2. Components
| Component | File | Responsibility |
|---|---|---|
| Application object | `webapp/core.py` `WebApp` | one active scan, document library, reader health, model lock, questions, re-read, delete |
| Scan | `webapp/session.py` `ScanSession` | threads, capture rules, per-page records, finalization, snapshot for the UI |
| Camera | `webapp/live_stream.py` | OpenCV capture that skips buffered frames after a stall (stays at the live edge) |
| Simulated camera | `webapp/demo.py` | plays the six synthetic pages in `llm_validation/images/` |
| Frame selection | `prototype/frame_selector.py` (C3/C5), B0 and timed in `session.py` | decides which frame becomes a page |
| OCR engine | `webapp/gemini_reader.py` `RemoteFirstEngine` | parallel page reading, retries, rate-limit waiting, offline fallback |
| OCR providers | `GeminiReader`, `webapp/ocrspace_reader.py` | one HTTP client each: `read(frame)`, `check()`, `probe()` |
| Local OCR | `prototype/ocr_engine.py`, `webapp/ocr_threads.py` | RapidOCR; built lazily, 4 ONNX threads |
| Watermark filter | `webapp/watermark.py` | removes the DroidCam overlay text, also when fused onto a sentence |
| Post-processing | `postprocess/` | normalize, segment, conservative word-frequency correction, confidence, export schema v2 |
| Documents on disk | `webapp/workspace.py` | `exports/web/<scan-id>/`, history, delete, re-read a failed page |
| Model client | `webapp/llm_runtime.py` | Ollama HTTP client; abortable requests; CPU thread cap |
| Question answering | `prototype/llm_stage.py` `DocumentAssistant` | prompts, JSON-schema answers, quote/number verification, legibility warnings |
| Server | `webapp/server.py` | aiohttp routes, SSE, MJPEG, Host/Origin guard |

## 3. Scan lifecycle and the sequencing rule
`ScanSession.phase`: `connecting → scanning → finalizing → finished`, or `failed` / `cancelled`.
Finalize stages (reported live to the UI): `stop_capture → read_pages → clean_text → save`. The camera gets 8 s to close; if it
does not, finalization continues with the pages already captured.

**Invariant: OCR first, model second.** `WebApp._llm_blocker` refuses the model while a scan is active, and for a document that
came from the current scan until `ScanSession.llm_blocker()` reports finalized and the OCR engine idle. All model work (the
preparation step and questions) holds one lock; a second request gets HTTP 409. `WebApp.start_scan` first calls `cancel_llm`
(bumps an epoch, aborts the in-flight request at the socket, retries until the lock is free), so **a new scan always stops the
model**; a question cut off this way is not saved.

## 4. Reading pages (providers and fallback)
- Provider selection: `MAVIS_OCR_PROVIDER` = `gemini` (default) or `ocrspace`; keys from `GEMINI_API_KEY` / `OCRSPACE_API_KEY`.
- Rate limit (HTTP 429 or OCR.space's "within N seconds" message): the page waits for the quota (up to 180 s, 6 times) and the
  request pace halves each time; the UI shows a countdown. Daily quota or longer waits fail the page with the reason.
- 5xx / timeout: retried (2 by default). Key rejected, bad request, blocked image: fail at once with the reason.
- **Offline** (a TCP connection to the provider cannot be opened): that page is read by the local engine; the provider is probed
  again every 15 s. Only a lost connection triggers the local engine, never a provider error.
- A page with no text (for example only the camera overlay) is "blank": kept out of the document, not counted as a failure.
- A failed or blank page keeps its photo; `POST /api/documents/{id}/pages/{n}/reread` reads it again from the saved photo and
  rebuilds the snapshot (same document id, new hash).

## 5. Data structures (on disk, per document: `exports/web/<scan-id>/`)
| File | Content |
|---|---|
| `export/raw_output.json`, `corrected_output.json`, `metadata.json` | export **schema v2**: raw OCR text per page; cleaned text, confidence, structure; document identity, summary, `processing` |
| `snapshot.json` | the frozen document the model sees: `{document_id, timestamp, pages:[{page_number, raw_text, normalized_text?, corrected_text, confidence_score, confidence_label, structure, corrections_applied}], summary, processing:{status, pages_ok, pages_failed}}`; its SHA-256 is recorded in `meta.json` and in every answer |
| `meta.json` | title, source, capture mode, per-page state/chars/time/provider/error, `blank_pages`, `readers`, `snapshot_sha256` |
| `pages/page_N.jpg` | the captured page photos (numbered in capture order) |
| `chat.json` | question/answer turns (each answer stores the verified result object) |

**Answer object** (`DocumentAssistant.ask`): `mode, request, model, error{code,message}, answer, answerable_from_document,
evidence[{page, quote, verified_in_ocr_text, match: exact|approx|none, match_ratio}], general_knowledge, source:
document|document_unverified|not_in_document|general_knowledge, unsupported_numbers, warnings, ocr_legibility, document_sha256,
timing, ocr_text_unmodified`. The turn saved to `chat.json` and returned by the API keeps all of these except `ocr_text_unmodified`.
Page numbers are **capture numbers**; a failed page leaves a gap, never a renumbering.

## 6. Verified interfaces a future integration layer can call
1. **HTTP** (`webapp/server.py`): `GET /api/state`, `GET /api/events` (SSE of the same state), `POST /api/scan/start|stop|cancel|dismiss`,
   `GET /api/scan/video`, `GET /api/scan/pages/{n}/image`, `GET /api/documents`, `GET|PATCH|DELETE /api/documents/{id}`,
   `POST /api/documents/{id}/ask|prepare`, `POST /api/documents/{id}/pages/{n}/reread`, `GET /api/documents/{id}/export/{kind}`,
   `POST /api/reader/check`, `POST /api/llm/check`. Refusals are HTTP 409 with `{"error":{"code","message"}}`.
2. **In process**: `WebApp.start_scan / stop_scan / cancel_scan / dismiss_scan / ask / prepare / reread_page / delete_document /
   snapshot`; `ScanSession.snapshot()`; `Library.get(id).detail()`; `DocumentAssistant.ask(doc, request, mode, extra_context)`.
3. **On disk**: `snapshot.json` and the schema v2 export are the most stable contracts. Anything under `webapp/` internals may change.

The scan-start body is `{"kind": "droidcam"|"demo", "ip", "port", "selector": "b0"|"c3"|"interval", "interval_s"}`.
This repository is the **standalone OCR + LLM module**: it runs on its own (`python -m webapp`) and needs no other module. The
controller-facing subset of the HTTP surface (activation, stop/cancel, status, results, behaviour while busy, and the missing
capabilities) is specified in `INTEGRATION_CONTRACT.md`.

## 7. Threading and performance notes (measured on the development machine)
- Reading pages by cloud API costs the machine almost nothing (about 1.5 % CPU) and the live picture stays about 40 ms old.
- The local OCR fallback uses 4 ONNX threads (faster than all 18 on this hybrid CPU, and leaves cores for the browser).
- The local model is capped at 8 threads (`MAVIS_LLM_THREADS`): about 5 % slower than uncapped but the browser stays responsive.
- First question after a scan takes about 30 s on CPU; the preparation step (about a minute) reads the whole document into the
  model once so later questions reuse it.
