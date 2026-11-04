# Controller contract: the standalone OCR + LLM module

**Scope.** This document is the minimum interface a *controller* (for example the MAVIS gesture-recognition controller) can rely on
to drive the standalone OCR + LLM module that this repository contains: scan pages with a camera, read them, finalize a document,
and ask a local model about it. It describes what exists today and labels what is missing. It does not define speech recognition,
web scraping, shared context or overall orchestration, and it does not add anything to the module.

**Labels.**
- **VERIFIED**: read in the code (`webapp/server.py`, `webapp/core.py`, `webapp/session.py`, `webapp/static/app.js`) *and* exercised
  over real HTTP on 2026-10-29 against an identical copy of the code, with a mock OCR API and a stub model in place of Gemini and
  Gemma (so timing and answer text are not representative, behaviour of the routes is).
- **CODE ONLY**: read in the code, not exercised in that run.
- **GAP**: not supported. Nothing below the GAP list should be assumed to exist.

Stability: the on-disk files (`snapshot.json`, schema v2 export) are the most stable, the HTTP routes next, `webapp/` internals may
change. The routes carry **no version field** (GAP 10), so a controller should pin to a release of this repository.

## 1. The interface in one table
All routes are on `http://127.0.0.1:8765` (`python -m webapp --port N` changes the port). Request and response bodies are JSON.

| Operation | Route | Body | Success | Refusals |
|---|---|---|---|---|
| Activate: start a scan | `POST /api/scan/start` | `{"kind": "droidcam"\|"demo", "ip", "port", "selector": "b0"(default)\|"c3"\|"interval", "interval_s": 2..30}` | `200 {"ok": true}` (returns before the camera is connected) | `400 error` bad parameters; `409 refused` scan already running, no OCR key, provider rejected the key |
| Stop scanning and finalize | `POST /api/scan/stop` | none | `200 {"ok": true}` (returns at once; finalizing runs in the background) | `409 refused` "There is no scan to stop." unless phase is `scanning` |
| Cancel while connecting | `POST /api/scan/cancel` | none | `200 {"ok": true}` | `409 refused` "There is nothing to cancel." unless phase is `connecting` |
| Leave a finished scan | `POST /api/scan/dismiss` | none | `200 {"ok": true}` (also when nothing exists) | `409 refused` while the scan is still active |
| Read state (poll) | `GET /api/state` | none | `200` state object, section 2.3 | none |
| Follow state (push) | `GET /api/events` | none | Server-sent events, whole state per message, section 2.3 | none |
| Prepare the model for a document | `POST /api/documents/{id}/prepare` | none | `200 {"prepare": {status, ...}}` | `409 refused` unknown document, scan in progress, document not yet finalized |
| Ask | `POST /api/documents/{id}/ask` | `{"mode": "answer"\|"summarize", "question": "..."}` (`question` is ignored for `summarize`) | `200 {"turn": {...}}` after the model has finished (seconds to minutes) | `409 refused` / `409 busy`, section 5 |
| Read a document and its conversation | `GET /api/documents/{id}` | none | `200` document with `pages` and `turns` | `404 not_found` |
| Discard a document | `DELETE /api/documents/{id}` | none | `200 {"ok": true}` | `409 refused` while a scan is running |

Other routes exist (page photos, export, re-read a page, rename, provider and model health checks, MJPEG preview); they are listed
in ARCHITECTURE.md §6 and are not needed for the operations below.

**Errors.** Every refusal has the shape `{"error": {"code", "message"}}`. HTTP status and `code` are: `400 error` (bad parameter),
`404 not_found`, `409 refused` (not allowed in the current state), `409 busy` (the model is occupied), `403` (Host or Origin is
not localhost; plain text). **The `message` strings are the only way to tell two `refused` cases apart (GAP 4).**

**Document id.** The scan id (`scan-YYYYMMDD-HHMMSS`, in `state.scan.id` from the moment the scan starts) *is* the document id of
the document that scan produces. VERIFIED.

## 2. Operations in detail

### 2.1 Activation (start a scan)  VERIFIED
`POST /api/scan/start`.
- `kind: "droidcam"` needs `ip` (letters, digits, dots, hyphens) and `port` (1 to 65535); the camera URL is
  `http://<ip>:<port>/video`. There is **no server-side default camera**: the controller supplies the address on every start
  (GAP 8). `kind: "demo"` uses the bundled simulated camera (it still needs a working OCR provider).
- `selector` defaults to `b0` (smart capture). `c3` is opt-in. `interval` captures one page every `interval_s` seconds
  (2 to 30). Anything else is `400 "Unknown capture mode."`. `c5` is accepted by the session class but **not** by the route.
- Rejections observed: no `kind` → `400 "Choose a camera source."`; port 99999 → `400 "The port must be a number from 1 to 65535."`;
  `interval_s` 1 → `400 "Choose a capture interval between 2 and 30 seconds."`.
- The route returns as soon as the session is created. The camera connects afterwards, so `200` does **not** mean the camera
  works: watch `state.scan.phase` (`connecting` → `scanning`, or `failed` with `state.scan.error`).
- Starting also stops any model work in progress (section 5). Starting with no OCR key set is refused with the key-setup text
  (VERIFIED in the fresh-install run); a key the provider rejected is refused with the provider message.
- Pages are captured automatically by the chosen selector. **There is no "capture now" command** (GAP 9).

### 2.2 Stop and cancel
There are two mechanisms, and they are not interchangeable. Neither exists for a running question (GAP 2).

| Situation | What works | What does not (observed) |
|---|---|---|
| Phase `connecting` | `POST /api/scan/cancel` → phase `cancelled`, nothing is kept | `stop` → `409 "There is no scan to stop."` |
| Phase `scanning` | `POST /api/scan/stop` → phase `finalizing`; every queued page is still read, then the document is cleaned, frozen and saved | `cancel` → `409 "There is nothing to cancel."` |
| Phase `finalizing` | wait; the second `stop` is `409` | there is no abort (GAP 1) |
| Phase `finished` / `failed` / `cancelled` | `POST /api/scan/dismiss` clears the screen state (optional; a new scan may start without it) | |

So "abort the scan and keep nothing" while scanning is only possible as a composition: `stop`, wait for `finished`, then
`DELETE /api/documents/{id}`. The pages are still read and sent to the OCR provider in between.

### 2.3 Status  VERIFIED (fields), PROVISIONAL (the mapping below is this document's, not the module's)
There is **no single status field** (GAP 3). A controller derives one from `GET /api/state`:

| Controller status | Derived from |
|---|---|
| idle | `scan` is `null` (or its phase is `finished`/`failed`/`cancelled`) and `busy` is `null` and no entry of `prepare` is `queued`/`running` |
| connecting | `scan.phase == "connecting"` |
| capturing | `scan.phase == "scanning"` (`scan.capture.pages_captured`, `scan.connection.fps`) |
| processing OCR | `scan.ocr.busy` or `scan.ocr.queue_depth > 0` while phase is `scanning` or `finalizing` (OCR runs while capturing) |
| finalizing | `scan.phase == "finalizing"`; `scan.stages` = `stop_capture` → `read_pages` → `clean_text` → `save`, each `pending`/`active`/`done` |
| preparing the model | `prepare[<id>].status` is `queued` or `running` (the model is loaded and the whole document is read into it; about a minute on CPU) |
| answering | `busy` is not `null`: `{doc_id, mode, question, started_at}` |
| completed (scan) | `scan.phase == "finished"` and `scan.finalized` and `scan.document_id` is set |
| completed (question) | the `ask` call returned `200` with a turn |
| error | see section 4 |

State object (keys VERIFIED): `now`, `reader` (`status`, `message`: cloud OCR key and reachability; `ok` when usable), `llm`
(`status`: `checking`, `ok`, `unavailable`, `model_missing`; `message`), `busy`, `prepare` (map of document id →
`{status, done, total, message, started_at, ended_at, warnings}`), `docs_version` (increases whenever a document is added,
changed or deleted), `limits`, `scan` (`null` or `{id, phase, error, source, selector, connection{state, error, fps, ...},
capture{status, accepting, pages_captured, ...}, ocr{queue_depth, busy, completed, failed, ...}, pages[], stages[], activity[],
finalized, empty, document_id}`).

`prepare.status` values: `queued`, `running`, `ready`, `cancelled`, `unavailable`, `failed` (CODE ONLY except `ready`).

Observed sequence for one scan (mock OCR, one page): `connecting` → `scanning` → (`stop`) → `finalizing` with stage `stop_capture`
active → `clean_text` active → `finished` with `prepare` = `ready` within the same poll. The document becomes askable only when
`phase == "finished"`, `finalized` is true, OCR is idle, and `prepare` is not running.

**Push.** `GET /api/events` is `text/event-stream`. The server checks the state every 0.5 s and writes the **whole state** as
`data: <json>` whenever anything except `now` changed (11 messages in 5 s while scanning), plus a `: keepalive` comment every 4 s.
No event names, no sequence ids, no replay after a reconnect (GAP 6). Polling `GET /api/state` returns the same object.

### 2.4 Results and errors  VERIFIED
- **Scan finished**: `state.scan.phase == "finished"`, `finalized: true`, `document_id` = scan id; `scan.ocr.completed` / `failed`
  count pages. A page that could not be read stays in `scan.pages` with `state: "error"` and its reason; it is never dropped.
- **Scan finished with nothing readable**: `phase == "finished"`, `empty: true`, `document_id: null`. There is no document and
  `ask` would be refused ("Unknown document.").
- **Scan failed**: `phase == "failed"` with `scan.error` (camera unreachable; scanner could not start; finalization failed).
  A later scan may start immediately.
- **Scan cancelled**: `phase == "cancelled"` (only from `connecting`).
- **Answer**: `200 {"turn": {id, ts, mode, question, result}}`; the turn is also saved to the document
  (`GET /api/documents/{id}` → `turns`, and `chat.json`). `result` carries `answer`, `source`
  (`document`, `document_unverified`, `not_in_document`, `general_knowledge`), `evidence[{page, quote, verified_in_ocr_text, ...}]`,
  `warnings`, `document_sha256`, `timing`, and `error`.
- **Model failure is HTTP 200 with `turn.result.error`**, not an HTTP error: `{"code", "message"}`, where code is one of
  `runtime_unavailable`, `model_missing`, `timeout`, `runtime_error`, `bad_output`, `context_too_large`, `output_truncated`
  (CODE ONLY; the probe used a stub model that does not fail). A controller must check `turn.result.error` before using `answer`.
  Such a turn is saved too.
- **Cancelled question**: `409 refused` with the message "The assistant was stopped because a new scan was started." and **nothing
  is saved** (VERIFIED, section 5).
- **Preparation**: `prepare[<id>].status` is `ready`, or `unavailable` (Ollama off, model missing), `failed`, or `cancelled`
  (a new scan started). A document can still be asked about while `prepare` is not `ready`, except while it is `running`/`queued`
  (then `409 busy`).

## 3. Preconditions for any operation
Reader and model readiness are **in the state**, not in a health route: `reader.status == "ok"` (or `checking`) and
`llm.status == "ok"`. Starting a scan needs a provider key in the environment (`GEMINI_API_KEY` or `OCRSPACE_API_KEY`); with no
network the pages are read by the local OCR engine instead. Asking needs Ollama running with `gemma3:4b`. No key or model file
ever passes through this interface and none is returned by it.

## 4. Failure summary
| Where | Signal |
|---|---|
| bad request | `400` + `{"error":{"code":"error"}}` |
| not allowed now | `409` + `code: "refused"` |
| model occupied | `409` + `code: "busy"` |
| unknown document on `GET`/`DELETE`/`PATCH` | `404 not_found`; on `ask` and `prepare` it is `409 refused "Unknown document."` |
| camera / scanner / finalization failure | `scan.phase == "failed"`, `scan.error` |
| a page could not be read | `scan.pages[i].state == "error"` with a reason |
| OCR provider unusable | `reader.status` (`rejected`, `model_missing`, ...) and `409` on start |
| local model unusable | `llm.status` and `prepare.status == "unavailable"`; inside an answer: `turn.result.error` |

## 5. Lifecycle and safety: a trigger arrives while something is running
Hard rules the module enforces itself (invariants preserved): the model never runs while a scan is `connecting`, `scanning` or
`finalizing` or before the document is finalized with OCR idle; at most one model request runs at a time; a new scan always wins
over the model; the model never edits OCR text; B0 is the default capture mode.

| Trigger arrives while... | Result (VERIFIED unless marked) |
|---|---|
| a scan is `connecting`/`scanning`/`finalizing`, `start` again | `409 refused` "A scan is already running. Finish it first." Nothing changes; safe to repeat |
| a scan is active, `ask` | `409 refused`. For the running scan's own id, "Unknown document." (it does not exist until finalized); for an older document, "A scan is in progress. Questions open once scanning has finished and every page has been read." (CODE ONLY for the second wording) |
| a scan is active, `prepare` | `409 refused` (CODE ONLY) |
| a question is being answered, `start` | **Accepted.** The model request is aborted, the question is **not saved**, the waiting `ask` call returns `409 refused` "The assistant was stopped because a new scan was started.", then the scan starts. The `start` call waits for the abort (2 s in the probe; up to 15 s by the code's limit) |
| the model is being prepared, `start` | Accepted; preparation ends as `cancelled` (CODE ONLY) |
| a question is being answered, `ask` again | `409 busy` "The assistant is still working on the previous request." (also while preparing: "...still preparing the document.") |
| a scan is finished but not dismissed, `start` | Accepted; a finished scan does not block the next one |
| a scan is `connecting`, `stop` | `409` (use `cancel`) |
| a scan is `scanning`/`finalizing`, `cancel` | `409` (no abort, GAP 1) |
| a scan is running, `DELETE` a document | `409 refused` "A scan is running. Stop it before deleting scans." (CODE ONLY) |

Consequences for a controller: `start`, `stop`, `cancel` are safe to repeat (extra copies are refused, not queued); **`ask` is not
idempotent** (a repeated question runs again and stores a second turn); a trigger can never queue behind another one, it is
accepted or refused immediately, except that `ask` and `prepare` block until the model finishes. All clients share one state:
the browser UI and a controller both act on the same single scan, and nothing records who started it (GAP 5).

## 6. Standalone operation  VERIFIED
- Start the module with `python -m webapp [--port N] [--no-browser]` (SETUP.md). The browser UI is just another client of the
  routes above; the probe drove every operation in this document with a plain HTTP client and no browser.
- Readiness: the process answers `GET /api/state`. There is no dedicated health route and no shutdown route; stop it by ending
  the process (GAP 10). Ollama is a separate process the module only talks to.
- Binding and access: `127.0.0.1` only; requests whose `Host` is not `127.0.0.1`, `localhost` or `::1` get `403`; a request with a
  browser `Origin` header that is not local gets `403`. A non-browser client that sends no `Origin` is accepted. There is no
  authentication (SECURITY_AND_DATA_FLOW.md).
- Long calls: set the HTTP client timeout on `ask` and `prepare` to at least the module's own limit
  (`MAVIS_LLM_TIMEOUT_S`, default 600 s). A real answer on CPU takes about 30 s.
- Data written: `exports/web/<scan-id>/` (photos, text, conversations), git-ignored, never returned except through the routes.
- The camera is DroidCam, which serves **one client at a time**; a second consumer of the same phone camera (for example a
  gesture recogniser) cannot attach to it while a scan runs (GAP 8).

## 7. GAPS (not supported today)
| # | Gap | Effect on a controller |
|---|---|---|
| 1 | No way to abort a scan once it is `scanning` or `finalizing` (`cancel` works only while `connecting`) | "Cancel" during a scan means stop + keep, or stop + delete |
| 2 | No route to stop a running question or preparation. `WebApp.cancel_llm()` exists in process but is not routed; only starting a scan stops them | A controller cannot cancel an answer on its own |
| 3 | No aggregate status or health field; no `completed` event | The controller derives status from several fields (section 2.3) |
| 4 | Refusal codes are generic (`refused`, `busy`); unknown-document is `409` on `ask`/`prepare` but `404` on `GET` | Distinguish cases by message text, which is not a stable contract |
| 5 | No command ids, idempotency keys, ownership or locking between clients | Duplicate `ask` runs twice; a person using the UI and a controller can interfere |
| 6 | SSE sends the whole state, with no event types, ids, resume or per-question completion event; resolution 0.5 s | A reconnecting controller re-reads the full state; a question's completion is seen only as the `ask` response or `busy` → `null` |
| 7 | `ask` is one long synchronous HTTP call with no job id and no way to fetch the result later by job. What happens to the model request if the client disconnects is **not verified** | A controller that times out cannot cancel or poll; it can read `turns` later with `GET /api/documents/{id}` |
| 8 | No server-side camera default; DroidCam address is sent on every start; DroidCam is single-client | The controller needs the phone address and cannot share the camera |
| 9 | No manual capture command; capture is automatic (B0 default, C3, interval) | A gesture cannot mean "take this page now" |
| 10 | No contract/version field, readiness route or shutdown route | Pin to a repository release; end the process to stop |
| 11 | No authentication | Localhost only by design; any local process can call the routes |
| 12 | The browser UI does not refresh an already open conversation for an answer created through the API: the pending bubble appears and clears, but the new answer is shown only after the document is reopened (`app.js` loads `turns` once per document view) | Cosmetic; the answer is stored and returned to the caller |

## 8. Not in this document
Speech recognition, web scraping, gesture recognition, shared context between modules and overall orchestration are outside this
contract. Any addition to the module (new routes, a status field, an abort, stable codes) is a change request needing approval, not
part of this description. If approved, GAPS 1, 2, 3 and 4 are the smallest set that would let a controller start, stop, observe and
cancel without parsing messages.
