# Security and data flow

MAVIS is a single-user application for your own computer. This note says what data exists, where it goes, and what is and is not
protected. It describes behaviour read from the code; it is not a security audit or a legal opinion.

## 1. Data flow
| Data | Where it goes | When |
|---|---|---|
| Page **photos** (JPEG, up to 1920 px wide; OCR.space: re-encoded under 900 KB) | The OCR provider you configured (Google Gemini or OCR.space), over HTTPS | Online, every captured page |
| Page photos | Nowhere (local OCR engine) | Only while the computer is offline |
| OCR text, cleaned text, snapshot, answers, conversation | Local disk only: `exports/web/<scan-id>/` | Always |
| Questions and document text sent to the language model | Local Ollama process on `127.0.0.1:11434` | After finalization, on request |
| API keys | Environment variables of the MAVIS process; sent to the provider in an HTTP header | Each provider request |

Nothing is sent to any other service. There is no telemetry, analytics or update check in the code.

**Provider data use.**
- *Google Gemini, unpaid use* (checked against https://ai.google.dev/gemini-api/terms): Google may use submitted content and the
  responses to provide, improve and develop its products and machine-learning technologies, and human reviewers may review
  inputs and outputs. The terms say: "Do not submit sensitive, confidential, or personal information to the Unpaid Services."
  Paid use is described there as not used to improve products. Terms change; re-check before relying on this.
- *OCR.space*: its terms and retention policy were **not reviewed** for this note. Its API page says it does not store page
  content, but that is a statement to verify, not something MAVIS enforces.
- Do not scan confidential or personal documents through a provider unless you have accepted its terms. The UI states
  "Page photos are sent to ..." on the home screen, the sidebar and the document view.

## 2. Stored data
- `exports/web/<scan-id>/` contains the page photos, OCR text and the full conversation **in plain text**. It is git-ignored; never
  commit or share it. Anyone with access to the folder can read it.
- The home page has a delete button for each scan; it permanently removes that folder.
- Keys are never written by MAVIS, never included in HTTP responses (`/api/state` returns only whether a key is present),
  never logged, and never put in a URL (tests assert this for both providers).

## 3. Network exposure
- The server binds to `127.0.0.1` only and rejects requests whose `Host` is not localhost, and cross-site `POST`s (`Origin` not
  localhost). There is **no authentication**: anything running as you on this machine can use the API. Do not bind it to another
  address or expose the port (tunnels, port forwarding) without adding authentication.
- `MAVIS_GEMINI_URL` / `MAVIS_OCRSPACE_URL` override where the key is sent; setting them to an untrusted host leaks the key.

## 4. Untrusted content
- Text read from photos is untrusted input to the language model. Mitigations present: the model has no tools or web access;
  answers are JSON with an evidence quote that is checked against the OCR text; numbers not in the document are flagged; answers
  from garbled pages are marked unverified; the model never edits the OCR text. Prompt injection inside a scanned page can still
  influence the wording of an answer. The answer's "source" label and evidence quotes are the safeguard to read.
- Page content is never executed or rendered as HTML (the UI escapes it).

## 5. Keys and secrets practice
Use `setx` / environment variables. Windows stores user environment variables unencrypted in the registry. Rotate a key if it
was ever pasted into chat, a log or a commit. The repository ships no keys; `.gitignore` excludes `.env` files and `exports/`.

## 6. Known gaps
- No authentication or per-user separation; no encryption at rest; no automatic data retention limit.
- Only Windows 11 / Python 3.11 was exercised. Local file permissions are those of your user account.
- Dependency versions are pinned but no vulnerability scan has been run.
- Report problems privately to the repository owner before disclosing publicly.
