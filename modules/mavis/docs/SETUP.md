# Setup and troubleshooting

## Verification status (read this first)
| Item | Status |
|---|---|
| Install from `requirements-lock.txt` into a **brand-new** Python 3.11.6 virtual environment on Windows 11 | **Verified (2026-10-29):** all 43 pinned packages installed from PyPI with `--no-deps` (82 s), exactly one OpenCV distribution (`opencv-python-headless` 4.8.0, no GUI, no contrib) |
| Test suite in that fresh environment, from a fresh copy of this tree | **Verified:** 149 passed, 5 skipped (4 benchmark-parity tests and 1 old-backup test, expected without the research files) in 4 of 5 full runs; `tests/test_postprocessing.py` and `tests/test_wordfreq_correction.py` pass. **Known intermittent failure:** `tests/test_gemini_reader.py::test_a_failed_page_is_listed_with_its_photo_and_reason_and_can_be_read_again` failed in 1 full run (148 passed, 1 failed, 5 skipped) and in 3 of 20 isolated runs. Cause: a start-up race in `webapp/session.py`; if Stop arrives within milliseconds of the scan entering `scanning` (before its worker threads have started), finalization fails with "cannot join thread before it is started". Not fixed in this release (production code is unchanged); a human pressing Stop cannot normally hit it, a script or controller can |
| `pip check` in that environment | Reports one line, **expected**: `rapidocr 3.9.2 requires opencv-python, which is not installed` (see "Why `--no-deps`" below) |
| Real local OCR in the fresh environment (offline fallback code path) | **Verified:** an invoice image read correctly in 1.6 s, using model files copied in from the development machine (no download) |
| App start from the fresh environment with no key | **Verified:** `python -m webapp` serves the UI; starting a scan is refused (HTTP 409) with the key setup text |
| Other Python versions (3.10, 3.13), macOS, Linux | Not tested |
| Cloud providers with real keys | Gemini and OCR.space were each used by the project owner from the UI; the automated tests use local mocks only. OCR.space was also called once with its public demo key |
| First-time local OCR **model download** (offline fallback) | Not verified: the fresh-environment run used model files copied from the development machine. The download code path was read (hash-checked fetch from ModelScope), not executed from a clean checkout |

## 1. Prerequisites
- Windows 11, Python 3.11 (python.org installer). The web app does not need Tk.
- [Ollama](https://ollama.com) installed and running. Then: `ollama pull gemma3:4b` (about 3.3 GB). CPU only is fine; expect about 5 GB of RAM while the model is loaded.
- An API key for **one** OCR provider:
  - Google Gemini: create a key in Google AI Studio. Free tiers have small per-minute and per-day limits that change.
  - OCR.space: free key from https://ocr.space/ocrapi/freekey (25,000 requests a month, 500 a day per IP, 1 MB per image).
- Optional: a phone with the DroidCam app on the same network. Without one, use the simulated camera (it also needs a working OCR provider).

## 2. Install
```bash
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --no-deps -r requirements-lock.txt
```
**Why `--no-deps` and why headless OpenCV.** The product uses only core OpenCV (VideoCapture with FFmpeg, resize, JPEG
encode/decode, Laplacian, Farneback optical flow) and never opens an OpenCV window, so `opencv-python-headless` is enough.
`rapidocr` declares a dependency on `opencv-python`; installing that as well would put two OpenCV distributions into one `cv2`
folder, where they overwrite each other (the development machine had three at once). If you already have several installed:
`pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless`, then install the lock file again.

## 3. Configure
Keys are read from environment variables only. On Windows, `setx` saves a variable for **new** terminals, so open a new terminal
afterwards. Never put a key in a file that is committed.
```bash
setx GEMINI_API_KEY "..."                      # Gemini (default provider)
setx OCRSPACE_API_KEY "..."                    # or OCR.space
setx MAVIS_OCR_PROVIDER ocrspace               # only if you want OCR.space
```
| Variable | Default | Meaning |
|---|---|---|
| `MAVIS_OCR_PROVIDER` | `gemini` | `gemini` or `ocrspace` |
| `GEMINI_API_KEY` / `OCRSPACE_API_KEY` | none | the provider's key (required; scanning is refused without it) |
| `MAVIS_READER` | unset | `local`: development override, always read with the local engine |
| `MAVIS_GEMINI_MODEL` | `gemini-2.5-flash` | model name (names change; check Google AI Studio) |
| `MAVIS_GEMINI_URL`, `MAVIS_OCRSPACE_URL` | provider URLs | API base URL (tests point these at a mock). Never point them at a host you do not trust: the key is sent there |
| `MAVIS_GEMINI_PARALLEL` / `MAVIS_GEMINI_RPM` | 3 / 10 | pages read at once / requests per minute |
| `MAVIS_GEMINI_THINKING_BUDGET` | 0 | 0 turns off model "thinking" (much faster pages); negative keeps the model default |
| `MAVIS_OCRSPACE_ENGINE` / `_LANGUAGE` / `_TABLE` | 2 / `eng` / 0 | OCR.space engine (3 = most accurate, slower), language (`auto` to detect), line-by-line mode |
| `MAVIS_OCRSPACE_PARALLEL` / `MAVIS_OCRSPACE_RPM` | 2 / 20 | as above |
| `MAVIS_OCR_RETRIES` (or `MAVIS_GEMINI_RETRIES`) | 2 | retries for server errors and timeouts |
| `MAVIS_OCR_THREADS` | 4 | threads for the local OCR fallback (0 = library default) |
| `MAVIS_LLM_MODEL` / `MAVIS_LLM_URL` | `gemma3:4b` / `http://127.0.0.1:11434` | local model and Ollama address |
| `MAVIS_LLM_NUM_CTX` | 16384 | model context size (a document larger than this is refused, never truncated) |
| `MAVIS_LLM_THREADS` | 8 | CPU threads the model may use (0 = all cores; lower it if the browser stalls during answers) |
| `MAVIS_LLM_TIMEOUT_S` / `MAVIS_LLM_KEEP_ALIVE` | 600 / `10m` | request timeout / how long Ollama keeps the model loaded |
| `MAVIS_LLM_RECOVER` | off in the web app | `1` re-checks the first pages for OCR mistakes with the model (no measured accuracy benefit, minutes of CPU) |

## 4. Run
```bash
.venv\Scripts\python -m webapp              # http://127.0.0.1:8765 ; options: --port N, --no-browser
```
The home page shows the OCR provider status, the local model status and the camera. Connect DroidCam (the app shows an address such
as `192.168.1.5:4747`; MAVIS reads `http://<ip>:<port>/video`) or press "Try it with a simulated camera".

### Local OCR models (offline fallback only)
`rapidocr` downloads its PP-OCRv5 models from ModelScope on first use. It compares an **already cached** file with its pinned SHA-256
value (a matching file is reused, a mismatching one is downloaded again); a **fresh download is not hash-checked** after it has
been saved. The models are cached in `prototype/models/` (git-ignored, not part of this repository). While online MAVIS never loads them. To have the
fallback ready before you ever go offline, build the reader once while connected (**not yet verified from a clean checkout**):
```bash
.venv\Scripts\python -c "import sys; sys.path[:0]=['prototype','.']; from ocr_engine import PrototypeOCREngine as E; E._init_reader()"
```

## 5. Tests
```bash
.venv\Scripts\python -m pip install --no-deps -r requirements-dev.txt
.venv\Scripts\python -m pytest prototype/test_ocr_engine.py tests -q
set PYTHONPATH=.& .venv\Scripts\python tests\test_postprocessing.py
set PYTHONPATH=.& .venv\Scripts\python tests\test_wordfreq_correction.py
```
No key, Ollama or camera is needed. The suite takes a few minutes (real threads, mock servers). The standalone scripts write
into `tests/exports/` (git-ignored).

## 6. Troubleshooting
| Symptom | Cause and fix |
|---|---|
| "Page reading needs a ... API key" and scans refuse to start | The key variable is not visible to the process. Run `setx`, then start MAVIS from a **new** terminal |
| Home page says the key was rejected / model not found | Check the key; for Gemini set `MAVIS_GEMINI_MODEL` to a model your key can use. (OCR.space cannot verify a key without using a request, so a bad key shows on the first page) |
| Pages wait with a countdown ("rate limit") | The provider's free quota. Pages are kept and read when it clears; lower `*_RPM`, or wait. A *daily* quota fails the page; use "Read this page again" later or switch provider |
| A page is listed "Could not be read" | The reason is shown on the page card. Open the pages panel and press **Read this page again** |
| "Offline: local reader will be used" | No route to the provider. Pages are read locally (slower, needs the local models, see above) |
| "AI assistant unavailable" | Start Ollama and run `ollama pull gemma3:4b`; then "Check again" |
| Answers take about 30 s | Expected on CPU. The first one after a scan is preceded by a preparation step of about a minute |
| Browser or PC stalls while answering | Lower `MAVIS_LLM_THREADS` (for example 6) |
| Cannot connect to the camera | Phone and PC on the same network; use the address DroidCam shows; check the Windows firewall; DroidCam serves one client at a time |
| Poor text | Mostly capture conditions: lamp light from above, no shadow or glare on the page, page fills the frame, phone parallel to the page, 1080p in DroidCam |
| "Address already in use" | Another MAVIS is running; use `--port` |
| A native DLL is blocked (once seen: `pyclipper`) | Windows Application Control policy. Do not bypass it; ask the administrator. Online scanning does not need that library |
