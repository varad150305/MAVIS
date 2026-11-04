# MAVIS

One app with three modes, built from three existing projects:

| Mode | Project | What it does | Where |
|---|---|---|---|
| 1 · **Drishti** | `modules/drishti` (Drishti zip) | AI eyes: describe scenes, read text, money, colours, prices, translation, hazards, voice, Indian languages | `/drishti/` |
| 2 · **MAVIS OCR** | `modules/mavis` (varad150305/mavis-ocr-llm) | Scan printed pages, OCR them, then ask questions with page-cited answers | `/mavis/` |
| 3 · **Gesture** | `modules/gesture` (YogitaAher20/gesture_recognition) | Choose a feature with your hand: ☝️ OCR · ✌️ translation · 🤟 price comparison · 👍 confirm · 🖐 back | `/gesture/` |

**No code was lost or changed.** The three projects sit in `modules/` exactly as delivered: 234 files, each one checked by
SHA-256 in `ORIGINALS.sha256`. Run `python -m hub.verify_originals` at any time to check. Everything new is in `hub/` (the server that joins
them), `tests/` and the start scripts.

## Start it (Windows)

1. Install **Python 3.11** from python.org (tick "Add python.exe to PATH").
2. Double-click **`setup.bat`** (once, takes a few minutes).
3. Optional: open `keys.env` and add your keys. `GROQ_API_KEY` gives Drishti cloud AI; `GEMINI_API_KEY` lets OCR mode
   read pages with Gemini (without it, pages are read on this computer by MAVIS's local OCR engine). For OCR mode's
   question answering, install [Ollama](https://ollama.com) and run `ollama pull gemma3:4b`.
4. Double-click **`run.bat`**. The launcher opens at **http://127.0.0.1:8000**.

macOS/Linux: `./run.sh`.

### On your phone

The launcher (on the computer) shows a **QR code**. Put the phone on the **same Wi-Fi**, scan it, and the first time tap
**Advanced → Proceed** on the certificate warning. The warning appears because the certificate was made on your computer,
for your computer. Phones only allow the camera on HTTPS pages, so this step is needed.

- The first time `run.bat` starts, Windows asks whether Python may use the network. Allow **Private networks**,
  otherwise the phone cannot connect.
- The link contains a private access key. Without it, other devices on the Wi-Fi get a "link needed" page, so your camera
  and documents stay private. The key and certificate live in `.hub/`. Delete that folder to make new ones.

## Working offline

All three modes run without internet. Open **Offline readiness** (link at the bottom of the launcher, `/offline/`) once
while online:

- **Drishti**: click **Download Drishti's models**. Drishti's local models (SmolVLM-500M for scene description, COCO-SSD,
  Tesseract.js, YAMNet, Depth-Anything) normally download from the internet into each browser. MAVIS routes those
  downloads through this computer (`/hub/mirror/...`) and keeps a copy in `offline-cache/`, so afterwards every device,
  including phones, gets them from this computer with no internet. SmolVLM is about 2 GB. Using Drishti normally while
  online fills the same copy. Drishti's files are not changed; the page is rewritten as it is served.
- **OCR mode**: click **Download the local OCR models** (RapidOCR PP-OCRv5) and install Ollama with `gemma3:4b`.
  Without a Gemini key, or with no internet, pages are read locally.
- **Gesture**: always offline (MediaPipe and the classifier are installed with MAVIS).

The page shows what is already saved for each mode. To try it, start with `MAVIS_OFFLINE_ONLY=1`: the hub then never
contacts the internet and serves only what was saved.

## The gesture fix

**Problem:** `mavis_gesture.py` opens the camera of the computer it runs on and shows its output with `cv2.imshow()`, a
window on that computer. So the output always "went back to the system" and never appeared on the phone.

**Fix:** in Gesture mode, the camera belongs to the device you are holding. The page sends its frames to the hub, and
every frame goes through the **unchanged** gesture code (`HandTracker → GestureClassifier → GestureStabilizer →
FeatureController`). This is the in-process integration the gesture project's own `CLAUDE.md` recommends ("Option 2"). The
result comes back to the same page and is drawn there: hand skeleton, recognised gesture, confidence, hold progress, finger
states, and the select → confirm → active state. It also speaks the result aloud if you want.
**Classic view** shows the original OpenCV window (`draw_recognition` + `draw_state_panel`) as a picture on your phone
instead of on the computer.

A confirmed feature can open its mode: OCR opens MAVIS, translation opens Drishti's Translate mode, and price comparison
opens Drishti's Prices mode. Turn on "Open the feature automatically", or tap the button.

## What else the hub adds (without touching the originals)

- **MAVIS on the phone.** MAVIS's own app is mounted at `/mavis/`. Its page is served with paths pointing to `/mavis/...`;
  the files on disk are untouched. On its home screen there is a new **"Use this device's camera"** option: the phone's
  camera becomes MAVIS's camera with no DroidCam app. The hub serves the frames to MAVIS in DroidCam's own MJPEG format,
  so MAVIS scans through its normal code path. DroidCam and the simulated camera still work as before.
- **Drishti's API routes.** On Vercel, Drishti uses `api/groq.js`, `sarvam.js` and `cloudflare.js`. The hub provides the
  same routes (`/api/groq` …), with the same responses and limits, using keys from `keys.env`. `/drishti/?mode=prices`
  opens Drishti directly in a mode. Drishti's Vercel/Netlify/GitHub Pages deployment files are unchanged and still work on
  their own.
- A **Modes** button on every screen, keys 1/2/3 on the launcher, light and dark themes, and layouts for phone and desktop.

## Tests

```bat
.venv\Scripts\python -m pip install --no-deps -r requirements-dev.txt
.venv\Scripts\python -m pytest tests -q                                       & rem hub: 16 tests
cd modules\gesture && ..\..\.venv\Scripts\python -m unittest discover -s tests -t .   & rem 112 tests
cd modules\mavis && ..\..\.venv\Scripts\python -m pytest prototype/test_ocr_engine.py tests -q   & rem 149 passed, 5 skipped
```

End-to-end testing (real browser, virtual camera feeds built from real hand photos and document pages, desktop and
phone over HTTPS) is described in the black book; `tests/e2e/stand_ins.py` lets you run MAVIS's whole flow without a
Gemini key or Ollama (test stand-ins, clearly not the real services).

All of these pass in one shared environment (`requirements-lock.txt`). That file explains how the two projects'
dependencies were combined. The main point: exactly one OpenCV package, at the same 4.8.0.76 version MAVIS pins. Drishti's
own browser tests (`modules/drishti/drishti-web/tests`) were also run against Drishti served by the hub, with the same
results as Drishti served on its own.

## Layout

```
hub/            the new server: launcher, gesture service, OCR mount + camera relay, Drishti API routes, offline mirror, access/HTTPS
  static/       launcher, gesture screen, offline readiness page, small add-ons for the OCR and Drishti pages
modules/        the three original projects, unchanged (see ORIGINALS.sha256)
tests/          hub and offline tests (e2e/: test-only stand-ins)
offline-cache/  saved model files (created on first use)
setup.bat, run.bat, run.sh, keys.env.example, requirements-lock.txt
```

Ports: 8000 (this computer), 8443 (phones, HTTPS), 8766 (internal, this computer only). Change them with
`run.bat --port … --https-port … --relay-port …`, or use `--no-lan` for this computer only.

Not included in the web app: Drishti's Flutter app (`modules/drishti/drishti-app`) is kept in full as source. Build it
with Flutter as before. Its 530 MB of build caches were left out because Flutter re-creates them. The gesture project's
legacy tray app (`tray_app.py` …) is kept but not started, because it controls the mouse and keyboard.
