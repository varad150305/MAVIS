# Third-party components and terms (DRAFT: not legal advice)

This repository's own licence has **not been chosen** (no LICENSE file). This note records what was actually checked about the
third-party material MAVIS installs or calls. "Verified" means read from the installed package metadata or from the cited page
on 2026-10-29; "not verified" means exactly that. Nothing below was invented; where a fact could not be confirmed it says so.

## 1. Python packages (installed from PyPI by the user; not redistributed here)
Licence strings below are the ones in each package's installed metadata (verified). The full 43-package closure in
`requirements-lock.txt` has **not** been reviewed; run a licence report on a clean environment before publishing binaries.

| Package | Version | Licence per metadata |
|---|---|---|
| aiohttp | 3.13.3 | Apache-2.0 AND MIT |
| numpy | 1.26.4 | BSD |
| opencv-python-headless | 4.8.0.76 | Apache 2.0 (wheel packaging) |
| wordfreq | 3.1.1 | Apache-2.0 |
| ftfy | 6.3.1 | Apache-2.0 |
| rapidocr | 3.9.2 | Apache-2.0 |
| onnxruntime | 1.23.2 | MIT |
| pillow | 12.0.0 | MIT-CMU |
| pyyaml | 6.0.2 | MIT |
| shapely | 2.1.2 | BSD 3-Clause |
| pyclipper | 1.4.0 | MIT |
| requests | 2.32.5 | Apache-2.0 |
| protobuf | 4.25.3 | 3-Clause BSD |
| tqdm | 4.67.1 | MPL-2.0 AND MIT |
| regex | 2026.2.28 | Apache-2.0 AND CNRI-Python |
| pytest (dev only) | 9.1.1 | MIT |

Not verified: the licence of the data files shipped inside `wordfreq` (it bundles word-frequency data compiled from several
sources; read its README); the FFmpeg binaries inside the OpenCV wheels (the OpenCV build report shows "FFMPEG: YES (prebuilt
binaries)"; their licence terms were not assessed).

## 2. Local OCR models (offline fallback): **not bundled**
- MAVIS uses RapidOCR's PP-OCRv5 mobile detector (`ch_PP-OCRv5_det_mobile.onnx`), English recogniser
  (`en_PP-OCRv5_rec_mobile.onnx`) and classifier (`ch_ppocr_mobile_v2.0_cls_mobile.onnx`).
- Verified: the three files used during development are byte-identical (SHA-256) to the entries in RapidOCR 3.9.2's own model
  catalog, and RapidOCR downloads them from `modelscope.cn` on first use. RapidOCR checks those hashes only for a file that is
  already cached (a mismatching cached file is downloaded again); it does not verify a freshly downloaded file against the hash
  (read from its `download_file.py`; the download itself has not been run from a clean checkout).
- Verified from the RapidOCR README (2026-10-29): code under Apache-2.0; models derived from PaddleOCR; "The applicable
  upstream models are distributed under the Apache License, Version 2.0" and converted artifacts are redistributed under the same
  terms; RapidOCR claims no ownership of the weights. PaddleOCR's own repository states Apache-2.0.
- Not verified: RapidOCR's `MODEL_LICENSES.md` (referenced but not readable), and a model-specific licence statement for PP-OCRv5
  on the PaddleOCR page.
- Decision: because redistribution rights were not fully confirmed and the files are fetched automatically, **the model files
  are not included in this repository** (`prototype/models/` is git-ignored). If they are ever bundled, include the Apache-2.0
  text and the upstream notices.

## 3. Language model: Gemma 3 through Ollama (user-installed; not redistributed)
- `ollama pull gemma3:4b` downloads the weights under the **Gemma Terms of Use** (https://ai.google.dev/gemma/terms). Summary of
  that page, verified 2026-10-29: it is a custom agreement, **not an open-source licence**; use is bound by the Gemma Prohibited
  Use Policy; anyone who *distributes* Gemma must pass on a notice and the use restrictions; Google claims no rights in outputs
  and the user is responsible for them. This repository does not distribute the weights. The page notes a separate licence for
  later Gemma versions; check the terms of whichever model you configure.
- Ollama itself: licence not reviewed here.

## 4. Online services (called with the user's own key)
- Google Gemini API: Gemini API Additional Terms; summary of the unpaid-use data clauses is in `SECURITY_AND_DATA_FLOW.md`.
- OCR.space API: terms not reviewed.
The user is responsible for complying with each provider's terms and for what they submit.

## 5. This project
No licence selected. Until the owner adds a LICENSE file, nobody has been granted rights to use, copy or modify the code.
