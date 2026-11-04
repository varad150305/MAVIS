# Licences and model notices

**Verify each licence on its official page before redistributing this folder or using it commercially.** The entries below are the licences these projects publish, as known at packaging time. The model cards themselves were not part of the cached assets and are not included.

## Bundled model assets (`models/`)

| asset | origin | licence (as published) | note |
|---|---|---|---|
| `Systran/faster-whisper-tiny` @ d90ca5fe | CTranslate2 conversion of OpenAI Whisper tiny, by SYSTRAN | MIT (Whisper: MIT) | ASR |
| `Helsinki-NLP/opus-mt-en-hi` @ 75d7f7c9 | OPUS-MT, University of Helsinki | Apache-2.0 on the Hugging Face model card (OPUS-MT models are also distributed as CC-BY 4.0); **check the card** | draft MT |
| `facebook/nllb-200-distilled-600M` @ f8d333a0 tokenizer files, and its local CTranslate2 int8 conversion | Meta AI (NLLB-200) | **CC-BY-NC 4.0: non-commercial use only** | final MT. **A commercial MAVIS deployment needs a different final-MT model or permission** |
| Silero VAD (inside the `silero-vad` pip package, not in `models/`) | Silero Team | MIT | VAD |

## Python packages (installed by `setup_demo.ps1`, not bundled)

| package | licence (as published) |
|---|---|
| torch, torchaudio | BSD-3-Clause |
| faster-whisper, ctranslate2, onnxruntime, sounddevice (+ PortAudio), silero-vad, PyYAML, sacremoses | MIT |
| transformers, tokenizers, huggingface-hub, safetensors, sentencepiece | Apache-2.0 |
| numpy | BSD-3-Clause |
| opencv-python | Apache-2.0 (bundles FFmpeg, LGPL) |
| av (PyAV) | BSD-3-Clause; bundles FFmpeg (LGPL) |

## Code

- `product/`: the standalone product layer.
- The frozen engine (`main.py`, `asr/`, `audio/`, `vad/`, `streaming/`, `translation/`, `utils/`, `evaluation/`): copied unchanged from the originating research project. See `SOURCE_PROVENANCE.json`.

No licence file existed in the originating project. Clarify the terms with the project owner before redistributing.
