"""Verify (and optionally re-acquire) the package-local model assets.

    .venv\\Scripts\\python.exe -B scripts\\verify_models.py            # check sizes + SHA-256
    .venv\\Scripts\\python.exe -B scripts\\verify_models.py --load     # also load every model once
    .venv\\Scripts\\python.exe -B scripts\\verify_models.py --download # re-acquire missing assets (network)

The expected files and hashes are the `models` section of SOURCE_PROVENANCE.json.
Everything lives under <package>/models; nothing is read from or written to a
global cache.

--download (fallback; needs internet, ~3.5 GB transient download):
  * Whisper tiny, opus-mt-en-hi, NLLB tokenizer: huggingface_hub
    snapshot_download of the pinned revisions into models/huggingface/hub
  * NLLB-200 int8: downloads facebook/nllb-200-distilled-600M at the pinned
    revision (~2.4 GB PyTorch weights) and converts it with
    product.tools.convert_ct2 into models/nllb-200-distilled-600M-ct2-int8.
  This fallback was NOT exercised when the package was built (the assets were
  copied from the accepted installation); a re-converted model may not be
  byte-identical, so its hashes are reported, not enforced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PINNED = {
    "Systran/faster-whisper-tiny": ("d90ca5fe260221311c53c58e660288d3deb8d356", None),
    "Helsinki-NLP/opus-mt-en-hi": ("75d7f7c9232b2891c7d65fe4ef635616c72be867",
                                   ["config.json", "generation_config.json", "pytorch_model.bin", "source.spm",
                                    "target.spm", "tokenizer_config.json", "vocab.json"]),
    "facebook/nllb-200-distilled-600M": ("f8d333a098d19b4fd9a8b18f94170487ad3f821d",
                                         ["config.json", "generation_config.json", "sentencepiece.bpe.model",
                                          "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]),
}
NLLB_CT2 = ROOT / "models" / "nllb-200-distilled-600M-ct2-int8"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check() -> list[str]:
    expected = json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))["models"]
    problems = []
    for m in expected:
        p = ROOT / m["dest_path"]
        if not p.exists():
            problems.append(f"MISSING  {m['dest_path']}  ({m['note']})")
        elif p.stat().st_size != m["bytes"] or sha(p) != m["sha256"]:
            problems.append(f"CHANGED  {m['dest_path']}  (size/hash differs from the accepted asset)")
    total = sum(m["bytes"] for m in expected) / 2**20
    print(f"{len(expected)} model files expected ({total:.0f} MB), {len(problems)} problem(s)")
    return problems


def load() -> None:
    import product  # noqa: F401  (sets the package-local HF cache + offline mode)
    from product import runtime

    runtime.prefer_offline_models()
    import numpy as np

    from asr.whisper_engine import WhisperEngine
    from product.final_mt import NllbCt2Translator
    from translation.translator import TranslationEngine

    w = WhisperEngine("tiny", "cpu", "int8", language="en", beam_size=1, temperature=0.0, cpu_threads=4,
                      vad_filter=False, word_timestamps=True)
    w.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    print("Whisper tiny: loaded")
    t = TranslationEngine(model_name="Helsinki-NLP/opus-mt-en-hi", device="cpu", max_input_tokens=256, num_beams=1,
                          max_new_tokens=256, revision=PINNED["Helsinki-NLP/opus-mt-en-hi"][0])
    print("opus-mt-en-hi: loaded")
    n = NllbCt2Translator(NLLB_CT2, threads=4)
    print("NLLB-200 int8: loaded")
    from silero_vad import load_silero_vad

    load_silero_vad(onnx=False)
    print("Silero VAD: loaded (bundled with the silero-vad package)")
    del w, t, n


def download() -> None:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    import product  # noqa: F401  (HF_HOME = package-local cache)

    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download

    for repo, (rev, files) in PINNED.items():
        print(f"downloading {repo}@{rev[:8]} ...")
        snapshot_download(repo, revision=rev, allow_patterns=files)
        ref = ROOT / "models" / "huggingface" / "hub" / ("models--" + repo.replace("/", "--")) / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(rev, encoding="utf-8")
    if not (NLLB_CT2 / "model.bin").exists():
        rev = PINNED["facebook/nllb-200-distilled-600M"][0]
        print("downloading the full NLLB-200 600M weights for conversion (~2.4 GB) ...")
        snapshot_download("facebook/nllb-200-distilled-600M", revision=rev)
        os.environ["HF_HUB_OFFLINE"] = "1"
        from product.tools import convert_ct2

        convert_ct2.main(["facebook/nllb-200-distilled-600M", rev, str(NLLB_CT2), "--quantization", "int8",
                          "--copy", "tokenizer.json", "sentencepiece.bpe.model", "special_tokens_map.json",
                          "tokenizer_config.json"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="load every model once (slow: ~10-30 s)")
    ap.add_argument("--download", action="store_true", help="re-acquire missing assets (internet)")
    args = ap.parse_args()
    problems = check()
    if problems and args.download:
        download()
        problems = check()
        for p in problems:
            print("  (after download) " + p + "  - a re-downloaded/re-converted file may legitimately differ")
        problems = [p for p in problems if p.startswith("MISSING")]
    for p in problems:
        print("  " + p)
    if problems:
        print("Model assets are incomplete. Restore the models/ folder from the handoff package, or run\n"
              "  .\\setup_models.ps1 -Download   (internet; see DEMO_READINESS.md / CLAUDE.md 'Model assets')")
        return 1
    if args.load:
        load()
    print("models OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
