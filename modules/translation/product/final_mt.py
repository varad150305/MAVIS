"""Product-only final-quality translation: NLLB-200 distilled 600M (CTranslate2 int8).

Live drafts keep using the research MT (opus-mt-en-hi via
translation.translator, scheduled by MT3-C). When a segment closes, its
complete English source is translated once more by this engine and that
result is shown as the segment's final Hindi.

Why (product_outputs/diagnostics/2026-10-28_live_quality_report.md):
opus-mt-en-hi mistranslates clean sentences ("artificial" -> "कलापूर्ण",
"reliable" -> "पूर्ण", drops "technology") and hallucinates on garbled
input (the "Bible" draft is its deterministic output for the session's
misrecognized source); NLLB-600M preserved the meaning of every clean test
sentence. It is ~6x slower than opus-mt on this CPU, too slow for every
draft but affordable once per segment.

The model is a local conversion of the cached Hugging Face snapshot
(`python -m product.tools.convert_ct2 facebook/nllb-200-distilled-600M
f8d333a098d19b4fd9a8b18f94170487ad3f821d product_outputs/models/
nllb-200-distilled-600M-ct2-int8 --copy tokenizer.json
sentencepiece.bpe.model special_tokens_map.json tokenizer_config.json`).
Nothing is downloaded at run time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "nllb-200-distilled-600M-ct2-int8"  # handoff: package-local
NLLB_TOKENIZER = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"


@dataclass
class FinalResponse:
    """The fields MTWorker reads from a translate_request() response."""

    target_text: str
    inference_time_ms: float


class NllbCt2Translator:
    """English -> Hindi with a CTranslate2 NLLB model (greedy, like the research MT)."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, threads: int = 4, max_decoding_length: int = 256,
                 source_language: str = "eng_Latn", target_language: str = "hin_Deva") -> None:
        import ctranslate2
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        if not (model_dir / "model.bin").exists():
            raise FileNotFoundError(
                f"{model_dir} has no converted model; see product/final_mt.py for the conversion command")
        self.model_dir = model_dir
        self.threads = threads
        self.max_decoding_length = max_decoding_length
        self.target_language = target_language
        self.tokenizer = AutoTokenizer.from_pretrained(NLLB_TOKENIZER, revision=NLLB_REVISION, src_lang=source_language)
        self.translator = ctranslate2.Translator(str(model_dir), device="cpu", compute_type="int8",
                                                 intra_threads=threads, inter_threads=1)

    def translate(self, text: str) -> str:
        text = " ".join(text.split())
        if not text:
            return ""
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        result = self.translator.translate_batch([tokens], target_prefix=[[self.target_language]], beam_size=1,
                                                 max_decoding_length=self.max_decoding_length)
        hypothesis = result[0].hypotheses[0]
        if hypothesis and hypothesis[0] == self.target_language:
            hypothesis = hypothesis[1:]
        return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(hypothesis), skip_special_tokens=True).strip()

    def translate_request(self, request: Any) -> FinalResponse:
        start = time.perf_counter()
        text = self.translate(request.source_text)
        return FinalResponse(text, (time.perf_counter() - start) * 1000.0)

    def warm_up(self) -> None:
        self.translate("Hello.")

    def settings(self) -> dict[str, Any]:
        return {"model": "facebook/nllb-200-distilled-600M", "revision": NLLB_REVISION, "backend": "ctranslate2",
                "compute_type": "int8", "model_dir": str(self.model_dir), "threads": self.threads,
                "beam_size": 1, "max_decoding_length": self.max_decoding_length,
                "target_language": self.target_language}
