"""Convert a locally cached Hugging Face seq2seq model to CTranslate2 (offline).

    .venv\Scripts\python.exe -m product.tools.convert_ct2 MODEL REVISION OUT_DIR [--quantization int8] [--copy FILE ...]

Product-only helper. Works around ctranslate2 4.8 passing `dtype=` to
transformers 4.52's `from_pretrained` (which expects `torch_dtype=`), so
neither package needs upgrading. Nothing is downloaded (HF offline mode).
"""

from __future__ import annotations

import argparse

from product import runtime


def main(argv: list[str] | None = None) -> int:
    runtime.prefer_offline_models()
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("revision")
    parser.add_argument("out_dir")
    parser.add_argument("--quantization", default="int8")
    parser.add_argument("--copy", nargs="*", default=[], help="tokenizer files to copy next to the converted model")
    args = parser.parse_args(argv)

    from ctranslate2.converters import TransformersConverter

    class _Converter(TransformersConverter):
        def load_model(self, model_class, model_name_or_path, **kwargs):
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            return model_class.from_pretrained(model_name_or_path, **kwargs)

        def load_tokenizer(self, tokenizer_class, model_name_or_path, **kwargs):
            kwargs.setdefault("revision", args.revision)
            return tokenizer_class.from_pretrained(model_name_or_path, **kwargs)

    converter = _Converter(args.model, copy_files=args.copy or None, revision=args.revision)
    print(converter.convert(args.out_dir, quantization=args.quantization, force=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
