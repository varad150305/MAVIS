"""Product-facing live translation layer.

Composes the existing (research-frozen) modules into a runnable live
application without modifying any of them. Nothing under product/ is read
by research or evaluation code.
"""


# --- handoff package (destination-only addition) ---------------------------
# Model assets live inside this package (models/), never in a global or the
# research repository's cache: every process that imports `product` (the UI,
# the session worker, the CLI, tests) uses <package>/models/huggingface as
# its Hugging Face cache and stays offline. MAVIS_HF_HOME overrides the
# location explicitly (e.g. a shared host model store).
import os as _os
from pathlib import Path as _Path

PACKAGE_ROOT = _Path(__file__).resolve().parent.parent
_os.environ["HF_HOME"] = _os.environ.get("MAVIS_HF_HOME") or str(PACKAGE_ROOT / "models" / "huggingface")
for _var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    _os.environ.pop(_var, None)
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
