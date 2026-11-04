"""Download Yogita's translation models with her own script (scripts/verify_models.py --download), run from her
package folder with her Python 3.10 environment.

Hugging Face stores downloads as symlinks; Windows refuses symlinks unless Developer Mode is on ("WinError 1314").
This wrapper tells Hugging Face that symlinks are unsupported, so it copies the files instead. Her code is unchanged.
"""
import runpy
import sys

import huggingface_hub.file_download as fd

fd.are_symlinks_supported = lambda *a, **k: False
sys.argv = ["scripts/verify_models.py", "--download"]
runpy.run_path("scripts/verify_models.py", run_name="__main__")
