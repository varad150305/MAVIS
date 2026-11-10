"""Run from modules/translation with its own venv. Fetch any file Yogita's verify_models.py lists as MISSING straight into its snapshot folder (plain files, no symlinks).
Uses her PINNED list (repo, revision, files). Her code is not changed."""
import os, sys
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "0"; os.environ["TRANSFORMERS_OFFLINE"] = "0"
sys.path.insert(0, os.getcwd())
import runpy
vm = runpy.run_path("scripts/verify_models.py")          # defines PINNED without running main
from huggingface_hub import hf_hub_download
hub = Path("models") / "huggingface" / "hub"
for repo, (rev, files) in vm["PINNED"].items():
    if not files:
        continue
    snap = hub / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
    snap.mkdir(parents=True, exist_ok=True)
    ref = hub / ("models--" + repo.replace("/", "--")) / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True); ref.write_text(rev, encoding="utf-8")
    for f in files:
        if (snap / f).exists():
            continue
        print("fetching", repo, f, flush=True)
        hf_hub_download(repo, f, revision=rev, local_dir=str(snap))
print("repair done", flush=True)
