"""Checks that the three original projects in modules/ are complete and unchanged.

    python -m hub.verify_originals

ORIGINALS.sha256 lists every file of the three projects as delivered (Drishti zip, mavis-ocr-llm and
gesture_recognition repositories) with its SHA-256. Files created later by running the apps (exports, caches,
__pycache__, downloaded models) are not part of the originals and are ignored.
"""
import hashlib
import sys

from .paths import SUITE

MANIFEST = SUITE / "ORIGINALS.sha256"


def main() -> int:
    missing, changed, ok = [], [], 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, rel = line.split("  ", 1)
        f = SUITE / rel
        if not f.exists():
            missing.append(rel)
        elif hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            changed.append(rel)
        else:
            ok += 1
    print(f"{ok} original files intact, {len(changed)} changed, {len(missing)} missing")
    for rel in changed:
        print("  changed:", rel)
    for rel in missing:
        print("  missing:", rel)
    return 0 if not (missing or changed) else 1


if __name__ == "__main__":
    sys.exit(main())
