"""Where the three original projects live, and how their code is put on the import path.

The projects are kept exactly as they were delivered, under modules/. The hub never edits them; it imports them.
"""
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent.parent
MODULES = SUITE / "modules"
DRISHTI = MODULES / "drishti"
DRISHTI_WEB = DRISHTI / "drishti-web"
MAVIS = MODULES / "mavis"
GESTURE = MODULES / "gesture"
HUB_STATIC = Path(__file__).resolve().parent / "static"
DATA = SUITE / ".hub"            # access key, TLS certificate (git-ignored, created on first run)


def add_import_paths():
    """MAVIS is imported as the package `webapp` (with its own prototype/ folder on the path, as its server does itself).
    The gesture controller is a flat folder of modules (events, stabilizer, ...), imported by name.
    No module names collide between the two projects."""
    for p in (MAVIS / "prototype", MAVIS, GESTURE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
