"""Test-wide settings: never open the real webcam during tests (the UI's camera
background reads MAVIS_UI_CAMERA; video tests pass a generated video file)."""

import os

os.environ.setdefault("MAVIS_UI_CAMERA", "off")
