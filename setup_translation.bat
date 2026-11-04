@echo off
rem MAVIS - one-time setup of Yogita's live translation module: its own Python 3.10 environment and models (~1 GB).
rem Needs Python 3.10 64-bit from python.org (with tcl/tk) and internet once.
cd /d "%~dp0modules\translation"
powershell -NoProfile -ExecutionPolicy Bypass -File setup_demo.ps1 -NoTests
if not exist "models\nllb-200-distilled-600M-ct2-int8\model.bin" (
  echo Downloading the translation models ^(about 3.5 GB download, then a conversion^)...
  .venv\Scripts\python.exe -B "%~dp0hub\translation_models.py"
)
.venv\Scripts\python.exe -B scripts\verify_models.py
pause
