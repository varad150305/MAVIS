@echo off
title MAVIS - finishing the translation models
cd /d "%~dp0..\modules\translation"
.venv\Scripts\python.exe -B "%~dp0repair_translation_models.py"
.venv\Scripts\python.exe -B scripts\verify_models.py
pause
