@echo off
rem MAVIS - start (Windows). Opens http://127.0.0.1:8000 and prints the link for your phone.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe call setup.bat
if exist keys.env (
  for /f "usebackq eol=# tokens=1,* delims==" %%a in ("keys.env") do if not "%%b"=="" set "%%a=%%b"
)
.venv\Scripts\python -m hub %*
pause
