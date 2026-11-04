@echo off
rem MAVIS - start (Windows). Everything the demo needs: Ollama (if installed), then MAVIS, then the browser.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe call setup.bat

rem Ollama answers questions in the OCR + LLM module. Start it if it is installed and not running yet.
tasklist /fi "imagename eq ollama.exe" 2>nul | find /i "ollama.exe" >nul
if errorlevel 1 (
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
    echo Starting Ollama...
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
    timeout /t 5 /nobreak >nul
  ) else (
    echo Ollama is not installed: OCR + LLM can scan pages but cannot answer questions. See https://ollama.com
  )
)

if exist keys.env (
  for /f "usebackq eol=# tokens=1,* delims==" %%a in ("keys.env") do if not "%%b"=="" set "%%a=%%b"
)
.venv\Scripts\python -m hub %*
pause
