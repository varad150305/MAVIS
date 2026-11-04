@echo off
rem MAVIS - one-time setup (Windows). Needs Python 3.11 from python.org.
cd /d "%~dp0"
py -3.11 --version >nul 2>&1
if errorlevel 1 (
  echo.
  echo  Python 3.11 was not found. Install it from https://www.python.org/downloads/release/python-3119/
  echo  ^(tick "Add python.exe to PATH"^), then run setup.bat again.
  echo.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo Creating the Python environment...
  py -3.11 -m venv .venv || goto :fail
)
echo Installing packages ^(first time: a few minutes^)...
.venv\Scripts\python -m pip install --upgrade pip >nul
.venv\Scripts\python -m pip install --no-deps -r requirements-lock.txt || goto :fail
if not exist keys.env copy keys.env.example keys.env >nul
echo.
echo  Setup done. Put your API keys in keys.env ^(optional^), then double-click run.bat
echo.
pause
exit /b 0
:fail
echo.
echo  Setup failed. Scroll up for the error.
pause
exit /b 1
