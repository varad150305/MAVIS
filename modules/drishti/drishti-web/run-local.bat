@echo off
cd /d C:\Drishti\drishti-web
start "Drishti server" cmd /k "npx --yes serve -l 5500 ."
timeout /t 4 /nobreak >nul
start "" http://localhost:5500
