<#
.SYNOPSIS
  Console version of the translator (same pipeline as the UI). Arguments are
  passed to `python -m product.live_translate`.

.EXAMPLE
  .\run_cli_demo.ps1 --list-devices                    # list microphones (* = default)
  .\run_cli_demo.ps1                                   # default microphone; Ctrl+C stops (drains, writes files)
  .\run_cli_demo.ps1 --device 1 --save-audio           # microphone 1, keep input.wav
  .\run_cli_demo.ps1 --wav runtime_data\demo\demo_script_tts.wav --duration 30   # replay a 16 kHz WAV at 1x
  .\run_cli_demo.ps1 --help
#>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Write-Host "Run .\setup_demo.ps1 first." -ForegroundColor Red; exit 1 }
& $venvPy -m product.live_translate @args
exit $LASTEXITCODE
