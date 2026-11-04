<#
.SYNOPSIS
  Start the standalone desktop UI (English speech -> Hindi translation).
.DESCRIPTION
  Equivalent to:  .\.venv\Scripts\python.exe -m product.ui   (run from this folder)
  Nothing is captured until you press Start. Session files go to runtime_data\sessions\.
#>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Write-Host "Run .\setup_demo.ps1 first." -ForegroundColor Red; exit 1 }
& $venvPy -m product.ui
exit $LASTEXITCODE
