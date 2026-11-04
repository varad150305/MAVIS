<#
.SYNOPSIS
  Create this package's own virtual environment and install the translation module.

.DESCRIPTION
  1. finds Python 3.10 (tested: 3.10.11, Windows x64)
  2. creates .venv inside this folder (never anywhere else)
  3. installs CPU-only torch/torchaudio from the PyTorch CPU index, then
     requirements-demo.txt (and requirements-test.txt unless -NoTests)
  4. runs pip check and an import check
  5. runs setup_models.ps1 (verifies the bundled models)

.EXAMPLE
  cd "<path>\mavis_translation_demo_handoff"
  .\setup_demo.ps1
.EXAMPLE
  .\setup_demo.ps1 -Python "C:\Python310\python.exe"
.EXAMPLE
  .\setup_demo.ps1 -UseLock      # exact versions of every package (requirements-demo.lock.txt)
#>
param(
    [string]$Python = "",
    [switch]$NoTests,
    [switch]$UseLock,
    [switch]$Recreate
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

function Fail($msg) { Write-Host "SETUP FAILED: $msg" -ForegroundColor Red; exit 1 }

# ---- 1. Python 3.10 ---------------------------------------------------------
$candidates = @()
if ($Python) { $candidates += ,@($Python) }
$candidates += ,@("py", "-3.10")
$candidates += ,@("python")
$pyCmd = $null
foreach ($c in $candidates) {
    try {
        $exe = $c[0]; $pre = @(); if ($c.Count -gt 1) { $pre = $c[1..($c.Count - 1)] }
        $v = & $exe @pre -c "import sys, platform; print('%d.%d.%d %s' % (sys.version_info[:3] + (platform.machine(),)))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v -match '^3\.10\.\d+ (AMD64|x86_64)') { $pyCmd = $c; Write-Host "Python: $v ($($c -join ' '))"; break }
        elseif ($v) { Write-Host "skipping $($c -join ' '): Python $v (need 3.10 x64)" }
    } catch { }
}
if (-not $pyCmd) { Fail "Python 3.10 (64-bit) not found. Install Python 3.10.x from python.org (tick 'tcl/tk' - the desktop UI needs tkinter) or pass -Python <path to python.exe>." }

# ---- 2. venv in this folder ----------------------------------------------------
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if ($Recreate -and (Test-Path (Join-Path $Root ".venv"))) { Remove-Item -Recurse -Force (Join-Path $Root ".venv") }
if (-not (Test-Path $venvPy)) {
    $exe = $pyCmd[0]; $pre = @(); if ($pyCmd.Count -gt 1) { $pre = $pyCmd[1..($pyCmd.Count - 1)] }
    & $exe @pre -m venv (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) { Fail "could not create .venv" }
}
& $venvPy -c "import tkinter" 2>$null
if ($LASTEXITCODE -ne 0) { Fail "this Python has no tkinter (needed by the desktop UI). Reinstall Python with 'tcl/tk and IDLE' selected." }

# ---- 3. packages -------------------------------------------------------------
& $venvPy -m pip install --upgrade "pip==25.1.1"
if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed (network?)" }
if ($UseLock) {
    & $venvPy -m pip install -r requirements-demo.lock.txt
    if ($LASTEXITCODE -ne 0) { Fail "installing requirements-demo.lock.txt failed" }
} else {
    & $venvPy -m pip install -r requirements-torch-cpu.txt
    if ($LASTEXITCODE -ne 0) { Fail "CPU torch install failed (needs https://download.pytorch.org/whl/cpu)" }
    & $venvPy -m pip install -r requirements-demo.txt
    if ($LASTEXITCODE -ne 0) { Fail "installing requirements-demo.txt failed" }
    if (-not $NoTests) {
        & $venvPy -m pip install -r requirements-test.txt
        if ($LASTEXITCODE -ne 0) { Fail "installing requirements-test.txt failed" }
    }
}

# ---- 4. consistency -------------------------------------------------------------
& $venvPy -m pip check
if ($LASTEXITCODE -ne 0) { Fail "pip check reported inconsistent dependencies (see above)" }
& $venvPy -B scripts\check_environment.py
if ($LASTEXITCODE -ne 0) { Fail "environment check failed (see above)" }

# ---- 5. models ----------------------------------------------------------------
& (Join-Path $Root "setup_models.ps1")
if ($LASTEXITCODE -ne 0) { Fail "model check failed - see setup_models.ps1 output" }

Write-Host ""
Write-Host "Setup complete. Start the desktop UI with:  .\run_demo.ps1" -ForegroundColor Green
Write-Host "  (or directly: .\.venv\Scripts\python.exe -m product.ui)"
exit 0
