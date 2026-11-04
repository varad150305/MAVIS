<#
.SYNOPSIS
  Verify the package-local model assets (bundled in models\, ~1.0 GB).

.DESCRIPTION
  The package ships the exact models the product was accepted with:
    models\huggingface\hub\models--Systran--faster-whisper-tiny      Whisper tiny (ASR)
    models\huggingface\hub\models--Helsinki-NLP--opus-mt-en-hi        draft MT (pinned revision)
    models\huggingface\hub\models--facebook--nllb-200-distilled-600M  NLLB tokenizer files only
    models\nllb-200-distilled-600M-ct2-int8                           NLLB-200 600M, CTranslate2 int8 (final MT)
  Silero VAD's weights are inside the silero-vad pip package.
  Default: check every file's size and SHA-256 against SOURCE_PROVENANCE.json.

.EXAMPLE
  .\setup_models.ps1            # verify (fast)
  .\setup_models.ps1 -Load      # verify and load each model once (~10-30 s)
  .\setup_models.ps1 -Download  # re-acquire missing assets from Hugging Face (internet, ~3.5 GB transient; fallback)
#>
param([switch]$Load, [switch]$Download)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Write-Host "Run .\setup_demo.ps1 first (no .venv in this folder)." -ForegroundColor Red; exit 1 }
$a = @("-B", "scripts\verify_models.py")
if ($Load) { $a += "--load" }
if ($Download) { $a += "--download" }
& $venvPy @a
exit $LASTEXITCODE
