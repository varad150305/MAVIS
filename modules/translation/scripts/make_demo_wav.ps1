<#
.SYNOPSIS
  Create a 16 kHz mono WAV of the demo script with a built-in Windows voice
  (System.Speech; no download). Used by the end-to-end smoke test and for
  rehearsing the UI without a microphone (UI: Source = WAV file).
.EXAMPLE
  .\scripts\make_demo_wav.ps1                       # -> runtime_data\demo\demo_script_tts.wav
  .\scripts\make_demo_wav.ps1 -Out runtime_data\demo\short.wav -Lines 3
#>
param([string]$Out = "", [int]$Lines = 0)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Out) { $Out = Join-Path $Root "runtime_data\demo\demo_script_tts.wav" }
elseif (-not [System.IO.Path]::IsPathRooted($Out)) { $Out = Join-Path $Root $Out }
New-Item -ItemType Directory -Force (Split-Path -Parent $Out) | Out-Null
$text = Get-Content (Join-Path $Root "demo_assets\demo_script.txt") -Encoding UTF8
if ($Lines -gt 0) { $text = $text | Select-Object -First $Lines }
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile($Out, $fmt)
$s.Rate = -1
foreach ($line in $text) { $s.Speak($line); }
$s.Dispose()
Write-Host "wrote $Out"
