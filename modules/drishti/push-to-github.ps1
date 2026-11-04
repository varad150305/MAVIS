# Drishti — Push to GitHub
# Run this once from PowerShell after creating your GitHub repo.
# Usage: Right-click this file → "Run with PowerShell"

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  Drishti — Push to GitHub" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# ── Ask for GitHub details ──
$username = Read-Host "Enter your GitHub username"
$repoName = Read-Host "Enter repo name (press Enter for 'drishti')"
if ([string]::IsNullOrWhiteSpace($repoName)) { $repoName = "drishti" }

$remoteUrl = "https://github.com/$username/$repoName.git"

Write-Host ""
Write-Host "Will push to: $remoteUrl" -ForegroundColor Yellow
Write-Host ""

# ── Change to the project folder ──
Set-Location $PSScriptRoot

# ── Clean up any broken .git from previous attempt ──
if (Test-Path ".git") {
    Write-Host "Removing old .git folder..." -ForegroundColor Gray
    Remove-Item -Recurse -Force ".git"
}

# ── Init + configure ──
Write-Host "Initialising git repo..." -ForegroundColor Green
git init
git config user.email "aartik.ramkumar@gmail.com"
git config user.name "Aarthik"
git branch -M main

# ── Stage everything ──
Write-Host "Staging files..." -ForegroundColor Green
git add .
git status --short

# ── Commit ──
Write-Host "Committing..." -ForegroundColor Green
git commit -m "Initial commit — Drishti v1.0

Features:
- 10-mode AI vision (describe, read, currency, color, people, navigate, objects, prices, food, safety)
- Voice commands in 9 Indian languages + Hinglish
- Emergency SOS (voice + triple-tap + GPS SMS)
- Smart Live mode (motion detection)
- WhatsApp share on every output
- Snap history (last 5 in localStorage)
- Groq API + Sarvam AI TTS
- Deploy configs for Vercel, Netlify, GitHub Pages"

# ── Add remote + push ──
Write-Host ""
Write-Host "Adding remote and pushing..." -ForegroundColor Green
git remote add origin $remoteUrl
git push -u origin main

Write-Host ""
Write-Host "====================================" -ForegroundColor Green
Write-Host "  Done! Your repo is live at:" -ForegroundColor Green
Write-Host "  https://github.com/$username/$repoName" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next: Enable GitHub Pages at" -ForegroundColor Yellow
Write-Host "https://github.com/$username/$repoName/settings/pages" -ForegroundColor Yellow
Write-Host "Set Source to 'GitHub Actions' — the workflow will deploy automatically." -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to close"
