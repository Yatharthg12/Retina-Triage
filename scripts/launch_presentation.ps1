$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath "artifacts\models\best_model.keras")) {
    Write-Host "Installing presentation assets..." -ForegroundColor Cyan
    python scripts\setup_presentation.py
}

Write-Host "Verifying end-to-end presentation workflow..." -ForegroundColor Cyan
python scripts\verify_presentation.py
if ($LASTEXITCODE -ne 0) {
    throw "Presentation verification failed."
}

Write-Host ""
Write-Host "RetinaTriage is starting at http://127.0.0.1:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor DarkGray
python scripts\run_presentation.py
