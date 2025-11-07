# Stop the trading bot

Write-Host "Stopping trading bot..." -ForegroundColor Yellow

# Stop all Python processes
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "Bot stopped!" -ForegroundColor Green
Write-Host ""
Write-Host "To restart: .\start_bot.ps1" -ForegroundColor Cyan
Write-Host ""
