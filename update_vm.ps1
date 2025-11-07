# Update VM with latest code from GitHub
# Run this script on the VM to get all latest changes

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  UPDATING BOT FROM GITHUB" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project directory
cd C:\Users\abylay_dos\Desktop\bolashak

# Stop any running Python processes
Write-Host "Stopping any running bots..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Backing up local changes..." -ForegroundColor Yellow
git stash save "backup_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')"

Write-Host ""
Write-Host "Fetching latest code from GitHub..." -ForegroundColor Yellow
git fetch origin

Write-Host ""
Write-Host "Resetting to match GitHub (discarding local changes)..." -ForegroundColor Yellow
git reset --hard origin/main

Write-Host ""
Write-Host "Cleaning untracked files..." -ForegroundColor Yellow
git clean -fd

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  UPDATE COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the bot, run:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python live_trading\demo_bot_with_risk.py" -ForegroundColor White
Write-Host ""
