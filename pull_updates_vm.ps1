# Run this script ON THE VM to pull risk management updates
# Location: C:\Users\abylay_dos\Desktop\bolashak

Write-Host "=== PULLING RISK MANAGEMENT UPDATES FROM GITHUB ===" -ForegroundColor Cyan
Write-Host ""

# Navigate to project folder
Set-Location C:\Users\abylay_dos\Desktop\bolashak

# Check if git is initialized
if (-not (Test-Path .git)) {
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
    git remote add origin https://github.com/abylaydospayev/bolashak.git
}

# Check for local changes
Write-Host "Checking for local changes..." -ForegroundColor Yellow
$status = git status --porcelain

if ($status) {
    Write-Host ""
    Write-Host "⚠️  You have local changes. Stashing them..." -ForegroundColor Yellow
    git stash
    Write-Host "✅ Local changes stashed" -ForegroundColor Green
}

# Pull latest changes
Write-Host ""
Write-Host "Pulling latest changes from GitHub..." -ForegroundColor Yellow
git pull origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ SUCCESS! Risk management files updated:" -ForegroundColor Green
    Write-Host "   ✓ risk_manager.py" -ForegroundColor White
    Write-Host "   ✓ demo_bot_with_risk.py" -ForegroundColor White
    Write-Host "   ✓ .env.example (template)" -ForegroundColor White
    Write-Host "   ✓ DEPLOYMENT_GUIDE.md" -ForegroundColor White
    Write-Host "   ✓ VM_SETUP_CHECKLIST.txt" -ForegroundColor White
    Write-Host ""
    
    # Check if .env needs updating
    Write-Host "=== NEXT STEPS ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. UPDATE .env FILE" -ForegroundColor Yellow
    Write-Host "   Add these lines to: live_trading\.env" -ForegroundColor White
    Write-Host ""
    Write-Host "   MAX_POSITIONS=3" -ForegroundColor Gray
    Write-Host "   MIN_INTERVAL_SECONDS=300" -ForegroundColor Gray
    Write-Host "   STOP_LOSS_PIPS=30" -ForegroundColor Gray
    Write-Host "   TAKE_PROFIT_PIPS=50" -ForegroundColor Gray
    Write-Host "   MAX_DAILY_LOSS=500" -ForegroundColor Gray
    Write-Host "   BUY_THRESHOLD=0.75" -ForegroundColor Gray
    Write-Host "   SELL_THRESHOLD=0.25" -ForegroundColor Gray
    Write-Host "   MIN_PROBABILITY_DIFF=0.2" -ForegroundColor Gray
    Write-Host ""
    Write-Host "   VERIFY: MT5_LOT_SIZE=0.1  (NOT 11.77!)" -ForegroundColor Red
    Write-Host ""
    
    Write-Host "2. ENABLE AUTOTRADING IN MT5" -ForegroundColor Yellow
    Write-Host "   Press Ctrl+E in MT5 terminal" -ForegroundColor White
    Write-Host "   Verify button turns GREEN" -ForegroundColor White
    Write-Host ""
    
    Write-Host "3. STOP OLD BOT (if running)" -ForegroundColor Yellow
    Write-Host "   Press Ctrl+C in the PowerShell window" -ForegroundColor White
    Write-Host ""
    
    Write-Host "4. START NEW BOT" -ForegroundColor Yellow
    Write-Host "   cd C:\Users\abylay_dos\Desktop\bolashak" -ForegroundColor White
    Write-Host "   .\.venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "   python live_trading\demo_bot_with_risk.py" -ForegroundColor White
    Write-Host ""
    
    Write-Host "=== DEPLOYMENT COMPLETE ===" -ForegroundColor Green
    Write-Host ""
    
} else {
    Write-Host ""
    Write-Host "❌ Error pulling from GitHub" -ForegroundColor Red
    Write-Host "This might be the first pull. Try:" -ForegroundColor Yellow
    Write-Host "  git fetch origin main" -ForegroundColor White
    Write-Host "  git checkout -b main origin/main" -ForegroundColor White
}
