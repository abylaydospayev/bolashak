# ============================================
# COPY AND PASTE THIS INTO VM POWERSHELL
# ============================================

# Step 1: Navigate to project folder
cd C:\Users\abylay_dos\Desktop\bolashak

# Step 2: Initialize git and pull updates
git init
git remote add origin https://github.com/abylaydospayev/bolashak.git
git fetch origin main
git checkout -b main origin/main

# Step 3: Update .env file (you'll need to edit this manually)
Write-Host ""
Write-Host "=== NOW UPDATE .env FILE ===" -ForegroundColor Yellow
Write-Host "Open: live_trading\.env" -ForegroundColor White
Write-Host "Add these lines at the end:" -ForegroundColor White
Write-Host ""
Write-Host "MAX_POSITIONS=3" -ForegroundColor Cyan
Write-Host "MIN_INTERVAL_SECONDS=300" -ForegroundColor Cyan
Write-Host "STOP_LOSS_PIPS=30" -ForegroundColor Cyan
Write-Host "TAKE_PROFIT_PIPS=50" -ForegroundColor Cyan
Write-Host "MAX_DAILY_LOSS=500" -ForegroundColor Cyan
Write-Host "BUY_THRESHOLD=0.75" -ForegroundColor Cyan
Write-Host "SELL_THRESHOLD=0.25" -ForegroundColor Cyan
Write-Host "MIN_PROBABILITY_DIFF=0.2" -ForegroundColor Cyan
Write-Host ""
Write-Host "VERIFY: MT5_LOT_SIZE=0.1  (NOT 11.77!)" -ForegroundColor Red
Write-Host ""
Write-Host "Press Enter when .env is updated..." -ForegroundColor Green
Read-Host

# Step 4: Reminder to enable AutoTrading
Write-Host ""
Write-Host "=== ENABLE AUTOTRADING IN MT5 ===" -ForegroundColor Yellow
Write-Host "1. Open MT5 terminal" -ForegroundColor White
Write-Host "2. Press Ctrl+E" -ForegroundColor White
Write-Host "3. Verify button turns GREEN" -ForegroundColor White
Write-Host ""
Write-Host "Press Enter when AutoTrading is enabled..." -ForegroundColor Green
Read-Host

# Step 5: Stop old bot (if running)
Write-Host ""
Write-Host "=== STOP OLD BOT ===" -ForegroundColor Yellow
Write-Host "If demo_bot.py is running, press Ctrl+C in that window" -ForegroundColor White
Write-Host ""
Write-Host "Press Enter to continue..." -ForegroundColor Green
Read-Host

# Step 6: Start new bot with risk management
Write-Host ""
Write-Host "=== STARTING NEW BOT WITH RISK MANAGEMENT ===" -ForegroundColor Green
Write-Host ""

.\.venv\Scripts\Activate.ps1
python live_trading\demo_bot_with_risk.py
