# Deploy Risk Management System to VM
# Run this script AFTER you've connected via RDP

Write-Host "=== RISK MANAGEMENT DEPLOYMENT SCRIPT ===" -ForegroundColor Cyan
Write-Host ""

# Files to deploy
$localPath = "C:\Users\abyla\Desktop\bolashak\live_trading"
$vmPath = "C:\Users\abylay_dos\Desktop\bolashak\live_trading"

Write-Host "This script will guide you through deploying the risk management system." -ForegroundColor White
Write-Host ""
Write-Host "STEP 1: Connect to VM" -ForegroundColor Yellow
Write-Host "  - The RDP session should already be open" -ForegroundColor Gray
Write-Host "  - If not, run: mstsc /v:20.9.129.81" -ForegroundColor Gray
Write-Host ""

Write-Host "STEP 2: Copy these files from local PC to VM:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  NEW FILES:" -ForegroundColor Green
Write-Host "    $localPath\risk_manager.py" -ForegroundColor White
Write-Host "    $localPath\demo_bot_with_risk.py" -ForegroundColor White
Write-Host ""
Write-Host "  UPDATED FILE:" -ForegroundColor Yellow
Write-Host "    $localPath\.env" -ForegroundColor White
Write-Host ""
Write-Host "  DESTINATION:" -ForegroundColor Cyan
Write-Host "    $vmPath\" -ForegroundColor White
Write-Host ""

Write-Host "STEP 3: Verify .env on VM has these lines:" -ForegroundColor Yellow
Write-Host "  MAX_POSITIONS=3" -ForegroundColor White
Write-Host "  MIN_INTERVAL_SECONDS=300" -ForegroundColor White
Write-Host "  STOP_LOSS_PIPS=30" -ForegroundColor White
Write-Host "  TAKE_PROFIT_PIPS=50" -ForegroundColor White
Write-Host "  MAX_DAILY_LOSS=500" -ForegroundColor White
Write-Host "  BUY_THRESHOLD=0.75" -ForegroundColor White
Write-Host "  SELL_THRESHOLD=0.25" -ForegroundColor White
Write-Host "  MT5_LOT_SIZE=0.1  # CRITICAL: Must be 0.1, not 11.77!" -ForegroundColor Red
Write-Host ""

Write-Host "STEP 4: Enable AutoTrading in MT5 on VM" -ForegroundColor Yellow
Write-Host "  - Open MT5" -ForegroundColor Gray
Write-Host "  - Press Ctrl+E (or click AutoTrading button)" -ForegroundColor Gray
Write-Host "  - Verify button turns GREEN" -ForegroundColor Gray
Write-Host ""

Write-Host "STEP 5: Stop old bot (if running)" -ForegroundColor Yellow
Write-Host "  - Find PowerShell window running demo_bot.py" -ForegroundColor Gray
Write-Host "  - Press Ctrl+C to stop it" -ForegroundColor Gray
Write-Host ""

Write-Host "STEP 6: Start new bot on VM" -ForegroundColor Yellow
Write-Host "  cd C:\Users\abylay_dos\Desktop\bolashak" -ForegroundColor White
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python live_trading\demo_bot_with_risk.py" -ForegroundColor White
Write-Host ""

Write-Host "STEP 7: Monitor the bot" -ForegroundColor Yellow
Write-Host "  - Open another PowerShell window on VM" -ForegroundColor Gray
Write-Host "  - Run: python live_trading\monitor.py" -ForegroundColor Gray
Write-Host ""

Write-Host "=== Quick File Copy Instructions ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "OPTION A: Drag and Drop via RDP" -ForegroundColor Yellow
Write-Host "  1. In RDP session, open: $vmPath" -ForegroundColor Gray
Write-Host "  2. On local PC, open: $localPath" -ForegroundColor Gray
Write-Host "  3. Drag files from local to VM window" -ForegroundColor Gray
Write-Host ""

Write-Host "OPTION B: Copy/Paste via RDP" -ForegroundColor Yellow
Write-Host "  1. Select file on local PC" -ForegroundColor Gray
Write-Host "  2. Ctrl+C to copy" -ForegroundColor Gray
Write-Host "  3. In RDP session, navigate to destination" -ForegroundColor Gray
Write-Host "  4. Ctrl+V to paste" -ForegroundColor Gray
Write-Host ""

Write-Host "Press Enter to open the local files folder..." -ForegroundColor Green
$null = Read-Host

# Open local folder
Start-Process "explorer.exe" -ArgumentList $localPath

Write-Host ""
Write-Host "Local folder opened!" -ForegroundColor Green
Write-Host "Now connect to VM and copy the files." -ForegroundColor Yellow
Write-Host ""
Write-Host "Opening RDP connection to VM..." -ForegroundColor Cyan
Start-Sleep -Seconds 2

# Open RDP
mstsc /v:20.9.129.81

Write-Host ""
Write-Host "=== DEPLOYMENT COMPLETE ===" -ForegroundColor Green
Write-Host ""
Write-Host "Don't forget to:" -ForegroundColor Yellow
Write-Host "  ✓ Enable AutoTrading (Ctrl+E in MT5)" -ForegroundColor White
Write-Host "  ✓ Verify MT5_LOT_SIZE=0.1 in .env" -ForegroundColor White
Write-Host "  ✓ Start new bot: python live_trading\demo_bot_with_risk.py" -ForegroundColor White
Write-Host ""
