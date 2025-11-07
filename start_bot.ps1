# Start the USDJPY trading bot with auto-restart and logging
# This script runs the bot in a new window and logs to trading.log

Write-Host "Starting USDJPY Trading Bot..." -ForegroundColor Green

$scriptBlock = @"
cd C:\Users\abylay_dos\Desktop\bolashak
.\.venv\Scripts\Activate.ps1
Write-Host "Bot starting with auto-restart..." -ForegroundColor Cyan
while (`$true) {
    try {
        python live_trading\demo_bot_with_risk.py 2>&1 | Tee-Object -FilePath 'trading.log' -Append
    }
    catch {
        Write-Host "Error: `$_" -ForegroundColor Red
    }
    Write-Host "Bot stopped. Restarting in 30 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
}
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $scriptBlock

Write-Host ""
Write-Host "Bot started in new window!" -ForegroundColor Green
Write-Host "To view logs: .\view_logs.ps1" -ForegroundColor Cyan
Write-Host "To stop bot: .\stop_bot.ps1" -ForegroundColor Yellow
Write-Host ""
