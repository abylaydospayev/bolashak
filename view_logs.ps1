# View trading bot logs in real-time

$logFile = "C:\Users\abylay_dos\Desktop\bolashak\trading.log"

if (Test-Path $logFile) {
    Write-Host "Viewing bot logs (Ctrl+C to exit)..." -ForegroundColor Cyan
    Write-Host ""
    Get-Content $logFile -Wait -Tail 50
} else {
    Write-Host "Log file not found: $logFile" -ForegroundColor Red
    Write-Host "Make sure the bot is running with: .\start_bot.ps1" -ForegroundColor Yellow
}
