# View trading bot logs in real-time

$today = Get-Date -Format "yyyyMMdd"
$logFile = "live_trading\logs\demo_bot_$today.log"

if (Test-Path $logFile) {
    Write-Host "Viewing bot logs (Ctrl+C to exit)..." -ForegroundColor Cyan
    Write-Host "Log file: $logFile" -ForegroundColor Yellow
    Write-Host ""
    Get-Content $logFile -Wait -Tail 50
} else {
    Write-Host "Log file not found: $logFile" -ForegroundColor Red
    Write-Host ""
    Write-Host "Looking for other log files..." -ForegroundColor Yellow
    $allLogs = Get-ChildItem live_trading\logs\*.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($allLogs) {
        $latestLog = $allLogs[0]
        Write-Host "Found: $($latestLog.Name)" -ForegroundColor Green
        Write-Host ""
        Get-Content $latestLog.FullName -Wait -Tail 50
    } else {
        Write-Host "No log files found. Make sure the bot is running with: .\start_bot.ps1" -ForegroundColor Red
    }
}
