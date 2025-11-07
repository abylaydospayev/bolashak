# Check which PowerShell window is running the bot

Write-Host "Checking for running bot processes..." -ForegroundColor Cyan
Write-Host ""

$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue

if ($pythonProcesses) {
    Write-Host "Python processes running:" -ForegroundColor Green
    $pythonProcesses | Format-Table Id, CPU, WorkingSet, StartTime -AutoSize
    
    Write-Host ""
    Write-Host "PowerShell windows:" -ForegroundColor Yellow
    Get-Process powershell | Select-Object Id, MainWindowTitle | Format-Table -AutoSize
    
    Write-Host ""
    Write-Host "To view bot logs: .\view_logs.ps1" -ForegroundColor Cyan
    Write-Host "To stop bot: .\stop_bot.ps1" -ForegroundColor Yellow
} else {
    Write-Host "No Python processes running - bot is not active" -ForegroundColor Red
    Write-Host ""
    Write-Host "To start bot: .\start_bot.ps1" -ForegroundColor Cyan
}

Write-Host ""
