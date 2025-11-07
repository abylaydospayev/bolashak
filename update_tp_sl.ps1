# Update bot settings to optimized 20/30 TP/SL
# Run this on the VM after pulling latest code

Write-Host "Updating bot to optimized 20 SL / 30 TP settings..." -ForegroundColor Cyan
Write-Host ""

$envFile = "C:\Users\abylay_dos\Desktop\bolashak\live_trading\.env"

if (Test-Path $envFile) {
    # Read current content
    $content = Get-Content $envFile
    
    # Update SL and TP
    $content = $content -replace 'STOP_LOSS_PIPS=30', 'STOP_LOSS_PIPS=20'
    $content = $content -replace 'TAKE_PROFIT_PIPS=50', 'TAKE_PROFIT_PIPS=30'
    
    # Save
    $content | Set-Content $envFile
    
    Write-Host "Updated .env file:" -ForegroundColor Green
    Write-Host "  STOP_LOSS_PIPS=20" -ForegroundColor White
    Write-Host "  TAKE_PROFIT_PIPS=30" -ForegroundColor White
    Write-Host ""
    Write-Host "Monte Carlo Results:" -ForegroundColor Yellow
    Write-Host "  Win Rate: 76.6%" -ForegroundColor White
    Write-Host "  Expectancy: $66.68/trade" -ForegroundColor White
    Write-Host "  Profit Factor: 5.71" -ForegroundColor White
    Write-Host "  Max Drawdown: 0.24%" -ForegroundColor White
    Write-Host ""
    Write-Host "Restart the bot to apply changes:" -ForegroundColor Cyan
    Write-Host "  .\stop_bot.ps1" -ForegroundColor White
    Write-Host "  .\start_bot.ps1" -ForegroundColor White
} else {
    Write-Host "ERROR: .env file not found at $envFile" -ForegroundColor Red
}
