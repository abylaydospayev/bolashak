# Quick Setup Script for MT5 Demo Bot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MT5 DEMO BOT SETUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if .env exists
$envFile = "live_trading\.env"
if (-Not (Test-Path $envFile)) {
    Write-Host "Step 1: Creating .env file..." -ForegroundColor Yellow
    Copy-Item "live_trading\.env.example" $envFile
    Write-Host "✅ Created $envFile" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Edit live_trading\.env with your MT5 credentials!" -ForegroundColor Red
    Write-Host "   Required fields:" -ForegroundColor Yellow
    Write-Host "   - MT5_TERMINAL_PATH (path to terminal64.exe)" -ForegroundColor Yellow
    Write-Host "   - MT5_LOGIN (your account number)" -ForegroundColor Yellow
    Write-Host "   - MT5_PASSWORD (your account password)" -ForegroundColor Yellow
    Write-Host "   - MT5_SERVER (broker server name)" -ForegroundColor Yellow
    Write-Host ""
    
    # Open .env in notepad for editing
    $response = Read-Host "Open .env in notepad now? (Y/n)"
    if ($response -eq "" -or $response -eq "Y" -or $response -eq "y") {
        notepad $envFile
    }
} else {
    Write-Host "✅ .env file already exists" -ForegroundColor Green
}

Write-Host ""

# Step 2: Test connection
Write-Host "Step 2: Testing MT5 connection..." -ForegroundColor Yellow
Write-Host "Make sure:" -ForegroundColor Cyan
Write-Host "  1. MT5 terminal is installed" -ForegroundColor Cyan
Write-Host "  2. You've edited .env with correct credentials" -ForegroundColor Cyan
Write-Host "  3. MT5 terminal is running (or terminal path is correct)" -ForegroundColor Cyan
Write-Host ""

$response = Read-Host "Run connection test now? (Y/n)"
if ($response -eq "" -or $response -eq "Y" -or $response -eq "y") {
    Write-Host ""
    & .\.venv\Scripts\python.exe live_trading\test_mt5_connection.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "✅ CONNECTION SUCCESSFUL!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now run the demo bot:" -ForegroundColor Cyan
        Write-Host "  .\.venv\Scripts\python.exe live_trading\demo_bot.py" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Or monitor live:" -ForegroundColor Cyan
        Write-Host "  .\.venv\Scripts\python.exe live_trading\monitor.py" -ForegroundColor Yellow
    } else {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Red
        Write-Host "❌ CONNECTION FAILED" -ForegroundColor Red
        Write-Host "========================================" -ForegroundColor Red
        Write-Host ""
        Write-Host "Troubleshooting:" -ForegroundColor Yellow
        Write-Host "  1. Check .env file has correct credentials" -ForegroundColor Cyan
        Write-Host "  2. Ensure MT5 terminal is installed" -ForegroundColor Cyan
        Write-Host "  3. Try launching MT5 terminal manually first" -ForegroundColor Cyan
        Write-Host "  4. Verify login/password/server are correct" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "See live_trading\README_MT5.md for more help" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
