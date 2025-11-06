# Deploy files to Ubuntu VM "Sabyr"
# Run this on your Windows PC

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Deploy to Azure VM 'Sabyr'" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get VM IP from user
$vmIP = Read-Host "Enter VM Public IP address"

if ([string]::IsNullOrWhiteSpace($vmIP)) {
    Write-Host "❌ No IP provided. Exiting." -ForegroundColor Red
    exit 1
}

$username = "abylay_dos"
$remotePath = "${username}@${vmIP}:~/bolashak/"

Write-Host ""
Write-Host "Target: $remotePath" -ForegroundColor Yellow
Write-Host ""
Write-Host "This will copy:" -ForegroundColor Cyan
Write-Host "  - models/" -ForegroundColor Gray
Write-Host "  - live_trading/" -ForegroundColor Gray
Write-Host "  - build_features_enhanced.py" -ForegroundColor Gray
Write-Host "  - requirements.txt" -ForegroundColor Gray
Write-Host ""

$confirm = Read-Host "Continue? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "[1/5] Copying models..." -ForegroundColor Cyan

# Copy models
scp -r models\* "${username}@${vmIP}:~/bolashak/models/"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to copy models" -ForegroundColor Red
    exit 1
}

Write-Host "[2/5] Copying live_trading..." -ForegroundColor Cyan

# Copy live_trading (excluding logs and cache)
scp live_trading\*.py "${username}@${vmIP}:~/bolashak/live_trading/"
scp live_trading\.env.example "${username}@${vmIP}:~/bolashak/live_trading/"
scp live_trading\*.md "${username}@${vmIP}:~/bolashak/live_trading/"
scp live_trading\*.txt "${username}@${vmIP}:~/bolashak/live_trading/"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to copy live_trading files" -ForegroundColor Red
    exit 1
}

Write-Host "[3/5] Copying feature engineering..." -ForegroundColor Cyan

# Copy feature engineering
scp build_features_enhanced.py "${username}@${vmIP}:~/bolashak/"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to copy build_features_enhanced.py" -ForegroundColor Red
    exit 1
}

Write-Host "[4/5] Copying setup script..." -ForegroundColor Cyan

# Copy setup script
scp setup_ubuntu.sh "${username}@${vmIP}:~/bolashak/"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to copy setup_ubuntu.sh" -ForegroundColor Red
    exit 1
}

Write-Host "[5/5] Copying documentation..." -ForegroundColor Cyan

# Copy documentation
scp UBUNTU_SETUP.md "${username}@${vmIP}:~/bolashak/"
scp README.md "${username}@${vmIP}:~/bolashak/" 2>$null  # Optional

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✅ Files Copied Successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. SSH to VM:" -ForegroundColor Yellow
Write-Host "   ssh $username@$vmIP" -ForegroundColor White
Write-Host ""
Write-Host "2. Configure .env file:" -ForegroundColor Yellow
Write-Host "   nano ~/bolashak/live_trading/.env" -ForegroundColor White
Write-Host ""
Write-Host "   Add your Oanda API token and Account ID" -ForegroundColor Gray
Write-Host "   Get them from: https://www.oanda.com/demo-account/login" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Test connection:" -ForegroundColor Yellow
Write-Host "   cd ~/bolashak/live_trading" -ForegroundColor White
Write-Host "   source ../.venv/bin/activate" -ForegroundColor White
Write-Host "   python3 oanda_client.py" -ForegroundColor White
Write-Host ""
Write-Host "4. Start bot:" -ForegroundColor Yellow
Write-Host "   python3 ubuntu_bot.py" -ForegroundColor White
Write-Host ""
Write-Host "See UBUNTU_SETUP.md for detailed instructions" -ForegroundColor Gray
Write-Host ""
