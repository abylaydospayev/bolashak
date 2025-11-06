# Setup Script for Windows VM - FTMO Trading Bot
# Run this INSIDE the Windows VM after connecting via RDP

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  FTMO Trading Bot - Windows VM Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[ERROR] Please run as Administrator" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/6] Installing Chocolatey package manager..." -ForegroundColor Cyan
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

Write-Host "[2/6] Installing Python 3.12..." -ForegroundColor Cyan
choco install python312 -y
refreshenv

Write-Host "[3/6] Installing Git..." -ForegroundColor Cyan
choco install git -y
refreshenv

Write-Host "[4/6] Downloading MetaTrader 5..." -ForegroundColor Cyan
$mt5Url = "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"
$mt5Path = "C:\mt5setup.exe"
Invoke-WebRequest -Uri $mt5Url -OutFile $mt5Path

Write-Host "[5/6] Installing MetaTrader 5..." -ForegroundColor Yellow
Write-Host "   Please complete the MT5 installation wizard" -ForegroundColor Gray
Start-Process $mt5Path -Wait

Write-Host "[6/6] Creating project directory..." -ForegroundColor Cyan
New-Item -Path "C:\bolashak" -ItemType Directory -Force
cd C:\bolashak

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Copy project files from your PC:" -ForegroundColor Yellow
Write-Host "   - Open RDP window" -ForegroundColor Gray
Write-Host "   - On your PC, navigate to: C:\Users\abyla\Desktop\bolashak" -ForegroundColor Gray
Write-Host "   - Drag and drop the entire folder into the RDP window" -ForegroundColor Gray
Write-Host "   - Files will copy to C:\bolashak on the VM" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Setup Python environment:" -ForegroundColor Yellow
Write-Host "   cd C:\bolashak" -ForegroundColor White
Write-Host "   python -m venv .venv" -ForegroundColor White
Write-Host "   .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "   pip install MetaTrader5 python-dotenv pandas numpy scikit-learn joblib" -ForegroundColor White
Write-Host ""
Write-Host "3. Configure FTMO credentials:" -ForegroundColor Yellow
Write-Host "   cd C:\bolashak\live_trading" -ForegroundColor White
Write-Host "   Copy-Item .env.example .env" -ForegroundColor White
Write-Host "   notepad .env" -ForegroundColor White
Write-Host "   # Add your FTMO MT5 credentials" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Test connection:" -ForegroundColor Yellow
Write-Host "   .\.venv\Scripts\python.exe live_trading\test_mt5_connection.py" -ForegroundColor White
Write-Host ""
Write-Host "5. Start trading bot:" -ForegroundColor Yellow
Write-Host "   .\.venv\Scripts\python.exe live_trading\demo_bot.py" -ForegroundColor White
Write-Host ""
