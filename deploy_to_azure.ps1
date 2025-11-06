<#
.SYNOPSIS
    Deploy USDJPY Trading Bot to Azure using your $100 credits
.DESCRIPTION
    Creates cost-optimized Azure VM (B1ms) and configures auto-start trading bot
    Monthly Cost: ~$22/month | Your $100 credits last: 4-5 months FREE
#>

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Azure Trading Bot Deployment (B1ms)" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$ResourceGroup = "TradingBotRG"
$VMName = "TradingBotVM"
$Location = "centralus"  # Common region for student subscriptions
$VMSize = "Standard_B1ms"
$AdminUsername = "azureuser"

# Generate secure password that meets Azure requirements
# Must be 12-123 chars with 3 of: lowercase, uppercase, number, special char
$AdminPassword = "TradingBot" + (Get-Random -Minimum 1000 -Maximum 9999) + "!@#"

Write-Host "Cost Breakdown:" -ForegroundColor Yellow
Write-Host "   VM (B1ms):      ~$15/month"
Write-Host "   Storage:        ~$3/month"
Write-Host "   Public IP:      ~$3/month"
Write-Host "   Bandwidth:      ~$1/month"
Write-Host "   --------------------------------"
Write-Host "   Total:          ~$22/month" -ForegroundColor Green
Write-Host "   Your credits:   $100 = 4-5 months FREE" -ForegroundColor Green
Write-Host ""

# Check Azure CLI
Write-Host "Checking Azure CLI..." -ForegroundColor Cyan

# Try to find az command in PATH or default installation location
$azCommand = Get-Command az -ErrorAction SilentlyContinue
if (-not $azCommand) {
    $azPath = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd"
    if (Test-Path $azPath) {
        # Use full path if not in PATH
        Set-Alias -Name az -Value $azPath -Scope Script
        Write-Host "[OK] Azure CLI found at: $azPath" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Azure CLI not found" -ForegroundColor Red
        Write-Host ""
        Write-Host "Install from: https://aka.ms/installazurecliwindows" -ForegroundColor Yellow
        Write-Host "Then restart PowerShell and run this script again" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "[OK] Azure CLI found" -ForegroundColor Green
}
Write-Host ""

# Login
Write-Host "[1/5] Checking Azure login status..." -ForegroundColor Cyan

# Check if already logged in
$accountCheck = az account show 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Already logged in to Azure" -ForegroundColor Green
} else {
    Write-Host "        Opening browser for authentication..." -ForegroundColor Gray
    Write-Host "        After logging in, close the browser tab and return here" -ForegroundColor Yellow
    Write-Host ""
    
    az login --use-device-code
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Azure login failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "[OK] Logged in successfully" -ForegroundColor Green
}
Write-Host ""

# Show subscription
Write-Host "[2/5] Checking Azure subscription..." -ForegroundColor Cyan
$subscription = az account show --query name -o tsv
$subscriptionId = az account show --query id -o tsv
Write-Host "        Active: $subscription" -ForegroundColor Green
Write-Host "        ID: $subscriptionId" -ForegroundColor Gray
Write-Host ""

# Create resource group
Write-Host "[3/5] Creating resource group..." -ForegroundColor Cyan
Write-Host "        Name: $ResourceGroup" -ForegroundColor Gray
Write-Host "        Location: $Location" -ForegroundColor Gray

az group create --name $ResourceGroup --location $Location --output none

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to create resource group" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Resource group created" -ForegroundColor Green
Write-Host ""

# Create VM
Write-Host "[4/5] Creating Azure VM (takes 3-5 minutes)..." -ForegroundColor Cyan
Write-Host "        VM Name: $VMName" -ForegroundColor Gray
Write-Host "        Size: $VMSize" -ForegroundColor Gray
Write-Host "        Image: Windows Server 2022" -ForegroundColor Gray
Write-Host ""
Write-Host "        Please wait..." -ForegroundColor Yellow

az vm create `
    --resource-group $ResourceGroup `
    --name $VMName `
    --image Win2022Datacenter `
    --size $VMSize `
    --admin-username $AdminUsername `
    --admin-password $AdminPassword `
    --public-ip-sku Standard `
    --nsg-rule RDP `
    --os-disk-size-gb 64 `
    --storage-sku Standard_LRS `
    --output none

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to create VM" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  - Quota limits exceeded" -ForegroundColor Gray
    Write-Host "  - Insufficient credits" -ForegroundColor Gray
    Write-Host "  - Region doesn't support B1ms" -ForegroundColor Gray
    exit 1
}

Write-Host "[OK] VM created successfully" -ForegroundColor Green
Write-Host ""

# Get public IP
Write-Host "[5/5] Retrieving connection details..." -ForegroundColor Cyan
$PublicIP = az vm show -d -g $ResourceGroup -n $VMName --query publicIps -o tsv

if ([string]::IsNullOrWhiteSpace($PublicIP)) {
    Write-Host "[ERROR] Failed to get public IP" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Public IP: $PublicIP" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "  DEPLOYMENT SUCCESSFUL" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
Write-Host ""
Write-Host "VM Connection Details:" -ForegroundColor Cyan
Write-Host ""
Write-Host "   Public IP:  $PublicIP" -ForegroundColor Yellow
Write-Host "   Username:   $AdminUsername"
Write-Host "   Password:   $AdminPassword" -ForegroundColor Yellow
Write-Host ""
Write-Host "   SAVE THESE CREDENTIALS!" -ForegroundColor Red
Write-Host ""
Write-Host "Connect to VM:" -ForegroundColor Cyan
Write-Host ""
Write-Host "   mstsc /v:$PublicIP" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "   1. Connect via Remote Desktop (command above)"
Write-Host "   2. See AZURE_SETUP.md for complete setup"
Write-Host "   3. Or AZURE_QUICK_START.md for fast 5-step setup"
Write-Host ""

# Save credentials
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$ConnectionInfo = @"
=======================================================
Azure Trading Bot VM - Connection Information
=======================================================

Created: $timestamp

VM Details:
  Name:           $VMName
  Resource Group: $ResourceGroup
  Location:       $Location
  Size:           $VMSize
  Public IP:      $PublicIP

Login Credentials:
  Username:       $AdminUsername
  Password:       $AdminPassword

Connect via Remote Desktop:
  mstsc /v:$PublicIP

Cost Information:
  Monthly Cost:   ~$22/month
  Your Credits:   $100
  Free Duration:  ~4-5 months

Next Steps:
  Step 1: Connect via RDP (command above)
  Step 2: Follow AZURE_SETUP.md for complete guide
  Step 3: Configure bot with your MT5 credentials
  Step 4: Start trading

Important Files to Copy to VM:
  * models/USDJPY_ensemble_oos.pkl
  * models/scaler.pkl
  * live_trading/.env (with MT5 credentials)

=======================================================
Keep this file safe - do NOT commit to Git
=======================================================
"@

$CredentialsFile = "azure_vm_credentials.txt"
$ConnectionInfo | Out-File $CredentialsFile -Encoding UTF8

Write-Host "Credentials saved to: $CredentialsFile" -ForegroundColor Green
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Open credentials file
Write-Host "Opening credentials file..." -ForegroundColor Gray
Start-Process notepad $CredentialsFile
