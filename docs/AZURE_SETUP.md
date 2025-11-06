# Azure VM Setup for Trading Bot (B1ms)

##  Cost: ~$22/month = 4-5 months FREE with your $100 credits

---

## Quick Deploy Overview

**Total Time:** 30-45 minutes from start to running bot  
**VM Spec:** B1ms (1 vCPU, 2GB RAM) - Perfect for MT5 + Python bot  
**OS:** Windows Server 2022  
**Monthly Cost:** ~$22 (FREE for 4-5 months with credits)

---

## Step 1: Deploy Azure VM (On Your PC)

### Prerequisites
- Azure account with $100 student credits
- Azure CLI installed (script will prompt if missing)

### Deploy

```powershell
cd C:\Users\abyla\Desktop\bolashak
.\deploy_to_azure.ps1
```

**What it does:**
-  Logs you into Azure
-  Creates resource group `TradingBotRG`
-  Creates Windows Server 2022 VM (B1ms)
-  Configures networking and RDP access
-  Generates secure credentials
-  Saves connection info to `azure_vm_credentials.txt`

**Time:** ~5 minutes

**Output:**
```
 DEPLOYMENT SUCCESSFUL!
Public IP: 20.12.34.56
Username: azureuser
Password: Abc123XyzABC!@#$
```

 **SAVE THESE CREDENTIALS!** They're also in `azure_vm_credentials.txt`

---

## Step 2: Connect to Azure VM

### Option A: Command Line
```powershell
mstsc /v:20.12.34.56  # Use your actual IP from deployment
```

### Option B: GUI
1. Press `Win + R`
2. Type `mstsc` and press Enter
3. Paste your VM's public IP
4. Click "Connect"
5. Login with credentials from Step 1

**First time:** Windows will warn about certificate. Click "Yes" to connect.

---

## Step 3: Setup VM Software (On Azure VM)

### 3.1 Install Chocolatey (Package Manager)

Open **PowerShell as Administrator** on the VM:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

**Time:** 30 seconds

### 3.2 Install Python and Git

```powershell
# Install both at once
choco install python312 git -y

# Refresh environment
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Verify
python --version  # Should show Python 3.12.x
git --version     # Should show git version
```

**Time:** 2-3 minutes

### 3.3 Install MetaTrader 5

```powershell
# Download MT5 installer
Invoke-WebRequest -Uri "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe" -OutFile "C:\mt5setup.exe"

# Run installer
Start-Process "C:\mt5setup.exe" -Wait

# Clean up
Remove-Item "C:\mt5setup.exe"
```

**Manual steps in MT5 installer:**
1. Click "Next" through the wizard
2. Accept license agreement
3. Install to default location: `C:\Program Files\MetaTrader 5\`
4. Uncheck "Run MetaTrader 5" (we'll configure it first)
5. Click "Finish"

**Time:** 3-5 minutes

---

## Step 4: Transfer Your Project to VM

### Option A: Git Clone (If you have a repo)

```powershell
cd C:\
git clone https://github.com/yourusername/bolashak.git
cd bolashak
```

### Option B: Manual Copy via RDP (Recommended)

**On your local PC:**
1. Open File Explorer to `C:\Users\abyla\Desktop\bolashak`
2. In the RDP window to Azure VM, open File Explorer
3. Drag and drop the entire `bolashak` folder from your PC to `C:\` on the VM

**This copies everything:**
- Python scripts
- Models
- Configuration files
- Everything!

**Time:** 5-10 minutes (depends on internet speed)

---

## Step 5: Setup Python Environment (On Azure VM)

```powershell
cd C:\bolashak

# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install MetaTrader5 python-dotenv pandas numpy scikit-learn joblib

# Verify installations
python -c "import MetaTrader5 as mt5; print(f'MT5 version: {mt5.__version__}')"
```

**Expected output:**
```
Successfully installed MetaTrader5-5.0.xxx...
MT5 version: 5.0.xxx
```

**Time:** 2-3 minutes

---

## Step 6: Configure MT5 Credentials

### 6.1 Create .env File

```powershell
cd C:\bolashak\live_trading

# Copy template
Copy-Item .env.example .env

# Edit with notepad
notepad .env
```

### 6.2 Update Credentials in .env

**Change these values:**

```ini
MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_LOGIN=1600037272
MT5_PASSWORD=8?p1?$$*kW3
MT5_SERVER=OANDA-Demo-1
MT5_SYMBOL=USDJPY.sim
```

**Important:**
-  Keep the path as shown (standard MT5 install location)
-  Use your actual MT5 login/password
-  Symbol must be `USDJPY.sim` for OANDA demo

**Save and close notepad.**

**Time:** 1 minute

---

## Step 7: Test MT5 Connection

```powershell
cd C:\bolashak
.\.venv\Scripts\python.exe live_trading\test_mt5_connection.py
```

**Expected output:**
```
 MT5 initialized
Account #1600037272
Balance: $98,972.55
Server: OANDA-Demo-1

Fetching 100 M15 bars for USDJPY.sim...
 Retrieved 100 bars
Latest price: 153.850

 MT5 shutdown
```

**If you see errors:**

**Error: "MT5 initialize() failed"**
- Solution: Open MT5 manually once and login to save credentials
- Run: `Start-Process "C:\Program Files\MetaTrader 5\terminal64.exe"`
- Login with your credentials
- Close MT5
- Try test again

**Error: "Symbol USDJPY.sim not found"**
- Solution: Check available symbols
- Run: `.\.venv\Scripts\python.exe live_trading\list_symbols.py`
- Find the correct USDJPY symbol name
- Update `.env` file

**Time:** 2 minutes (or 10 if troubleshooting)

---

## Step 8: Configure Auto-Start

### 8.1 Create Auto-Restart Script

```powershell
$StartScript = @'
Set-Location C:\bolashak
.\.venv\Scripts\Activate.ps1

# Log file for startup script
$logFile = "C:\bolashak\live_trading\logs\startup.log"

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content $logFile "[$timestamp] Starting trading bot..."
    Write-Host "[$timestamp] Starting USDJPY trading bot..."
    
    try {
        python live_trading\demo_bot.py
        $exitCode = $LASTEXITCODE
        Add-Content $logFile "[$timestamp] Bot exited with code: $exitCode"
    }
    catch {
        $errorMsg = $_.Exception.Message
        Add-Content $logFile "[$timestamp] Error: $errorMsg"
        Write-Host "[$timestamp] Error: $errorMsg" -ForegroundColor Red
    }
    
    Add-Content $logFile "[$timestamp] Bot stopped. Restarting in 30 seconds..."
    Write-Host "[$timestamp] Bot stopped. Restarting in 30 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
}
'@

# Save script
$StartScript | Out-File "C:\bolashak\start_bot.ps1" -Encoding UTF8

Write-Host " Auto-restart script created: C:\bolashak\start_bot.ps1" -ForegroundColor Green
```

### 8.2 Create Scheduled Task (Auto-start on VM boot)

```powershell
# Define task action
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File C:\bolashak\start_bot.ps1" `
    -WorkingDirectory "C:\bolashak"

# Trigger: Run at system startup
$trigger = New-ScheduledTaskTrigger -AtStartup

# Run as SYSTEM account (runs even when no user logged in)
$principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

# Settings: Auto-restart on failure
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# Register the task
Register-ScheduledTask `
    -TaskName "TradingBot" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "USDJPY automated trading bot - runs 24/7 with auto-restart" `
    -Force

Write-Host " Scheduled task 'TradingBot' created successfully" -ForegroundColor Green
```

### 8.3 Verify Scheduled Task

```powershell
# Check task exists
Get-ScheduledTask -TaskName "TradingBot"

# Expected output:
# TaskName: TradingBot
# State: Ready
```

**Time:** 2 minutes

---

## Step 9: Start the Bot

### Manual Start (First Time Test)

```powershell
cd C:\bolashak
.\.venv\Scripts\python.exe live_trading\demo_bot.py
```

**Expected output:**
```
[2025-11-05 21:45:00] [INFO] ================================================================================
[2025-11-05 21:45:00] [INFO] USDJPY DEMO BOT STARTING
[2025-11-05 21:45:00] [INFO] ================================================================================
[2025-11-05 21:45:01] [INFO]  Loaded model: models\USDJPY_ensemble_oos.pkl
[2025-11-05 21:45:01] [INFO] Connecting to MT5: OANDA-Demo-1
 MT5 initialized
[2025-11-05 21:45:02] [INFO] Account Balance: $98,972.55
[2025-11-05 21:45:02] [INFO] Account Equity: $98,972.55
[2025-11-05 21:45:02] [INFO] Starting trading loop...
[2025-11-05 21:45:02] [INFO] Risk per trade: 1.0%
[2025-11-05 21:45:02] [INFO] Max positions: 3
[2025-11-05 21:45:02] [INFO] Check interval: 60s
[2025-11-05 21:45:03] [INFO] [1] Price: 153.850, Prob: 0.273, H1 Trend: 1, Positions: 0
```

 **Success!** The bot is running!

**To stop:** Press `Ctrl+C`

### Enable Background Running

Now that you've confirmed it works, let it run in background via scheduled task:

```powershell
# Start the scheduled task
Start-ScheduledTask -TaskName "TradingBot"

# Verify it's running
Get-ScheduledTask -TaskName "TradingBot" | Select-Object State
# State should be "Running"
```

**Now you can disconnect from RDP** - the bot keeps running! 

---

## Step 10: Verify Auto-Start on Reboot

Test that the bot survives a VM reboot:

```powershell
# Reboot the VM
Restart-Computer -Force
```

**Wait 2-3 minutes**, then reconnect via RDP.

### Check Bot Status

```powershell
# Is the scheduled task running?
Get-ScheduledTask -TaskName "TradingBot" | Select-Object TaskName, State

# Is Python process running?
Get-Process python -ErrorAction SilentlyContinue

# View recent logs
Get-Content C:\bolashak\live_trading\logs\demo_bot.log -Tail 20
```

**Expected:**
- Task State: Running
- Python process: Found
- Logs showing recent predictions (within last 2 minutes)

 **If all checks pass, your bot is fully configured for 24/7 operation!**

---

## Monitoring & Maintenance

### View Live Logs

Connect via RDP and run:

```powershell
# Tail logs (live updates)
Get-Content C:\bolashak\live_trading\logs\demo_bot.log -Tail 50 -Wait

# Search for trades
Get-Content C:\bolashak\live_trading\logs\demo_bot.log | Select-String "TRADE"

# Check for errors
Get-Content C:\bolashak\live_trading\logs\demo_bot.log | Select-String "ERROR"
```

### Restart Bot

```powershell
# Stop
Stop-ScheduledTask -TaskName "TradingBot"

# Wait a few seconds
Start-Sleep -Seconds 5

# Start
Start-ScheduledTask -TaskName "TradingBot"
```

### Check Azure Costs

**On your local PC:**

```powershell
# Login
az login

# View current month costs
az consumption usage list --output table --query "[?contains(instanceName, 'TradingBot')]"

# Or check in Azure Portal
# https://portal.azure.com  Cost Management + Billing
```

---

## Cost Optimization

### Current Setup Cost Breakdown

| Item | Monthly Cost |
|------|--------------|
| B1ms VM (1 vCPU, 2GB) | $15.04 |
| Standard LRS Storage (64GB) | $2.56 |
| Standard Public IP | $3.00 |
| Bandwidth (<10GB) | $0.87 |
| **Total** | **~$21.47** |

**Your $100 credits last:** ~4.6 months (139 days)

### Save More Money

**1. Auto-shutdown on weekends** (market closed):

```powershell
az vm auto-shutdown `
    --resource-group TradingBotRG `
    --name TradingBotVM `
    --time 2100 `
    --email your@email.com
```

Saves ~$5/month (weekends off)

**2. Manual stop when not needed:**

```powershell
# Stop VM (saves ~70% cost, still pay for storage)
az vm deallocate -g TradingBotRG -n TradingBotVM

# Start when needed
az vm start -g TradingBotRG -n TradingBotVM
```

---

## Troubleshooting

### Bot Not Starting After Reboot

```powershell
# Check task history
Get-WinEvent -LogName "Microsoft-Windows-TaskScheduler/Operational" -MaxEvents 50 |
    Where-Object {$_.Message -like "*TradingBot*"} |
    Format-Table TimeCreated, Message -AutoSize

# Manually trigger task
Start-ScheduledTask -TaskName "TradingBot"

# Check startup log
Get-Content C:\bolashak\live_trading\logs\startup.log -Tail 20
```

### MT5 Connection Lost

```powershell
# Restart MT5 terminal process
Get-Process MetaTrader | Stop-Process
Start-Sleep -Seconds 3
Start-Process "C:\Program Files\MetaTrader 5\terminal64.exe"

# Test connection
.\.venv\Scripts\python.exe live_trading\test_mt5_connection.py
```

### High Memory Usage (B1ms only has 2GB)

```powershell
# Check memory
Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 10 Name, @{Name="RAM(MB)";Expression={$_.WorkingSet / 1MB}}

# If memory is full, restart bot
Restart-ScheduledTask -TaskName "TradingBot"
```

If consistently running out of memory, upgrade to B2s:

```powershell
az vm resize -g TradingBotRG -n TradingBotVM --size Standard_B2s
# Cost increases to ~$30/month
```

### VM Won't Connect

```powershell
# Check VM status
az vm get-instance-view -g TradingBotRG -n TradingBotVM --query instanceView.statuses

# Restart VM
az vm restart -g TradingBotRG -n TradingBotVM

# Check if your IP changed and RDP port is blocked
# Azure Portal  TradingBotVM  Networking  Add inbound rule for RDP from your IP
```

---

## Security Checklist

- [ ] Changed default RDP password (from deployment)
- [ ] Restricted RDP access to your IP only (Azure Portal  Networking)
- [ ] `.env` file contains real credentials (not in Git)
- [ ] Windows Firewall enabled on VM
- [ ] Scheduled task running as SYSTEM (prevents accidental stop)
- [ ] Auto-shutdown configured (optional cost savings)
- [ ] Credentials saved in safe location (password manager)

---

## Success Checklist

After 24-48 hours, verify:

- [ ] Bot is still running (`Get-Process python`)
- [ ] Logs show predictions every 60 seconds
- [ ] No error messages in logs
- [ ] MT5 connection stable
- [ ] VM auto-starts bot after reboot (tested)
- [ ] Azure costs are ~$0.70/day (check portal)
- [ ] Can disconnect RDP and bot keeps running

---

## Cleanup (When Done Testing)

**To delete everything and stop charges:**

```powershell
# Delete entire resource group (VM, storage, IP, everything)
az group delete --name TradingBotRG --yes --no-wait

# This stops ALL charges immediately
```

**To keep data but stop charges temporarily:**

```powershell
# Deallocate VM (saves ~70%, keeps disk)
az vm deallocate -g TradingBotRG -n TradingBotVM

# Restart when needed
az vm start -g TradingBotRG -n TradingBotVM
```

---

## Next Steps

1. **Monitor for 4-8 weeks** on demo account
2. **Compare live results to backtest**
   - Win rate: Should be 75-80% (backtest: 79.9%)
   - Max drawdown: Should be <10% (backtest: 7.7%)
   - Profit: Should match within 5%
3. **If performance matches backtest**, consider going live with real capital
4. **If performance diverges >10%**, investigate and debug before live trading

---

** Congratulations! Your trading bot is now running 24/7 on Azure!**

**Monthly cost:** $22 (FREE for 4-5 months with your $100 credits)

**Expected profit:** Based on backtest, $1,000/month with $100k capital (1% monthly ROI)

For questions or issues, check the troubleshooting section or logs at `C:\bolashak\live_trading\logs\`

