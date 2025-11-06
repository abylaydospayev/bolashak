# Azure Trading Bot - Quick Start (30 Minutes)

**VM:** B1ms (1 vCPU, 2GB RAM) | **Cost:** ~$22/month | **FREE:** 4-5 months with your $100 credits

---

##  Deploy in 5 Steps

### **On Your Local PC:**

#### **Step 1: Deploy Azure VM (5 min)**

```powershell
cd C:\Users\abyla\Desktop\bolashak
.\deploy_to_azure.ps1
```

**Saves credentials to:** `azure_vm_credentials.txt`

#### **Step 2: Connect via RDP (1 min)**

```powershell
mstsc /v:<PUBLIC_IP_FROM_STEP_1>
```

Login with username/password from deployment output.

---

### **On Azure VM (via RDP):**

Open **PowerShell as Administrator** and paste these commands:

#### **Step 3: Install Software (10 min)**

```powershell
# Install Chocolatey
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Python 3.12 and Git
choco install python312 git -y

# Install MT5
Invoke-WebRequest -Uri "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe" -OutFile "C:\mt5setup.exe"
Start-Process "C:\mt5setup.exe" -Wait
Remove-Item "C:\mt5setup.exe"
```

#### **Step 4: Copy Project Files (10 min)**

**Option A - Drag & Drop:**
1. On your PC, open `C:\Users\abyla\Desktop\bolashak`
2. In RDP window, open `C:\`
3. Drag the `bolashak` folder from your PC to `C:\` on VM

**Option B - Git Clone:**
```powershell
cd C:\
git clone https://github.com/yourusername/bolashak.git
```

#### **Step 5: Setup & Start Bot (5 min)**

```powershell
cd C:\bolashak

# Create Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install MetaTrader5 python-dotenv pandas numpy scikit-learn joblib

# Configure MT5 credentials
cd live_trading
Copy-Item .env.example .env
notepad .env  
# Edit: Update MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
# Save and close

# Test connection
cd C:\bolashak
.\.venv\Scripts\python.exe live_trading\test_mt5_connection.py
# Should see:  MT5 initialized, Balance: $98,972.55

# Create auto-start script
$script = @'
Set-Location C:\bolashak
.\.venv\Scripts\Activate.ps1
while ($true) {
    Write-Host "[$(Get-Date)] Starting bot..."
    python live_trading\demo_bot.py
    Write-Host "[$(Get-Date)] Restarting in 30s..."
    Start-Sleep -Seconds 30
}
'@
$script | Out-File "C:\bolashak\start_bot.ps1" -Encoding UTF8

# Create scheduled task (auto-start on boot)
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File C:\bolashak\start_bot.ps1"
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName "TradingBot" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force

# Start the bot
Start-ScheduledTask -TaskName "TradingBot"

# Verify it's running
Get-ScheduledTask -TaskName "TradingBot"
Get-Content C:\bolashak\live_trading\logs\demo_bot.log -Tail 5
```

---

##  Done!

**You can now disconnect from RDP.** The bot runs 24/7 in the background!

---

##  Monitor Your Bot

**Reconnect via RDP anytime:**
```powershell
mstsc /v:<YOUR_VM_IP>
```

**View live logs:**
```powershell
Get-Content C:\bolashak\live_trading\logs\demo_bot.log -Tail 50 -Wait
```

**Check bot status:**
```powershell
Get-ScheduledTask -TaskName "TradingBot"  # Should say "Running"
Get-Process python  # Should show python.exe
```

**Restart bot:**
```powershell
Restart-ScheduledTask -TaskName "TradingBot"
```

---

##  Cost Tracking

**Check your Azure costs:**
1. Go to https://portal.azure.com
2. Click "Cost Management + Billing"
3. View your $100 credit balance

**Expected:** ~$0.70/day = ~$22/month

**Your $100 lasts:** ~4-5 months

---

##  Quick Troubleshooting

**Bot not running?**
```powershell
# Check logs for errors
Get-Content C:\bolashak\live_trading\logs\demo_bot.log | Select-String "ERROR"

# Restart
Start-ScheduledTask -TaskName "TradingBot"
```

**MT5 connection failed?**
```powershell
# Open MT5 and login manually once
Start-Process "C:\Program Files\MetaTrader 5\terminal64.exe"

# Then test again
.\.venv\Scripts\python.exe live_trading\test_mt5_connection.py
```

---

##  Full Documentation

- **Complete Setup Guide:** `AZURE_SETUP.md`
- **Deployment Script:** `deploy_to_azure.ps1`
- **VM Credentials:** `azure_vm_credentials.txt`

---

##  Success Metrics (After 24-48 hours)

- [ ] Bot logs show predictions every 60 seconds
- [ ] No ERROR messages in logs
- [ ] Scheduled task shows "Running" state
- [ ] Bot survives VM reboot (test with `Restart-Computer`)
- [ ] Azure costs are ~$0.70/day

---

** Your bot is now running 24/7 on Azure for FREE (using your $100 credits)!**

**Total time:** 30 minutes | **Monthly cost:** $22 | **Free duration:** 4-5 months

