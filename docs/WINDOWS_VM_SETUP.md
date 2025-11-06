# Windows VM FTMO Setup - Quick Guide

## VM Details
- **Name**: sabyrbolashak
- **IP**: 20.9.129.81
- **OS**: Windows Server 2022
- **Username**: abylay_dos
- **Cost**: ~$30-40/month (runs 24/7 for FTMO)

---

## Step 1: Connect to VM (1 minute)

On your **Windows PC**:

```powershell
mstsc /v:20.9.129.81
```

- **Username**: `abylay_dos`
- **Password**: (your VM password)

---

## Step 2: Initial Setup (10 minutes)

**Inside the VM** (via RDP), open PowerShell as Administrator:

1. Right-click PowerShell  **Run as Administrator**

2. Copy the setup script from your PC:
   - On your PC: Open `C:\Users\abyla\Desktop\bolashak\setup_windows_vm.ps1`
   - Copy all content (Ctrl+A, Ctrl+C)
   - In VM PowerShell: Paste and press Enter

   OR download directly:
   ```powershell
   # Download setup script
   Invoke-WebRequest -Uri "https://raw.githubusercontent.com/your-repo/bolashak/main/setup_windows_vm.ps1" -OutFile "C:\setup_vm.ps1"
   
   # Run it
   C:\setup_vm.ps1
   ```

This installs:
-  Chocolatey
-  Python 3.12
-  Git
-  MetaTrader 5

---

## Step 3: Copy Project Files (5 minutes)

### Option A: Drag & Drop (Easiest)

1. On your **PC**, open: `C:\Users\abyla\Desktop\bolashak`
2. Select ALL files and folders
3. Drag them into the **RDP window**
4. Drop into `C:\` drive on the VM
5. Files will copy to `C:\bolashak`

### Option B: Git Clone (If you have a repo)

```powershell
cd C:\
git clone https://github.com/your-username/bolashak.git
```

---

## Step 4: Setup Python Environment (3 minutes)

**In VM PowerShell**:

```powershell
cd C:\bolashak

# Create virtual environment
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Install packages
pip install MetaTrader5 python-dotenv pandas numpy scikit-learn joblib
```

---

## Step 5: Configure FTMO Credentials (2 minutes)

```powershell
cd C:\bolashak\live_trading

# Create .env from example
Copy-Item .env.example .env

# Edit .env
notepad .env
```

**Update these lines**:
```ini
# FTMO MT5 Credentials
MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_LOGIN=your_ftmo_account_number
MT5_PASSWORD=your_ftmo_password
MT5_SERVER=FTMO-Server-1  # Check FTMO email for exact server name
MT5_SYMBOL=USDJPY
```

**Save** (Ctrl+S) and **close** Notepad.

---

## Step 6: Test Connection (1 minute)

```powershell
cd C:\bolashak
.\.venv\Scripts\python.exe live_trading\test_mt5_connection.py
```

**Expected output**:
```
 MT5 initialized
Account balance: $100,000
 USDJPY bars fetched (100 bars)
```

If you see errors:
- Check MT5 is installed at `C:\Program Files\MetaTrader 5\terminal64.exe`
- Verify FTMO credentials in `.env`
- Make sure MT5 terminal is closed (bot needs exclusive access)

---

## Step 7: Start Bot Manually (Test First)

```powershell
cd C:\bolashak
.\.venv\Scripts\python.exe live_trading\demo_bot.py
```

**You should see**:
```
================================================================================
USDJPY TRADING BOT STARTED
================================================================================
 MT5 initialized successfully
 Account Balance: $100,000.00
[1] Price: 153.850 | BuyProb: 0.241 | Positions: 0
```

Let it run for **5-10 minutes** to ensure no errors.

Press **Ctrl+C** to stop.

---

## Step 8: Setup Auto-Start on Boot (5 minutes)

This ensures bot runs 24/7, even after VM reboots.

**In VM PowerShell (as Administrator)**:

```powershell
# Create scheduled task
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File C:\bolashak\start_ftmo_forever.ps1"

$trigger = New-ScheduledTaskTrigger -AtStartup

$principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -RestartCount 999

Register-ScheduledTask `
    -TaskName "FTMO-TradingBot" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "FTMO Trading Bot - Runs 24/7"

# Start the task now (without waiting for reboot)
Start-ScheduledTask -TaskName "FTMO-TradingBot"
```

**Verify it's running**:
```powershell
Get-ScheduledTask -TaskName "FTMO-TradingBot" | Get-ScheduledTaskInfo
```

Should show: `Running` or `Ready`

---

## Step 9: Disconnect RDP (Bot Keeps Running!)

Just close the RDP window. The bot will continue running 24/7.

---

## Monitoring

### Check if bot is running:

```powershell
# Connect via RDP
mstsc /v:20.9.129.81

# In VM PowerShell:
Get-ScheduledTask -TaskName "FTMO-TradingBot" | Get-ScheduledTaskInfo

# Check logs:
Get-Content C:\bolashak\live_trading\logs\demo_bot.log -Tail 50 -Wait
```

### View auto-start logs:

```powershell
Get-Content C:\bolashak\live_trading\logs\autostart.log -Tail 50 -Wait
```

### Check running processes:

```powershell
Get-Process python
```

Should show `python.exe` running.

---

## FTMO Challenge Rules (CRITICAL!)

| Rule | Limit | Status |
|------|-------|--------|
| **Max Daily Loss** | 5% |  Bot set to 3% (safer) |
| **Max Total Loss** | 10% |  Bot set to 10% |
| **Profit Target (Phase 1)** | 8% |  Target in 4-6 weeks |
| **Profit Target (Phase 2)** | 5% |  Target in 4-6 weeks |
| **Min Trading Days** | 10 days |  Bot trades daily |

---

## Troubleshooting

### Bot not starting:

```powershell
# Check scheduled task
Get-ScheduledTaskInfo -TaskName "FTMO-TradingBot"

# View task history
Get-WinEvent -LogName "Microsoft-Windows-TaskScheduler/Operational" | Where-Object {$_.Message -like "*FTMO*"} | Select-Object -First 10
```

### MT5 connection errors:

1. Open MT5 manually on the VM
2. Login with FTMO credentials
3. Verify it connects
4. Close MT5
5. Restart bot

### Python errors:

```powershell
# Reinstall packages
cd C:\bolashak
.\.venv\Scripts\Activate.ps1
pip install --force-reinstall MetaTrader5 python-dotenv pandas numpy scikit-learn joblib
```

---

## Important Notes

###  NEVER Shutdown the VM

- VM must run 24/7 for FTMO challenge
- Azure charges even when stopped (storage costs)
- Only shutdown if you want to stop trading

###  VM Auto-Restarts

- Windows updates may reboot VM
- Scheduled task auto-starts bot after reboot
- Check logs after any VM restart

###  Cost Management

- VM costs ~$30-40/month
- Your $100 credits last ~2.5-3 months
- After credits expire, add credit card to Azure

---

## Summary

| Step | Time | Status |
|------|------|--------|
| 1. Connect RDP | 1 min | |
| 2. Run setup script | 10 min | |
| 3. Copy project files | 5 min | |
| 4. Setup Python | 3 min | |
| 5. Configure FTMO | 2 min | |
| 6. Test connection | 1 min | |
| 7. Test bot | 5 min | |
| 8. Setup auto-start | 5 min | |
| **Total** | **32 min** | |

---

## Next Steps

1.  **Week 1-2**: Monitor bot daily, verify trades
2.  **Week 3-6**: Let bot run toward 8% profit (Phase 1)
3.  **Week 7-12**: Complete Phase 2 (5% profit)
4.  **Week 13+**: Get funded account, start earning!

---

**Your Windows VM "sabyrbolashak" is ready for 24/7 FTMO trading!** 

VM IP: `20.9.129.81`  
Username: `abylay_dos`

**Connect now**: `mstsc /v:20.9.129.81`

