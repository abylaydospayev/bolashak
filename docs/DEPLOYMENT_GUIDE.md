# Risk Management Deployment Guide

## Overview
This guide will help you deploy the new risk management system to your Azure VM.

## What's New

### 1. Risk Manager Module (`risk_manager.py`)
-  Maximum 3 concurrent positions
-  5-minute cooldown between trades
-  30 pip stop loss / 50 pip take profit on every trade
-  $500 daily loss limit
-  Signal strength validation (BUY > 0.75, SELL < 0.25)

### 2. Enhanced Bot (`demo_bot_with_risk.py`)
-  Integrated risk management
-  Professional risk controls
-  Better logging and position tracking

### 3. Updated Configuration (`.env`)
-  All risk parameters now configurable
-  Safer default values

## Deployment Steps

### Option 1: Git Workflow (Recommended)

#### On Local PC:
```powershell
# Navigate to project folder
cd C:\Users\abyla\Desktop\bolashak

# Check what files changed
git status

# Add new/modified files
git add live_trading/risk_manager.py
git add live_trading/demo_bot_with_risk.py
git add live_trading/.env
git add live_trading/.env.example

# Commit changes
git commit -m "Add comprehensive risk management system"

# Push to repository (if you have a remote setup)
git push origin main
```

#### On Azure VM (via RDP):
```powershell
# Open PowerShell on VM
cd C:\Users\abylay_dos\Desktop\bolashak

# Pull latest changes
git pull origin main

# If you get errors, you may need to stash local changes first:
git stash
git pull origin main
```

### Option 2: Direct Copy via RDP (Faster)

1. **Connect to VM via RDP** (IP: 20.9.129.81)
2. **Open two windows side by side:**
   - Left: Your local PC folder: `C:\Users\abyla\Desktop\bolashak\live_trading`
   - Right: VM folder: `C:\Users\abylay_dos\Desktop\bolashak\live_trading`
3. **Copy these files from local to VM:**
   - `risk_manager.py` (NEW)
   - `demo_bot_with_risk.py` (NEW)
   - `.env` (UPDATED - has new risk parameters)

## Critical Configuration Changes

### Update `.env` on VM

Open `C:\Users\abylay_dos\Desktop\bolashak\live_trading\.env` and add these lines:

```bash
# ==============================================
# RISK MANAGEMENT
# ==============================================
MAX_POSITIONS=3
MIN_INTERVAL_SECONDS=300
STOP_LOSS_PIPS=30
TAKE_PROFIT_PIPS=50
MAX_DAILY_LOSS=500

# ==============================================
# SIGNAL THRESHOLDS
# ==============================================
BUY_THRESHOLD=0.75
SELL_THRESHOLD=0.25
MIN_PROBABILITY_DIFF=0.2
```

**CRITICAL:** Also verify `MT5_LOT_SIZE=0.1` (NOT 11.77!)

## Enable AutoTrading in MT5

 **THIS IS CRITICAL - Without this, no trades will execute!**

1. Open MT5 terminal on the VM
2. Press **Ctrl + E** (or click the AutoTrading button in toolbar)
3. Verify the button turns **GREEN**
4. You should see "AutoTrading enabled" in the terminal log

## Start the New Bot

### Stop Old Bot (if running):
1. Find the PowerShell window running `demo_bot.py`
2. Press **Ctrl + C** to stop it
3. Close that terminal window

### Start New Bot with Risk Management:

```powershell
# Navigate to project folder
cd C:\Users\abylay_dos\Desktop\bolashak

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Start new bot with risk management
python live_trading\demo_bot_with_risk.py
```

## Verify It's Working

You should see output like this:

```
================================================================================
USDJPY DEMO BOT STARTING (WITH RISK MANAGEMENT)
================================================================================
[2024-11-06 15:30:00] [INFO]  Loaded model: models\USDJPY_ensemble_oos.pkl
[2024-11-06 15:30:01] [INFO]  Risk Manager initialized:
[2024-11-06 15:30:01] [INFO]    - Max positions: 3
[2024-11-06 15:30:01] [INFO]    - Min interval: 300s
[2024-11-06 15:30:01] [INFO]    - Stop loss: 30 pips
[2024-11-06 15:30:01] [INFO]    - Take profit: 50 pips
[2024-11-06 15:30:01] [INFO]    - Max daily loss: $500
[2024-11-06 15:30:01] [INFO]    - Buy threshold: 0.75
[2024-11-06 15:30:01] [INFO]    - Sell threshold: 0.25
[2024-11-06 15:30:02] [INFO] Connecting to MT5: OANDA-Demo-1
[2024-11-06 15:30:03] [INFO] Account Balance: $98,972.55
[2024-11-06 15:30:03] [INFO] Account Equity: $98,972.55
[2024-11-06 15:30:03] [INFO] Starting trading loop...
[2024-11-06 15:30:03] [INFO] Lot size: 0.1
[2024-11-06 15:30:03] [INFO] Check interval: 60s
```

## Monitor the Bot

### Use the Monitor Script:

Open a **second** PowerShell window on the VM:

```powershell
cd C:\Users\abylay_dos\Desktop\bolashak
.\.venv\Scripts\Activate.ps1
python live_trading\monitor.py
```

This will show:
-  Account balance and equity
-  Open positions
-  AutoTrading status
-  Recent trades
-  Risk warnings

### Check Logs:

```powershell
# View latest log file
cd C:\Users\abylay_dos\Desktop\bolashak\live_trading\logs
Get-Content demo_bot_20241106.log -Tail 50
```

## What to Expect

### With Risk Management:
-  Maximum 3 positions at a time
-  Each position has SL (-30 pips) and TP (+50 pips)
-  5-minute cooldown between trades
-  Only strong signals (>0.75 or <0.25) get executed
-  Daily loss limit of $500
-  0.1 lot size = ~$10 per pip (manageable risk)

### First Trade Example:
```
[2024-11-06 15:35:00] [INFO] [1] Price: 152.845, Prob: 0.185, H1: -1, Pos: 0
[2024-11-06 15:35:00] [INFO]  SELL SIGNAL: prob=0.185, H1 downtrend, STRONG
[2024-11-06 15:35:01] [INFO]  SELL executed: 0.1 lots @ 152.845, SL=153.145, TP=152.345
```

### Risk Rejection Example:
```
[2024-11-06 15:40:00] [INFO] [2] Price: 152.800, Prob: 0.220, H1: -1, Pos: 3
[2024-11-06 15:40:00] [INFO]  Max positions reached (3), waiting...
```

## Troubleshooting

### Problem: "Order failed, retcode=10027"
**Solution:** AutoTrading is disabled. Press Ctrl+E in MT5.

### Problem: "Max positions reached"
**Cause:** You already have 3 positions open (this is GOOD - it's working!)
**Action:** Wait for some positions to close (hit SL or TP)

### Problem: "Signal too weak, skipping"
**Cause:** Model probability not strong enough (between 0.25-0.75)
**Action:** This is normal. Bot only trades very confident signals.

### Problem: "Time interval not met"
**Cause:** Less than 5 minutes since last trade
**Action:** This is working correctly - prevents overtrading

### Problem: "Daily loss limit reached"
**Cause:** Lost $500 today
**Action:** Bot stops trading for the day. Resets at midnight UTC.

## Safety Checklist

Before leaving the bot running, verify:

-  AutoTrading is ENABLED in MT5 (green button)
-  MT5_LOT_SIZE=0.1 in .env (not 11.77!)
-  MAX_POSITIONS=3 in .env
-  STOP_LOSS_PIPS=30 and TAKE_PROFIT_PIPS=50 in .env
-  Bot shows "Risk Manager initialized" on startup
-  Account has sufficient balance ($98,972 is plenty)
-  Symbol is USDJPY.sim (not USDJPY)
-  VM stays on (don't shut it down)

## Daily Monitoring

### Quick Check (5 minutes):
1. RDP into VM
2. Check bot is still running (PowerShell window open)
3. Run `python live_trading\monitor.py`
4. Verify:
   - Balance hasn't dropped dramatically
   - No more than 3 positions open
   - All positions have SL/TP set
   - AutoTrading still enabled

### Weekly Review (30 minutes):
1. Check logs for patterns
2. Calculate win rate and average P/L
3. Adjust thresholds if needed (BUY_THRESHOLD, SELL_THRESHOLD)
4. Consider adjusting lot size based on performance

## Performance Tuning (Optional)

After 1 week of trading, you can adjust:

### More Conservative:
```bash
MAX_POSITIONS=2           # Only 2 positions
BUY_THRESHOLD=0.80        # Higher threshold (fewer but stronger signals)
SELL_THRESHOLD=0.20       # Lower threshold
MT5_LOT_SIZE=0.05         # Smaller position size
```

### More Aggressive:
```bash
MAX_POSITIONS=5           # More positions
BUY_THRESHOLD=0.70        # Lower threshold (more signals)
SELL_THRESHOLD=0.30       # Higher threshold
MT5_LOT_SIZE=0.2          # Larger position size
```

 **Never set MAX_POSITIONS > 10 or LOT_SIZE > 1.0 without careful testing!**

## Emergency Stop

If something goes wrong:

1. **Immediate Stop:**
   - Press Ctrl+C in the bot's PowerShell window
   - Or: Disable AutoTrading in MT5 (Ctrl+E)

2. **Close All Positions:**
   - In MT5, go to "Trade" tab
   - Right-click each position  "Close"

3. **Prevent Restart:**
   - Delete or rename `demo_bot_with_risk.py`
   - Or: Set `MAX_DAILY_LOSS=0` in .env

## Support

If you need help:
1. Check the logs: `live_trading\logs\demo_bot_YYYYMMDD.log`
2. Run the monitor: `python live_trading\monitor.py`
3. Review this guide
4. Check MT5 for errors (Tools  Journal)

---

**Remember:** This is a demo account. The worst that can happen is you lose demo money. Use this to test and refine your strategy before considering any real trading!

