# Ubuntu Bot Deployment - Summary

##  What Was Created

### 1. **Ubuntu Trading Bot** (`live_trading/ubuntu_bot.py`)
- Full-featured trading bot for Linux/Ubuntu
- Uses Oanda REST API (no MT5 needed)
- Same strategy as Windows version:
  - Ensemble model (RandomForest + GradientBoosting)
  - Multi-timeframe features (M15, M30, H1, H4)
  - 80% probability threshold
  - Risk management (1% per trade, 3% daily loss limit)
  - Auto SL/TP based on ATR

### 2. **Oanda API Client** (`live_trading/oanda_client.py`)
- Already exists in your project 
- REST API integration for:
  - Account info
  - Historical data (M1, M5, M15, M30, H1, H4, D1)
  - Market orders (buy/sell)
  - Position management
  - SL/TP support

### 3. **Deployment Scripts**
- **`deploy_to_ubuntu.ps1`** - Windows PowerShell script to copy files to VM
- **`setup_ubuntu.sh`** - Bash script to setup Ubuntu environment

### 4. **Documentation**
- **`UBUNTU_QUICK_START.md`** - 15-minute setup guide (5 steps)
- **`UBUNTU_SETUP.md`** - Complete guide with troubleshooting
- **`.env.example`** - Updated with Oanda configuration

---

##  Your Azure VM "Sabyr"

| Detail | Value |
|--------|-------|
| **Name** | Sabyr |
| **Location** | westus3 |
| **Size** | Standard_B2ats_v2 (2 vCPU, 1GB RAM) |
| **OS** | Ubuntu 24.04 LTS |
| **Username** | abylay_dos |
| **Auth** | SSH key (already configured) |
| **Cost** | ~$10-12/month |
| **Your Credits** | $100 = **8-10 months FREE!**  |

---

##  Cost Comparison

| Option | Monthly Cost | Your Credits Last |
|--------|-------------|------------------|
| **Ubuntu VM (current)** | **$10-12** | **8-10 months**  |
| Windows B1ms | $22 | 4-5 months |
| Windows B2ats_v2 | $84 | 1.2 months  |
| Keep PC on 24/7 | $30-50 (electricity) | N/A |

**You're saving ~$10-60/month by using Ubuntu!** 

---

##  Next Steps (Follow Quick Start)

### Option A: Quick Start (15 minutes)
Follow **`UBUNTU_QUICK_START.md`**:

1. Get VM Public IP from Azure Portal (1 min)
2. Copy files with `deploy_to_ubuntu.ps1` (3 min)
3. Get Oanda API token (2 min)
4. Configure `.env` on VM (5 min)
5. Start bot (4 min)

Total: **15 minutes** 

### Option B: Detailed Setup (30 minutes)
Follow **`UBUNTU_SETUP.md`** for:
- Step-by-step instructions
- Auto-start with systemd service
- Advanced monitoring
- Troubleshooting guide

---

##  What You Need

### 1. VM Public IP
Get from: https://portal.azure.com  Virtual Machines  Sabyr

### 2. Oanda API Credentials
- **Login**: https://www.oanda.com/demo-account/login
- **Username**: `1600037272`
- **Password**: `8?p1?$$*kW3`
- Generate: **Personal Access Token**
- Copy: **Account ID** (starts with `101-`)

### 3. Files to Deploy
Already prepared:
-  `models/USDJPY_ensemble_oos.pkl`
-  `models/scaler.pkl`
-  `live_trading/ubuntu_bot.py`
-  `live_trading/oanda_client.py`
-  `build_features_enhanced.py`

---

##  Expected Performance

Based on backtest results:

| Metric | Backtest | Target (Live) |
|--------|----------|--------------|
| Win Rate | 79.97% | 75-80%  |
| Max Drawdown | 7.71% | <10%  |
| Total Profit | +$94,000 | Monitor  |
| Sharpe Ratio | 2.15 | >2.0  |
| AUC Score | 0.795 | N/A |

---

##  Monitoring Commands

### Check bot status:
```bash
ssh abylay_dos@<VM_IP>
screen -r tradingbot
```

### View logs:
```bash
ssh abylay_dos@<VM_IP>
tail -f ~/bolashak/live_trading/logs/ubuntu_bot.log
```

### Check account:
```bash
ssh abylay_dos@<VM_IP>
cd ~/bolashak/live_trading
source ../.venv/bin/activate
python3 -c "
from oanda_client import OandaClient
c = OandaClient()
c.initialize()
info = c.get_account_info()
print(f'Balance: \${info[\"balance\"]:,.2f}')
print(f'Equity: \${info[\"equity\"]:,.2f}')
"
```

---

##  Key Features

### Risk Management
-  1% risk per trade
-  3% daily loss limit
-  10% max drawdown protection
-  Max 3 positions simultaneously
-  Dynamic position sizing based on ATR

### Auto Stop Loss / Take Profit
-  SL: 2.0  ATR from entry
-  TP: 3.0  ATR from entry
-  Risk-reward ratio: 1:1.5

### Signal Quality
-  Only trade when probability  80%
-  Confirm trend with H1 timeframe
-  Multi-timeframe analysis (M30, H1, H4)

### Bot Resilience
-  Auto-reconnect if API fails
-  Logs all trades and errors
-  Runs 24/7 in background (screen/systemd)

---

##  Deployment Workflow

```
Windows PC                    Azure VM "Sabyr"
            

1. deploy_to_ubuntu.ps1     Copy files via SCP
2. SSH to VM                Login to Ubuntu
3. Edit .env                Add API credentials
4. Start bot                Runs 24/7
5. Disconnect SSH           Bot keeps running
6. Monitor logs             Check performance
```

---

##  Checklist

Before starting:
- [ ] Get VM Public IP from Azure Portal
- [ ] Get Oanda API token
- [ ] Get Oanda Account ID
- [ ] Run `deploy_to_ubuntu.ps1` on Windows
- [ ] SSH to VM: `ssh abylay_dos@<VM_IP>`
- [ ] Edit `.env` with API credentials
- [ ] Test connection: `python3 oanda_client.py`
- [ ] Start bot: `python3 ubuntu_bot.py`
- [ ] Run in background: `screen -S tradingbot`
- [ ] Detach screen: `Ctrl+A, then D`

After 24 hours:
- [ ] Check logs for errors
- [ ] Verify trades are being placed (when signals occur)
- [ ] Compare performance to backtest
- [ ] Monitor Azure costs

After 1 week:
- [ ] Calculate win rate
- [ ] Track drawdown
- [ ] Adjust threshold if needed
- [ ] Consider scaling up position size

---

##  Support Resources

### Documentation
- **Quick Start**: `UBUNTU_QUICK_START.md`
- **Full Guide**: `UBUNTU_SETUP.md`
- **Oanda API**: https://developer.oanda.com/rest-live-v20/

### Azure Portal
- **VM Management**: https://portal.azure.com
- **Cost Tracking**: Cost Management  Cost analysis
- **Resource Group**: Bolashak

### Oanda
- **Demo Login**: https://www.oanda.com/demo-account/login
- **API Access**: Manage API Access (after login)
- **Support**: https://www.oanda.com/support/

---

##  Summary

You now have:
1.  **Ubuntu bot code** ready for deployment
2.  **Azure VM** already created (Sabyr, westus3)
3.  **Deployment scripts** to copy files
4.  **Complete documentation** (quick + detailed)
5.  **8-10 months FREE** hosting ($100 credits)

**Next action**: Follow **`UBUNTU_QUICK_START.md`** to get your bot running in 15 minutes! 

---

##  Pro Tips

1. **Start with small position size** (0.01 lots) for 1 week
2. **Monitor daily** for first week to catch issues
3. **Compare to backtest** - should be within 5%
4. **Don't overtrade** - let the 80% threshold filter signals
5. **Track costs** - should be ~$0.30-0.40/day
6. **Backup logs weekly** - `scp` them to your PC
7. **Update model monthly** - retrain with latest data

---

**Good luck with your 24/7 trading bot on Ubuntu!** 

