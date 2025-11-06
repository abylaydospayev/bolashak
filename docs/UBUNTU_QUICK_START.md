# Ubuntu Bot - Quick Start (5 Steps, 15 Minutes)

Your Azure VM "Sabyr" is ready! This guide gets your bot running in 15 minutes.

---

## VM Details
- **Name**: Sabyr
- **Location**: westus3
- **Cost**: ~$10-12/month (8-10 months FREE with your $100 credits!)
- **OS**: Ubuntu 24.04 LTS

---

## Step 1: Get VM Public IP (1 minute)

1. Go to: https://portal.azure.com
2. Click: **Virtual Machines**  **Sabyr**
3. Copy the **Public IP address** (e.g., `20.123.45.67`)

---

## Step 2: Copy Files to VM (3 minutes)

On your **Windows PC** (in this directory):

```powershell
.\deploy_to_ubuntu.ps1
```

Enter the VM IP when prompted, then press `y` to confirm.

This copies:
-  Models (USDJPY_ensemble_oos.pkl, scaler.pkl)
-  Bot code (ubuntu_bot.py, oanda_client.py)
-  Feature engineering (build_features_enhanced.py)

---

## Step 3: Get Oanda API Token (2 minutes)

1. Login: https://www.oanda.com/demo-account/login
   - Username: `1600037272`
   - Password: `8?p1?$$*kW3`

2. Click your name (top right)  **Manage API Access**

3. Click **Generate**  **Personal Access Token**

4. **Copy** the token (starts with something like `abc123...`)

5. **Copy** your Account ID (looks like `101-004-12345678-001`)

---

## Step 4: Configure Bot on VM (5 minutes)

```powershell
# SSH to VM (replace <VM_IP> with your actual IP)
ssh abylay_dos@<VM_IP>
```

Type `yes` when asked to continue connecting.

Once connected:

```bash
# Navigate to project
cd ~/bolashak/live_trading

# Edit .env file
nano .env
```

Update these lines:
```ini
OANDA_API_TOKEN=paste_your_token_here
OANDA_ACCOUNT_ID=paste_your_account_id_here
```

**Save**: Press `Ctrl+O`, then `Enter`  
**Exit**: Press `Ctrl+X`

---

## Step 5: Start Bot (4 minutes)

### Test connection first:

```bash
cd ~/bolashak/live_trading
source ../.venv/bin/activate
python3 oanda_client.py
```

Should show:
```
 Connected to Oanda (practice)
Account Balance: $98,972.55
 Retrieved 100 bars
```

### Start the bot:

```bash
python3 ubuntu_bot.py
```

You should see:
```
================================================================================
USDJPY TRADING BOT STARTED (Ubuntu/Oanda API)
VM: Sabyr (westus3) | Symbol: USDJPY
================================================================================
 Connected to Oanda (practice)
 Account Balance: $98,972.55
[1] Price: 153.850 | BuyProb: 0.241 | H1Trend: 1 | Positions: 0
```

 **Bot is running!**

---

## Keep Bot Running 24/7

The bot stops when you close SSH. To run 24/7:

```bash
# Install screen
sudo apt install screen -y

# Start screen session
screen -S tradingbot

# Start bot
cd ~/bolashak/live_trading
source ../.venv/bin/activate
python3 ubuntu_bot.py

# Detach (bot keeps running): Press Ctrl+A, then D
```

Now you can:
-  Close SSH - bot keeps running
-  Close your PC - bot keeps running
-  Disconnect from internet - bot keeps running on Azure

### Reconnect later:

```bash
ssh abylay_dos@<VM_IP>
screen -r tradingbot
```

### Stop bot:

```bash
# Reconnect to screen
screen -r tradingbot

# Press Ctrl+C

# Exit screen
exit
```

---

## Monitoring

### View logs:

```bash
ssh abylay_dos@<VM_IP>
cd ~/bolashak/live_trading/logs
tail -f ubuntu_bot.log
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
print(f'Profit: \${info[\"profit\"]:,.2f}')
"
```

---

## Cost Breakdown

| Item | Cost/Month |
|------|-----------|
| VM (B2ats_v2) | $10-12 |
| Storage | $1-2 |
| Bandwidth | $0-1 |
| **Total** | **~$12/month** |

Your **$100 credits = 8-10 months FREE!** 

---

## Troubleshooting

**Bot crashes:**
```bash
ssh abylay_dos@<VM_IP>
cd ~/bolashak/live_trading/logs
tail -50 ubuntu_bot.log
```

**Can't connect to Oanda:**
- Check `.env` has correct API token
- Verify Account ID is correct
- Token might be expired - generate new one

**Model file not found:**
```powershell
# On Windows PC
scp C:\Users\abyla\Desktop\bolashak\models\* abylay_dos@<VM_IP>:~/bolashak/models/
```

---

## Next Steps

1.  **Monitor for 1 week** - Check logs daily
2.  **Track performance** - Should match backtest (79.9% win rate)
3.  **Adjust if needed** - Tune threshold (currently 0.80)
4.  **Scale up** - Increase position size if profitable
5.  **Go live** - Change `OANDA_ENVIRONMENT=live` when ready

---

## Quick Reference

| Action | Command |
|--------|---------|
| SSH to VM | `ssh abylay_dos@<VM_IP>` |
| Start bot (screen) | `screen -S tradingbot; cd ~/bolashak/live_trading; source ../.venv/bin/activate; python3 ubuntu_bot.py` |
| Detach screen | Press `Ctrl+A`, then `D` |
| Reconnect | `screen -r tradingbot` |
| Stop bot | `Ctrl+C` in screen session |
| View logs | `tail -f ~/bolashak/live_trading/logs/ubuntu_bot.log` |

---

**Your bot is now trading 24/7 on Azure!** 

For detailed setup and advanced features, see: **UBUNTU_SETUP.md**

