# Ubuntu Trading Bot Setup Guide
### Deploy USDJPY Bot to Azure VM "Sabyr" (westus3)

---

## VM Information
- **Name**: Sabyr
- **Location**: westus3
- **Size**: Standard_B2ats_v2 (2 vCPU, 1GB RAM)
- **OS**: Ubuntu 24.04 LTS
- **Cost**: ~$10-12/month (Your $100 credits = 8-10 months FREE!)
- **Username**: abylay_dos
- **Authentication**: SSH key

---

## Step 1: Get Public IP Address

1. Go to Azure Portal: https://portal.azure.com
2. Navigate to: **Virtual Machines**  **Sabyr**
3. Copy the **Public IP address** (something like `20.xxx.xxx.xxx`)

---

## Step 2: Connect to VM via SSH

### From Windows (PowerShell):

```powershell
# Replace <PUBLIC_IP> with your actual IP
ssh abylay_dos@<PUBLIC_IP>
```

Example:
```powershell
ssh abylay_dos@20.123.45.67
```

### From Linux/Mac:

```bash
ssh abylay_dos@<PUBLIC_IP>
```

**First time connection**: Type `yes` when asked to continue connecting.

---

## Step 3: Install Python and Dependencies

```bash
# Update package list
sudo apt update

# Install Python 3.12 and pip
sudo apt install python3.12 python3-pip git -y

# Verify installation
python3 --version
pip3 --version
```

Expected output:
```
Python 3.12.x
pip 24.x
```

---

## Step 4: Clone or Copy Your Project

### Option A: Clone from Git (if you have a repo)

```bash
cd ~
git clone https://github.com/your-username/bolashak.git
cd bolashak
```

### Option B: Copy files from local PC (recommended)

On your **Windows PC** (PowerShell):

```powershell
# Navigate to project directory
cd C:\Users\abyla\Desktop\bolashak

# Copy entire project to VM (replace <PUBLIC_IP>)
scp -r * abylay_dos@<PUBLIC_IP>:~/bolashak/
```

This will copy all files to the VM.

---

## Step 5: Setup Python Environment on VM

**On the VM** (SSH session):

```bash
cd ~/bolashak

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install pandas numpy scikit-learn joblib python-dotenv requests ta-lib

# If ta-lib fails, install dependencies:
sudo apt install build-essential -y
pip install pandas numpy scikit-learn joblib python-dotenv requests
```

---

## Step 6: Get Oanda API Token

1. **Login to Oanda**: https://www.oanda.com/demo-account/login
   - Username: `1600037272`
   - Password: `8?p1?$$*kW3`

2. **Generate API Token**:
   - Click your name (top right)  **Manage API Access**
   - Click **Generate**  **Personal Access Token**
   - Copy the token (looks like: `abc123-def456-ghi789...`)
   - **IMPORTANT**: Save it somewhere safe (you can't see it again)

3. **Get Account ID**:
   - Same page, look for **Account ID**
   - Usually starts with `101-` (e.g., `101-004-12345678-001`)
   - Copy it

---

## Step 7: Configure .env File

**On the VM**:

```bash
cd ~/bolashak/live_trading

# Create .env file
nano .env
```

Paste this content (replace with your actual values):

```ini
# Oanda API Configuration
OANDA_API_TOKEN=your_api_token_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENVIRONMENT=practice

# Symbol to trade
SYMBOL=USDJPY
```

**Replace**:
- `your_api_token_here`  Your actual API token from Step 6
- `your_account_id_here`  Your actual Account ID from Step 6

**Save and exit**:
- Press `Ctrl + O` (save)
- Press `Enter`
- Press `Ctrl + X` (exit)

---

## Step 8: Test Oanda Connection

```bash
cd ~/bolashak/live_trading
source ../.venv/bin/activate
python3 oanda_client.py
```

Expected output:
```
 Connected to Oanda (practice)

Account Balance: $98,972.55
Equity: $98,972.55

Fetching USD_JPY M15 bars...
 Retrieved 100 bars
```

If you see errors, check:
- API token is correct
- Account ID is correct
- `.env` file has no typos

---

## Step 9: Start the Trading Bot

```bash
cd ~/bolashak/live_trading
source ../.venv/bin/activate
python3 ubuntu_bot.py
```

You should see:
```
================================================================================
USDJPY TRADING BOT STARTED (Ubuntu/Oanda API)
VM: Sabyr (westus3) | Symbol: USDJPY
Threshold: 0.8 | Max Positions: 3
================================================================================
 Connected to Oanda (practice)
 Account Balance: $98,972.55
 Account Equity: $98,972.55
 Model loaded from /home/abylay_dos/bolashak/models/USDJPY_ensemble_oos.pkl
 Fetched and prepared 250 M15 bars with 37 features
[1] Price: 153.850 | BuyProb: 0.241 | H1Trend: 1 | ATR: 0.1234 | Positions: 0
```

The bot is now running! Press `Ctrl+C` to stop it.

---

## Step 10: Run Bot 24/7 in Background

### Option A: Using `screen` (recommended)

```bash
# Install screen
sudo apt install screen -y

# Start a new screen session named "tradingbot"
screen -S tradingbot

# Inside screen, start the bot
cd ~/bolashak/live_trading
source ../.venv/bin/activate
python3 ubuntu_bot.py

# Detach from screen (bot keeps running):
# Press: Ctrl+A, then D

# You can now disconnect from SSH and bot keeps running!
```

**To reconnect later**:
```bash
ssh abylay_dos@<PUBLIC_IP>
screen -r tradingbot
```

**To stop the bot**:
```bash
# Reconnect to screen
screen -r tradingbot

# Press Ctrl+C to stop bot

# Exit screen
exit
```

### Option B: Using `systemd` service (auto-start on reboot)

Create a service file:

```bash
sudo nano /etc/systemd/system/tradingbot.service
```

Paste this content:

```ini
[Unit]
Description=USDJPY Trading Bot
After=network.target

[Service]
Type=simple
User=abylay_dos
WorkingDirectory=/home/abylay_dos/bolashak/live_trading
Environment="PATH=/home/abylay_dos/bolashak/.venv/bin"
ExecStart=/home/abylay_dos/bolashak/.venv/bin/python3 /home/abylay_dos/bolashak/live_trading/ubuntu_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Save and enable:

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable tradingbot

# Start service now
sudo systemctl start tradingbot

# Check status
sudo systemctl status tradingbot

# View logs
sudo journalctl -u tradingbot -f
```

**Service commands**:
```bash
# Start bot
sudo systemctl start tradingbot

# Stop bot
sudo systemctl stop tradingbot

# Restart bot
sudo systemctl restart tradingbot

# View logs (live)
sudo journalctl -u tradingbot -f

# View logs (last 100 lines)
sudo journalctl -u tradingbot -n 100
```

---

## Step 11: Monitor the Bot

### Check logs:

```bash
cd ~/bolashak/live_trading/logs
tail -f ubuntu_bot.log
```

### Check account status:

```bash
cd ~/bolashak/live_trading
source ../.venv/bin/activate
python3 -c "
from oanda_client import OandaClient
c = OandaClient()
c.initialize()
info = c.get_account_info()
print(f'Balance: \${info[\"balance\"]:,.2f}')
print(f'Equity: \${info[\"equity\"]:,.2f}')
print(f'Profit: \${info[\"profit\"]:,.2f}')
"
```

### Check VM resource usage:

```bash
# CPU and memory
htop

# Disk space
df -h

# Network usage
vnstat
```

---

## Troubleshooting

### Bot crashes with "Module not found"
```bash
cd ~/bolashak
source .venv/bin/activate
pip install pandas numpy scikit-learn joblib python-dotenv requests
```

### Can't connect to Oanda
- Check `.env` file has correct API token and Account ID
- Verify token is still valid (generate new one if needed)
- Check VM has internet connection: `ping google.com`

### Model file not found
```bash
# Check if models exist
ls -la ~/bolashak/models/

# If missing, copy from Windows PC:
# On Windows:
scp C:\Users\abyla\Desktop\bolashak\models\* abylay_dos@<PUBLIC_IP>:~/bolashak/models/
```

### VM is slow/unresponsive
```bash
# Check memory usage
free -h

# Check CPU usage
top

# If needed, restart VM from Azure Portal
```

### Bot not placing trades
- Check logs: `tail -f ~/bolashak/live_trading/logs/ubuntu_bot.log`
- Probability might be below threshold (0.80)
- Check positions: May have reached max (3 positions)

---

## Cost Monitoring

Check Azure costs:
1. Go to: https://portal.azure.com
2. Navigate to: **Cost Management**  **Cost analysis**
3. Filter by: **Resource group**  **Bolashak**

Expected cost: ~$0.30-0.40/day (~$10-12/month)

Your $100 credits will last **8-10 months**! 

---

## Backup Strategy

### Backup logs weekly:

```bash
# On your Windows PC
scp abylay_dos@<PUBLIC_IP>:~/bolashak/live_trading/logs/*.log C:\Users\abyla\Desktop\bot_logs\
```

### Backup model updates:

If you retrain the model locally, deploy to VM:

```bash
# On Windows PC
scp C:\Users\abyla\Desktop\bolashak\models\USDJPY_ensemble_oos.pkl abylay_dos@<PUBLIC_IP>:~/bolashak/models/

# Then restart bot on VM
ssh abylay_dos@<PUBLIC_IP>
sudo systemctl restart tradingbot
```

---

## Next Steps

1.  **Monitor for 1 week** - Check logs daily
2.  **Track performance** - Compare to backtest (79.9% win rate)
3.  **Adjust if needed** - Tune threshold, position size, etc.
4.  **Scale up** - If profitable, increase position size
5.  **Go live** - Switch to `OANDA_ENVIRONMENT=live` when ready

---

## Quick Reference

| Action | Command |
|--------|---------|
| SSH to VM | `ssh abylay_dos@<PUBLIC_IP>` |
| Start bot (screen) | `screen -S tradingbot; cd ~/bolashak/live_trading; source ../.venv/bin/activate; python3 ubuntu_bot.py` |
| Detach screen | `Ctrl+A, then D` |
| Reconnect screen | `screen -r tradingbot` |
| View logs | `tail -f ~/bolashak/live_trading/logs/ubuntu_bot.log` |
| Start service | `sudo systemctl start tradingbot` |
| Stop service | `sudo systemctl stop tradingbot` |
| Check status | `sudo systemctl status tradingbot` |
| View service logs | `sudo journalctl -u tradingbot -f` |

---

**VM Details**:
- Name: Sabyr
- Location: westus3  
- Size: Standard_B2ats_v2
- OS: Ubuntu 24.04 LTS
- Username: abylay_dos
- Cost: ~$10-12/month

**Good luck with your 24/7 trading bot!** 

