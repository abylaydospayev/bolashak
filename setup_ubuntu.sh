#!/bin/bash
# Quick deployment script for Ubuntu VM "Sabyr"
# Run this on the VM after SSH connection

set -e  # Exit on error

echo "=========================================="
echo "  USDJPY Bot - Ubuntu Setup Script"
echo "  VM: Sabyr (westus3)"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "❌ Please don't run as root (no sudo)"
   exit 1
fi

# Update system
echo "[1/7] Updating system packages..."
sudo apt update -qq
sudo apt upgrade -y -qq

# Install Python and dependencies
echo "[2/7] Installing Python 3.12..."
sudo apt install python3.12 python3-pip git htop screen -y -qq

# Verify Python
python3 --version
pip3 --version

# Create project directory
echo "[3/7] Setting up project directory..."
mkdir -p ~/bolashak
cd ~/bolashak

# Create virtual environment
echo "[4/7] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install Python packages
echo "[5/7] Installing Python packages..."
echo "   - pandas, numpy, scikit-learn, joblib, python-dotenv, requests"
pip install pandas numpy scikit-learn joblib python-dotenv requests -q

# Create directories
echo "[6/7] Creating project structure..."
mkdir -p ~/bolashak/models
mkdir -p ~/bolashak/live_trading/logs
mkdir -p ~/bolashak/data

# Create .env template if doesn't exist
if [ ! -f ~/bolashak/live_trading/.env ]; then
    echo "[7/7] Creating .env template..."
    cat > ~/bolashak/live_trading/.env << 'EOF'
# Oanda API Configuration
OANDA_API_TOKEN=your_api_token_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENVIRONMENT=practice

# Symbol to trade
SYMBOL=USDJPY
EOF
    echo "✅ .env template created at ~/bolashak/live_trading/.env"
    echo "⚠️  Edit it with: nano ~/bolashak/live_trading/.env"
else
    echo "[7/7] .env file already exists, skipping..."
fi

echo ""
echo "=========================================="
echo "  ✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Copy your project files from Windows PC:"
echo "   scp -r C:\\Users\\abyla\\Desktop\\bolashak\\* abylay_dos@<VM_IP>:~/bolashak/"
echo ""
echo "2. Get Oanda API token:"
echo "   - Login: https://www.oanda.com/demo-account/login"
echo "   - Username: 1600037272"
echo "   - Password: 8?p1?$$*kW3"
echo "   - Generate token in 'Manage API Access'"
echo ""
echo "3. Edit .env file:"
echo "   nano ~/bolashak/live_trading/.env"
echo ""
echo "4. Test connection:"
echo "   cd ~/bolashak/live_trading"
echo "   source ../.venv/bin/activate"
echo "   python3 oanda_client.py"
echo ""
echo "5. Start trading bot:"
echo "   python3 ubuntu_bot.py"
echo ""
echo "=========================================="
echo ""
