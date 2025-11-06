"""Test script to initialize MT5 and fetch bars.

Usage:
  1. Copy .env.example to .env
  2. Edit .env with your MT5 credentials
  3. Run:
    .\.venv\Scripts\python.exe live_trading\test_mt5_connection.py

OR set env variables in PowerShell:
  $env:MT5_TERMINAL_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
  $env:MT5_LOGIN = "12345678"
  $env:MT5_PASSWORD = "mypassword"
  $env:MT5_SERVER = "Your-Broker-Server"
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from live_trading.mt5_client import init_mt5, shutdown_mt5, get_bars, get_account_info

# Load .env file
load_dotenv(Path(__file__).parent / '.env')

TERMINAL = os.getenv('MT5_TERMINAL_PATH')
LOGIN = os.getenv('MT5_LOGIN')
PASSWORD = os.getenv('MT5_PASSWORD')
SERVER = os.getenv('MT5_SERVER')

if not LOGIN or not PASSWORD or not SERVER:
    print("Please set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER environment variables (see .env.example)")
    exit(1)

try:
    login_int = int(LOGIN)
except Exception:
    print("MT5_LOGIN must be an integer")
    exit(1)

print("Initializing MT5...")
ok = init_mt5(TERMINAL, login_int, PASSWORD, SERVER)
if not ok:
    print("Failed to init MT5. Ensure terminal is installed and terminal path is correct.")
    exit(1)

print("Fetching 100 M15 bars for USDJPY.sim...")
df = get_bars('USDJPY.sim', timeframe='M15', count=100)
if df is None:
    print("Failed to fetch bars. Make sure USDJPY.sim is in Market Watch.")
else:
    print(f"✅ Retrieved {len(df)} bars")
    print("\nLatest 5 bars:")
    print(df.tail())

print('Account info:')
info = get_account_info()
print(info)

shutdown_mt5()
print('Done')
