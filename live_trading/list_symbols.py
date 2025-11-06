"""List all available symbols in MT5 terminal."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
load_dotenv(Path(__file__).parent / '.env')

TERMINAL = os.getenv('MT5_TERMINAL_PATH')
LOGIN = int(os.getenv('MT5_LOGIN', 0))
PASSWORD = os.getenv('MT5_PASSWORD')
SERVER = os.getenv('MT5_SERVER')

print("Initializing MT5...")
if TERMINAL:
    ok = mt5.initialize(terminal_path=TERMINAL, login=LOGIN, password=PASSWORD, server=SERVER)
else:
    ok = mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER)

if not ok:
    print(f"Failed to initialize MT5: {mt5.last_error()}")
    exit(1)

print("✅ MT5 initialized\n")

# Get all symbols
symbols = mt5.symbols_get()

if symbols is None or len(symbols) == 0:
    print("No symbols found")
else:
    print(f"Found {len(symbols)} symbols\n")
    
    # Filter for common forex pairs
    forex_pairs = []
    for s in symbols:
        name = s.name
        # Look for USD, EUR, GBP, JPY pairs
        if any(x in name.upper() for x in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']):
            forex_pairs.append(name)
    
    print(f"Forex pairs ({len(forex_pairs)}):")
    print("-" * 60)
    for pair in sorted(forex_pairs):
        print(f"  {pair}")
    
    print("\n" + "=" * 60)
    print("Looking for USDJPY...")
    print("=" * 60)
    usdjpy_variants = [s for s in forex_pairs if 'USD' in s.upper() and 'JPY' in s.upper()]
    if usdjpy_variants:
        print(f"✅ Found {len(usdjpy_variants)} USDJPY variant(s):")
        for v in usdjpy_variants:
            print(f"  ➜ {v}")
        print(f"\nUse this in your .env file: MT5_SYMBOL={usdjpy_variants[0]}")
    else:
        print("❌ No USDJPY variants found")
        print("\nTroubleshooting:")
        print("1. Open MT5 terminal")
        print("2. Press Ctrl+U to open 'Symbols'")
        print("3. Search for 'USDJPY' or 'USD/JPY'")
        print("4. Right-click → 'Show in Market Watch'")
        print("5. Note the exact symbol name and update .env file")

mt5.shutdown()
