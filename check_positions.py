"""
Check current open positions
"""
import MetaTrader5 as mt5
from datetime import datetime

# Initialize MT5
if not mt5.initialize():
    print(f"MT5 initialization failed: {mt5.last_error()}")
    exit()

print(f"\n{'='*60}")
print("CURRENT OPEN POSITIONS")
print(f"{'='*60}\n")

# Get all positions
positions = mt5.positions_get()

if positions is None:
    print(f"ERROR: Failed to get positions: {mt5.last_error()}")
elif len(positions) == 0:
    print("No open positions")
else:
    print(f"Total positions: {len(positions)}\n")
    for i, pos in enumerate(positions, 1):
        print(f"Position {i}:")
        print(f"  Ticket: {pos.ticket}")
        print(f"  Symbol: {pos.symbol}")
        print(f"  Type: {'BUY' if pos.type == 0 else 'SELL'}")
        print(f"  Volume: {pos.volume}")
        print(f"  Price Open: {pos.price_open}")
        print(f"  Current Price: {pos.price_current}")
        print(f"  SL: {pos.sl}")
        print(f"  TP: {pos.tp}")
        print(f"  Profit: ${pos.profit:.2f}")
        print(f"  Time: {datetime.fromtimestamp(pos.time)}")
        print()

# Check account mode
account_info = mt5.account_info()
if account_info:
    print(f"\n{'='*60}")
    print("ACCOUNT MODE")
    print(f"{'='*60}\n")
    
    # Account trade mode
    trade_modes = {
        0: "ACCOUNT_TRADE_MODE_DEMO",
        1: "ACCOUNT_TRADE_MODE_CONTEST", 
        2: "ACCOUNT_TRADE_MODE_REAL"
    }
    
    # Margin mode
    margin_modes = {
        0: "ACCOUNT_MARGIN_MODE_RETAIL_NETTING",
        1: "ACCOUNT_MARGIN_MODE_EXCHANGE",
        2: "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING"
    }
    
    print(f"Login: {account_info.login}")
    print(f"Trade Mode: {trade_modes.get(account_info.trade_mode, 'Unknown')}")
    print(f"Margin Mode: {margin_modes.get(account_info.margin_mode, 'Unknown')}")
    print(f"Balance: ${account_info.balance:.2f}")
    print(f"Equity: ${account_info.equity:.2f}")
    print(f"Profit: ${account_info.profit:.2f}")

# Check recent deals
print(f"\n{'='*60}")
print("LAST 10 DEALS")
print(f"{'='*60}\n")

from datetime import datetime, timedelta
now = datetime.now()
from_date = now - timedelta(hours=1)

deals = mt5.history_deals_get(from_date, now)

if deals is None:
    print(f"ERROR: Failed to get deals: {mt5.last_error()}")
elif len(deals) == 0:
    print("No recent deals")
else:
    for i, deal in enumerate(deals[-10:], 1):
        deal_types = {0: 'BUY', 1: 'SELL', 2: 'BALANCE', 3: 'CREDIT', 4: 'CHARGE', 5: 'CORRECTION', 6: 'BONUS', 7: 'COMMISSION', 8: 'DAILY', 9: 'MONTHLY', 10: 'ANNUAL'}
        entry_types = {0: 'IN', 1: 'OUT', 2: 'INOUT', 3: 'OUT_BY'}
        
        print(f"Deal {i}:")
        print(f"  Ticket: {deal.ticket}")
        print(f"  Order: {deal.order}")
        print(f"  Position: {deal.position_id}")
        print(f"  Symbol: {deal.symbol}")
        print(f"  Type: {deal_types.get(deal.type, 'Unknown')}")
        print(f"  Entry: {entry_types.get(deal.entry, 'Unknown')}")
        print(f"  Volume: {deal.volume}")
        print(f"  Price: {deal.price}")
        print(f"  Profit: ${deal.profit:.2f}")
        print(f"  Time: {datetime.fromtimestamp(deal.time)}")
        print()

mt5.shutdown()
