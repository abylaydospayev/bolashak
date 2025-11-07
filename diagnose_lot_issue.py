"""
Analyze why we have 19.5 lots instead of 0.5 lots
"""
import MetaTrader5 as mt5
from pathlib import Path
from datetime import datetime

print("\n" + "="*80)
print("ANALYZING LOT SIZE ISSUE")
print("="*80 + "\n")

# Initialize MT5
if not mt5.initialize():
    print("Failed to initialize MT5")
    exit(1)

# Check current positions
print("1. CURRENT POSITIONS:")
print("-"*80)
positions = mt5.positions_get()
if positions:
    for pos in positions:
        print(f"  Ticket: {pos.ticket}")
        print(f"  Symbol: {pos.symbol}")
        print(f"  Type: {'BUY' if pos.type == 0 else 'SELL'}")
        print(f"  Volume: {pos.volume} lots  ⚠️ SHOULD BE 0.5!")
        print(f"  Entry: {pos.price_open}")
        print(f"  Current: {pos.price_current}")
        print(f"  Profit: ${pos.profit:.2f}")
        print(f"  Time: {datetime.fromtimestamp(pos.time)}")
        print()
else:
    print("  No open positions\n")

# Check recent deals (trades executed)
print("2. RECENT DEALS (Last 50):")
print("-"*80)
from_date = datetime.now().replace(hour=0, minute=0, second=0)
deals = mt5.history_deals_get(from_date, datetime.now())

if deals:
    print(f"  Total deals today: {len(deals)}\n")
    
    # Group by position_id to see what combined
    from collections import defaultdict
    positions_dict = defaultdict(list)
    
    for deal in deals:
        if deal.symbol == "USDJPY.sim":
            positions_dict[deal.position_id].append(deal)
    
    print(f"  Unique positions opened: {len(positions_dict)}\n")
    
    # Analyze each position
    for pos_id, deals_list in positions_dict.items():
        print(f"  Position ID {pos_id}:")
        total_volume = 0
        for deal in deals_list:
            deal_types = {0: 'BUY', 1: 'SELL', 2: 'BALANCE'}
            entry_types = {0: 'IN', 1: 'OUT', 2: 'INOUT', 3: 'OUT_BY'}
            
            print(f"    Deal {deal.ticket}: {deal_types.get(deal.type, 'Unknown')} "
                  f"{deal.volume} lots @ {deal.price} "
                  f"({entry_types.get(deal.entry, 'Unknown')}) "
                  f"Time: {datetime.fromtimestamp(deal.time).strftime('%H:%M:%S')}")
            
            if deal.entry == 0:  # IN (opening trade)
                total_volume += deal.volume
            elif deal.entry == 1:  # OUT (closing trade)
                total_volume -= deal.volume
        
        print(f"    Net Volume: {total_volume} lots")
        if total_volume > 0.5:
            print(f"    ⚠️ WARNING: This position has {total_volume/0.5:.1f}x the expected volume!")
        print()
else:
    print("  No deals found today\n")

# Check account mode
print("3. ACCOUNT SETTINGS:")
print("-"*80)
account_info = mt5.account_info()
if account_info:
    margin_modes = {
        0: "NETTING (same symbol positions merge)",
        1: "EXCHANGE",
        2: "HEDGING (separate positions allowed)"
    }
    print(f"  Account: {account_info.login}")
    print(f"  Margin Mode: {margin_modes.get(account_info.margin_mode, 'Unknown')}")
    print(f"  Balance: ${account_info.balance:.2f}")
    print(f"  Equity: ${account_info.equity:.2f}")
    print(f"  Margin: ${account_info.margin:.2f}")
    print(f"  Free Margin: ${account_info.margin_free:.2f}")
    print(f"  Margin Level: {account_info.margin_level:.2f}%")
    print()

# Analyze log file
print("4. BOT LOG ANALYSIS:")
print("-"*80)
today = datetime.now().strftime("%Y%m%d")
log_file = Path(f"live_trading/logs/demo_bot_{today}.log")

if log_file.exists():
    trades_executed = []
    with open(log_file) as f:
        for line in f:
            if "executed:" in line and "lots @" in line:
                try:
                    timestamp = line.split("]")[0].replace("[", "")
                    parts = line.split("executed: ")[1]
                    lots = float(parts.split(" lots")[0])
                    trades_executed.append({
                        'time': timestamp,
                        'lots': lots,
                        'line': line.strip()
                    })
                except:
                    pass
    
    print(f"  Trades logged: {len(trades_executed)}")
    print(f"\n  Individual trades:")
    for i, t in enumerate(trades_executed, 1):
        print(f"    {i}. {t['time']} - {t['lots']} lots")
    
    total_lots = sum(t['lots'] for t in trades_executed)
    print(f"\n  Total lots executed: {total_lots}")
    print(f"  Expected (0.5 per trade): {len(trades_executed) * 0.5}")
    
    if total_lots > len(trades_executed) * 0.5:
        print(f"\n  ⚠️ ISSUE: Each trade used {total_lots/len(trades_executed):.1f} lots instead of 0.5!")
    elif len(trades_executed) > 1:
        print(f"\n  ⚠️ ISSUE: {len(trades_executed)} trades merged into 1 position (NETTING MODE)")
else:
    print(f"  Log file not found: {log_file}")

mt5.shutdown()

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80 + "\n")
