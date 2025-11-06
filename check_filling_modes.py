"""
Check what filling modes are supported for USDJPY.sim symbol
"""
import MetaTrader5 as mt5

# Initialize MT5
if not mt5.initialize():
    print(f"MT5 initialization failed: {mt5.last_error()}")
    exit()

symbol = "USDJPY.sim"

# Get symbol info
symbol_info = mt5.symbol_info(symbol)

if symbol_info is None:
    print(f"Failed to get symbol info for {symbol}")
    mt5.shutdown()
    exit()

print(f"\n{'='*60}")
print(f"SYMBOL INFORMATION FOR {symbol}")
print(f"{'='*60}\n")

print(f"Name: {symbol_info.name}")
print(f"Description: {symbol_info.description}")
print(f"Visible: {symbol_info.visible}")
print(f"Selected: {symbol_info.select}")

# Check filling modes
print(f"\n{'='*60}")
print("FILLING MODES")
print(f"{'='*60}\n")

print(f"Filling mode flags: {symbol_info.filling_mode}")

# Decode filling modes
filling_modes = []
if symbol_info.filling_mode & 1:  # SYMBOL_FILLING_FOK
    filling_modes.append("FOK (Fill or Kill)")
if symbol_info.filling_mode & 2:  # SYMBOL_FILLING_IOC
    filling_modes.append("IOC (Immediate or Cancel)")
if symbol_info.filling_mode & 4:  # SYMBOL_FILLING_RETURN
    filling_modes.append("RETURN")

if filling_modes:
    print("Supported filling modes:")
    for mode in filling_modes:
        print(f"  - {mode}")
else:
    print("No filling modes detected or unsupported value")

# Trade modes
print(f"\n{'='*60}")
print("TRADE MODES")
print(f"{'='*60}\n")

print(f"Trade mode: {symbol_info.trade_mode}")

trade_modes = {
    0: "TRADE_DISABLED - Trading disabled",
    1: "TRADE_LONGONLY - Only long positions allowed",
    2: "TRADE_SHORTONLY - Only short positions allowed",
    3: "TRADE_CLOSEONLY - Only position closing allowed",
    4: "TRADE_FULL - No trade restrictions"
}

print(f"  {trade_modes.get(symbol_info.trade_mode, 'Unknown')}")

# Order modes
print(f"\n{'='*60}")
print("ORDER MODES")
print(f"{'='*60}\n")

print(f"Order mode flags: {symbol_info.order_mode}")

# Lot info
print(f"\n{'='*60}")
print("LOT INFORMATION")
print(f"{'='*60}\n")

print(f"Min lot: {symbol_info.volume_min}")
print(f"Max lot: {symbol_info.volume_max}")
print(f"Lot step: {symbol_info.volume_step}")

# Current prices
tick = mt5.symbol_info_tick(symbol)
if tick:
    print(f"\n{'='*60}")
    print("CURRENT PRICES")
    print(f"{'='*60}\n")
    print(f"Bid: {tick.bid}")
    print(f"Ask: {tick.ask}")
    print(f"Spread: {tick.ask - tick.bid}")

mt5.shutdown()
print("\nDone!")
