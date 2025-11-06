"""
Test script to verify bot can execute trades on MT5
Makes one small BUY and one small SELL trade, then closes them
"""

import MetaTrader5 as mt5
import time
from datetime import datetime

def test_connection():
    """Test MT5 connection"""
    print("\n" + "="*60)
    print("TESTING MT5 CONNECTION")
    print("="*60)
    
    if not mt5.initialize():
        print(f"ERROR: MT5 initialize failed: {mt5.last_error()}")
        return False
    
    account_info = mt5.account_info()
    if account_info is None:
        print(f"ERROR: Failed to get account info: {mt5.last_error()}")
        mt5.shutdown()
        return False
    
    print(f"\nAccount Info:")
    print(f"  Login: {account_info.login}")
    print(f"  Balance: ${account_info.balance:.2f}")
    print(f"  Equity: ${account_info.equity:.2f}")
    print(f"  Margin: ${account_info.margin:.2f}")
    print(f"  Free Margin: ${account_info.margin_free:.2f}")
    
    return True

def test_symbol(symbol="USDJPY.sim"):
    """Test symbol availability"""
    print(f"\n" + "="*60)
    print(f"TESTING SYMBOL: {symbol}")
    print("="*60)
    
    # Check if symbol is available
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"ERROR: Symbol {symbol} not found")
        print("\nTrying alternative symbols...")
        
        # Try without .sim
        alt_symbol = symbol.replace('.sim', '')
        symbol_info = mt5.symbol_info(alt_symbol)
        if symbol_info is not None:
            symbol = alt_symbol
            print(f"SUCCESS: Using {symbol} instead")
        else:
            print(f"ERROR: {alt_symbol} also not found")
            return None
    
    # Enable symbol in Market Watch
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"ERROR: Failed to enable {symbol}")
            return None
    
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"ERROR: Failed to get tick for {symbol}")
        return None
    
    print(f"\nSymbol Info:")
    print(f"  Bid: {tick.bid}")
    print(f"  Ask: {tick.ask}")
    print(f"  Spread: {(tick.ask - tick.bid) * 100:.1f} pips")
    print(f"  Point: {symbol_info.point}")
    print(f"  Digits: {symbol_info.digits}")
    print(f"  Lot Size: {symbol_info.volume_min} - {symbol_info.volume_max}")
    
    return symbol

def place_test_order(symbol, order_type, lot_size=0.01):
    """Place a test order"""
    print(f"\n" + "="*60)
    print(f"PLACING {order_type} ORDER")
    print("="*60)
    
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    symbol_info = mt5.symbol_info(symbol)
    
    if tick is None or symbol_info is None:
        print("ERROR: Failed to get symbol info")
        return None
    
    # Prepare order
    if order_type == "BUY":
        price = tick.ask
        sl = price - 30 * symbol_info.point * 100  # 30 pips below
        tp = price + 50 * symbol_info.point * 100  # 50 pips above
        order_type_mt5 = mt5.ORDER_TYPE_BUY
    else:  # SELL
        price = tick.bid
        sl = price + 30 * symbol_info.point * 100  # 30 pips above
        tp = price - 50 * symbol_info.point * 100  # 50 pips below
        order_type_mt5 = mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type_mt5,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "test_bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,  # Changed from IOC to FOK
    }
    
    print(f"\nOrder Details:")
    print(f"  Symbol: {symbol}")
    print(f"  Type: {order_type}")
    print(f"  Volume: {lot_size} lots")
    print(f"  Price: {price}")
    print(f"  SL: {sl} (30 pips)")
    print(f"  TP: {tp} (50 pips)")
    
    # Send order
    result = mt5.order_send(request)
    
    if result is None:
        print(f"\nERROR: order_send failed: {mt5.last_error()}")
        return None
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"\nERROR: Order failed with code {result.retcode}")
        print(f"  Comment: {result.comment}")
        return None
    
    print(f"\nSUCCESS: Order placed!")
    print(f"  Order: {result.order}")
    print(f"  Volume: {result.volume}")
    print(f"  Price: {result.price}")
    
    return result.order

def close_position(symbol, order_ticket):
    """Close a specific position - finds position by symbol and order"""
    print(f"\n" + "="*60)
    print(f"CLOSING POSITION (Order: {order_ticket})")
    print("="*60)
    
    # Get all positions for this symbol
    positions = mt5.positions_get(symbol=symbol)
    
    if positions is None or len(positions) == 0:
        print("ERROR: No open positions found for symbol")
        return False
    
    # Find the position (use first one for this symbol)
    position = positions[0]
    print(f"Found position ticket: {position.ticket}")
    
    # Prepare close request
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("ERROR: Failed to get tick")
        return False
    
    # Close opposite to open
    if position.type == mt5.ORDER_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,  # Use position ticket, not order ticket
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": "test_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,  # Changed from IOC to FOK
    }
    
    result = mt5.order_send(request)
    
    if result is None:
        print(f"ERROR: order_send failed: {mt5.last_error()}")
        return False
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"ERROR: Close failed with code {result.retcode}")
        print(f"  Comment: {result.comment}")
        return False
    
    print(f"\nSUCCESS: Position closed!")
    print(f"  Profit: ${position.profit:.2f}")
    
    return True

def main():
    print("\n" + "="*60)
    print("BOT TRADING TEST")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script will:")
    print("  1. Connect to MT5")
    print("  2. Place a small BUY order on USDJPY (0.01 lots)")
    print("  3. Place a small BUY order on EURUSD (0.01 lots)")
    print("  4. Wait 10 seconds")
    print("  5. Close both positions")
    print("\nNote: Using different symbols to avoid netting mode cancellation")
    print("\n" + "="*60)
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Test connection
    if not test_connection():
        print("\nConnection test failed. Exiting.")
        return
    
    # Test USDJPY symbol
    usdjpy = test_symbol("USDJPY.sim")
    if usdjpy is None:
        print("\nUSDJPY symbol test failed. Exiting.")
        mt5.shutdown()
        return
    
    # Test EURUSD symbol  
    eurusd = test_symbol("EURUSD.sim")
    if eurusd is None:
        print("\nEURUSD symbol test failed. Exiting.")
        mt5.shutdown()
        return
    
    # Place BUY order on USDJPY
    usdjpy_ticket = place_test_order(usdjpy, "BUY", 0.01)
    if usdjpy_ticket is None:
        print("\nUSDJPY BUY order failed. Exiting.")
        mt5.shutdown()
        return
    
    time.sleep(2)  # Wait 2 seconds
    
    # Place BUY order on EURUSD (using BUY for both to test different symbols)
    eurusd_ticket = place_test_order(eurusd, "BUY", 0.01)
    if eurusd_ticket is None:
        print("\nEURUSD BUY order failed. Closing USDJPY position...")
        close_position(usdjpy, usdjpy_ticket)
        mt5.shutdown()
        return
    
    print("\n" + "="*60)
    print("BOTH ORDERS PLACED SUCCESSFULLY!")
    print("="*60)
    print("\nWaiting 10 seconds before closing...")
    
    for i in range(10, 0, -1):
        print(f"  {i}...", end='\r')
        time.sleep(1)
    
    print("\nClosing positions...")
    
    # Close positions
    close_position(usdjpy, usdjpy_ticket)
    time.sleep(1)
    close_position(eurusd, eurusd_ticket)
    
    # Final account info
    print("\n" + "="*60)
    print("FINAL ACCOUNT INFO")
    print("="*60)
    
    account_info = mt5.account_info()
    if account_info:
        print(f"\nBalance: ${account_info.balance:.2f}")
        print(f"Equity: ${account_info.equity:.2f}")
    
    mt5.shutdown()
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print("\nIf you saw:")
    print("  - Connection successful")
    print("  - BUY order placed")
    print("  - SELL order placed")
    print("  - Both positions closed")
    print("\nThen your bot is working correctly!")
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user")
        mt5.shutdown()
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        mt5.shutdown()
