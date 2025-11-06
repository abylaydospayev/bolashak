"""
MetaTrader5 client wrapper for live trading.

Usage:
- Set environment variables (see .env.example) or export in PowerShell
- Ensure MetaTrader 5 terminal is installed and the terminal.exe path is provided
- Run `python live_trading/test_mt5_connection.py` to validate

This module exposes:
- init_mt5(terminal_path, login, password, server)
- shutdown_mt5()
- get_bars(symbol, timeframe, n)
- place_market_order(symbol, lot, sl=None, tp=None, deviation=20)
- get_open_positions()
- close_position_by_ticket(ticket)

Notes:
- The `MetaTrader5` Python package is required: `pip install MetaTrader5`
- The terminal must be running or the terminal path must be correct for `mt5.initialize()` to work.
"""
import os
import time
import math
from datetime import datetime

try:
    import MetaTrader5 as mt5
except Exception as e:
    mt5 = None
    print("MetaTrader5 package not available. Install with: pip install MetaTrader5")


def init_mt5(terminal_path: str, login: int, password: str, server: str) -> bool:
    """Initialize MT5 terminal connection.

    Parameters:
    - terminal_path: full path to terminal64.exe
    - login: account number (int)
    - password: account password
    - server: broker server name

    Returns True if initialized.
    """
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed")

    # Try initialize with terminal path
    if terminal_path:
        ok = mt5.initialize(terminal_path=terminal_path, login=login, password=password, server=server)
    else:
        ok = mt5.initialize(login=login, password=password, server=server)

    if not ok:
        last_error = mt5.last_error()
        print(f"Failed to initialize MT5: {last_error}")
        return False

    print("✅ MT5 initialized")
    # Wait a moment for connection
    time.sleep(1)
    return True


def shutdown_mt5():
    if mt5 is None:
        return
    try:
        mt5.shutdown()
        print("✅ MT5 shutdown")
    except Exception as e:
        print(f"Error shutting down MT5: {e}")


def get_account_info():
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed")
    info = mt5.account_info()
    return info


def get_bars(symbol: str, timeframe: str = 'M15', count: int = 500):
    """Fetch recent bars. timeframe strings: 'M1','M5','M15','M30','H1','H4','D'
    Returns pandas.DataFrame with time,indexed by datetime and columns open,high,low,close,volume
    """
    import pandas as pd

    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed")

    tf_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D': mt5.TIMEFRAME_D1
    }

    tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M15)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time','open','high','low','close','tick_volume']]
    df = df.rename(columns={'tick_volume':'volume'})
    df.set_index('time', inplace=True)
    return df


def place_market_order(symbol: str, lot: float, sl: float = None, tp: float = None, deviation: int = 20):
    """Place a market order.

    Parameters:
    - symbol: e.g., 'USDJPY'
    - lot: number of lots (0.01, 0.1, 1.0)
    - sl/tp: absolute price levels (not pips)
    - deviation: maximum slippage in points
    """
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed")

    # Determine direction
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        raise RuntimeError(f"Symbol {symbol} not found in Market Watch")

    # Ensure symbol is enabled for trading
    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    point = symbol_info.point
    price = mt5.symbol_info_tick(symbol).ask

    # units in lots need to be converted to volume (volume field in lots)
    volume = float(lot)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": deviation,
        "magic": 424242,
        "comment": "python_mt5_order",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    if sl is not None:
        request['sl'] = sl
    if tp is not None:
        request['tp'] = tp

    result = mt5.order_send(request)
    if result is None:
        print("Order send returned None")
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE and result.retcode != 10009:
        print(f"Order failed, retcode={result.retcode}, result={result}")
        return result

    print(f"✅ Order placed: ticket={result.order}, volume={volume}, symbol={symbol}")
    return result


def get_open_positions():
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed")
    positions = mt5.positions_get()
    return positions


def close_position_by_ticket(ticket: int):
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed")

    position = mt5.positions_get(ticket=ticket)
    if position is None or len(position) == 0:
        print(f"No position with ticket={ticket}")
        return None

    pos = position[0]
    symbol = pos.symbol
    volume = pos.volume
    side = pos.type  # 0=BUY,1=SELL

    if side == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": 424242,
        "comment": "python_mt5_close",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    print(f"Close order result: {result}")
    return result


if __name__ == '__main__':
    print("mt5_client module loaded. Use functions from other scripts to operate MT5.")
