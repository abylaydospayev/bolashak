"""
Real-time monitoring dashboard for demo bot.

Shows:
- Current account status
- Open positions
- Recent trades
- Performance metrics
- Model predictions

Usage:
    .\.venv\Scripts\python.exe live_trading\monitor.py
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from live_trading.mt5_client import init_mt5, shutdown_mt5, get_account_info, get_open_positions, get_trade_history

# Load environment variables
load_dotenv(Path(__file__).parent / '.env')

TERMINAL_PATH = os.getenv('MT5_TERMINAL_PATH')
LOGIN = int(os.getenv('MT5_LOGIN', 0))
PASSWORD = os.getenv('MT5_PASSWORD')
SERVER = os.getenv('MT5_SERVER')

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def display_dashboard(account, positions, trades):
    """Display trading dashboard."""
    clear_screen()
    
    print("=" * 80)
    print(f"USDJPY DEMO BOT - LIVE MONITOR".center(80))
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("=" * 80)
    
    # Account Info
    print("\n ACCOUNT STATUS")
    print("-" * 80)
    if account:
        balance = account.balance
        equity = account.equity
        margin_used = account.margin
        margin_free = account.margin_free
        pl = equity - balance
        
        print(f"  Balance:        ${balance:>12,.2f}")
        print(f"  Equity:         ${equity:>12,.2f}")
        print(f"  P&L:            ${pl:>12,.2f}  ({(pl/balance*100):>6.2f}%)")
        print(f"  Margin Used:    ${margin_used:>12,.2f}")
        print(f"  Margin Free:    ${margin_free:>12,.2f}")
    else:
        print("  Unable to fetch account info")
    
    # Open Positions
    print("\n OPEN POSITIONS")
    print("-" * 80)
    if positions and len(positions) > 0:
        for pos in positions:
            side = "BUY" if pos.type == 0 else "SELL"
            pl_color = "" if pos.profit > 0 else ""
            print(f"  {pl_color} {pos.symbol:<10} {side:<6} {pos.volume:>8.2f} lots @ {pos.price_open:>10.5f}")
            print(f"     Current: {pos.price_current:>10.5f}  |  P&L: ${pos.profit:>10.2f}  |  Ticket: {pos.ticket}")
    else:
        print("  No open positions")
    
    # Recent Trades
    print("\n RECENT TRADES (Last 10)")
    print("-" * 80)
    if trades and len(trades) > 0:
        wins = [t for t in trades if t.profit > 0]
        losses = [t for t in trades if t.profit <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        
        print(f"  Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
        print()
        
        for trade in trades[:10]:
            side = "BUY" if trade.type == 0 else "SELL"
            result = " WIN" if trade.profit > 0 else " LOSS"
            print(f"  {result} {trade.symbol:<10} {side:<6} {trade.volume:>8.2f} lots  |  P&L: ${trade.profit:>10.2f}")
            print(f"     Entry: {trade.price_open:>10.5f}  |  Exit: {trade.price_current:>10.5f}  |  {trade.time_update}")
            print()
    else:
        print("  No trade history available")
    
    print("\n" + "=" * 80)
    print("Press Ctrl+C to exit".center(80))
    print("=" * 80)


def main():
    """Main monitoring loop."""
    print("Initializing MT5...")
    if not init_mt5(TERMINAL_PATH, LOGIN, PASSWORD, SERVER):
        print(" Failed to initialize MT5")
        return
    
    print(" Connected to MT5")
    print("Starting monitor... (updates every 5 seconds)")
    time.sleep(2)
    
    try:
        while True:
            # Fetch current data
            account = get_account_info()
            positions = get_open_positions()
            
            # Get recent trades (note: MT5 API doesn't have this exact method, using positions history)
            trades = []  # Placeholder - would need to implement trade history fetching
            
            # Display dashboard
            display_dashboard(account, positions, trades)
            
            # Wait before refresh
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n\n  Monitor stopped")
    
    finally:
        shutdown_mt5()


if __name__ == '__main__':
    main()

