"""
Risk Manager Module for Live Trading
Provides comprehensive risk management controls
"""
import os
import time
from datetime import datetime
from dotenv import load_dotenv


class RiskManager:
    """Advanced risk management for live trading"""
    
    def __init__(self):
        load_dotenv()
        
        # Position limits
        self.max_positions = int(os.getenv('MAX_POSITIONS', 3))
        self.min_interval = int(os.getenv('MIN_INTERVAL_SECONDS', 300))
        
        # Stop loss & take profit
        self.stop_loss_pips = float(os.getenv('STOP_LOSS_PIPS', 30))
        self.take_profit_pips = float(os.getenv('TAKE_PROFIT_PIPS', 50))
        
        # Daily loss limit
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', 500))
        
        # Signal thresholds
        self.buy_threshold = float(os.getenv('BUY_THRESHOLD', 0.7))
        self.sell_threshold = float(os.getenv('SELL_THRESHOLD', 0.3))
        self.min_prob_diff = float(os.getenv('MIN_PROBABILITY_DIFF', 0.15))
        
        # Trading state
        self.last_trade_time = 0
        self.daily_loss = 0.0
        self.last_reset_date = datetime.now().date()
        
        print(f"✅ Risk Manager initialized:")
        print(f"   Max Positions: {self.max_positions}")
        print(f"   Min Interval: {self.min_interval}s")
        print(f"   Stop Loss: {self.stop_loss_pips} pips")
        print(f"   Take Profit: {self.take_profit_pips} pips")
        print(f"   Max Daily Loss: ${self.max_daily_loss}")
    
    def can_open_trade(self, mt5):
        """Check if we can open a new position"""
        # Reset daily loss at midnight
        self._reset_daily_if_needed()
        
        # Check 1: Daily loss limit
        if self.daily_loss >= self.max_daily_loss:
            print(f"❌ Daily loss limit reached: ${self.daily_loss:.2f} / ${self.max_daily_loss}")
            return False
        
        # Check 2: Max positions
        try:
            positions = mt5.positions_total()
        except:
            positions = 0
            
        if positions >= self.max_positions:
            print(f"⚠️ Max positions ({self.max_positions}) reached. Current: {positions}")
            return False
        
        # Check 3: Time interval
        current_time = time.time()
        time_since_last = current_time - self.last_trade_time
        if self.last_trade_time > 0 and time_since_last < self.min_interval:
            remaining = int(self.min_interval - time_since_last)
            print(f"⏳ Wait {remaining}s before next trade (min interval: {self.min_interval}s)")
            return False
        
        return True
    
    def update_last_trade_time(self):
        """Update timestamp of last trade"""
        self.last_trade_time = time.time()
    
    def record_trade_result(self, profit):
        """Record trade result for daily tracking"""
        if profit < 0:
            self.daily_loss += abs(profit)
            print(f"📉 Daily loss updated: ${self.daily_loss:.2f} / ${self.max_daily_loss}")
    
    def _reset_daily_if_needed(self):
        """Reset daily counters at midnight"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            print(f"🔄 Resetting daily counters (new day: {today})")
            self.daily_loss = 0.0
            self.last_reset_date = today
    
    def is_signal_strong(self, probability, signal_type):
        """Check if signal is strong enough to trade"""
        if signal_type == 'BUY':
            is_strong = probability >= self.buy_threshold
            diff = probability - 0.5
        elif signal_type == 'SELL':
            is_strong = probability <= self.sell_threshold
            diff = 0.5 - probability
        else:
            return False
        
        # Additional check: probability should be significantly different from 0.5
        if diff < self.min_prob_diff:
            print(f"⚠️ Signal too weak (diff: {diff:.3f} < {self.min_prob_diff})")
            return False
        
        return is_strong
    
    def calculate_sl_tp(self, entry_price, signal_type, symbol='USDJPY'):
        """Calculate stop loss and take profit prices"""
        # For USDJPY: 1 pip = 0.01
        pip_value = 0.01
        
        if signal_type == 'BUY':
            sl = entry_price - (self.stop_loss_pips * pip_value)
            tp = entry_price + (self.take_profit_pips * pip_value)
        elif signal_type == 'SELL':
            sl = entry_price + (self.stop_loss_pips * pip_value)
            tp = entry_price - (self.take_profit_pips * pip_value)
        else:
            return None, None
        
        return round(sl, 3), round(tp, 3)
    
    def get_position_summary(self, mt5):
        """Get summary of current positions"""
        try:
            positions = mt5.positions_get()
        except:
            return "No open positions (MT5 not connected)"
        
        if not positions:
            return "No open positions"
        
        summary = f"\n📊 Open Positions: {len(positions)}"
        total_profit = 0.0
        
        for pos in positions:
            total_profit += pos.profit
            type_str = "BUY" if pos.type == 0 else "SELL"
            summary += f"\n   {type_str} {pos.volume} lots @ {pos.price_open} | P/L: ${pos.profit:.2f}"
        
        summary += f"\n💰 Total P/L: ${total_profit:.2f}"
        return summary
