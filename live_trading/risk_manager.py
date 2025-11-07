"""
Risk Manager Module for Live Trading
ATR-based volatility adaptive risk management with fixed % risk per trade
"""
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import MetaTrader5 as mt5


class RiskManager:
    """ATR-based adaptive risk management for live trading"""
    
    def __init__(self):
        load_dotenv()
        
        # === ATR-BASED VOLATILITY SETTINGS ===
        self.atr_enabled = os.getenv('ATR_VOLATILITY_ADJUSTMENT', 'true').lower() == 'true'
        self.atr_period = int(os.getenv('ATR_PERIOD', 14))
        self.atr_baseline_pips = float(os.getenv('ATR_BASELINE_PIPS', 30))  # Baseline ATR (pips)
        
        # SL/TP limits (pips)
        self.min_sl_pips = float(os.getenv('MIN_SL_PIPS', 20))
        self.max_sl_pips = float(os.getenv('MAX_SL_PIPS', 80))
        self.risk_reward_ratio = float(os.getenv('RISK_REWARD_RATIO', 1.6))  # TP = SL * 1.6
        
        # Fixed % risk per trade
        self.risk_percent_per_trade = float(os.getenv('RISK_PERCENT_PER_TRADE', 0.5))  # 0.5% default
        self.pip_value_per_lot = float(os.getenv('PIP_VALUE_PER_LOT', 9.17))  # USDJPY ≈ $9.17/pip/lot
        
        # Lot size limits
        self.min_lot_size = float(os.getenv('MIN_LOT_SIZE', 0.10))
        self.max_lot_size = float(os.getenv('MAX_LOT_SIZE', 1.50))
        self.max_total_lots = float(os.getenv('MAX_TOTAL_LOTS', 3.0))  # Across all positions
        
        # Extreme volatility protection
        self.extreme_vol_factor = float(os.getenv('EXTREME_VOL_FACTOR', 2.5))  # Skip if ATR > 2.5x baseline
        self.high_vol_factor = float(os.getenv('HIGH_VOL_FACTOR', 2.0))  # Half size if ATR > 2.0x
        
        # Position limits
        self.max_positions = int(os.getenv('MAX_POSITIONS', 3))
        self.min_interval = int(os.getenv('MIN_INTERVAL_SECONDS', 900))  # 15 min default spacing
        
        # Daily loss limits (FTMO safe)
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', 4948))  # Hard stop
        self.daily_loss_soft_stop_pct = float(os.getenv('DAILY_LOSS_SOFT_STOP_PCT', 3.0))  # 3% soft stop
        
        # Signal thresholds
        self.buy_threshold = float(os.getenv('BUY_THRESHOLD', 0.7))
        self.sell_threshold = float(os.getenv('SELL_THRESHOLD', 0.3))
        self.min_prob_diff = float(os.getenv('MIN_PROBABILITY_DIFF', 0.15))
        
        # Trading state
        self.last_trade_time = 0
        self.daily_loss = 0.0
        self.last_reset_date = datetime.now().date()
        self.current_atr = None
        self.current_sl_pips = self.atr_baseline_pips
        self.current_tp_pips = self.atr_baseline_pips * self.risk_reward_ratio
        self.current_lot_size = None
        
        print(f"[RISK] Risk Manager initialized:")
        print(f"   ATR-Based Volatility: {'[ENABLED]' if self.atr_enabled else '[DISABLED]'}")
        if self.atr_enabled:
            print(f"   ATR Period: {self.atr_period}")
            print(f"   ATR Baseline: {self.atr_baseline_pips} pips")
            print(f"   SL Range: {self.min_sl_pips}-{self.max_sl_pips} pips")
            print(f"   Risk:Reward Ratio: 1:{self.risk_reward_ratio}")
            print(f"   Risk per Trade: {self.risk_percent_per_trade}%")
            print(f"   Lot Range: {self.min_lot_size}-{self.max_lot_size}")
            print(f"   Max Total Lots: {self.max_total_lots}")
        print(f"   Max Positions: {self.max_positions}")
        print(f"   Min Interval: {self.min_interval}s")
        print(f"   Daily Loss: Soft {self.daily_loss_soft_stop_pct}% / Hard ${self.max_daily_loss}")
    
    def calculate_atr(self, symbol='USDJPY', timeframe=mt5.TIMEFRAME_H1, period=14):
        """Calculate ATR in pips for the given symbol and timeframe
        
        Args:
            symbol: Trading symbol
            timeframe: MT5 timeframe constant (default H1)
            period: ATR period (default 14)
            
        Returns:
            ATR value in pips, or None if calculation fails
        """
        try:
            # Get recent bars
            bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
            if bars is None or len(bars) < period + 1:
                return None
            
            # Calculate True Range for each bar
            tr_values = []
            for i in range(1, len(bars)):
                high = bars[i]['high']
                low = bars[i]['low']
                prev_close = bars[i-1]['close']
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                tr_values.append(tr)
            
            # ATR is simple moving average of TR
            atr = sum(tr_values[-period:]) / period
            
            # Convert to pips (for USDJPY: 1 pip = 0.01)
            atr_pips = atr / 0.01
            
            return round(atr_pips, 2)
            
        except Exception as e:
            print(f"[WARNING] ATR calculation failed: {e}")
            return None
    
    def adjust_for_volatility(self, account_balance, symbol='USDJPY'):
        """Calculate position size, SL, and TP based on ATR volatility
        
        This is the core volatility adaptation logic:
        1. Calculate ATR from H1 chart
        2. Compute SL/TP based on ATR (maintains R:R ratio)
        3. Calculate lot size to risk fixed % of balance
        4. Apply extreme volatility protection
        
        Args:
            account_balance: Current account balance
            symbol: Trading symbol
            
        Returns:
            tuple: (lot_size, sl_pips, tp_pips, atr_pips, vol_factor) or None if extreme volatility
        """
        if not self.atr_enabled:
            # Fallback to baseline values
            return (
                self.max_lot_size,
                self.atr_baseline_pips,
                round(self.atr_baseline_pips * self.risk_reward_ratio),
                self.atr_baseline_pips,
                1.0
            )
        
        # Calculate current ATR
        atr_pips = self.calculate_atr(symbol, mt5.TIMEFRAME_H1, self.atr_period)
        if atr_pips is None:
            print("[WARNING] Using baseline values (ATR unavailable)")
            atr_pips = self.atr_baseline_pips
        
        # Calculate volatility factor
        vol_factor = atr_pips / self.atr_baseline_pips
        
        # === 1) STOP LOSS FROM ATR ===
        # SL scales with volatility but is clamped to safe limits
        sl_pips = round(self.atr_baseline_pips * vol_factor)
        sl_pips = max(self.min_sl_pips, min(self.max_sl_pips, sl_pips))
        
        # === 2) TAKE PROFIT FROM SL ===
        # Maintain consistent R:R ratio
        tp_pips = round(sl_pips * self.risk_reward_ratio)
        
        # === 3) LOT SIZE FROM FIXED % RISK ===
        # risk_$ = balance * risk%
        # lot_size = risk_$ / (SL_pips * pip_value_per_lot)
        risk_dollars = account_balance * (self.risk_percent_per_trade / 100.0)
        lot_size = risk_dollars / (sl_pips * self.pip_value_per_lot)
        
        # Clamp lot size to broker/risk limits
        lot_size = max(self.min_lot_size, min(self.max_lot_size, round(lot_size, 2)))
        
        # === 4) EXTREME VOLATILITY PROTECTION ===
        if vol_factor >= self.extreme_vol_factor:
            print(f"[ALERT] EXTREME VOLATILITY: ATR {atr_pips:.1f} pips ({vol_factor:.2f}x baseline)")
            print(f"   Skipping new trades until volatility normalizes")
            return None
        
        # High volatility: reduce size by 50%
        if vol_factor >= self.high_vol_factor:
            lot_size = max(self.min_lot_size, round(lot_size * 0.5, 2))
            print(f"[WARNING] HIGH VOLATILITY: ATR {atr_pips:.1f} pips ({vol_factor:.2f}x) - Halving position size")
        
        # Store current state
        self.current_atr = atr_pips
        self.current_sl_pips = sl_pips
        self.current_tp_pips = tp_pips
        self.current_lot_size = lot_size
        
        # Print volatility report
        vol_status = "[LOW]" if vol_factor < 1.2 else "[NORMAL]" if vol_factor < 1.8 else "[HIGH]"
        print(f"[VOL] Volatility Analysis:")
        print(f"   Status: {vol_status} | ATR: {atr_pips:.1f} pips (factor: {vol_factor:.2f}x)")
        print(f"   Stop Loss: {sl_pips} pips")
        print(f"   Take Profit: {tp_pips} pips (R:R = 1:{tp_pips/sl_pips:.2f})")
        print(f"   Position Size: {lot_size} lots (risk: ${risk_dollars:.2f} = {self.risk_percent_per_trade}%)")
        
        return lot_size, sl_pips, tp_pips, atr_pips, vol_factor
    
    def get_lot_size(self):
        """Get current lot size (after volatility adjustment)"""
        return self.current_lot_size if self.current_lot_size else self.max_lot_size
    
    def can_open_trade(self, mt5_client, account_balance):
        """Check if we can open a new position
        
        Args:
            mt5_client: MT5 connection object
            account_balance: Current account balance
            
        Returns:
            bool: True if can trade, False otherwise
        """
        # Reset daily loss at midnight
        self._reset_daily_if_needed()
        
        # === 1) DAILY LOSS CHECKS ===
        # Hard stop: max daily loss in dollars
        if self.daily_loss >= self.max_daily_loss:
            print(f"[STOP] HARD STOP: Daily loss limit ${self.daily_loss:.2f} / ${self.max_daily_loss}")
            return False
        
        # Soft stop: max daily loss as % of balance
        daily_loss_pct = (self.daily_loss / account_balance) * 100.0
        if daily_loss_pct >= self.daily_loss_soft_stop_pct:
            print(f"[STOP] SOFT STOP: Daily loss {daily_loss_pct:.2f}% >= {self.daily_loss_soft_stop_pct}%")
            print(f"   Pausing new trades to protect capital")
            return False
        
        # === 2) EXPOSURE LIMITS ===
        try:
            positions = mt5_client.positions_get()
            if positions is None:
                positions_count = 0
                total_volume = 0.0
            else:
                positions_count = len(positions)
                total_volume = sum(pos.volume for pos in positions)
        except:
            positions_count = 0
            total_volume = 0.0
        
        # Check max positions
        if positions_count >= self.max_positions:
            print(f"[STOP] Max positions ({self.max_positions}) reached")
            return False
        
        # Check total volume across all positions
        if total_volume >= self.max_total_lots:
            print(f"[STOP] Max total volume ({self.max_total_lots} lots) reached. Current: {total_volume:.2f}")
            return False
        
        # === 3) TIME INTERVAL CHECK ===
        current_time = time.time()
        time_since_last = current_time - self.last_trade_time
        if self.last_trade_time > 0 and time_since_last < self.min_interval:
            remaining = int(self.min_interval - time_since_last)
            print(f"[WAIT] Wait {remaining}s before next trade (min spacing: {self.min_interval}s)")
            return False
        
        return True
    
    def update_last_trade_time(self):
        """Update timestamp of last trade"""
        self.last_trade_time = time.time()
    
    def record_trade_result(self, profit):
        """Record trade result for daily tracking"""
        if profit < 0:
            self.daily_loss += abs(profit)
            print(f"[LOSS] Daily loss updated: ${self.daily_loss:.2f} / ${self.max_daily_loss}")
    
    def _reset_daily_if_needed(self):
        """Reset daily counters at midnight"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            print(f"[RESET] Resetting daily counters (new day: {today})")
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
            print(f"[WARNING] Signal too weak (diff: {diff:.3f} < {self.min_prob_diff})")
            return False
        
        return is_strong
    
    def calculate_sl_tp(self, entry_price, signal_type, symbol='USDJPY'):
        """Calculate stop loss and take profit prices using current ATR-adjusted pips
        
        Args:
            entry_price: Entry price for the trade
            signal_type: 'BUY' or 'SELL'
            symbol: Trading symbol
            
        Returns:
            tuple: (sl_price, tp_price)
        """
        # For USDJPY: 1 pip = 0.01
        pip_value = 0.01
        
        if signal_type == 'BUY':
            sl = entry_price - (self.current_sl_pips * pip_value)
            tp = entry_price + (self.current_tp_pips * pip_value)
        elif signal_type == 'SELL':
            sl = entry_price + (self.current_sl_pips * pip_value)
            tp = entry_price - (self.current_tp_pips * pip_value)
        else:
            return None, None
        
        return round(sl, 3), round(tp, 3)
    
    def get_position_summary(self, mt5_client):
        """Get summary of current positions"""
        try:
            positions = mt5_client.positions_get()
        except:
            return "No open positions (MT5 not connected)"
        
        if not positions:
            return "No open positions"
        
        summary = f"\n[POS] Open Positions: {len(positions)}"
        total_profit = 0.0
        total_volume = 0.0
        
        for pos in positions:
            total_profit += pos.profit
            total_volume += pos.volume
            type_str = "BUY" if pos.type == 0 else "SELL"
            summary += f"\n   {type_str} {pos.volume} lots @ {pos.price_open} | P/L: ${pos.profit:.2f}"
        
        summary += f"\n[P/L] Total P/L: ${total_profit:.2f}"
        summary += f"\n[VOL] Total Volume: {total_volume:.2f} / {self.max_total_lots} lots"
        return summary

