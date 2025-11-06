"""
Position sizing strategies for risk management.

Implements:
1. Fixed fractional (risk % of equity per trade)
2. Kelly criterion
3. Volatility-adjusted sizing
"""
import numpy as np


class PositionSizer:
    """Calculate position sizes based on various strategies."""
    
    def __init__(self, strategy='fixed_fractional', risk_pct=0.01, max_risk_pct=0.02):
        """
        Parameters:
        -----------
        strategy : str
            'fixed_fractional', 'kelly', or 'volatility_adjusted'
        risk_pct : float
            Risk per trade as fraction of equity (e.g., 0.01 = 1%)
        max_risk_pct : float
            Maximum risk per trade (safety cap)
        """
        self.strategy = strategy
        self.risk_pct = risk_pct
        self.max_risk_pct = max_risk_pct
    
    def calculate_size(self, equity, stop_loss_pips, pip_value, 
                      win_rate=None, avg_win=None, avg_loss=None,
                      volatility=None):
        """
        Calculate position size.
        
        Parameters:
        -----------
        equity : float
            Current account equity
        stop_loss_pips : float
            Stop loss distance in pips
        pip_value : float
            Value of 1 pip (e.g., 0.0001 for USDJPY)
        win_rate : float (optional)
            Win rate for Kelly criterion
        avg_win : float (optional)
            Average win for Kelly criterion
        avg_loss : float (optional)
            Average loss for Kelly criterion
        volatility : float (optional)
            Current volatility measure for volatility_adjusted
            
        Returns:
        --------
        position_size : float
            Position size in lots
        """
        if self.strategy == 'fixed_fractional':
            return self._fixed_fractional(equity, stop_loss_pips, pip_value)
        
        elif self.strategy == 'kelly':
            return self._kelly_criterion(equity, stop_loss_pips, pip_value,
                                        win_rate, avg_win, avg_loss)
        
        elif self.strategy == 'volatility_adjusted':
            return self._volatility_adjusted(equity, stop_loss_pips, pip_value, volatility)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _fixed_fractional(self, equity, stop_loss_pips, pip_value):
        """
        Fixed fractional position sizing.
        Risk a fixed % of equity per trade.
        
        Parameters:
        -----------
        pip_value : float
            Pip value in USD per standard lot (100k units)
            e.g., $10 for EURUSD, $6.67 for USDJPY @ 150
        """
        # Calculate risk amount in currency
        risk_amount = equity * self.risk_pct
        
        # Calculate position size
        # risk_amount = position_size * stop_loss_pips * pip_value
        # position_size = risk_amount / (stop_loss_pips * pip_value)
        
        if stop_loss_pips > 0 and pip_value > 0:
            position_size = risk_amount / (stop_loss_pips * pip_value)
        else:
            position_size = 0.01  # Minimum
        
        # Cap at maximum risk
        max_position = (equity * self.max_risk_pct) / (stop_loss_pips * pip_value) if stop_loss_pips > 0 and pip_value > 0 else 0.01
        position_size = min(position_size, max_position)
        
        # Minimum 0.01 lots, round to 2 decimals
        position_size = max(0.01, round(position_size, 2))
        
        return position_size
    
    def _kelly_criterion(self, equity, stop_loss_pips, pip_value,
                        win_rate, avg_win, avg_loss):
        """
        Kelly criterion position sizing.
        
        Formula: f = (p*W - (1-p)*L) / W
        where:
            f = fraction of capital to risk
            p = win rate
            W = average win
            L = average loss
        """
        if win_rate is None or avg_win is None or avg_loss is None:
            # Fall back to fixed fractional
            return self._fixed_fractional(equity, stop_loss_pips, pip_value)
        
        if avg_win <= 0:
            return 0.01
        
        # Kelly fraction
        kelly_f = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) / avg_win
        
        # Use half-Kelly for safety (less aggressive)
        kelly_f = kelly_f * 0.5
        
        # Cap Kelly fraction
        kelly_f = max(0, min(kelly_f, self.max_risk_pct))
        
        # Calculate position size
        if stop_loss_pips > 0:
            position_size = (equity * kelly_f) / (stop_loss_pips * 100000)
        else:
            position_size = 0.01
        
        # Minimum 0.01 lots, round to 2 decimals
        position_size = max(0.01, round(position_size, 2))
        
        return position_size
    
    def _volatility_adjusted(self, equity, stop_loss_pips, pip_value, volatility):
        """
        Volatility-adjusted position sizing.
        Reduce position size in high volatility, increase in low volatility.
        """
        if volatility is None:
            return self._fixed_fractional(equity, stop_loss_pips, pip_value)
        
        # Normalize volatility (assume average volatility = 1.0)
        # Higher volatility  smaller position
        # Lower volatility  larger position
        volatility_adjustment = 1.0 / max(volatility, 0.5)
        volatility_adjustment = min(volatility_adjustment, 2.0)  # Cap at 2x
        
        # Adjusted risk
        adjusted_risk_pct = self.risk_pct * volatility_adjustment
        adjusted_risk_pct = min(adjusted_risk_pct, self.max_risk_pct)
        
        # Calculate position size
        risk_amount = equity * adjusted_risk_pct
        
        if stop_loss_pips > 0:
            position_size = risk_amount / (stop_loss_pips * 100000)
        else:
            position_size = 0.01
        
        # Minimum 0.01 lots, round to 2 decimals
        position_size = max(0.01, round(position_size, 2))
        
        return position_size


def test_position_sizer():
    """Test position sizing strategies."""
    print("\n" + "="*60)
    print("POSITION SIZING TEST")
    print("="*60 + "\n")
    
    equity = 100000  # $100k account
    stop_loss_pips = 50  # 50 pip stop
    pip_value = 0.01  # USDJPY
    
    # Test different strategies
    strategies = [
        ('Fixed Fractional (1%)', PositionSizer('fixed_fractional', risk_pct=0.01)),
        ('Fixed Fractional (2%)', PositionSizer('fixed_fractional', risk_pct=0.02)),
        ('Kelly Criterion', PositionSizer('kelly', risk_pct=0.01)),
        ('Volatility Adjusted (Low Vol)', PositionSizer('volatility_adjusted', risk_pct=0.01)),
        ('Volatility Adjusted (High Vol)', PositionSizer('volatility_adjusted', risk_pct=0.01)),
    ]
    
    print(f"Account Equity: ${equity:,.0f}")
    print(f"Stop Loss: {stop_loss_pips} pips")
    print(f"Pip Value: {pip_value}\n")
    
    print(f"{'Strategy':<35} {'Position Size':<15} {'Risk $':<15} {'Risk %':<10}")
    print(""*75)
    
    for name, sizer in strategies:
        if 'Kelly' in name:
            # Use sample stats
            size = sizer.calculate_size(equity, stop_loss_pips, pip_value,
                                       win_rate=0.55, avg_win=1000, avg_loss=800)
        elif 'Low Vol' in name:
            size = sizer.calculate_size(equity, stop_loss_pips, pip_value,
                                       volatility=0.7)
        elif 'High Vol' in name:
            size = sizer.calculate_size(equity, stop_loss_pips, pip_value,
                                       volatility=1.5)
        else:
            size = sizer.calculate_size(equity, stop_loss_pips, pip_value)
        
        risk_amount = size * stop_loss_pips * 100000
        risk_pct = risk_amount / equity * 100
        
        print(f"{name:<35} {size:>6.2f} lots      ${risk_amount:>10,.0f}      {risk_pct:>5.1f}%")
    
    print("\n" + "="*60)
    print("COMPARISON vs FIXED 1 LOT:")
    print("="*60 + "\n")
    
    fixed_1_lot_risk = 1.0 * stop_loss_pips * 100000
    fixed_1_lot_pct = fixed_1_lot_risk / equity * 100
    
    print(f"Fixed 1 lot: ${fixed_1_lot_risk:,.0f} risk ({fixed_1_lot_pct:.1f}% of equity)")
    print(f"\nPROBLEM: Risking {fixed_1_lot_pct:.1f}% per trade is WAY too high!")
    print(f"With 50 pip SL, one trade risks ${fixed_1_lot_risk:,.0f}")
    print(f"\n SOLUTION: Use 0.{int(equity * 0.01 / (stop_loss_pips * 100000) * 100):02d} lots (1% risk) = ${equity * 0.01:,.0f} risk")
    print("="*60 + "\n")


if __name__ == '__main__':
    test_position_sizer()

