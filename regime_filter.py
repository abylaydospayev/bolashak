"""
Regime Filter - Only trade in favorable market conditions.

Based on analysis of USDJPY walk-forward results:
- Fold 1 (profitable): RSI 49.4, ATR 0.097, Return -0.38%
- Correlations: Return% (-0.709), ATR (-0.647), RSI (-0.502) with PnL

Profitable regime characteristics:
1. Neutral momentum (RSI 45-55)
2. Normal volatility (ATR < 0.11)
3. No strong trend (abs(return) < 2%)
"""
import pandas as pd
import numpy as np
from pathlib import Path


class RegimeFilter:
    """
    Filter to detect favorable trading regimes.
    Only allows trading when market conditions match profitable characteristics.
    """
    
    def __init__(self, 
                 rsi_lower=45, 
                 rsi_upper=55,
                 atr_max=0.11,
                 return_threshold=0.02,
                 lookback_periods=100):
        """
        Parameters:
        -----------
        rsi_lower : float
            Minimum RSI (below = oversold, skip)
        rsi_upper : float
            Maximum RSI (above = overbought, skip)
        atr_max : float
            Maximum ATR (above = too volatile, skip)
        return_threshold : float
            Maximum absolute return over lookback (above = strong trend, skip)
        lookback_periods : int
            Periods to calculate return over
        """
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.atr_max = atr_max
        self.return_threshold = return_threshold
        self.lookback_periods = lookback_periods
        
        # Statistics tracking
        self.total_checks = 0
        self.passed_checks = 0
        self.rejection_reasons = {
            'high_rsi': 0,
            'low_rsi': 0,
            'high_atr': 0,
            'strong_trend': 0,
            'passed': 0
        }
    
    def should_trade(self, df, verbose=False):
        """
        Check if current market conditions are favorable for trading.
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV data with indicators (must have 'rsi14', 'atr14', 'close')
        verbose : bool
            If True, print rejection reason
            
        Returns:
        --------
        should_trade : bool
            True if conditions are favorable
        reason : str
            Explanation of decision
        """
        self.total_checks += 1
        
        # Get latest values (handle both rsi14 and rsi_14 naming)
        rsi_col = 'rsi14' if 'rsi14' in df.columns else 'rsi_14'
        atr_col = 'atr14' if 'atr14' in df.columns else 'atr_14'
        
        current_rsi = df[rsi_col].iloc[-1]
        current_atr = df[atr_col].iloc[-1]
        
        # Calculate return over lookback
        if len(df) >= self.lookback_periods:
            price_now = df['close'].iloc[-1]
            price_then = df['close'].iloc[-self.lookback_periods]
            return_pct = (price_now / price_then) - 1
        else:
            return_pct = 0.0
        
        # Check 1: RSI in neutral range?
        if current_rsi > self.rsi_upper:
            self.rejection_reasons['high_rsi'] += 1
            reason = f"SKIP: RSI too high ({current_rsi:.1f} > {self.rsi_upper})"
            if verbose:
                print(reason)
            return False, reason
        
        if current_rsi < self.rsi_lower:
            self.rejection_reasons['low_rsi'] += 1
            reason = f"SKIP: RSI too low ({current_rsi:.1f} < {self.rsi_lower})"
            if verbose:
                print(reason)
            return False, reason
        
        # Check 2: ATR within normal range?
        if current_atr > self.atr_max:
            self.rejection_reasons['high_atr'] += 1
            reason = f"SKIP: ATR too high ({current_atr:.4f} > {self.atr_max})"
            if verbose:
                print(reason)
            return False, reason
        
        # Check 3: No strong trend?
        if abs(return_pct) > self.return_threshold:
            self.rejection_reasons['strong_trend'] += 1
            reason = f"SKIP: Strong trend ({return_pct*100:+.2f}% > {self.return_threshold*100:.1f}%)"
            if verbose:
                print(reason)
            return False, reason
        
        # All checks passed!
        self.passed_checks += 1
        self.rejection_reasons['passed'] += 1
        reason = f"TRADE: RSI={current_rsi:.1f}, ATR={current_atr:.4f}, Return={return_pct*100:+.2f}%"
        
        if verbose:
            print(reason)
        
        return True, reason
    
    def get_statistics(self):
        """
        Get statistics on regime filtering.
        
        Returns:
        --------
        stats : dict
            Dictionary with filtering statistics
        """
        pass_rate = self.passed_checks / self.total_checks if self.total_checks > 0 else 0
        
        stats = {
            'total_checks': self.total_checks,
            'passed': self.passed_checks,
            'rejected': self.total_checks - self.passed_checks,
            'pass_rate': pass_rate,
            'rejection_breakdown': self.rejection_reasons.copy()
        }
        
        return stats
    
    def print_statistics(self):
        """Print human-readable statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("REGIME FILTER STATISTICS")
        print("="*60)
        print(f"Total Checks:        {stats['total_checks']:,}")
        print(f"Passed (Trade):      {stats['passed']:,} ({stats['pass_rate']*100:.1f}%)")
        print(f"Rejected (Skip):     {stats['rejected']:,} ({(1-stats['pass_rate'])*100:.1f}%)")
        print("\nRejection Breakdown:")
        print(f"  - High RSI:        {stats['rejection_breakdown']['high_rsi']:,}")
        print(f"  - Low RSI:         {stats['rejection_breakdown']['low_rsi']:,}")
        print(f"  - High ATR:        {stats['rejection_breakdown']['high_atr']:,}")
        print(f"  - Strong Trend:    {stats['rejection_breakdown']['strong_trend']:,}")
        print("="*60)
    
    def reset_statistics(self):
        """Reset tracking statistics."""
        self.total_checks = 0
        self.passed_checks = 0
        self.rejection_reasons = {
            'high_rsi': 0,
            'low_rsi': 0,
            'high_atr': 0,
            'strong_trend': 0,
            'passed': 0
        }


def test_regime_filter():
    """Test the regime filter on historical data."""
    print("Testing Regime Filter...")
    
    # Load features
    feat_path = Path('features') / 'USDJPY.sim_features.csv'
    df = pd.read_csv(feat_path).dropna()
    
    print(f"\nLoaded {len(df):,} bars")
    
    # Create filter with default settings
    rf = RegimeFilter(
        rsi_lower=45,
        rsi_upper=55,
        atr_max=0.11,
        return_threshold=0.02
    )
    
    # Test on sample of data
    sample_size = min(1000, len(df))
    print(f"\nTesting on {sample_size} samples...\n")
    
    results = []
    for i in range(len(df) - sample_size, len(df)):
        df_slice = df.iloc[:i+1]
        should_trade, reason = rf.should_trade(df_slice)
        results.append(should_trade)
    
    # Print statistics
    rf.print_statistics()
    
    # Show regime distribution
    print("\n" + "="*60)
    print("REGIME CHARACTERISTICS")
    print("="*60)
    
    # All bars
    print("\nAll Bars:")
    print(f"  RSI:    Mean={df['rsi14'].mean():.1f}, Median={df['rsi14'].median():.1f}")
    print(f"  ATR:    Mean={df['atr14'].mean():.4f}, Median={df['atr14'].median():.4f}")
    
    # Favorable regime bars
    favorable_mask = pd.Series(results, index=df.iloc[-sample_size:].index)
    favorable_df = df.iloc[-sample_size:][favorable_mask]
    
    if len(favorable_df) > 0:
        print(f"\nFavorable Regime Bars ({len(favorable_df):,}):")
        print(f"  RSI:    Mean={favorable_df['rsi14'].mean():.1f}, Median={favorable_df['rsi14'].median():.1f}")
        print(f"  ATR:    Mean={favorable_df['atr14'].mean():.4f}, Median={favorable_df['atr14'].median():.4f}")
    
    print("="*60)
    
    return rf


if __name__ == '__main__':
    test_regime_filter()
