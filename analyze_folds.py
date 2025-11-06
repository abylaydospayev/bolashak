"""
Analyze what made Fold 1 profitable vs other folds.
Compares market conditions across different time periods.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Import from build_features (has load_mt5_data)
import make_dataset
import indicators

def analyze_regime(df):
    """Analyze market regime characteristics."""
    # Add indicators
    df['ema_20'] = indicators.ema(df['close'], 20)
    df['ema_50'] = indicators.ema(df['close'], 50)
    df['rsi_14'] = indicators.rsi(df['close'], 14)
    df['atr_14'] = indicators.atr(df, 14)
    
    # Calculate regime features
    stats = {
        'start_date': df['time'].min(),
        'end_date': df['time'].max(),
        'bars': len(df),
        
        # Price stats
        'avg_close': df['close'].mean(),
        'std_close': df['close'].std(),
        'min_close': df['close'].min(),
        'max_close': df['close'].max(),
        'total_return': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100,
        
        # Volatility
        'avg_atr': df['atr_14'].mean(),
        'std_atr': df['atr_14'].std(),
        'max_atr': df['atr_14'].max(),
        
        # Trend
        'ema_crossovers': ((df['ema_20'] > df['ema_50']).astype(int).diff().abs()).sum(),
        'pct_trending': ((df['ema_20'] - df['ema_50']).abs() > df['atr_14']).mean() * 100,
        
        # RSI
        'avg_rsi': df['rsi_14'].mean(),
        'pct_overbought': (df['rsi_14'] > 70).mean() * 100,
        'pct_oversold': (df['rsi_14'] < 30).mean() * 100,
        
        # Returns
        'avg_return': df['close'].pct_change().mean() * 100,
        'std_return': df['close'].pct_change().std() * 100,
        'max_drawdown': ((df['close'] / df['close'].cummax()) - 1).min() * 100,
    }
    
    return stats

def main():
    symbol = 'USDJPY.sim'
    print(f"Analyzing {symbol} across all folds...")
    print("="*80)
    
    # Load full data from CSV
    data_path = Path('data') / f"{symbol}_M15.csv"
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    print(f"Total data: {len(df)} bars from {df['time'].min()} to {df['time'].max()}")
    
    # Define fold splits (10 folds, 50/5 split)
    total_samples = len(df)
    train_size = 0.5
    test_size = 0.05
    
    # Initial training size
    initial_train_samples = int(total_samples * train_size)
    test_samples = int(total_samples * test_size)
    
    fold_stats = []
    
    for fold in range(1, 11):
        train_end = initial_train_samples + (fold - 1) * test_samples
        test_start = train_end
        test_end = test_start + test_samples
        
        if test_end > total_samples:
            break
            
        # Get test period data
        test_data = df.iloc[test_start:test_end].copy()
        
        print(f"\n{'='*80}")
        print(f"FOLD {fold}")
        print(f"{'='*80}")
        print(f"Period: {test_data['time'].min()} to {test_data['time'].max()}")
        
        stats = analyze_regime(test_data)
        
        # Print key metrics
        print(f"\nPrice Movement:")
        print(f"  Avg Close: {stats['avg_close']:.3f}")
        print(f"  Total Return: {stats['total_return']:+.2f}%")
        print(f"  Max Drawdown: {stats['max_drawdown']:.2f}%")
        
        print(f"\nVolatility:")
        print(f"  Avg ATR: {stats['avg_atr']:.5f}")
        print(f"  Std ATR: {stats['std_atr']:.5f}")
        print(f"  Max ATR: {stats['max_atr']:.5f}")
        print(f"  Avg Return Std: {stats['std_return']:.4f}%")
        
        print(f"\nTrend:")
        print(f"  EMA Crossovers: {stats['ema_crossovers']}")
        print(f"  % Trending: {stats['pct_trending']:.1f}%")
        
        print(f"\nMomentum:")
        print(f"  Avg RSI: {stats['avg_rsi']:.1f}")
        print(f"  % Overbought: {stats['pct_overbought']:.1f}%")
        print(f"  % Oversold: {stats['pct_oversold']:.1f}%")
        
        fold_stats.append({
            'fold': fold,
            **stats
        })
    
    # Create comparison DataFrame
    comparison = pd.DataFrame(fold_stats)
    
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    # Key metrics to compare
    metrics = [
        ('total_return', 'Return %'),
        ('max_drawdown', 'Max DD %'),
        ('avg_atr', 'Avg ATR'),
        ('pct_trending', '% Trending'),
        ('avg_rsi', 'Avg RSI'),
        ('ema_crossovers', 'Crossovers'),
    ]
    
    for col, label in metrics:
        print(f"{label:15s}: ", end="")
        for fold in range(1, 11):
            val = comparison[comparison['fold'] == fold][col].values[0]
            if fold == 1:  # Highlight profitable fold
                print(f"[{val:7.2f}]* ", end="")
            else:
                print(f" {val:7.2f}  ", end="")
        print()
    
    print("\n* = Profitable fold")
    
    # Calculate correlations with profitability
    # Load walk-forward results
    wf_results_path = Path('results') / f'{symbol}_walkforward.csv'
    if wf_results_path.exists():
        wf_results = pd.read_csv(wf_results_path)
        
        # Merge with regime stats
        merged = comparison.merge(wf_results[['fold', 'total_pnl']], on='fold')
        
        print(f"\n{'='*80}")
        print("CORRELATION WITH PROFITABILITY")
        print(f"{'='*80}\n")
        
        correlations = []
        for col, label in metrics:
            corr = merged[[col, 'total_pnl']].corr().iloc[0, 1]
            correlations.append((label, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for label, corr in correlations:
            direction = "" if corr > 0 else ""
            strength = abs(corr)
            if strength > 0.5:
                marker = ""
            elif strength > 0.3:
                marker = ""
            elif strength > 0.1:
                marker = ""
            else:
                marker = ""
            
            print(f"{label:20s}: {corr:+.3f} {direction} {marker}")
        
        print("\nInterpretation:")
        print("   = Strong correlation (|r| > 0.5)")
        print("     = Moderate correlation (|r| > 0.3)")
        print("       = Weak correlation (|r| > 0.1)")
        
    # Save detailed stats
    output_path = Path('results') / f'{symbol}_regime_analysis.csv'
    comparison.to_csv(output_path, index=False)
    print(f"\nDetailed stats saved to: {output_path}")

if __name__ == "__main__":
    main()

