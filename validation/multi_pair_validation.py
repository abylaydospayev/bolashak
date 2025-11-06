"""
Multi-Pair Walk-Forward Validation

Tests the trading system across multiple currency pairs to validate generalization.
Handles different pip values and spread costs per pair.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime

from walk_forward_h1_filter import run_walk_forward_with_filters


# Symbol-specific configurations
SYMBOL_CONFIGS = {
    'USDJPY': {
        'pip': 0.01,
        'pip_value_usd': 10.0,  # For 1 lot
        'spread_pips': 0.8,
        'commission_per_lot': 7.0,
        'slippage_pips': 0.4
    },
    'EURUSD': {
        'pip': 0.0001,
        'pip_value_usd': 10.0,
        'spread_pips': 0.6,
        'commission_per_lot': 7.0,
        'slippage_pips': 0.3
    },
    'GBPUSD': {
        'pip': 0.0001,
        'pip_value_usd': 10.0,
        'spread_pips': 1.0,
        'commission_per_lot': 7.0,
        'slippage_pips': 0.5
    },
    'AUDUSD': {
        'pip': 0.0001,
        'pip_value_usd': 10.0,
        'spread_pips': 0.8,
        'commission_per_lot': 7.0,
        'slippage_pips': 0.4
    }
}


def update_config_for_symbol(symbol):
    """Update config.yaml with symbol-specific parameters."""
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    if symbol in SYMBOL_CONFIGS:
        symbol_cfg = SYMBOL_CONFIGS[symbol]
        cfg['spread_pips'] = symbol_cfg['spread_pips']
        cfg['commission_per_lot'] = symbol_cfg['commission_per_lot']
        cfg['slippage_pips'] = symbol_cfg['slippage_pips']
        
        # Save updated config
        with open('config.yaml', 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        
        print(f"Updated config for {symbol}:")
        print(f"  Spread: {symbol_cfg['spread_pips']} pips")
        print(f"  Commission: ${symbol_cfg['commission_per_lot']}/lot")
        print(f"  Slippage: {symbol_cfg['slippage_pips']} pips")
    else:
        print(f"WARNING: No specific config for {symbol}, using defaults")


def run_multi_pair_validation(symbols, n_splits=4, stop_mult=1.5, tp_mult=2.5):
    """
    Run walk-forward validation across multiple currency pairs.
    
    Parameters:
    -----------
    symbols : list
        List of symbols to test
    n_splits : int
        Number of walk-forward folds
    stop_mult : float
        Stop loss multiplier
    tp_mult : float
        Take profit multiplier
    """
    
    print("="*80)
    print("MULTI-PAIR WALK-FORWARD VALIDATION")
    print("="*80)
    print(f"\nTesting {len(symbols)} pairs: {', '.join(symbols)}")
    print(f"Configuration: SL={stop_mult}ATR, TP={tp_mult}ATR")
    print(f"Folds: {n_splits}")
    print()
    
    all_results = []
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING: {symbol}")
        print(f"{'='*80}\n")
        
        # Check if enhanced features exist
        feat_path = Path('features') / f'{symbol}_features_enhanced.csv'
        if not feat_path.exists():
            print(f"ERROR: Enhanced features not found for {symbol}")
            print(f"Run: python build_features_enhanced.py --symbol {symbol}")
            continue
        
        # Update config for this symbol
        update_config_for_symbol(symbol)
        
        # Run walk-forward validation
        try:
            results_df = run_walk_forward_with_filters(
                symbol=symbol,
                n_splits=n_splits,
                use_h1_filter=True,
                stop_mult=stop_mult,
                tp_mult=tp_mult,
                max_features=30
            )
            
            if results_df is not None:
                # Add symbol column
                results_df['symbol'] = symbol
                all_results.append(results_df)
                
                # Summary for this pair
                total_pnl = results_df['pnl'].sum()
                total_trades = results_df['trades'].sum()
                avg_win_rate = results_df['win_rate'].mean()
                profitable_folds = (results_df['pnl'] > 0).sum()
                
                print(f"\n{symbol} SUMMARY:")
                print(f"  Total PnL: ${total_pnl:,.0f}")
                print(f"  Total Trades: {total_trades}")
                print(f"  Avg Win Rate: {avg_win_rate:.1f}%")
                print(f"  Profitable Folds: {profitable_folds}/{len(results_df)}")
                
        except Exception as e:
            print(f"ERROR testing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        output_file = f"validation/results/multi_pair_walkforward_sl{stop_mult}_tp{tp_mult}.csv"
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n{'='*80}")
        print("OVERALL MULTI-PAIR SUMMARY")
        print(f"{'='*80}\n")
        
        # Summary by symbol
        print(f"{'Symbol':<10} {'PnL':<15} {'Trades':<10} {'Win%':<10} {'Profitable Folds':<20}")
        print("-"*80)
        
        for symbol in symbols:
            symbol_data = combined_df[combined_df['symbol'] == symbol]
            if len(symbol_data) > 0:
                total_pnl = symbol_data['pnl'].sum()
                total_trades = symbol_data['trades'].sum()
                avg_wr = symbol_data['win_rate'].mean()
                prof_folds = (symbol_data['pnl'] > 0).sum()
                total_folds = len(symbol_data)
                
                status = "" if total_pnl > 0 else ""
                print(f"{symbol:<10} ${total_pnl:>12,.0f} {int(total_trades):<10} {avg_wr:<9.1f}% "
                      f"{prof_folds}/{total_folds:<17} {status}")
        
        print("-"*80)
        
        # Grand totals
        grand_total_pnl = combined_df['pnl'].sum()
        grand_total_trades = combined_df['trades'].sum()
        grand_avg_wr = combined_df['win_rate'].mean()
        profitable_symbols = len([s for s in symbols if combined_df[combined_df['symbol']==s]['pnl'].sum() > 0])
        
        print(f"{'TOTAL':<10} ${grand_total_pnl:>12,.0f} {int(grand_total_trades):<10} {grand_avg_wr:<9.1f}% "
              f"{profitable_symbols}/{len(symbols)} pairs profitable")
        
        print(f"\nResults saved to: {output_file}")
        
        return combined_df
    else:
        print("\nERROR: No results collected!")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-pair walk-forward validation')
    parser.add_argument('--symbols', type=str, nargs='+', 
                        default=['USDJPY', 'EURUSD'],
                        help='Symbols to test')
    parser.add_argument('--n_splits', type=int, default=4, help='Number of folds')
    parser.add_argument('--stop_mult', type=float, default=1.5, help='Stop loss multiplier')
    parser.add_argument('--tp_mult', type=float, default=2.5, help='Take profit multiplier')
    
    args = parser.parse_args()
    
    results = run_multi_pair_validation(
        symbols=args.symbols,
        n_splits=args.n_splits,
        stop_mult=args.stop_mult,
        tp_mult=args.tp_mult
    )

