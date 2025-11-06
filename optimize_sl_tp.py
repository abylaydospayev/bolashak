"""
SL/TP Grid Search Optimization
Tests different stop-loss and take-profit multipliers to find optimal risk/reward ratio.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import itertools
from backtest import backtest_with_position_sizing

def load_data_and_signals(symbol: str):
    """Load features and predictions"""
    feature_file = Path('features') / f'{symbol}_features_enhanced.csv'
    pred_file = Path('results') / f'{symbol}_ensemble_predictions.csv'
    
    if not feature_file.exists():
        raise FileNotFoundError(f"Features not found: {feature_file}")
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_file}")
    
    data = pd.read_csv(feature_file)
    preds = pd.read_csv(pred_file)
    
    # Merge predictions
    data = data.merge(preds[['datetime', 'prob_buy', 'prob_sell']], on='datetime', how='left')
    data['prob_buy'].fillna(0.5, inplace=True)
    data['prob_sell'].fillna(0.5, inplace=True)
    
    return data

def run_backtest_with_params(data, cfg, stop_mult, tp_mult, prob_buy, prob_sell):
    """Run backtest with specific SL/TP parameters"""
    
    # Apply thresholds
    data_bt = data.copy()
    data_bt['signal'] = 0
    data_bt.loc[data_bt['prob_buy'] >= prob_buy, 'signal'] = 1
    data_bt.loc[data_bt['prob_sell'] <= prob_sell, 'signal'] = -1
    
    # Override SL/TP in config
    cfg_custom = cfg.copy()
    cfg_custom['stop_atr_mult'] = stop_mult
    cfg_custom['tp_atr_mult'] = tp_mult
    
    # Run backtest
    results = backtest_with_position_sizing(data_bt, cfg_custom)
    
    return results

def optimize_sl_tp(symbol: str, output_file: str = None):
    """Grid search over SL/TP parameters"""
    
    print("="*80)
    print("SL/TP OPTIMIZATION - GRID SEARCH")
    print("="*80)
    
    # Load config
    cfg = yaml.safe_load(open('config.yaml'))
    prob_buy = cfg.get('prob_buy', 0.80)
    prob_sell = cfg.get('prob_sell', 0.20)
    
    # Load data
    print(f"\nLoading data for {symbol}...")
    data = load_data_and_signals(symbol)
    print(f"Loaded {len(data)} bars")
    
    # Define parameter grid
    stop_multipliers = [0.8, 1.0, 1.2, 1.5]
    tp_multipliers = [1.5, 1.8, 2.0, 2.5, 3.0]
    
    print(f"\nTesting {len(stop_multipliers)}  {len(tp_multipliers)} = {len(stop_multipliers) * len(tp_multipliers)} combinations...")
    print(f"Probability thresholds: {prob_buy}/{prob_sell}")
    
    # Store results
    results_list = []
    
    # Grid search
    for stop_mult in stop_multipliers:
        for tp_mult in tp_multipliers:
            print(f"\nTesting SL={stop_mult}ATR, TP={tp_mult}ATR...")
            
            try:
                results = run_backtest_with_params(data, cfg, stop_mult, tp_mult, prob_buy, prob_sell)
                
                # Calculate risk/reward ratio
                rr_ratio = tp_mult / stop_mult
                
                result_dict = {
                    'stop_mult': stop_mult,
                    'tp_mult': tp_mult,
                    'rr_ratio': rr_ratio,
                    'pnl': results['total_pnl'],
                    'trades': results['num_trades'],
                    'win_rate': results['win_rate'],
                    'max_dd_pct': results['max_dd_pct'],
                    'profit_factor': results.get('profit_factor', 0),
                    'avg_win': results.get('avg_win', 0),
                    'avg_loss': results.get('avg_loss', 0),
                    'sharpe': results.get('sharpe_ratio', 0)
                }
                
                results_list.append(result_dict)
                
                print(f"  PnL: ${result_dict['pnl']:,.0f} | Trades: {result_dict['trades']} | "
                      f"Win Rate: {result_dict['win_rate']:.1f}% | Max DD: {result_dict['max_dd_pct']:.2f}%")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Sort by PnL
    results_df = results_df.sort_values('pnl', ascending=False)
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS - TOP 10")
    print("="*80)
    print(f"\n{'Rank':<5} {'SL':<6} {'TP':<6} {'R:R':<6} {'PnL':<12} {'Trades':<8} {'Win%':<7} {'Max DD%':<9}")
    print("-"*80)
    
    for idx, row in results_df.head(10).iterrows():
        print(f"{results_df.index.get_loc(idx)+1:<5} "
              f"{row['stop_mult']:<6.1f} "
              f"{row['tp_mult']:<6.1f} "
              f"{row['rr_ratio']:<6.2f} "
              f"${row['pnl']:>10,.0f} "
              f"{int(row['trades']):<8} "
              f"{row['win_rate']:<6.1f}% "
              f"{row['max_dd_pct']:<8.2f}%")
    
    # Find best by different metrics
    print("\n" + "="*80)
    print("BEST PARAMETERS BY METRIC")
    print("="*80)
    
    best_pnl = results_df.iloc[0]
    print(f"\n1. Best PnL: SL={best_pnl['stop_mult']:.1f}, TP={best_pnl['tp_mult']:.1f}")
    print(f"   PnL: ${best_pnl['pnl']:,.0f}, Win Rate: {best_pnl['win_rate']:.1f}%")
    
    best_wr = results_df.loc[results_df['win_rate'].idxmax()]
    print(f"\n2. Best Win Rate: SL={best_wr['stop_mult']:.1f}, TP={best_wr['tp_mult']:.1f}")
    print(f"   Win Rate: {best_wr['win_rate']:.1f}%, PnL: ${best_wr['pnl']:,.0f}")
    
    best_sharpe = results_df.loc[results_df['sharpe'].idxmax()]
    print(f"\n3. Best Sharpe: SL={best_sharpe['stop_mult']:.1f}, TP={best_sharpe['tp_mult']:.1f}")
    print(f"   Sharpe: {best_sharpe['sharpe']:.3f}, PnL: ${best_sharpe['pnl']:,.0f}")
    
    # Best low drawdown with positive PnL
    profitable = results_df[results_df['pnl'] > 0]
    if len(profitable) > 0:
        best_dd = profitable.loc[profitable['max_dd_pct'].idxmin()]
        print(f"\n4. Best Risk-Adjusted (profitable with lowest DD):")
        print(f"   SL={best_dd['stop_mult']:.1f}, TP={best_dd['tp_mult']:.1f}")
        print(f"   PnL: ${best_dd['pnl']:,.0f}, Max DD: {best_dd['max_dd_pct']:.2f}%")
    
    # Save results
    if output_file is None:
        output_file = f"results/{symbol}_sl_tp_optimization.csv"
    
    Path(output_file).parent.mkdir(exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")
    
    return results_df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize SL/TP parameters')
    parser.add_argument('--symbol', type=str, default='USDJPY', help='Symbol to optimize')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file')
    
    args = parser.parse_args()
    
    results = optimize_sl_tp(args.symbol, args.output)

