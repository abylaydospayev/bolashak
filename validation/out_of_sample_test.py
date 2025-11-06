"""
Out-of-Sample Validation

Tests the trained ensemble model on completely unseen recent data
to verify the system isn't overfit and generalizes to new market conditions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import yaml
import sys
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_ensemble import EnsembleClassifier, create_ensemble
from backtest import backtest


def test_out_of_sample(symbol, start_date=None, stop_mult=1.5, tp_mult=2.5):
    """
    Test ensemble model on out-of-sample data.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol
    start_date : str
        Start date for OOS period (YYYY-MM-DD). If None, uses last 20% of data
    stop_mult : float
        Stop loss multiplier
    tp_mult : float
        Take profit multiplier
    """
    
    print("="*80)
    print(f"OUT-OF-SAMPLE VALIDATION: {symbol}")
    print("="*80)
    print(f"SL/TP: {stop_mult}ATR / {tp_mult}ATR\n")
    
    # Load enhanced features
    feat_path = Path('features') / f'{symbol}_features_enhanced.csv'
    if not feat_path.exists():
        print(f"ERROR: Enhanced features not found at {feat_path}")
        return None
    
    df = pd.read_csv(feat_path)
    
    # Handle time column (might be 'time' or 'datetime')
    time_col = 'time' if 'time' in df.columns else 'datetime'
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.dropna()
    
    print(f"Loaded {len(df):,} bars")
    print(f"Date range: {df[time_col].min()} to {df[time_col].max()}\n")
    
    # Split into train (old) and OOS (new)
    if start_date:
        oos_start = pd.to_datetime(start_date)
        train_df = df[df[time_col] < oos_start].copy()
        oos_df = df[df[time_col] >= oos_start].copy()
    else:
        # Use last 20% as OOS
        split_idx = int(len(df) * 0.80)
        train_df = df.iloc[:split_idx].copy()
        oos_df = df.iloc[split_idx:].copy()
    
    print(f"Training set: {len(train_df):,} bars ({train_df[time_col].min()} to {train_df[time_col].max()})")
    print(f"OOS set:      {len(oos_df):,} bars ({oos_df[time_col].min()} to {oos_df[time_col].max()})\n")
    
    # Prepare features
    feature_cols = [c for c in df.columns if c not in ['y', 'time', 'datetime', 'h1_trend']]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['y'].values
    X_oos = oos_df[feature_cols].values
    y_oos = oos_df['y'].values
    
    # Train ensemble (or load if exists)
    model_path = Path('models') / f'{symbol}_ensemble_oos.pkl'
    
    print("Training ensemble on in-sample data...")
    from train_ensemble import create_ensemble
    ensemble = create_ensemble(use_xgboost=False)
    ensemble.fit(X_train, y_train)
    
    # Save model
    joblib.dump(ensemble, model_path)
    print(f"Model saved to: {model_path}\n")
    
    # Predict on OOS
    print("Generating predictions on OOS data...")
    y_pred_proba = ensemble.predict_proba(X_oos)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    oos_auc = roc_auc_score(y_oos, y_pred_proba)
    oos_acc = accuracy_score(y_oos, y_pred)
    
    print(f"OOS Model Quality:")
    print(f"  AUC: {oos_auc:.4f}")
    print(f"  Accuracy: {oos_acc:.4f}\n")
    
    # Add predictions to OOS dataframe
    oos_df['prob_buy'] = y_pred_proba
    oos_df['prob_sell'] = 1 - y_pred_proba
    
    # Load config and override SL/TP
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    cfg['stop_atr_mult'] = stop_mult
    cfg['tp_atr_mult'] = tp_mult
    
    # Backtest on OOS
    print("Backtesting on OOS data...")
    bt_result = backtest(oos_df, oos_df['prob_buy'].values, cfg)
    
    if bt_result['trades'] > 0:
        n_trades = bt_result['trades']
        win_rate = bt_result['winrate']
        equity = bt_result['equity']
        initial_bal = cfg.get('initial_balance', 100000)
        total_pnl = equity - initial_bal
        wins = int(n_trades * win_rate)
        losses = n_trades - wins
        max_dd = bt_result['max_dd'] * 100
        
        print(f"\nOOS Backtest Results:")
        print(f"  Trades:      {n_trades}")
        print(f"  PnL:         ${total_pnl:,.0f}")
        print(f"  Win Rate:    {win_rate*100:.1f}% ({wins}W / {losses}L)")
        print(f"  Max DD:      {max_dd:.2f}%")
        
        # Save results
        results = {
            'symbol': symbol,
            'oos_start': oos_df[time_col].min(),
            'oos_end': oos_df[time_col].max(),
            'oos_bars': len(oos_df),
            'train_bars': len(train_df),
            'auc': oos_auc,
            'accuracy': oos_acc,
            'trades': n_trades,
            'pnl': total_pnl,
            'win_rate': win_rate * 100,
            'wins': wins,
            'losses': losses,
            'max_dd_pct': max_dd,
            'stop_mult': stop_mult,
            'tp_mult': tp_mult
        }
        
        results_df = pd.DataFrame([results])
        output_file = f"validation/results/{symbol}_oos_test_sl{stop_mult}_tp{tp_mult}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Determine success
        success = total_pnl > 0 and win_rate > 0.70 and oos_auc > 0.65
        
        print(f"\n{'='*80}")
        if success:
            print(" OUT-OF-SAMPLE TEST: PASSED")
            print("System successfully generalizes to unseen data!")
        else:
            print(" OUT-OF-SAMPLE TEST: FAILED")
            if total_pnl <= 0:
                print("  - Not profitable on OOS data")
            if win_rate <= 0.70:
                print(f"  - Win rate too low ({win_rate*100:.1f}% < 70%)")
            if oos_auc <= 0.65:
                print(f"  - Model quality degraded (AUC {oos_auc:.3f} < 0.65)")
        print(f"{'='*80}\n")
        
        return results
    else:
        print("\nNo trades executed on OOS data!")
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Out-of-sample validation')
    parser.add_argument('--symbol', type=str, default='USDJPY', help='Symbol to test')
    parser.add_argument('--start_date', type=str, default=None, 
                        help='OOS start date (YYYY-MM-DD), default=last 20%%')
    parser.add_argument('--stop_mult', type=float, default=1.5, help='Stop loss multiplier')
    parser.add_argument('--tp_mult', type=float, default=2.5, help='Take profit multiplier')
    
    args = parser.parse_args()
    
    test_out_of_sample(
        symbol=args.symbol,
        start_date=args.start_date,
        stop_mult=args.stop_mult,
        tp_mult=args.tp_mult
    )

