"""
Walk-forward validation with Lorentzian classifier and regime filtering.
Tests model on sequential time periods with expanding training window.
"""
import pandas as pd
import numpy as np
import argparse
import yaml
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import sys

# Import Lorentzian classifier
sys.path.insert(0, str(Path(__file__).parent))
from lorentzian_classifier import LorentzianClassifier

import backtest
import indicators

def calculate_confidence(proba):
    """Calculate confidence as distance from 0.5 (uncertain)."""
    return abs(proba - 0.5) * 2  # Scale to 0-1

def main():
    parser = argparse.ArgumentParser(description='Walk-forward validation with Lorentzian')
    parser.add_argument('--symbol', required=True, help='Trading symbol')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of walk-forward folds')
    parser.add_argument('--train_size', type=float, default=0.6, help='Initial training size (0-1)')
    parser.add_argument('--test_size', type=float, default=0.1, help='Test size per fold (0-1)')
    parser.add_argument('--k', type=int, default=8, help='Number of neighbors')
    parser.add_argument('--confidence_threshold', type=float, default=0.0, 
                       help='Minimum confidence to trade (0-1, 0=no filter)')
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open('config.yaml'))
    feat_dir = Path(cfg['feature_dir'])
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    symbol = args.symbol
    print(f"\n{'='*80}")
    print(f"Walk-Forward Validation (Lorentzian): {symbol}")
    print(f"{'='*80}\n")
    print(f"Parameters:")
    print(f"  Folds: {args.n_splits}")
    print(f"  Initial train: {args.train_size*100:.0f}%")
    print(f"  Test per fold: {args.test_size*100:.0f}%")
    print(f"  k neighbors: {args.k}")
    print(f"  Confidence filter: {'OFF' if args.confidence_threshold == 0 else f'{args.confidence_threshold:.2f}'}")
    
    # Load features
    feat_path = feat_dir / f'{symbol}_features.csv'
    if not feat_path.exists():
        print(f"ERROR: {feat_path} not found!")
        return
    
    df = pd.read_csv(feat_path)
    df['time'] = pd.to_datetime(df['time'])
    df_clean = df.dropna()
    
    print(f"\nTotal samples: {len(df_clean)}")
    
    # Feature columns
    feature_cols = [c for c in df_clean.columns 
                   if c not in ['y', 'time', 'fwd_ret', 'fwd_ret_net']]
    print(f"Features: {len(feature_cols)}")
    
    # Calculate fold splits
    total_samples = len(df_clean)
    initial_train_samples = int(total_samples * args.train_size)
    test_samples = int(total_samples * args.test_size)
    
    # Calculate actual number of folds
    max_folds = (total_samples - initial_train_samples) // test_samples
    n_folds = min(args.n_splits, max_folds)
    print(f"\nCreated {n_folds} folds\n")
    
    # Results storage
    fold_results = []
    all_trades = []
    
    for fold in range(1, n_folds + 1):
        print(f"{'='*80}")
        print(f"FOLD {fold}/{n_folds}")
        print(f"{'='*80}")
        
        # Define train/test split
        train_end = initial_train_samples + (fold - 1) * test_samples
        test_start = train_end
        test_end = test_start + test_samples
        
        if test_end > total_samples:
            print("Insufficient data for this fold, stopping.")
            break
        
        # Split data
        train_df = df_clean.iloc[:train_end]
        test_df = df_clean.iloc[test_start:test_end]
        
        print(f"Train: {len(train_df)} samples ({df_clean['time'].iloc[0]} to {df_clean['time'].iloc[train_end-1]})")
        print(f"Test:  {len(test_df)} samples ({df_clean['time'].iloc[test_start]} to {df_clean['time'].iloc[test_end-1]})")
        
        # Prepare features
        X_train = train_df[feature_cols].values
        y_train = train_df['y'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['y'].values
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Lorentzian
        print(f"\nTraining Lorentzian (k={args.k})...")
        clf = LorentzianClassifier(k=args.k, weight_by_distance=True)
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        print("Predicting...")
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate confidence
        confidence = calculate_confidence(y_pred_proba)
        
        # Apply confidence filter
        if args.confidence_threshold > 0:
            high_confidence_mask = confidence >= args.confidence_threshold
            filtered_count = np.sum(~high_confidence_mask)
            print(f"Confidence filter: Removed {filtered_count}/{len(y_pred)} predictions (low confidence)")
        else:
            high_confidence_mask = np.ones(len(y_pred), dtype=bool)
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\nMetrics:")
        print(f"  AUC: {auc:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Avg Confidence: {confidence.mean():.4f}")
        
        # Backtest
        print("\nRunning backtest...")
        
        # Prepare data for backtest
        test_df_copy = test_df.copy()
        test_df_copy['prob'] = y_pred_proba
        test_df_copy['confidence'] = confidence
        test_df_copy['high_confidence'] = high_confidence_mask
        
        # Load price data
        data_dir = Path(cfg['data_dir'])
        price_df = pd.read_csv(data_dir / f"{symbol}_M15.csv")
        price_df['time'] = pd.to_datetime(price_df['time'])
        
        # Run backtest with confidence filter
        bt_cfg = {
            'prob_buy': 0.6,
            'prob_sell': 0.4,
            'stop_atr_mult': 2.0,
            'tp_atr_mult': 3.0,
            'max_hold_bars': 3,
            'symbol': symbol
        }
        
        filtered_probs = y_pred_proba[high_confidence_mask]
        filtered_df = test_df.iloc[high_confidence_mask].copy()
        
        if len(filtered_df) > 0:
            results = backtest.backtest(filtered_df, filtered_probs, bt_cfg)
            trades = results['trades']
        else:
            trades = []
        
        # Calculate metrics
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            total_pnl = trades_df['pnl_net'].sum()
            num_trades = len(trades_df)
            win_rate = (trades_df['pnl_net'] > 0).mean() * 100
            
            # Profit factor
            gross_profit = trades_df[trades_df['pnl_net'] > 0]['pnl_net'].sum()
            gross_loss = abs(trades_df[trades_df['pnl_net'] < 0]['pnl_net'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            print(f"\nBacktest Results:")
            print(f"  Trades: {num_trades}")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Profit Factor: {profit_factor:.2f}")
            print(f"  Total PnL: ${total_pnl:,.2f}")
            
            # Store trades
            trades_df['fold'] = fold
            all_trades.append(trades_df)
        else:
            print(f"\nBacktest Results: No trades (all filtered by confidence)")
            total_pnl = 0
            num_trades = 0
            win_rate = 0
            profit_factor = 0
        
        # Store fold results
        fold_results.append({
            'fold': fold,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'auc': auc,
            'accuracy': acc,
            'avg_confidence': confidence.mean(),
            'trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl
        })
        
        print()
    
    # Aggregate results
    print(f"{'='*80}")
    print("AGGREGATE RESULTS")
    print(f"{'='*80}\n")
    
    results_df = pd.DataFrame(fold_results)
    
    print("Per-Fold Results:")
    print(results_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    total_trades = results_df['trades'].sum()
    avg_auc = results_df['auc'].mean()
    total_pnl = results_df['total_pnl'].sum()
    profitable_folds = (results_df['total_pnl'] > 0).sum()
    
    print(f"\nTotal Trades: {total_trades}")
    print(f"Average AUC: {avg_auc:.4f}")
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Profitable Folds: {profitable_folds}/{n_folds} ({profitable_folds/n_folds*100:.0f}%)")
    
    if total_trades > 0:
        all_trades_df = pd.concat(all_trades, ignore_index=True)
        overall_wr = (all_trades_df['pnl_net'] > 0).mean() * 100
        overall_gp = all_trades_df[all_trades_df['pnl_net'] > 0]['pnl_net'].sum()
        overall_gl = abs(all_trades_df[all_trades_df['pnl_net'] < 0]['pnl_net'].sum())
        overall_pf = overall_gp / overall_gl if overall_gl > 0 else np.inf
        
        print(f"Overall Win Rate: {overall_wr:.1f}%")
        print(f"Overall Profit Factor: {overall_pf:.2f}")
    
    # Save results
    results_file = results_dir / f'{symbol}_lorentzian_walkforward.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    if len(all_trades) > 0:
        trades_file = results_dir / f'{symbol}_lorentzian_trades.csv'
        all_trades_df = pd.concat(all_trades, ignore_index=True)
        all_trades_df.to_csv(trades_file, index=False)
        print(f"Trades saved to: {trades_file}")

if __name__ == '__main__':
    main()
