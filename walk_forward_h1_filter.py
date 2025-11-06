"""
Walk-forward validation with H1 TREND FILTER + SL/TP optimization.

Key improvements:
1. H1 trend filter: Only trade when ema20_h1 vs ema50_h1 shows clear trend
2. Configurable SL/TP multipliers for optimization
3. All previous enhancements (ensemble, multi-timeframe, etc.)
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
import yaml

from train_ensemble import create_ensemble
from backtest import backtest


def walk_forward_split(df, n_splits=5, train_size=0.6, test_size=0.1):
    """Create walk-forward splits."""
    n = len(df)
    initial_train = int(n * train_size)
    fold_size = int(n * test_size)
    
    splits = []
    for i in range(n_splits):
        test_start = initial_train + i * fold_size
        test_end = test_start + fold_size
        
        if test_end > n:
            break
        
        train_idx = list(range(0, test_start))
        test_idx = list(range(test_start, test_end))
        
        splits.append((train_idx, test_idx))
    
    return splits


def select_features_for_fold(X_train, y_train, feature_cols, max_features=30):
    """Select top features using mutual information."""
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    
    feature_scores = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    selected_features = feature_scores.head(max_features)['feature'].tolist()
    selected_indices = [feature_cols.index(f) for f in selected_features]
    
    return selected_indices, selected_features


def apply_h1_trend_filter(df):
    """
    Apply H1 trend filter - only trade when H1 shows clear trend.
    
    Rules:
    - BUY signals only when ema20_h1 > ema50_h1 (uptrend)
    - SELL signals only when ema20_h1 < ema50_h1 (downtrend)
    
    Returns:
    --------
    df with 'h1_trend' column: 1 = uptrend, -1 = downtrend, 0 = no trend
    """
    if 'ema20_h1' not in df.columns or 'ema50_h1' not in df.columns:
        print("WARNING: H1 EMA columns not found, trend filter disabled!")
        df['h1_trend'] = 0
        return df
    
    df['h1_trend'] = 0
    df.loc[df['ema20_h1'] > df['ema50_h1'], 'h1_trend'] = 1   # Uptrend
    df.loc[df['ema20_h1'] < df['ema50_h1'], 'h1_trend'] = -1  # Downtrend
    
    return df


def run_walk_forward_with_filters(symbol, n_splits, use_h1_filter=True,
                                   stop_mult=1.0, tp_mult=1.8, max_features=30):
    """
    Run walk-forward validation with H1 trend filter and custom SL/TP.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol
    n_splits : int
        Number of walk-forward folds
    use_h1_filter : bool
        If True, apply H1 trend filter
    stop_mult : float
        Stop loss multiplier ( ATR)
    tp_mult : float
        Take profit multiplier ( ATR)
    max_features : int
        Number of features to select per fold
    """
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD VALIDATION: ENSEMBLE MODEL + {symbol}")
    print(f"H1 TREND FILTER: {'ENABLED' if use_h1_filter else 'DISABLED'}")
    print(f"SL/TP: {stop_mult}ATR / {tp_mult}ATR (R:R = 1:{tp_mult/stop_mult:.2f})")
    print(f"{'='*80}\n")
    
    # Load enhanced features
    feat_path = Path('features') / f'{symbol}_features_enhanced.csv'
    if not feat_path.exists():
        print(f"ERROR: Enhanced features not found at {feat_path}")
        print(f"Run: python build_features_enhanced.py --symbol {symbol}")
        return
    
    df = pd.read_csv(feat_path)
    df = df.dropna()
    
    print(f"Loaded {len(df):,} bars with enhanced features")
    
    # Apply H1 trend filter
    if use_h1_filter:
        df = apply_h1_trend_filter(df)
        uptrend_bars = (df['h1_trend'] == 1).sum()
        downtrend_bars = (df['h1_trend'] == -1).sum()
        print(f"H1 Trends: {uptrend_bars:,} uptrend ({uptrend_bars/len(df)*100:.1f}%), "
              f"{downtrend_bars:,} downtrend ({downtrend_bars/len(df)*100:.1f}%)")
    
    # Prepare features
    feature_cols = [c for c in df.columns if c not in ['y', 'time', 'datetime', 'h1_trend']]
    
    X = df[feature_cols].values
    y = df['y'].values
    
    print(f"Features: {len(feature_cols)}")
    
    # Create splits
    splits = walk_forward_split(df, n_splits=n_splits)
    print(f"Created {len(splits)} walk-forward folds\n")
    
    # Load config and override SL/TP
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    cfg['stop_atr_mult'] = stop_mult
    cfg['tp_atr_mult'] = tp_mult
    
    # Results tracking
    all_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{''*80}")
        print(f"FOLD {fold_idx + 1}/{len(splits)}")
        print(f"{''*80}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        test_df = df.iloc[test_idx].copy()
        
        print(f"Train: {len(train_idx):,} bars | Test: {len(test_idx):,} bars")
        
        # Feature selection
        print("Selecting features...")
        selected_indices, selected_features = select_features_for_fold(
            X_train, y_train, feature_cols, max_features=max_features
        )
        
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        
        print(f"Selected {len(selected_features)} features")
        print(f"Top 5: {', '.join(selected_features[:5])}")
        
        # Create and train ensemble
        print("\nTraining ensemble...")
        ensemble = create_ensemble(use_xgboost=False)
        ensemble.fit(X_train_selected, y_train)
        
        # Predict
        y_pred_proba = ensemble.predict_proba(X_test_selected)[:, 1]
        
        # Add predictions to test_df
        test_df['prob_buy'] = y_pred_proba
        test_df['prob_sell'] = 1 - y_pred_proba
        
        # Apply H1 trend filter if enabled
        if use_h1_filter:
            print("\nApplying H1 trend filter...")
            
            # Count how many signals would be generated before filter
            prob_buy_thresh = cfg.get('prob_buy', 0.80)
            prob_sell_thresh = cfg.get('prob_sell', 0.20)
            
            initial_buy_signals = (test_df['prob_buy'] >= prob_buy_thresh).sum()
            initial_sell_signals = (test_df['prob_sell'] <= prob_sell_thresh).sum()
            initial_signals = initial_buy_signals + initial_sell_signals
            
            # Only allow BUY when uptrend (h1_trend == 1), SELL when downtrend (h1_trend == -1)
            # Set probabilities to neutral (0.5) where trend doesn't align
            test_df.loc[(test_df['prob_buy'] >= prob_buy_thresh) & (test_df['h1_trend'] != 1), 'prob_buy'] = 0.5
            test_df.loc[(test_df['prob_sell'] <= prob_sell_thresh) & (test_df['h1_trend'] != -1), 'prob_sell'] = 0.5
            
            # Recalculate to ensure probabilities sum to 1
            test_df['prob_sell'] = 1 - test_df['prob_buy']
            
            # Count filtered signals
            filtered_buy_signals = (test_df['prob_buy'] >= prob_buy_thresh).sum()
            filtered_sell_signals = (test_df['prob_sell'] <= prob_sell_thresh).sum()
            filtered_signals = filtered_buy_signals + filtered_sell_signals
            
            pct_kept = filtered_signals / initial_signals * 100 if initial_signals > 0 else 0
            print(f"  Signals: {initial_signals}  {filtered_signals} ({pct_kept:.1f}% kept)")
            print(f"    BUY: {initial_buy_signals}  {filtered_buy_signals}")
            print(f"    SELL: {initial_sell_signals}  {filtered_sell_signals}")
        
        # Backtest
        print("\nBacktesting...")
        bt_result = backtest(test_df, test_df['prob_buy'].values, cfg)
        
        if bt_result['trades'] > 0:
            n_trades = bt_result['trades']
            win_rate = bt_result['winrate']
            
            equity = bt_result['equity']
            initial_bal = cfg.get('initial_balance', 100000)
            total_pnl = equity - initial_bal
            wins = int(n_trades * win_rate)
            losses = n_trades - wins
            max_dd = bt_result['max_dd'] * 100
            
            print(f"  Trades:      {n_trades}")
            print(f"  PnL:         ${total_pnl:,.0f}")
            print(f"  Win Rate:    {win_rate*100:.1f}% ({wins}W / {losses}L)")
            print(f"  Max DD:      {max_dd:.2f}%")
        else:
            print("  No trades executed!")
            total_pnl = 0
            win_rate = 0
            wins, losses = 0, 0
            n_trades = 0
            max_dd = 0
        
        # Store results
        fold_result = {
            'fold': fold_idx + 1,
            'pnl': total_pnl,
            'trades': n_trades,
            'win_rate': win_rate * 100,
            'max_dd_pct': max_dd,
            'wins': wins,
            'losses': losses
        }
        all_results.append(fold_result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY - ALL FOLDS")
    print(f"{'='*80}\n")
    
    results_df = pd.DataFrame(all_results)
    
    print(f"{'Fold':<6} {'PnL':<12} {'Trades':<8} {'Win%':<8}")
    print(""*60)
    
    for _, row in results_df.iterrows():
        print(f"{int(row['fold']):<6} ${row['pnl']:>10,.0f} {int(row['trades']):<8} {row['win_rate']:<7.1f}%")
    
    print(""*60)
    
    total_pnl = results_df['pnl'].sum()
    total_trades = results_df['trades'].sum()
    avg_win_rate = results_df['win_rate'].mean()
    profitable_folds = (results_df['pnl'] > 0).sum()
    
    print(f"{'TOTAL':<6} ${total_pnl:>10,.0f} {int(total_trades):<8} {avg_win_rate:<7.1f}%")
    print(f"\nProfitable Folds: {profitable_folds}/{len(results_df)}")
    print(f"{'='*80}\n")
    
    # Save results
    output_file = f"results/{symbol}_walkforward_h1filter_sl{stop_mult}_tp{tp_mult}.csv"
    Path(output_file).parent.mkdir(exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Walk-forward with H1 trend filter')
    parser.add_argument('--symbol', type=str, default='USDJPY', help='Symbol')
    parser.add_argument('--n_splits', type=int, default=4, help='Number of folds')
    parser.add_argument('--use_h1_filter', type=int, default=1, help='Use H1 trend filter (1=yes, 0=no)')
    parser.add_argument('--stop_mult', type=float, default=1.0, help='Stop loss multiplier')
    parser.add_argument('--tp_mult', type=float, default=1.8, help='Take profit multiplier')
    parser.add_argument('--max_features', type=int, default=30, help='Max features to select')
    
    args = parser.parse_args()
    
    results = run_walk_forward_with_filters(
        symbol=args.symbol,
        n_splits=args.n_splits,
        use_h1_filter=bool(args.use_h1_filter),
        stop_mult=args.stop_mult,
        tp_mult=args.tp_mult,
        max_features=args.max_features
    )

