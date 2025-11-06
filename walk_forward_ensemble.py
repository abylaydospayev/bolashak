"""
Walk-forward validation with ENSEMBLE model and enhanced features.

Tests the improved system:
1. Multi-timeframe features
2. Feature selection (top 30)
3. Ensemble (RF + GradientBoosting)
4. Regime filter
5. Position sizing (1%)
"""
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_selection import mutual_info_classif
from datetime import datetime

from regime_filter import RegimeFilter
from backtest import backtest
from train_ensemble import EnsembleClassifier, create_ensemble
import yaml


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


def run_walk_forward_ensemble(symbol, n_splits, use_regime_filter=True,
                               max_features=30):
    """
    Run walk-forward validation with ensemble model.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol
    n_splits : int
        Number of walk-forward folds
    use_regime_filter : bool
        If True, apply regime filter
    max_features : int
        Number of features to select per fold
    """
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD VALIDATION: ENSEMBLE MODEL + {symbol}")
    if use_regime_filter:
        print("WITH REGIME FILTER + POSITION SIZING (1%)")
    else:
        print("WITH POSITION SIZING (1%), NO FILTER")
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
    
    # Ensure regime filter columns exist (they should be in M15 base features)
    # If not, we need to load the original features
    if 'rsi14' not in df.columns and 'atr14' not in df.columns:
        print("Warning: Base RSI/ATR not in enhanced features, loading original features...")
        base_feat_path = Path('features') / f'{symbol}_features.csv'
        if base_feat_path.exists():
            df_base = pd.read_csv(base_feat_path)
            # Merge rsi14 and atr14 from base features
            df = df.merge(df_base[['time', 'rsi14', 'atr14']], on='time', how='left')
            df = df.dropna()
            print(f"Merged base features, {len(df):,} bars remaining")
    
    # Prepare features
    feature_cols = [c for c in df.columns if c not in ['y', 'time']]
    
    X = df[feature_cols].values
    y = df['y'].values
    
    print(f"Features: {len(feature_cols)}")
    
    # Create splits
    splits = walk_forward_split(df, n_splits=n_splits)
    print(f"Created {len(splits)} walk-forward folds\n")
    
    # Initialize regime filter
    if use_regime_filter:
        regime_filter = RegimeFilter(
            rsi_lower=45,
            rsi_upper=55,
            atr_max=0.11,
            return_threshold=0.02
        )
    
    # Load config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
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
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Add predictions to test_df
        test_df['pred_proba'] = y_pred_proba
        test_df['pred'] = y_pred
        
        # Apply regime filter if enabled
        if use_regime_filter:
            print("\nApplying regime filter...")
            
            # Check each bar for favorable regime
            regime_allowed = []
            for i in range(len(test_df)):
                current_idx = test_idx[i]
                historical_df = df.iloc[:current_idx + 1]
                
                should_trade, _ = regime_filter.should_trade(historical_df, verbose=False)
                regime_allowed.append(should_trade)
            
            test_df['regime_allowed'] = regime_allowed
            
            n_allowed = sum(regime_allowed)
            pct_allowed = n_allowed / len(regime_allowed) * 100
            print(f"  Regime check: {n_allowed}/{len(regime_allowed)} bars allowed ({pct_allowed:.1f}%)")
            
            test_df_filtered = test_df[test_df['regime_allowed']].copy()
        else:
            test_df_filtered = test_df.copy()
        
        # Metrics on allowed trades
        if len(test_df_filtered) > 0:
            auc = roc_auc_score(test_df_filtered['y'], test_df_filtered['pred_proba'])
            acc = accuracy_score(test_df_filtered['y'], test_df_filtered['pred'])
            print(f"  Metrics: AUC={auc:.3f}, Acc={acc:.3f}")
        else:
            print("  No trades allowed by regime filter!")
            auc, acc = np.nan, np.nan
        
        # Backtest
        print("\nBacktesting...")
        
        bt_result = backtest(test_df_filtered, test_df_filtered['pred_proba'].values, cfg)
        
        if bt_result['trades'] > 0:
            n_trades = bt_result['trades']
            win_rate = bt_result['winrate']
            
            equity = bt_result['equity']
            initial_bal = cfg.get('initial_balance', 100000)
            total_pnl = equity - initial_bal
            wins = int(n_trades * win_rate)
            losses = n_trades - wins
            
            print(f"  Trades:      {n_trades}")
            print(f"  PnL:         ${total_pnl:,.0f}")
            print(f"  Win Rate:    {win_rate*100:.1f}% ({wins}W / {losses}L)")
            print(f"  Max DD:      {bt_result['max_dd']*100:.2f}%")
        else:
            print("  No trades executed!")
            total_pnl = 0
            win_rate = 0
            wins, losses = 0, 0
            n_trades = 0
        
        # Store results
        fold_result = {
            'fold': fold_idx + 1,
            'train_bars': len(train_idx),
            'test_bars': len(test_idx),
            'allowed_bars': len(test_df_filtered),
            'pct_allowed': len(test_df_filtered) / len(test_idx) * 100 if len(test_idx) > 0 else 0,
            'auc': auc,
            'accuracy': acc,
            'n_trades': n_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'wins': wins,
            'losses': losses
        }
        all_results.append(fold_result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY - ALL FOLDS")
    print(f"{'='*80}")
    
    results_df = pd.DataFrame(all_results)
    
    # Print table
    print(f"\n{'Fold':<6} {'PnL':<12} {'Trades':<8} {'Win%':<8} {'Allowed%':<10} {'AUC':<6}")
    print(f"{''*60}")
    for _, row in results_df.iterrows():
        print(f"{int(row['fold']):<6} ${row['total_pnl']:>10,.0f} {int(row['n_trades']):<8} "
              f"{row['win_rate']*100:>5.1f}%  {row['pct_allowed']:>6.1f}%     {row['auc']:.3f}")
    
    print(f"{''*60}")
    
    # Totals
    total_pnl = results_df['total_pnl'].sum()
    total_trades = results_df['n_trades'].sum()
    avg_auc = results_df['auc'].mean()
    avg_allowed_pct = results_df['pct_allowed'].mean()
    profitable_folds = (results_df['total_pnl'] > 0).sum()
    
    print(f"{'TOTAL':<6} ${total_pnl:>10,.0f} {int(total_trades):<8} "
          f"         {avg_allowed_pct:>6.1f}%     {avg_auc:.3f}")
    print(f"\nProfitable Folds: {profitable_folds}/{len(results_df)}")
    print(f"{'='*80}\n")
    
    # Print regime filter statistics if used
    if use_regime_filter:
        regime_filter.print_statistics()
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    suffix = '_ensemble_regime' if use_regime_filter else '_ensemble_no_filter'
    results_df.to_csv(output_dir / f'{symbol}_walkforward{suffix}.csv', index=False)
    print(f"\nResults saved to: {output_dir / f'{symbol}_walkforward{suffix}.csv'}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Walk-forward validation with ensemble')
    parser.add_argument('--symbol', type=str, default='USDJPY', help='Trading symbol')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of walk-forward folds')
    parser.add_argument('--use_regime_filter', type=int, default=1, help='1=use filter, 0=no filter')
    parser.add_argument('--max_features', type=int, default=30, help='Max features per fold')
    
    args = parser.parse_args()
    
    run_walk_forward_ensemble(
        symbol=args.symbol,
        n_splits=args.n_splits,
        use_regime_filter=bool(args.use_regime_filter),
        max_features=args.max_features
    )


if __name__ == '__main__':
    main()

