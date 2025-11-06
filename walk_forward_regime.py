"""
Walk-forward validation WITH REGIME FILTER.

Tests the impact of only trading in favorable market conditions.
"""
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from datetime import datetime

from regime_filter import RegimeFilter
from backtest import backtest
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


def train_rf_model(X_train, y_train):
    """Train RandomForest."""
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def run_walk_forward_with_regime_filter(symbol, n_splits, 
                                        use_regime_filter=True,
                                        rsi_lower=45, rsi_upper=55,
                                        atr_max=0.11, return_threshold=0.02):
    """
    Run walk-forward validation with optional regime filtering.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol
    n_splits : int
        Number of walk-forward folds
    use_regime_filter : bool
        If True, apply regime filter
    rsi_lower, rsi_upper : float
        RSI range for favorable regime
    atr_max : float
        Maximum ATR for favorable regime
    return_threshold : float
        Maximum absolute return for favorable regime
    """
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD VALIDATION: {symbol}")
    if use_regime_filter:
        print(f"WITH REGIME FILTER (RSI {rsi_lower}-{rsi_upper}, ATR<{atr_max}, Return<{return_threshold*100:.0f}%)")
    else:
        print("WITHOUT REGIME FILTER (trade all signals)")
    print(f"{'='*80}\n")
    
    # Load features
    feat_path = Path('features') / f'{symbol}_features.csv'
    df = pd.read_csv(feat_path)
    df = df.dropna()
    
    print(f"Loaded {len(df):,} bars")
    
    # Prepare features
    feature_cols = [c for c in df.columns 
                   if c not in ['y', 'time', 'fwd_ret', 'fwd_ret_net']]
    
    X = df[feature_cols].values
    y = df['y'].values
    
    # Create splits
    splits = walk_forward_split(df, n_splits=n_splits)
    print(f"Created {len(splits)} walk-forward folds\n")
    
    # Initialize regime filter
    if use_regime_filter:
        regime_filter = RegimeFilter(
            rsi_lower=rsi_lower,
            rsi_upper=rsi_upper,
            atr_max=atr_max,
            return_threshold=return_threshold
        )
    
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
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        print("Training RandomForest...")
        rf = train_rf_model(X_train_scaled, y_train)
        
        # Predict
        y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
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
                # Get data up to current bar (including train period for context)
                current_idx = test_idx[i]
                historical_df = df.iloc[:current_idx + 1]
                
                should_trade, _ = regime_filter.should_trade(historical_df, verbose=False)
                regime_allowed.append(should_trade)
            
            test_df['regime_allowed'] = regime_allowed
            
            # Stats
            n_allowed = sum(regime_allowed)
            pct_allowed = n_allowed / len(regime_allowed) * 100
            print(f"  Regime check: {n_allowed}/{len(regime_allowed)} bars allowed ({pct_allowed:.1f}%)")
            
            # Filter signals
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
        
        # Load config for backtest
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Run backtest
        bt_result = backtest(test_df_filtered, test_df_filtered['pred_proba'].values, cfg)
        
        if bt_result['trades'] > 0:
            n_trades = bt_result['trades']
            win_rate = bt_result['winrate']
            avg_win = bt_result['avg_win']
            avg_loss = bt_result['avg_loss']
            
            # Calculate total PnL
            equity = bt_result['equity']
            initial_bal = cfg.get('initial_balance', 100000)
            total_pnl = equity - initial_bal
            wins = int(n_trades * win_rate)
            losses = n_trades - wins
            
            print(f"  Trades:      {n_trades}")
            print(f"  PnL:         ${total_pnl:,.0f}")
            print(f"  Win Rate:    {win_rate*100:.1f}% ({wins}W / {losses}L)")
            print(f"  Avg Win:     ${avg_win:,.0f}")
            print(f"  Avg Loss:    ${avg_loss:,.0f}")
            print(f"  Max DD:      {bt_result['max_dd']*100:.2f}%")
        else:
            print("  No trades executed!")
            total_pnl = 0
            win_rate = 0
            wins, losses = 0, 0
            avg_win, avg_loss = 0, 0
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
            'losses': losses,
            'avg_win': avg_win,
            'avg_loss': avg_loss
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
    
    suffix = '_regime_filtered' if use_regime_filter else '_no_filter'
    results_df.to_csv(output_dir / f'{symbol}_walkforward{suffix}.csv', index=False)
    print(f"\nResults saved to: {output_dir / f'{symbol}_walkforward{suffix}.csv'}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Walk-forward validation with regime filter')
    parser.add_argument('--symbol', type=str, default='USDJPY.sim', help='Trading symbol')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of walk-forward folds')
    parser.add_argument('--use_regime_filter', type=int, default=1, help='1=use filter, 0=no filter')
    parser.add_argument('--rsi_lower', type=float, default=45, help='Minimum RSI')
    parser.add_argument('--rsi_upper', type=float, default=55, help='Maximum RSI')
    parser.add_argument('--atr_max', type=float, default=0.11, help='Maximum ATR')
    parser.add_argument('--return_threshold', type=float, default=0.02, help='Maximum abs return')
    
    args = parser.parse_args()
    
    run_walk_forward_with_regime_filter(
        symbol=args.symbol,
        n_splits=args.n_splits,
        use_regime_filter=bool(args.use_regime_filter),
        rsi_lower=args.rsi_lower,
        rsi_upper=args.rsi_upper,
        atr_max=args.atr_max,
        return_threshold=args.return_threshold
    )


if __name__ == '__main__':
    main()

