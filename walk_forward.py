"""
Walk-forward validation for forex trading models.

Implements rolling window backtesting to avoid look-ahead bias:
1. Train on expanding or rolling window
2. Test on next out-of-sample period
3. Retrain and repeat

This gives a more realistic estimate of live performance.
"""
import argparse, os, yaml, joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from datetime import datetime

def walk_forward_split(df, n_splits=5, train_size=0.6, test_size=0.1):
    """Create walk-forward splits.
    
    Args:
        df: DataFrame with time series data
        n_splits: Number of walk-forward folds
        train_size: Initial training set size (as fraction)
        test_size: Test set size per fold (as fraction)
    
    Returns:
        List of (train_idx, test_idx) tuples
    """
    n = len(df)
    initial_train = int(n * train_size)
    fold_size = int(n * test_size)
    
    splits = []
    for i in range(n_splits):
        test_start = initial_train + i * fold_size
        test_end = test_start + fold_size
        
        if test_end > n:
            break
        
        # Expanding window: use all data up to test_start
        train_idx = list(range(0, test_start))
        test_idx = list(range(test_start, test_end))
        
        splits.append((train_idx, test_idx))
    
    return splits

def train_rf_model(X_train, y_train, cfg):
    """Train RandomForest with config settings."""
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf

def calculate_trade_costs(entry_price, pip_size, cfg, symbol=''):
    """Calculate trading costs per round-turn."""
    spread_pips = cfg['spread_pips'] * 2
    slippage_pips = cfg['slippage_pips'] * 2
    total_pips = spread_pips + slippage_pips
    
    if 'JPY' in symbol.upper():
        pip_value_usd = (pip_size / entry_price) * 100000 if entry_price > 0 else 10.0
    else:
        pip_value_usd = pip_size * 100000
    
    cost_from_pips = total_pips * pip_value_usd
    commission = cfg.get('commission_per_lot', 7.0)
    
    return cost_from_pips + commission

def backtest_fold(df_fold, proba, cfg, symbol):
    """Backtest a single fold with strategy."""
    prob_buy = cfg['prob_buy']
    prob_sell = cfg['prob_sell']
    stop_k = cfg['stop_atr_mult']
    tp_k = cfg['tp_atr_mult']
    
    equity = 10000.0
    pos = 0
    entry_price = None
    entry_bar = None
    trades = []
    
    prices = df_fold['close'].values
    atr = df_fold['atr14'].values
    pip = 0.0001 if 'EUR' in symbol else 0.01
    
    for i in range(len(df_fold) - 1):
        if i >= len(proba):
            break
            
        p = proba[i]
        price = prices[i + 1]
        bar_atr = atr[i]
        
        # Exit logic
        if pos != 0 and entry_price is not None and entry_bar is not None:
            stop = entry_price - stop_k * bar_atr if pos > 0 else entry_price + stop_k * bar_atr
            tp = entry_price + tp_k * bar_atr if pos > 0 else entry_price - tp_k * bar_atr
            
            high = df_fold['high'].iloc[i]
            low = df_fold['low'].iloc[i]
            
            pnl = 0.0
            exit_now = False
            
            if pos > 0:
                if low <= stop:
                    exit_now = True
                    pnl = stop - entry_price
                elif high >= tp:
                    exit_now = True
                    pnl = tp - entry_price
            else:
                if high >= stop:
                    exit_now = True
                    pnl = entry_price - stop
                elif low <= tp:
                    exit_now = True
                    pnl = entry_price - tp
            
            if not exit_now and (i - entry_bar) >= 3:
                exit_now = True
                exit_price = prices[i]
                pnl = (exit_price - entry_price) if pos > 0 else (entry_price - exit_price)
            
            if exit_now:
                pnl_gross = pnl * 100000
                costs = calculate_trade_costs(entry_price, pip, cfg, symbol)
                pnl_net = pnl_gross - costs
                
                equity += pnl_net
                trades.append(pnl_net)
                pos = 0
                entry_price = None
                entry_bar = None
        
        # Entry logic
        if pos == 0:
            if p >= prob_buy:
                pos = +1
                entry_price = price
                entry_bar = i
            elif p <= prob_sell:
                pos = -1
                entry_price = price
                entry_bar = i
    
    # Calculate metrics
    if trades:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        winrate = len(wins) / len(trades) if trades else 0
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else (np.inf if wins else 0)
    else:
        winrate = pf = 0
    
    return {
        'equity': equity,
        'trades': len(trades),
        'winrate': winrate,
        'profit_factor': pf,
        'total_pnl': equity - 10000
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--n_splits', type=int, default=5, help='Number of walk-forward folds')
    parser.add_argument('--train_size', type=float, default=0.6, help='Initial training size')
    parser.add_argument('--test_size', type=float, default=0.1, help='Test size per fold')
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open('config.yaml'))
    feat_dir = cfg['feature_dir']
    
    csv_path = os.path.join(feat_dir, f"{args.symbol}_features.csv")
    df = pd.read_csv(csv_path, parse_dates=['time'])
    df.attrs['symbol'] = args.symbol
    
    feature_cols = [c for c in df.columns if c not in ('time', 'y')]
    X = df[feature_cols].values
    y = df['y'].values
    
    print("=" * 80)
    print(f"Walk-Forward Validation: {args.symbol}")
    print("=" * 80)
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Walk-forward splits: {args.n_splits}")
    print(f"Initial train size: {args.train_size:.1%}")
    print(f"Test size per fold: {args.test_size:.1%}")
    print()
    
    # Create walk-forward splits
    splits = walk_forward_split(df, args.n_splits, args.train_size, args.test_size)
    print(f"Created {len(splits)} folds")
    print()
    
    # Results storage
    fold_results = []
    all_predictions = []
    all_actuals = []
    
    # Walk forward through splits
    for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"Fold {fold_idx}/{len(splits)}")
        print("-" * 80)
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        df_test = df.iloc[test_idx].reset_index(drop=True)
        
        train_period = f"{df.iloc[train_idx[0]]['time']} to {df.iloc[train_idx[-1]]['time']}"
        test_period = f"{df.iloc[test_idx[0]]['time']} to {df.iloc[test_idx[-1]]['time']}"
        
        print(f"  Train: {len(train_idx)} samples ({train_period[:10]} to {train_period[-10:]})")
        print(f"  Test:  {len(test_idx)} samples ({test_period[:10]} to {test_period[-10:]})")
        
        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Train
        model = train_rf_model(X_train_s, y_train, cfg)
        
        # Predict
        proba = model.predict_proba(X_test_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
        
        # Evaluate predictions
        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, pred)
        
        print(f"  AUC: {auc:.4f}  ACC: {acc:.4f}")
        
        # Backtest
        backtest_result = backtest_fold(df_test, proba, cfg, args.symbol)
        
        print(f"  Trades: {backtest_result['trades']}  "
              f"WinRate: {backtest_result['winrate']:.2%}  "
              f"PF: {backtest_result['profit_factor']:.2f}  "
              f"PnL: ${backtest_result['total_pnl']:,.0f}")
        print()
        
        # Store results
        fold_results.append({
            'fold': fold_idx,
            'train_samples': len(train_idx),
            'test_samples': len(test_idx),
            'auc': auc,
            'acc': acc,
            **backtest_result
        })
        
        all_predictions.extend(proba)
        all_actuals.extend(y_test)
    
    # Summary statistics
    print("=" * 80)
    print("Walk-Forward Summary")
    print("=" * 80)
    
    df_results = pd.DataFrame(fold_results)
    
    print("\nPer-Fold Results:")
    print(df_results.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Aggregate Statistics")
    print("=" * 80)
    
    total_trades = df_results['trades'].sum()
    avg_auc = df_results['auc'].mean()
    avg_acc = df_results['acc'].mean()
    total_pnl = df_results['total_pnl'].sum()
    avg_winrate = df_results['winrate'].mean()
    avg_pf = df_results[df_results['profit_factor'] < np.inf]['profit_factor'].mean()
    
    overall_auc = roc_auc_score(all_actuals, all_predictions)
    
    print(f"Total Trades:        {total_trades}")
    print(f"Average AUC:         {avg_auc:.4f}")
    print(f"Overall AUC:         {overall_auc:.4f}")
    print(f"Average Accuracy:    {avg_acc:.4f}")
    print(f"Average Win Rate:    {avg_winrate:.2%}")
    print(f"Average Profit Factor: {avg_pf:.2f}")
    print(f"Total PnL:           ${total_pnl:,.0f}")
    print(f"Final Equity:        ${10000 * len(splits) + total_pnl:,.0f}")
    
    profitable_folds = (df_results['total_pnl'] > 0).sum()
    print(f"\nProfitable Folds:    {profitable_folds}/{len(splits)} ({profitable_folds/len(splits):.1%})")
    
    # Save results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{args.symbol}_walkforward.csv")
    df_results.to_csv(results_path, index=False)
    print(f"\n Results saved to {results_path}")

if __name__ == '__main__':
    main()

