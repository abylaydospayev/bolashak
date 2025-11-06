"""
Walk-forward validation using LSTM with regime filter.
"""
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf

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


def create_sequences(X, y, lookback=60):
    """Create sequences for LSTM."""
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def build_lstm_model(input_shape):
    """Build LSTM model."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def run_walk_forward_lstm(symbol, n_splits, lookback=60,
                          use_regime_filter=True,
                          rsi_lower=45, rsi_upper=55,
                          atr_max=0.11, return_threshold=0.02):
    """Run walk-forward validation using LSTM."""
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD VALIDATION: {symbol}")
    print(f"LSTM (lookback={lookback})")
    if use_regime_filter:
        print(f"WITH REGIME FILTER (RSI {rsi_lower}-{rsi_upper}, ATR<{atr_max}, Return<{return_threshold*100:.0f}%)")
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
        
        # Create sequences
        print(f"Creating sequences (lookback={lookback})...")
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, lookback)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, lookback)
        
        print(f"  Train sequences: {len(X_train_seq):,}")
        print(f"  Test sequences: {len(X_test_seq):,}")
        
        # Build and train LSTM
        print("Training LSTM...")
        model = build_lstm_model(input_shape=(lookback, X_train.shape[1]))
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=30,
            batch_size=32,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0
        )
        
        print(f"  Best epoch: {len(history.history['loss']) - 5}")
        
        # Predict
        print("Predicting...")
        y_pred_proba = model.predict(X_test_seq, verbose=0)[:, 1]
        
        # Align predictions with test_df (account for lookback offset)
        test_df_aligned = test_df.iloc[lookback:].copy()
        test_df_aligned['pred_proba'] = y_pred_proba
        test_df_aligned['pred'] = (y_pred_proba > 0.5).astype(int)
        
        # Apply regime filter if enabled
        if use_regime_filter:
            print("\nApplying regime filter...")
            
            # Check each bar for favorable regime
            regime_allowed = []
            for i in test_df_aligned.index:
                # Get data up to current bar
                historical_df = df.loc[:i]
                should_trade, _ = regime_filter.should_trade(historical_df, verbose=False)
                regime_allowed.append(should_trade)
            
            test_df_aligned['regime_allowed'] = regime_allowed
            
            # Stats
            n_allowed = sum(regime_allowed)
            pct_allowed = n_allowed / len(regime_allowed) * 100 if len(regime_allowed) > 0 else 0
            print(f"  Regime check: {n_allowed}/{len(regime_allowed)} bars allowed ({pct_allowed:.1f}%)")
            
            # Filter signals
            test_df_final = test_df_aligned[test_df_aligned['regime_allowed']].copy()
        else:
            test_df_final = test_df_aligned.copy()
        
        # Metrics
        if len(test_df_final) > 0:
            auc = roc_auc_score(test_df_final['y'], test_df_final['pred_proba'])
            acc = accuracy_score(test_df_final['y'], test_df_final['pred'])
            print(f"\nMetrics (final {len(test_df_final)} bars):")
            print(f"  AUC:      {auc:.3f}")
            print(f"  Accuracy: {acc:.3f}")
        else:
            print("\nNo trades allowed after filters!")
            auc, acc = np.nan, np.nan
        
        # Backtest
        if len(test_df_final) > 0:
            print("\nBacktesting...")
            
            # Load config
            with open('config.yaml', 'r') as f:
                cfg = yaml.safe_load(f)
            
            # Run backtest
            bt_result = backtest(test_df_final, test_df_final['pred_proba'].values, cfg)
            
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
                
                print(f"  Trades:        {n_trades}")
                print(f"  PnL:           ${total_pnl:,.0f}")
                print(f"  Win Rate:      {win_rate*100:.1f}% ({wins}W / {losses}L)")
                print(f"  Avg Win:       ${avg_win:,.0f}")
                print(f"  Avg Loss:      ${avg_loss:,.0f}")
                print(f"  Max DD:        {bt_result['max_dd']*100:.2f}%")
                print(f"  Profit Factor: {bt_result['profit_factor']:.2f}")
            else:
                print("  No trades executed!")
                total_pnl = 0
                win_rate = 0
                wins, losses = 0, 0
                avg_win, avg_loss = 0, 0
                n_trades = 0
        else:
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
            'test_sequences': len(X_test_seq),
            'regime_filtered': len(test_df_final),
            'pct_traded': len(test_df_final) / len(X_test_seq) * 100 if len(X_test_seq) > 0 else 0,
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
    print(f"\n{'Fold':<6} {'PnL':<12} {'Trades':<8} {'Win%':<8} {'Traded%':<10} {'AUC':<6}")
    print(f"{''*70}")
    for _, row in results_df.iterrows():
        print(f"{int(row['fold']):<6} ${row['total_pnl']:>10,.0f} {int(row['n_trades']):<8} "
              f"{row['win_rate']*100:>5.1f}%  {row['pct_traded']:>6.1f}%     {row['auc']:.3f}")
    
    print(f"{''*70}")
    
    # Totals
    total_pnl = results_df['total_pnl'].sum()
    total_trades = results_df['n_trades'].sum()
    avg_auc = results_df['auc'].mean()
    avg_traded_pct = results_df['pct_traded'].mean()
    profitable_folds = (results_df['total_pnl'] > 0).sum()
    
    print(f"{'TOTAL':<6} ${total_pnl:>10,.0f} {int(total_trades):<8} "
          f"         {avg_traded_pct:>6.1f}%     {avg_auc:.3f}")
    print(f"\nProfitable Folds: {profitable_folds}/{len(results_df)}")
    
    if profitable_folds > 0:
        print(" SUCCESS: Some folds profitable!")
    else:
        print("  WARNING: No profitable folds")
    
    print(f"{'='*80}\n")
    
    # Print regime filter statistics if used
    if use_regime_filter:
        regime_filter.print_statistics()
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    suffix = '_lstm_regime'
    results_df.to_csv(output_dir / f'{symbol}_walkforward{suffix}.csv', index=False)
    print(f"\nResults saved to: {output_dir / f'{symbol}_walkforward{suffix}.csv'}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Walk-forward with LSTM + regime filter')
    parser.add_argument('--symbol', type=str, default='USDJPY', help='Trading symbol')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of walk-forward folds')
    parser.add_argument('--lookback', type=int, default=60, help='LSTM lookback period')
    parser.add_argument('--use_regime_filter', type=int, default=1, help='1=use filter, 0=no filter')
    parser.add_argument('--rsi_lower', type=float, default=45, help='Minimum RSI')
    parser.add_argument('--rsi_upper', type=float, default=55, help='Maximum RSI')
    parser.add_argument('--atr_max', type=float, default=0.11, help='Maximum ATR')
    parser.add_argument('--return_threshold', type=float, default=0.02, help='Maximum abs return')
    
    args = parser.parse_args()
    
    run_walk_forward_lstm(
        symbol=args.symbol,
        n_splits=args.n_splits,
        lookback=args.lookback,
        use_regime_filter=bool(args.use_regime_filter),
        rsi_lower=args.rsi_lower,
        rsi_upper=args.rsi_upper,
        atr_max=args.atr_max,
        return_threshold=args.return_threshold
    )


if __name__ == '__main__':
    main()

