"""
Train EURUSD Ensemble Model
Matches the USDJPY training approach for consistency
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from make_dataset import load_ohlcv
from build_features_enhanced import add_multi_timeframe_features
from indicators import ema, rsi, atr, pct_change, sincos_time
from ensemble import EnsembleClassifier

def create_target(df, forward_bars=5, threshold_pips=10):
    """
    Create binary target for EURUSD
    1 = Price will go UP by threshold in next N bars
    0 = Otherwise
    
    EURUSD: 1 pip = 0.0001
    threshold_pips = 10 means 10 pips = 0.0010
    """
    pip_size = 0.0001
    threshold = threshold_pips * pip_size
    
    df['future_close'] = df['close'].shift(-forward_bars)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    df['target'] = (df['future_return'] > threshold).astype(int)
    
    return df

def train_eurusd_model():
    """Train EURUSD ensemble model (matches USDJPY approach)"""
    
    print("\n" + "="*70)
    print(" TRAINING EURUSD ENSEMBLE MODEL")
    print("="*70 + "\n")
    
    symbol = 'EURUSD.sim'
    
    # 1. Load data
    print(f" Loading {symbol} data...")
    df_m15 = load_ohlcv(symbol, timeframe='M15')
    print(f"   M15: {len(df_m15)} bars ({df_m15.index[0]} to {df_m15.index[-1]})")
    
    df_m30 = load_ohlcv(symbol, timeframe='M30')
    print(f"   M30: {len(df_m30)} bars")
    
    df_h1 = load_ohlcv(symbol, timeframe='H1')
    print(f"   H1: {len(df_h1)} bars")
    
    df_h4 = load_ohlcv(symbol, timeframe='H4')
    print(f"   H4: {len(df_h4)} bars")
    
    # 2. Build M15 base features
    print("\n Building M15 base features...")
    df_m15['ema20'] = ema(df_m15['close'], 20)
    df_m15['ema50'] = ema(df_m15['close'], 50)
    df_m15['rsi14'] = rsi(df_m15['close'], 14)
    df_m15['atr14'] = atr(df_m15, 14)
    df_m15['ema50_slope'] = df_m15['ema50'].diff(5)
    
    # Time features
    time_col = df_m15.index if df_m15.index.tz is not None else df_m15.index.tz_localize('UTC')
    sin_h, cos_h = sincos_time(time_col)
    df_m15['sin_hour'] = sin_h.values
    df_m15['cos_hour'] = cos_h.values
    
    # Returns and volatility
    df_m15['ret1'] = pct_change(df_m15['close'], 1)
    df_m15['atr_pct'] = df_m15['atr14'] / df_m15['close']
    
    # Price vs EMAs
    df_m15['price_vs_ema20'] = (df_m15['close'] - df_m15['ema20']) / df_m15['close']
    df_m15['price_vs_ema50'] = (df_m15['close'] - df_m15['ema50']) / df_m15['close']
    
    print(f"   Created {len(df_m15.columns)} M15 features")
    
    # 3. Add multi-timeframe features
    print("\n Adding multi-timeframe features...")
    df_enhanced = add_multi_timeframe_features(df_m15, df_m30, 'm30')
    df_enhanced = add_multi_timeframe_features(df_enhanced, df_h1, 'h1')
    df_enhanced = add_multi_timeframe_features(df_enhanced, df_h4, 'h4')
    
    # 4. Add market structure features
    print("\n Adding market structure features...")
    df_enhanced['higher_high'] = (df_enhanced['high'] > df_enhanced['high'].shift(1)).astype(int)
    df_enhanced['lower_low'] = (df_enhanced['low'] < df_enhanced['low'].shift(1)).astype(int)
    
    # Swing lows
    df_enhanced['swing_low'] = ((df_enhanced['low'] < df_enhanced['low'].shift(1)) & 
                                (df_enhanced['low'] < df_enhanced['low'].shift(2)) &
                                (df_enhanced['low'] < df_enhanced['low'].shift(-1)) & 
                                (df_enhanced['low'] < df_enhanced['low'].shift(-2))).astype(int)
    
    print(f"   Total features: {len(df_enhanced.columns)}")
    
    # 5. Create target
    print("\n Creating target variable...")
    df_enhanced = create_target(df_enhanced, forward_bars=5, threshold_pips=10)
    
    # Drop NaN
    df_clean = df_enhanced.dropna()
    print(f"   Clean data: {len(df_clean)} rows")
    
    # Target distribution
    target_dist = df_clean['target'].value_counts()
    print(f"\n   Target distribution:")
    print(f"   Class 0 (Down/Neutral): {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(df_clean)*100:.1f}%)")
    print(f"   Class 1 (Up): {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(df_clean)*100:.1f}%)")
    
    # 6. Define feature columns (match USDJPY order)
    feature_cols = [
        'price_vs_ema20_h1', 'momentum_5_h1', 'momentum_10_h1', 'rsi14_h1', 'trend_strength_h1',
        'ema20_h1', 'atr14_h1', 'ema50_h1', 'atr_pct_h1', 'momentum_5_m30', 'ema50_m30',
        'ema20_m30', 'momentum_10_m30', 'price_vs_ema20_m30', 'rsi14_m30', 'trend_strength_m30',
        'atr14_m30', 'ema20_h4', 'atr_pct_h4', 'rsi14_h4', 'price_vs_ema20_h4', 'trend_strength_h4',
        'ema50_h4', 'atr_pct_m30', 'momentum_10_h4', 'momentum_5_h4', 'atr14_h4', 'swing_low',
        'higher_high', 'trend_ema_m30', 'open', 'high', 'low', 'close', 'volume', 'rsi14', 'atr14'
    ]
    
    # Check which features exist
    available_features = [col for col in feature_cols if col in df_clean.columns]
    missing_features = [col for col in feature_cols if col not in df_clean.columns]
    
    if missing_features:
        print(f"\n  Missing features: {len(missing_features)}")
        print(f"   Using {len(available_features)} available features")
        feature_cols = available_features
    
    X = df_clean[feature_cols]
    y = df_clean['target']
    
    print(f"\n Feature matrix shape: {X.shape}")
    
    # 7. Train/test split (80/20, time-based)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"\n  Data split:")
    print(f"   Training: {len(X_train)} samples ({X_train.index[0]} to {X_train.index[-1]})")
    print(f"   Testing: {len(X_test)} samples ({X_test.index[0]} to {X_test.index[-1]})")
    
    # 8. Train ensemble model
    print("\n Training ensemble model...")
    print("   This may take a few minutes...")
    
    ensemble = EnsembleClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=50,
        min_samples_leaf=20,
        scale_features=True,
        random_state=42
    )
    
    ensemble.fit(X_train, y_train)
    
    # 9. Evaluate on test set
    print("\n EVALUATION RESULTS (Out-of-Sample):")
    print("="*70)
    
    # Predictions
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    
    # Accuracy
    accuracy = (y_pred == y_test).sum() / len(y_test)
    print(f"\n Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Win rate per class
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down/Neutral', 'Up']))
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f" ROC AUC: {roc_auc:.4f}")
    except:
        print("  Could not calculate ROC AUC")
        roc_auc = None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Down   Up")
    print(f"Actual Down   {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"Actual Up     {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Win rate when model predicts UP
    up_predictions = y_pred == 1
    if up_predictions.sum() > 0:
        win_rate_up = (y_test[up_predictions] == 1).sum() / up_predictions.sum()
        print(f"\n Win Rate (when predicting UP): {win_rate_up:.4f} ({win_rate_up*100:.2f}%)")
    
    # Win rate when model predicts DOWN
    down_predictions = y_pred == 0
    if down_predictions.sum() > 0:
        win_rate_down = (y_test[down_predictions] == 0).sum() / down_predictions.sum()
        print(f" Win Rate (when predicting DOWN): {win_rate_down:.4f} ({win_rate_down*100:.2f}%)")
    
    # 10. Feature importance
    print("\n Top 10 Most Important Features:")
    feature_importance = ensemble.get_feature_importance()
    if feature_importance is not None:
        feature_imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_imp_df.head(10).iterrows():
            print(f"   {row['feature']:30s} {row['importance']:.4f}")
    
    # 11. Save model
    print("\n Saving model...")
    Path('models').mkdir(exist_ok=True)
    
    model_path = f'models/EURUSD_ensemble_oos.pkl'
    joblib.dump(ensemble, model_path)
    print(f"    Saved: {model_path}")
    
    # Save feature list
    feature_file = 'models/EURUSD_features.txt'
    with open(feature_file, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"    Saved: {feature_file}")
    
    # Save results summary
    results_file = 'models/EURUSD_ensemble_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"EURUSD Ensemble Model Results\n")
        f.write(f"={'='*70}\n")
        f.write(f"Trained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Testing samples: {len(X_test)}\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        if roc_auc:
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
        if up_predictions.sum() > 0:
            f.write(f"Win Rate (UP): {win_rate_up:.4f} ({win_rate_up*100:.2f}%)\n")
        if down_predictions.sum() > 0:
            f.write(f"Win Rate (DOWN): {win_rate_down:.4f} ({win_rate_down*100:.2f}%)\n")
    
    print(f"    Saved: {results_file}")
    
    print("\n" + "="*70)
    print(" EURUSD MODEL TRAINING COMPLETE!")
    print("="*70 + "\n")
    
    print(f" Final Results:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    if roc_auc:
        print(f"   ROC AUC: {roc_auc:.4f}")
    if up_predictions.sum() > 0:
        print(f"   Win Rate (UP): {win_rate_up:.4f} ({win_rate_up*100:.2f}%)")
    print(f"\n Model saved and ready for live trading!")
    
    return ensemble, accuracy, roc_auc if roc_auc else 0

if __name__ == "__main__":
    model, acc, auc = train_eurusd_model()
    
    print(f"\n Next Steps:")
    print(f"   1. Run tests: pytest tests/test_eurusd_model.py -v")
    print(f"   2. Update bot to use EURUSD model")
    print(f"   3. Test on demo account before live trading")

