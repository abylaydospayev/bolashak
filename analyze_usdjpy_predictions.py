"""
Detailed Analysis of USDJPY Model Predictions
Understand prediction distribution and behavior
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from build_features_enhanced import add_multi_timeframe_features
from indicators import ema, rsi, atr, pct_change, sincos_time

def load_usdjpy_data():
    """Load USDJPY data from CSV files"""
    try:
        df_m15 = pd.read_csv('data/USDJPY.sim_M15.csv')
    except:
        df_m15 = pd.read_csv('data/USDJPY_M15.csv')
    
    df_m15['time'] = pd.to_datetime(df_m15['time'])
    df_m15 = df_m15.set_index('time')
    
    # Resample for other timeframes
    df_m30 = df_m15.resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_h1 = df_m15.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_h4 = df_m15.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return df_m15, df_m30, df_h1, df_h4

def build_features(df_m15, df_m30, df_h1, df_h4):
    """Build features matching training"""
    df_m15['ema20'] = ema(df_m15['close'], 20)
    df_m15['ema50'] = ema(df_m15['close'], 50)
    df_m15['rsi14'] = rsi(df_m15['close'], 14)
    df_m15['atr14'] = atr(df_m15, 14)
    df_m15['ema50_slope'] = df_m15['ema50'].diff(5)
    
    sin_h, cos_h = sincos_time(df_m15.index)
    df_m15['sin_hour'] = sin_h.values
    df_m15['cos_hour'] = cos_h.values
    
    df_m15['ret1'] = pct_change(df_m15['close'], 1)
    df_m15['atr_pct'] = df_m15['atr14'] / df_m15['close']
    df_m15['price_vs_ema20'] = (df_m15['close'] - df_m15['ema20']) / df_m15['close']
    df_m15['price_vs_ema50'] = (df_m15['close'] - df_m15['ema50']) / df_m15['close']
    
    # Multi-timeframe with reset index
    df_m15_reset = df_m15.reset_index()
    df_m30_reset = df_m30.reset_index()
    df_h1_reset = df_h1.reset_index()
    df_h4_reset = df_h4.reset_index()
    
    df_enhanced = add_multi_timeframe_features(df_m15_reset, df_m30_reset, 'm30')
    df_enhanced = add_multi_timeframe_features(df_enhanced, df_h1_reset, 'h1')
    df_enhanced = add_multi_timeframe_features(df_enhanced, df_h4_reset, 'h4')
    
    if 'time' in df_enhanced.columns:
        df_enhanced = df_enhanced.set_index('time')
    
    df_enhanced['higher_high'] = (df_enhanced['high'] > df_enhanced['high'].shift(1)).astype(int)
    df_enhanced['lower_low'] = (df_enhanced['low'] < df_enhanced['low'].shift(1)).astype(int)
    df_enhanced['swing_low'] = ((df_enhanced['low'] < df_enhanced['low'].shift(1)) & 
                                (df_enhanced['low'] < df_enhanced['low'].shift(2)) &
                                (df_enhanced['low'] < df_enhanced['low'].shift(-1)) & 
                                (df_enhanced['low'] < df_enhanced['low'].shift(-2))).astype(int)
    
    return df_enhanced

def create_target(df, forward_bars=5):
    """Create target variable"""
    threshold_pct = 0.0005  # 0.05% = ~7.5 pips at 150.00
    
    df['future_close'] = df['close'].shift(-forward_bars)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    df['target'] = (df['future_return'] > threshold_pct).astype(int)
    
    return df

def analyze_predictions(model, X, y, df_clean):
    """Analyze model predictions"""
    print("\n" + "="*70)
    print("PREDICTION ANALYSIS")
    print("="*70)
    
    # Get predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    print(f"\n1. PREDICTION DISTRIBUTION:")
    print(f"   Predicted UP (1):   {(y_pred==1).sum()} ({(y_pred==1).sum()/len(y_pred)*100:.1f}%)")
    print(f"   Predicted DOWN (0): {(y_pred==0).sum()} ({(y_pred==0).sum()/len(y_pred)*100:.1f}%)")
    
    print(f"\n2. PROBABILITY DISTRIBUTION:")
    print(f"   Mean prob UP:   {y_proba[:, 1].mean():.4f}")
    print(f"   Median prob UP: {np.median(y_proba[:, 1]):.4f}")
    print(f"   Std prob UP:    {y_proba[:, 1].std():.4f}")
    print(f"   Min prob UP:    {y_proba[:, 1].min():.4f}")
    print(f"   Max prob UP:    {y_proba[:, 1].max():.4f}")
    
    print(f"\n3. ACTUAL OUTCOME DISTRIBUTION:")
    print(f"   Actual UP (1):   {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    print(f"   Actual DOWN (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    
    # Confusion matrix
    print(f"\n4. CONFUSION MATRIX:")
    tp = ((y_pred==1) & (y==1)).sum()
    fp = ((y_pred==1) & (y==0)).sum()
    tn = ((y_pred==0) & (y==0)).sum()
    fn = ((y_pred==0) & (y==1)).sum()
    
    print(f"   True Positive (UP correctly):   {tp}")
    print(f"   False Positive (wrong UP):      {fp}")
    print(f"   True Negative (DOWN correctly): {tn}")
    print(f"   False Negative (missed UP):     {fn}")
    
    # Performance when using different thresholds
    print(f"\n5. THRESHOLD ANALYSIS:")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for thresh in thresholds:
        preds_at_thresh = (y_proba[:, 1] >= thresh).astype(int)
        if preds_at_thresh.sum() > 0:
            win_rate = (y[preds_at_thresh == 1] == 1).sum() / preds_at_thresh.sum()
            n_trades = preds_at_thresh.sum()
            print(f"   Threshold {thresh:.1f}: {n_trades:5d} trades, win rate: {win_rate:.3f}")
        else:
            print(f"   Threshold {thresh:.1f}: 0 trades")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Probability distribution
    ax = axes[0, 0]
    ax.hist(y_proba[:, 1], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', label='Default Threshold')
    ax.axvline(0.7, color='green', linestyle='--', label='Conservative (0.7)')
    ax.set_xlabel('Probability of UP')
    ax.set_ylabel('Frequency')
    ax.set_title('Probability Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Probability by outcome
    ax = axes[0, 1]
    ax.hist(y_proba[y==0, 1], bins=50, alpha=0.5, label='Actual DOWN', color='red')
    ax.hist(y_proba[y==1, 1], bins=50, alpha=0.5, label='Actual UP', color='green')
    ax.set_xlabel('Probability of UP')
    ax.set_ylabel('Frequency')
    ax.set_title('Probability by Actual Outcome')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Win rate by threshold
    ax = axes[1, 0]
    thresholds_fine = np.linspace(0.1, 0.9, 17)
    win_rates = []
    n_trades_list = []
    
    for thresh in thresholds_fine:
        preds_at_thresh = (y_proba[:, 1] >= thresh).astype(int)
        if preds_at_thresh.sum() > 0:
            win_rate = (y[preds_at_thresh == 1] == 1).sum() / preds_at_thresh.sum()
            n_trades = preds_at_thresh.sum()
        else:
            win_rate = 0
            n_trades = 0
        win_rates.append(win_rate)
        n_trades_list.append(n_trades)
    
    ax.plot(thresholds_fine, win_rates, 'o-', linewidth=2)
    ax.axhline(0.5, color='red', linestyle='--', label='Breakeven')
    ax.axvline(0.7, color='green', linestyle='--', alpha=0.5, label='Conservative (0.7)')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate vs Threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Number of trades by threshold
    ax = axes[1, 1]
    ax.plot(thresholds_fine, n_trades_list, 'o-', linewidth=2, color='orange')
    ax.axvline(0.7, color='green', linestyle='--', alpha=0.5, label='Conservative (0.7)')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Number of Trades')
    ax.set_title('Trade Frequency vs Threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/USDJPY_prediction_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: results/USDJPY_prediction_analysis.png")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        print(f"\n6. TOP 10 MOST IMPORTANT FEATURES:")
        importances = model.feature_importances_
        feature_names = X.columns
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_imp.head(10).iterrows():
            print(f"   {row['feature']:25s}: {row['importance']:.4f}")

def main():
    print("\n" + "="*70)
    print("🔍 USDJPY MODEL PREDICTION ANALYSIS")
    print("="*70)
    
    # Load model
    print("\n📦 Loading model...")
    model = joblib.load('models/USDJPY_ensemble_oos.pkl')
    
    # Load data
    print("\n📊 Loading data...")
    df_m15, df_m30, df_h1, df_h4 = load_usdjpy_data()
    
    # Build features
    print("\n🔧 Building features...")
    df_features = build_features(df_m15, df_m30, df_h1, df_h4)
    df_features = create_target(df_features)
    df_clean = df_features.dropna()
    
    # Get feature columns
    feature_cols = [
        'price_vs_ema20_h1', 'momentum_5_h1', 'momentum_10_h1', 'rsi14_h1', 'trend_strength_h1',
        'ema20_h1', 'atr14_h1', 'ema50_h1', 'atr_pct_h1', 'momentum_5_m30', 'ema50_m30',
        'ema20_m30', 'momentum_10_m30', 'price_vs_ema20_m30', 'rsi14_m30', 'trend_strength_m30',
        'atr14_m30', 'ema20_h4', 'atr_pct_h4', 'rsi14_h4', 'price_vs_ema20_h4', 'trend_strength_h4',
        'ema50_h4', 'atr_pct_m30', 'momentum_10_h4', 'momentum_5_h4', 'atr14_h4', 'swing_low',
        'higher_high', 'trend_ema_m30', 'open', 'high', 'low', 'close', 'volume', 'rsi14', 'atr14'
    ]
    
    available_features = [col for col in feature_cols if col in df_clean.columns]
    X = df_clean[available_features]
    y = df_clean['target']
    
    # Analyze
    analyze_predictions(model, X, y, df_clean)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
