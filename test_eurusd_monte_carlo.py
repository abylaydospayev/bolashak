"""
Monte Carlo Simulation for EURUSD Ensemble Model
Tests model robustness with random walk-forward validation
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from build_features_enhanced import add_multi_timeframe_features
from indicators import ema, rsi, atr, pct_change, sincos_time

def load_eurusd_data():
    """Load EURUSD data from CSV files"""
    
    # Try both possible filenames
    try:
        df_m15 = pd.read_csv('data/EURUSD.sim_M15.csv')
    except:
        df_m15 = pd.read_csv('data/EURUSD_M15.csv')
    
    df_m15['time'] = pd.to_datetime(df_m15['time'])
    df_m15 = df_m15.set_index('time')
    
    # For other timeframes, resample from M15
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

def build_eurusd_features(df_m15, df_m30, df_h1, df_h4):
    """Build features matching training"""
    
    # M15 base features
    df_m15['ema20'] = ema(df_m15['close'], 20)
    df_m15['ema50'] = ema(df_m15['close'], 50)
    df_m15['rsi14'] = rsi(df_m15['close'], 14)
    df_m15['atr14'] = atr(df_m15, 14)
    df_m15['ema50_slope'] = df_m15['ema50'].diff(5)
    
    # Time features
    sin_h, cos_h = sincos_time(df_m15.index)
    df_m15['sin_hour'] = sin_h.values
    df_m15['cos_hour'] = cos_h.values
    
    # Returns
    df_m15['ret1'] = pct_change(df_m15['close'], 1)
    df_m15['atr_pct'] = df_m15['atr14'] / df_m15['close']
    
    # Price vs EMAs
    df_m15['price_vs_ema20'] = (df_m15['close'] - df_m15['ema20']) / df_m15['close']
    df_m15['price_vs_ema50'] = (df_m15['close'] - df_m15['ema50']) / df_m15['close']
    
    # Multi-timeframe features - reset index first for compatibility
    df_m15_reset = df_m15.reset_index()
    df_m30_reset = df_m30.reset_index()
    df_h1_reset = df_h1.reset_index()
    df_h4_reset = df_h4.reset_index()
    
    df_enhanced = add_multi_timeframe_features(df_m15_reset, df_m30_reset, 'm30')
    df_enhanced = add_multi_timeframe_features(df_enhanced, df_h1_reset, 'h1')
    df_enhanced = add_multi_timeframe_features(df_enhanced, df_h4_reset, 'h4')
    
    # Set time index
    if 'time' in df_enhanced.columns:
        df_enhanced = df_enhanced.set_index('time')
    
    # Market structure
    df_enhanced['higher_high'] = (df_enhanced['high'] > df_enhanced['high'].shift(1)).astype(int)
    df_enhanced['lower_low'] = (df_enhanced['low'] < df_enhanced['low'].shift(1)).astype(int)
    df_enhanced['swing_low'] = ((df_enhanced['low'] < df_enhanced['low'].shift(1)) & 
                                (df_enhanced['low'] < df_enhanced['low'].shift(2)) &
                                (df_enhanced['low'] < df_enhanced['low'].shift(-1)) & 
                                (df_enhanced['low'] < df_enhanced['low'].shift(-2))).astype(int)
    
    return df_enhanced

def create_target(df, forward_bars=5, threshold_pips=10):
    """Create target variable"""
    pip_size = 0.0001
    threshold = threshold_pips * pip_size
    
    df['future_close'] = df['close'].shift(-forward_bars)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    df['target'] = (df['future_return'] > threshold).astype(int)
    
    return df

def monte_carlo_simulation(X, y, model, n_simulations=100, test_size=0.2):
    """
    Run Monte Carlo simulation with random train/test splits
    """
    print(f"\n Running {n_simulations} Monte Carlo simulations...")
    print(f"   Test size: {test_size*100:.0f}%")
    
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'win_rate': []
    }
    
    for i in range(n_simulations):
        # Random split
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * (1 - test_size))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        results['accuracy'].append(accuracy_score(y_test, y_pred))
        results['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        results['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        results['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        
        try:
            results['roc_auc'].append(roc_auc_score(y_test, y_proba))
        except:
            results['roc_auc'].append(np.nan)
        
        # Win rate when model predicts UP
        up_preds = y_pred == 1
        if up_preds.sum() > 0:
            win_rate = (y_test[up_preds] == 1).sum() / up_preds.sum()
            results['win_rate'].append(win_rate)
        else:
            results['win_rate'].append(np.nan)
        
        if (i + 1) % 10 == 0:
            print(f"   Completed {i+1}/{n_simulations} simulations...")
    
    return results

def plot_monte_carlo_results(results):
    """Plot Monte Carlo results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('EURUSD Ensemble Model - Monte Carlo Simulation Results', fontsize=16)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'win_rate']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Win Rate (UP signals)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        
        data = [x for x in results[metric] if not np.isnan(x)]
        
        ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.3f}')
        ax.axvline(np.median(data), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(data):.3f}')
        
        ax.set_xlabel(title)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{title} Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/EURUSD_monte_carlo.png', dpi=300, bbox_inches='tight')
    print(f"\n Saved plot: results/EURUSD_monte_carlo.png")
    
    return fig

def print_monte_carlo_summary(results):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("MONTE CARLO SIMULATION RESULTS - EURUSD ENSEMBLE")
    print("="*70)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'win_rate']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Win Rate']
    
    for metric, title in zip(metrics, titles):
        data = [x for x in results[metric] if not np.isnan(x)]
        
        print(f"\n{title}:")
        print(f"  Mean:   {np.mean(data):.4f}")
        print(f"  Median: {np.median(data):.4f}")
        print(f"  Std:    {np.std(data):.4f}")
        print(f"  Min:    {np.min(data):.4f}")
        print(f"  Max:    {np.max(data):.4f}")
        print(f"  95% CI: [{np.percentile(data, 2.5):.4f}, {np.percentile(data, 97.5):.4f}]")

def main():
    print("\n" + "="*70)
    print(" EURUSD ENSEMBLE MODEL - MONTE CARLO ANALYSIS")
    print("="*70)
    
    # Load model
    print("\n Loading EURUSD ensemble model...")
    model_path = Path('models/EURUSD_ensemble_oos.pkl')
    model = joblib.load(model_path)
    print(f"    Loaded: {model_path}")
    
    # Load data
    print("\n Loading EURUSD data from CSV...")
    df_m15, df_m30, df_h1, df_h4 = load_eurusd_data()
    print(f"   M15: {len(df_m15)} bars")
    print(f"   M30: {len(df_m30)} bars")
    print(f"   H1: {len(df_h1)} bars")
    print(f"   H4: {len(df_h4)} bars")
    
    # Build features
    print("\n Building features...")
    df_features = build_eurusd_features(df_m15, df_m30, df_h1, df_h4)
    print(f"   Created {len(df_features.columns)} features")
    
    # Create target
    print("\n Creating target...")
    df_features = create_target(df_features, forward_bars=5, threshold_pips=10)
    
    # Clean data
    df_clean = df_features.dropna()
    print(f"   Clean data: {len(df_clean)} rows")
    
    # Get feature columns
    feature_cols = [
        'price_vs_ema20_h1', 'momentum_5_h1', 'momentum_10_h1', 'rsi14_h1', 'trend_strength_h1',
        'ema20_h1', 'atr14_h1', 'ema50_h1', 'atr_pct_h1', 'momentum_5_m30', 'ema50_m30',
        'ema20_m30', 'momentum_10_m30', 'price_vs_ema20_m30', 'rsi14_m30', 'trend_strength_m30',
        'atr14_m30', 'ema20_h4', 'atr_pct_h4', 'rsi14_h4', 'price_vs_ema20_h4', 'trend_strength_h4',
        'ema50_h4', 'atr_pct_m30', 'momentum_10_h4', 'momentum_5_h4', 'atr14_h4', 'swing_low',
        'higher_high', 'trend_ema_m30', 'open', 'high', 'low', 'close', 'volume', 'rsi14', 'atr14'
    ]
    
    # Check available features
    available_features = [col for col in feature_cols if col in df_clean.columns]
    print(f"   Using {len(available_features)} features")
    
    X = df_clean[available_features]
    y = df_clean['target']
    
    print(f"\n   Target distribution:")
    print(f"   Class 0: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"   Class 1: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    # Run Monte Carlo simulation
    results = monte_carlo_simulation(X, y, model, n_simulations=100, test_size=0.2)
    
    # Print summary
    print_monte_carlo_summary(results)
    
    # Plot results
    print("\n Generating plots...")
    Path('results').mkdir(exist_ok=True)
    plot_monte_carlo_results(results)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/EURUSD_monte_carlo_results.csv', index=False)
    print(f" Saved: results/EURUSD_monte_carlo_results.csv")
    
    print("\n" + "="*70)
    print(" MONTE CARLO ANALYSIS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()

