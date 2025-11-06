"""
Create 3-way ensemble: RandomForest + LSTM + Lorentzian
Combines predictions with optimized weights.
"""
import pandas as pd
import numpy as np
import argparse
import yaml
import joblib
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from itertools import product

def optimize_weights(y_true, rf_probs, lstm_probs, lc_probs, metric='auc'):
    """
    Find optimal ensemble weights using grid search.
    
    Args:
        y_true: True labels
        rf_probs: RandomForest probabilities
        lstm_probs: LSTM probabilities
        lc_probs: Lorentzian probabilities
        metric: 'auc' or 'accuracy'
    
    Returns:
        best_weights: (w_rf, w_lstm, w_lc)
        best_score: Best metric value
    """
    best_score = 0
    best_weights = None
    
    # Grid search over weight combinations (must sum to 1.0)
    step = 0.1
    weights_range = np.arange(0, 1.0 + step, step)
    
    print("Optimizing ensemble weights...")
    results = []
    
    for w_rf in weights_range:
        for w_lstm in weights_range:
            w_lc = 1.0 - w_rf - w_lstm
            
            # Skip invalid combinations
            if w_lc < -0.01 or w_lc > 1.01:
                continue
            
            # Ensure weights sum to 1.0
            w_lc = max(0, min(1.0, w_lc))
            
            # Ensemble prediction
            ensemble_probs = w_rf * rf_probs + w_lstm * lstm_probs + w_lc * lc_probs
            
            # Calculate metric
            if metric == 'auc':
                score = roc_auc_score(y_true, ensemble_probs)
            else:
                preds = (ensemble_probs >= 0.5).astype(int)
                score = accuracy_score(y_true, preds)
            
            results.append({
                'w_rf': w_rf,
                'w_lstm': w_lstm,
                'w_lc': w_lc,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_weights = (w_rf, w_lstm, w_lc)
    
    # Show top 5 combinations
    results_df = pd.DataFrame(results).sort_values('score', ascending=False)
    print("\nTop 5 weight combinations:")
    print(results_df.head().to_string(index=False))
    
    return best_weights, best_score

def main():
    parser = argparse.ArgumentParser(description='Create 3-way ensemble (RF + LSTM + Lorentzian)')
    parser.add_argument('--symbol', required=True, help='Trading symbol')
    parser.add_argument('--optimize', action='store_true', help='Optimize weights on validation set')
    parser.add_argument('--w_rf', type=float, default=0.33, help='RF weight (if not optimizing)')
    parser.add_argument('--w_lstm', type=float, default=0.33, help='LSTM weight (if not optimizing)')
    parser.add_argument('--w_lc', type=float, default=0.34, help='Lorentzian weight (if not optimizing)')
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open('config.yaml'))
    feat_dir = Path(cfg['feature_dir'])
    model_dir = Path(cfg['model_dir'])
    
    symbol = args.symbol
    print(f"\n{'='*80}")
    print(f"3-Way Ensemble: RandomForest + LSTM + Lorentzian")
    print(f"Symbol: {symbol}")
    print(f"{'='*80}\n")
    
    # Load feature data
    feat_path = feat_dir / f'{symbol}_features.csv'
    df = pd.read_csv(feat_path).dropna()
    
    feature_cols = [c for c in df.columns if c not in ['y', 'time', 'fwd_ret', 'fwd_ret_net']]
    
    # Time-based split
    n = len(df)
    n_train = int(n * 0.7)
    n_val = int(n * 0.85)
    
    val_df = df.iloc[n_train:n_val]
    test_df = df.iloc[n_val:]
    
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples\n")
    
    # Load models
    print("Loading models...")
    
    # RandomForest
    rf_model = joblib.load(model_dir / f'{symbol}_rf.pkl')
    rf_scaler = joblib.load(model_dir / 'scaler.pkl')
    
    # LSTM
    try:
        import tensorflow as tf
        lstm_model = tf.keras.models.load_model(model_dir / f'{symbol}_lstm_best.keras')
        lstm_scaler = joblib.load(model_dir / f'{symbol}_lstm_scaler.pkl')
        lstm_available = True
    except Exception as e:
        print(f"WARNING: Could not load LSTM model: {e}")
        lstm_available = False
    
    # Lorentzian
    try:
        lc_model = joblib.load(model_dir / f'{symbol}_lorentzian.pkl')
        lc_scaler = joblib.load(model_dir / f'{symbol}_lorentzian_scaler.pkl')
        lc_available = True
    except Exception as e:
        print(f"WARNING: Could not load Lorentzian model: {e}")
        lc_available = False
    
    if not lstm_available or not lc_available:
        print("\nERROR: Need all 3 models (RF, LSTM, Lorentzian)")
        print("Train missing models:")
        if not lstm_available:
            print(f"  python train_lstm.py --symbol {symbol}")
        if not lc_available:
            print(f"  python train_lorentzian.py --symbol {symbol}")
        return
    
    print(" All models loaded\n")
    
    # Get predictions on validation set
    print("Getting validation predictions...")
    
    X_val = val_df[feature_cols].values
    y_val = val_df['y'].values
    
    # RF predictions
    X_val_rf = rf_scaler.transform(X_val)
    rf_probs_val = rf_model.predict_proba(X_val_rf)[:, 1]
    
    # LSTM predictions (need sequences)
    X_val_lstm = lstm_scaler.transform(X_val)
    lookback = 60
    
    # Create sequences for LSTM
    lstm_sequences = []
    valid_indices = []
    for i in range(lookback, len(X_val_lstm)):
        lstm_sequences.append(X_val_lstm[i-lookback:i])
        valid_indices.append(i)
    
    lstm_sequences = np.array(lstm_sequences)
    lstm_probs_val_full = lstm_model.predict(lstm_sequences, verbose=0).flatten()
    
    # Align with valid indices
    lstm_probs_val = np.full(len(X_val), np.nan)
    lstm_probs_val[valid_indices] = lstm_probs_val_full
    
    # Lorentzian predictions
    X_val_lc = lc_scaler.transform(X_val)
    lc_probs_val = lc_model.predict_proba(X_val_lc)[:, 1]
    
    # Remove NaN entries (from LSTM lookback)
    valid_mask = ~np.isnan(lstm_probs_val)
    rf_probs_val = rf_probs_val[valid_mask]
    lstm_probs_val = lstm_probs_val[valid_mask]
    lc_probs_val = lc_probs_val[valid_mask]
    y_val = y_val[valid_mask]
    
    print(f"Valid predictions: {len(y_val)}")
    
    # Individual model performance on validation
    print("\nValidation Set Performance:")
    print(f"  RF:         AUC {roc_auc_score(y_val, rf_probs_val):.4f}")
    print(f"  LSTM:       AUC {roc_auc_score(y_val, lstm_probs_val):.4f}")
    print(f"  Lorentzian: AUC {roc_auc_score(y_val, lc_probs_val):.4f}")
    
    # Optimize or use fixed weights
    if args.optimize:
        print("\nOptimizing weights on validation set...")
        best_weights, best_score = optimize_weights(
            y_val, rf_probs_val, lstm_probs_val, lc_probs_val
        )
        w_rf, w_lstm, w_lc = best_weights
        print(f"\nOptimal weights: RF={w_rf:.2f}, LSTM={w_lstm:.2f}, LC={w_lc:.2f}")
        print(f"Validation AUC: {best_score:.4f}")
    else:
        w_rf = args.w_rf
        w_lstm = args.w_lstm
        w_lc = args.w_lc
        print(f"\nUsing fixed weights: RF={w_rf:.2f}, LSTM={w_lstm:.2f}, LC={w_lc:.2f}")
    
    # Test set evaluation
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80 + "\n")
    
    X_test = test_df[feature_cols].values
    y_test = test_df['y'].values
    
    # RF predictions
    X_test_rf = rf_scaler.transform(X_test)
    rf_probs_test = rf_model.predict_proba(X_test_rf)[:, 1]
    
    # LSTM predictions
    X_test_lstm = lstm_scaler.transform(X_test)
    lstm_sequences = []
    valid_indices = []
    for i in range(lookback, len(X_test_lstm)):
        lstm_sequences.append(X_test_lstm[i-lookback:i])
        valid_indices.append(i)
    
    lstm_sequences = np.array(lstm_sequences)
    lstm_probs_test_full = lstm_model.predict(lstm_sequences, verbose=0).flatten()
    
    lstm_probs_test = np.full(len(X_test), np.nan)
    lstm_probs_test[valid_indices] = lstm_probs_test_full
    
    # Lorentzian predictions
    X_test_lc = lc_scaler.transform(X_test)
    lc_probs_test = lc_model.predict_proba(X_test_lc)[:, 1]
    
    # Remove NaN entries
    valid_mask = ~np.isnan(lstm_probs_test)
    rf_probs_test = rf_probs_test[valid_mask]
    lstm_probs_test = lstm_probs_test[valid_mask]
    lc_probs_test = lc_probs_test[valid_mask]
    y_test = y_test[valid_mask]
    
    # Individual performance
    rf_auc = roc_auc_score(y_test, rf_probs_test)
    lstm_auc = roc_auc_score(y_test, lstm_probs_test)
    lc_auc = roc_auc_score(y_test, lc_probs_test)
    
    print("Individual Model Performance:")
    print(f"  RF:         AUC {rf_auc:.4f}")
    print(f"  LSTM:       AUC {lstm_auc:.4f}")
    print(f"  Lorentzian: AUC {lc_auc:.4f}")
    
    # Ensemble prediction
    ensemble_probs_test = w_rf * rf_probs_test + w_lstm * lstm_probs_test + w_lc * lc_probs_test
    ensemble_auc = roc_auc_score(y_test, ensemble_probs_test)
    
    print(f"\n Ensemble:  AUC {ensemble_auc:.4f}")
    
    # Calculate improvement
    best_individual = max(rf_auc, lstm_auc, lc_auc)
    improvement = ((ensemble_auc - best_individual) / best_individual) * 100
    
    print(f"\nImprovement: {improvement:+.2f}% vs best individual model")
    
    # Save ensemble configuration
    config = {
        'symbol': symbol,
        'weights': {
            'rf': float(w_rf),
            'lstm': float(w_lstm),
            'lorentzian': float(w_lc)
        },
        'val_auc': {
            'rf': float(roc_auc_score(y_val, rf_probs_val)),
            'lstm': float(roc_auc_score(y_val, lstm_probs_val)),
            'lorentzian': float(roc_auc_score(y_val, lc_probs_val)),
            'ensemble': float(best_score) if args.optimize else None
        },
        'test_auc': {
            'rf': float(rf_auc),
            'lstm': float(lstm_auc),
            'lorentzian': float(lc_auc),
            'ensemble': float(ensemble_auc)
        }
    }
    
    config_path = model_dir / f'{symbol}_ensemble3_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nEnsemble configuration saved to: {config_path}")
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

