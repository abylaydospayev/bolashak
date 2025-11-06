"""
Ensemble model combining RandomForest and LSTM predictions.

Uses weighted averaging or stacking to combine predictions from:
- RandomForest (fast, stable, works on single bars)
- LSTM (temporal patterns, sequence-based)

Ensemble often outperforms individual models.
"""
import argparse, os, yaml, joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

def create_lstm_sequences(X, lookback):
    """Create sequences for LSTM."""
    X_seq = []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
    return np.array(X_seq)

def time_split(df, train_ratio=0.7, val_ratio=0.15):
    """Time-based split."""
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    return df.iloc[:n_train], df.iloc[n_train:n_val], df.iloc[n_val:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--rf_weight', type=float, default=0.5, help='Weight for RF (0-1)')
    parser.add_argument('--method', choices=['average', 'weighted'], default='weighted')
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open('config.yaml'))
    lookback = cfg['lookback']
    feat_dir = cfg['feature_dir']
    model_dir = cfg['model_dir']
    
    # Load data
    csv_path = os.path.join(feat_dir, f"{args.symbol}_features.csv")
    df = pd.read_csv(csv_path, parse_dates=['time'])
    
    feature_cols = [c for c in df.columns if c not in ('time', 'y')]
    X = df[feature_cols].values
    y = df['y'].values
    
    print("=" * 80)
    print(f"Ensemble Model: RandomForest + LSTM")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Method: {args.method}")
    print(f"RF Weight: {args.rf_weight:.2f}")
    print(f"LSTM Weight: {1 - args.rf_weight:.2f}")
    print()
    
    # Load models
    print("Loading models...")
    rf_path = os.path.join(model_dir, f"{args.symbol}_rf.pkl")
    rf_scaler_path = os.path.join(model_dir, "scaler.pkl")
    lstm_path = os.path.join(model_dir, f"{args.symbol}_lstm_best.keras")
    lstm_scaler_path = os.path.join(model_dir, f"{args.symbol}_lstm_scaler.pkl")
    
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"RF model not found: {rf_path}")
    if not os.path.exists(lstm_path):
        raise FileNotFoundError(f"LSTM model not found: {lstm_path}")
    
    rf_model = joblib.load(rf_path)
    rf_scaler = joblib.load(rf_scaler_path)
    lstm_model = tf.keras.models.load_model(lstm_path)
    lstm_scaler = joblib.load(lstm_scaler_path)
    
    print(" Models loaded successfully")
    print()
    
    # Split data
    train_df, val_df, test_df = time_split(df)
    
    # For each split, generate predictions
    results = {}
    
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        X_split = split_df[feature_cols].values
        y_split = split_df['y'].values
        
        # RF predictions (all samples)
        X_rf_scaled = rf_scaler.transform(X_split)
        rf_proba = rf_model.predict_proba(X_rf_scaled)[:, 1]
        
        # LSTM predictions (need sequences)
        X_lstm_scaled = lstm_scaler.transform(X_split)
        X_lstm_seq = create_lstm_sequences(X_lstm_scaled, lookback)
        lstm_proba = lstm_model.predict(X_lstm_seq, verbose=0).flatten()
        
        # Align: LSTM loses first 'lookback' samples
        rf_proba_aligned = rf_proba[lookback:]
        y_aligned = y_split[lookback:]
        
        if len(lstm_proba) != len(rf_proba_aligned):
            min_len = min(len(lstm_proba), len(rf_proba_aligned))
            lstm_proba = lstm_proba[:min_len]
            rf_proba_aligned = rf_proba_aligned[:min_len]
            y_aligned = y_aligned[:min_len]
        
        # Ensemble predictions
        if args.method == 'average':
            ensemble_proba = (rf_proba_aligned + lstm_proba) / 2
        else:  # weighted
            ensemble_proba = args.rf_weight * rf_proba_aligned + (1 - args.rf_weight) * lstm_proba
        
        # Evaluate each model
        rf_auc = roc_auc_score(y_aligned, rf_proba_aligned)
        rf_acc = accuracy_score(y_aligned, (rf_proba_aligned >= 0.5).astype(int))
        
        lstm_auc = roc_auc_score(y_aligned, lstm_proba)
        lstm_acc = accuracy_score(y_aligned, (lstm_proba >= 0.5).astype(int))
        
        ensemble_auc = roc_auc_score(y_aligned, ensemble_proba)
        ensemble_acc = accuracy_score(y_aligned, (ensemble_proba >= 0.5).astype(int))
        
        results[split_name] = {
            'samples': len(y_aligned),
            'rf_auc': rf_auc,
            'rf_acc': rf_acc,
            'lstm_auc': lstm_auc,
            'lstm_acc': lstm_acc,
            'ensemble_auc': ensemble_auc,
            'ensemble_acc': ensemble_acc,
            'rf_proba': rf_proba_aligned,
            'lstm_proba': lstm_proba,
            'ensemble_proba': ensemble_proba,
            'y': y_aligned
        }
    
    # Print results
    print("=" * 80)
    print("Ensemble Evaluation Results")
    print("=" * 80)
    print()
    
    print(f"{'Split':<10} {'Model':<12} {'AUC':<10} {'Accuracy':<10} {'Samples':<10}")
    print("-" * 80)
    
    for split_name in ['train', 'val', 'test']:
        r = results[split_name]
        print(f"{split_name:<10} {'RF':<12} {r['rf_auc']:>8.4f} {r['rf_acc']:>10.4f} {r['samples']:>10}")
        print(f"{'':<10} {'LSTM':<12} {r['lstm_auc']:>8.4f} {r['lstm_acc']:>10.4f}")
        print(f"{'':<10} {'ENSEMBLE':<12} {r['ensemble_auc']:>8.4f} {r['ensemble_acc']:>10.4f}")
        
        # Show improvement
        best_single = max(r['rf_auc'], r['lstm_auc'])
        improvement = r['ensemble_auc'] - best_single
        symbol = "" if improvement > 0 else ""
        print(f"{'':<10} {symbol} Improvement: {improvement:+.4f} vs best single model")
        print()
    
    # Find optimal weight on validation set
    print("=" * 80)
    print("Optimal Weight Search (Validation Set)")
    print("=" * 80)
    
    val_rf = results['val']['rf_proba']
    val_lstm = results['val']['lstm_proba']
    val_y = results['val']['y']
    
    best_weight = 0.5
    best_auc = 0
    
    print(f"\n{'Weight (RF)':<15} {'AUC':<10}")
    print("-" * 30)
    
    for w in np.linspace(0, 1, 21):
        weighted_proba = w * val_rf + (1 - w) * val_lstm
        auc = roc_auc_score(val_y, weighted_proba)
        print(f"{w:>13.2f}   {auc:>8.4f} {'  ' if auc > best_auc else ''}")
        if auc > best_auc:
            best_auc = auc
            best_weight = w
    
    print()
    print(f" Optimal RF weight: {best_weight:.2f} (AUC: {best_auc:.4f})")
    print(f"   LSTM weight: {1-best_weight:.2f}")
    
    # Test with optimal weight
    print()
    print("=" * 80)
    print("Test Set Performance with Optimal Weight")
    print("=" * 80)
    
    test_rf = results['test']['rf_proba']
    test_lstm = results['test']['lstm_proba']
    test_y = results['test']['y']
    
    optimal_ensemble = best_weight * test_rf + (1 - best_weight) * test_lstm
    optimal_auc = roc_auc_score(test_y, optimal_ensemble)
    optimal_acc = accuracy_score(test_y, (optimal_ensemble >= 0.5).astype(int))
    
    print(f"\nRF Test AUC:           {results['test']['rf_auc']:.4f}")
    print(f"LSTM Test AUC:         {results['test']['lstm_auc']:.4f}")
    print(f"Ensemble Test AUC:     {optimal_auc:.4f} (weight={best_weight:.2f})")
    print(f"Ensemble Test ACC:     {optimal_acc:.4f}")
    
    best_single_test = max(results['test']['rf_auc'], results['test']['lstm_auc'])
    improvement = optimal_auc - best_single_test
    print(f"\nImprovement vs best:   {improvement:+.4f} ({improvement/best_single_test:+.2%})")
    
    # Save ensemble predictions
    print()
    print("=" * 80)
    
    # Save optimal weight configuration
    ensemble_config = {
        'symbol': args.symbol,
        'rf_weight': float(best_weight),
        'lstm_weight': float(1 - best_weight),
        'val_auc': float(best_auc),
        'test_auc': float(optimal_auc),
        'test_acc': float(optimal_acc),
        'rf_model': rf_path,
        'lstm_model': lstm_path,
        'rf_scaler': rf_scaler_path,
        'lstm_scaler': lstm_scaler_path,
        'lookback': lookback
    }
    
    import json
    ensemble_path = os.path.join(model_dir, f"{args.symbol}_ensemble_config.json")
    with open(ensemble_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    print(f" Ensemble config saved to {ensemble_path}")
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Model':<15} {'Val AUC':<12} {'Test AUC':<12} {'Winner':<10}")
    print("-" * 80)
    print(f"{'RandomForest':<15} {results['val']['rf_auc']:>10.4f} {results['test']['rf_auc']:>12.4f}")
    print(f"{'LSTM':<15} {results['val']['lstm_auc']:>10.4f} {results['test']['lstm_auc']:>12.4f}")
    print(f"{'Ensemble':<15} {best_auc:>10.4f} {optimal_auc:>12.4f}   {'' if improvement > 0 else ''}")
    
    if improvement > 0:
        print(f"\n Ensemble outperforms individual models!")
    else:
        print(f"\n  Single model performs better. Consider using best individual model.")

if __name__ == '__main__':
    main()

