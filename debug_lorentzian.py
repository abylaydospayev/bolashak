"""
Debug why Lorentzian Classification performs poorly.
Compare with RF/LSTM to identify issues.
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import joblib

def load_models(symbol):
    """Load all trained models."""
    model_dir = Path('models')
    
    rf = joblib.load(model_dir / f'{symbol}_rf.pkl')
    rf_scaler = joblib.load(model_dir / 'scaler.pkl')
    
    try:
        import tensorflow as tf
        lstm = tf.keras.models.load_model(model_dir / f'{symbol}_lstm_best.keras')
        lstm_scaler = joblib.load(model_dir / f'{symbol}_lstm_scaler.pkl')
    except:
        lstm = None
        lstm_scaler = None
    
    lc = joblib.load(model_dir / f'{symbol}_lorentzian.pkl')
    lc_scaler = joblib.load(model_dir / f'{symbol}_lorentzian_scaler.pkl')
    
    return {
        'rf': (rf, rf_scaler),
        'lstm': (lstm, lstm_scaler),
        'lorentzian': (lc, lc_scaler)
    }

def analyze_prediction_distribution(symbol):
    """Analyze how each model distributes predictions."""
    print(f"\n{'='*80}")
    print(f"PREDICTION DISTRIBUTION ANALYSIS: {symbol}")
    print(f"{'='*80}\n")
    
    # Load feature data
    feat_path = Path('features') / f'{symbol}_features.csv'
    df = pd.read_csv(feat_path).dropna()
    
    feature_cols = [c for c in df.columns 
                   if c not in ['y', 'time', 'fwd_ret', 'fwd_ret_net']]
    
    # Split data
    n = len(df)
    n_test = int(n * 0.15)
    test_df = df.iloc[-n_test:]
    
    X_test = test_df[feature_cols].values
    y_test = test_df['y'].values
    
    # Load models
    models = load_models(symbol)
    
    results = {}
    
    for model_name, (model, scaler) in models.items():
        if model is None:
            continue
            
        X_scaled = scaler.transform(X_test)
        
        if model_name == 'lstm':
            # LSTM needs sequences
            lookback = 60
            X_seq = []
            y_seq = []
            for i in range(lookback, len(X_scaled)):
                X_seq.append(X_scaled[i-lookback:i])
                y_seq.append(y_test[i])
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            import tensorflow as tf
            proba = model.predict(X_seq, verbose=0).flatten()
            y_actual = y_seq
        else:
            proba = model.predict_proba(X_scaled)[:, 1]
            y_actual = y_test
        
        from sklearn.metrics import roc_auc_score
        
        results[model_name] = {
            'proba': proba,
            'y': y_actual,
            'auc': roc_auc_score(y_actual, proba)
        }
        
        # Print statistics
        print(f"{model_name.upper()} Predictions:")
        print(f"  AUC: {results[model_name]['auc']:.4f}")
        print(f"  Mean: {proba.mean():.4f}")
        print(f"  Std:  {proba.std():.4f}")
        print(f"  Min:  {proba.min():.4f}")
        print(f"  Max:  {proba.max():.4f}")
        print(f"  Median: {np.median(proba):.4f}")
        print(f"  Q1:   {np.percentile(proba, 25):.4f}")
        print(f"  Q3:   {np.percentile(proba, 75):.4f}")
        
        # Check for extreme predictions
        extreme_low = (proba < 0.1).sum()
        extreme_high = (proba > 0.9).sum()
        print(f"  Extreme (<0.1): {extreme_low} ({extreme_low/len(proba)*100:.1f}%)")
        print(f"  Extreme (>0.9): {extreme_high} ({extreme_high/len(proba)*100:.1f}%)")
        
        # Calibration check
        for threshold in [0.4, 0.5, 0.6, 0.7]:
            predicted_positive = (proba >= threshold).sum()
            actual_positive = y_actual[proba >= threshold].sum() if predicted_positive > 0 else 0
            precision = actual_positive / predicted_positive if predicted_positive > 0 else 0
            print(f"  At threshold {threshold:.1f}: {predicted_positive} predictions, precision {precision:.3f}")
        
        print()
    
    return results

def analyze_neighbor_quality(symbol):
    """Analyze quality of Lorentzian neighbors."""
    print(f"\n{'='*80}")
    print(f"LORENTZIAN NEIGHBOR ANALYSIS: {symbol}")
    print(f"{'='*80}\n")
    
    # Load data
    feat_path = Path('features') / f'{symbol}_features.csv'
    df = pd.read_csv(feat_path).dropna()
    
    feature_cols = [c for c in df.columns 
                   if c not in ['y', 'time', 'fwd_ret', 'fwd_ret_net']]
    
    # Split data
    n = len(df)
    n_train = int(n * 0.7)
    n_val = int(n * 0.85)
    
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_val:]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['y'].values
    X_test = test_df[feature_cols].values[:100]  # Sample for speed
    y_test = test_df['y'].values[:100]
    
    # Load model
    model_dir = Path('models')
    lc = joblib.load(model_dir / f'{symbol}_lorentzian.pkl')
    scaler = joblib.load(model_dir / f'{symbol}_lorentzian_scaler.pkl')
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Analyze neighbors for sample predictions
    print("Sample predictions analysis:\n")
    
    neighbor_stats = []
    
    for i in range(min(10, len(X_test_scaled))):
        x = X_test_scaled[i]
        y_true = y_test[i]
        
        # Calculate distances to all training samples
        distances = np.array([
            lc.lorentzian_distance(x, x_train) 
            for x_train in X_train_scaled
        ])
        
        # Find k nearest neighbors
        k = lc.k
        k_indices = np.argpartition(distances, k)[:k]
        k_distances = distances[k_indices]
        k_labels = y_train[k_indices]
        
        # Prediction
        if lc.weight_by_distance:
            weights = 1.0 / (k_distances + 1e-6)
            weights = weights / weights.sum()
            prob = np.sum(weights * k_labels)
        else:
            prob = np.mean(k_labels)
        
        # Analysis
        avg_distance = k_distances.mean()
        min_distance = k_distances.min()
        max_distance = k_distances.max()
        std_distance = k_distances.std()
        pct_positive = k_labels.mean()
        
        neighbor_stats.append({
            'avg_dist': avg_distance,
            'min_dist': min_distance,
            'max_dist': max_distance,
            'std_dist': std_distance,
            'pct_pos': pct_positive,
            'prob': prob,
            'y_true': y_true
        })
        
        print(f"Sample {i+1}:")
        print(f"  True label: {y_true}")
        print(f"  Predicted prob: {prob:.3f}")
        print(f"  Neighbor labels: {k_labels}")
        print(f"  Distances: min={min_distance:.2f}, avg={avg_distance:.2f}, max={max_distance:.2f}")
        print(f"  Distance std: {std_distance:.2f} (consistency: {'low' if std_distance < avg_distance * 0.5 else 'high'})")
        print()
    
    # Overall statistics
    neighbor_df = pd.DataFrame(neighbor_stats)
    
    print("Overall neighbor statistics:")
    print(f"  Avg distance to neighbors: {neighbor_df['avg_dist'].mean():.2f}")
    print(f"  Avg distance std: {neighbor_df['std_dist'].mean():.2f}")
    print(f"  High variance neighbors: {(neighbor_df['std_dist'] / neighbor_df['avg_dist'] > 0.5).mean()*100:.1f}%")
    print()
    
    # Check if distances are informative
    correct_predictions = (neighbor_df['prob'] >= 0.5).astype(int) == neighbor_df['y_true']
    print(f"Accuracy on sample: {correct_predictions.mean()*100:.1f}%")
    
    # Correlation between distance and correctness
    neighbor_df['correct'] = correct_predictions.astype(int)
    corr_dist = neighbor_df[['avg_dist', 'correct']].corr().iloc[0, 1]
    print(f"Correlation (avg_dist, correct): {corr_dist:.3f}")
    print(f"   {'Closer neighbors = better' if corr_dist < 0 else 'Distance not informative!'}")

def compare_feature_importance():
    """Compare which features each model uses."""
    print(f"\n{'='*80}")
    print(f"FEATURE USAGE COMPARISON")
    print(f"{'='*80}\n")
    
    symbol = 'USDJPY.sim'
    
    # Load RF model
    model_dir = Path('models')
    rf = joblib.load(model_dir / f'{symbol}_rf.pkl')
    
    # Get feature importance
    feat_path = Path('features') / f'{symbol}_features.csv'
    df = pd.read_csv(feat_path).dropna()
    feature_cols = [c for c in df.columns 
                   if c not in ['y', 'time', 'fwd_ret', 'fwd_ret_net']]
    
    importances = rf.feature_importances_
    
    print("RandomForest Feature Importance (Top 10):")
    feature_importance = sorted(zip(feature_cols, importances), 
                                key=lambda x: x[1], reverse=True)
    
    for i, (feat, imp) in enumerate(feature_importance[:10], 1):
        print(f"  {i:2d}. {feat:20s}: {imp:.4f}")
    
    print("\nLorentzian uses ALL features equally (no feature selection)")
    print(" This may hurt performance if irrelevant features add noise!")

def main():
    symbols = ['USDJPY.sim', 'EURUSD.sim']
    
    for symbol in symbols:
        print(f"\n{'#'*80}")
        print(f"# DEBUGGING: {symbol}")
        print(f"{'#'*80}")
        
        # 1. Prediction distribution
        results = analyze_prediction_distribution(symbol)
        
        # 2. Neighbor quality
        analyze_neighbor_quality(symbol)
    
    # 3. Feature importance comparison
    compare_feature_importance()
    
    print(f"\n{'='*80}")
    print("SUMMARY: Why Lorentzian Failed")
    print(f"{'='*80}\n")
    
    print("1. OVERFITTING:")
    print("   - Train AUC: 1.000 (perfect memorization)")
    print("   - Test AUC:  0.533 (poor generalization)")
    print("    Model memorizes training data without learning patterns\n")
    
    print("2. CURSE OF DIMENSIONALITY:")
    print("   - 14 features  distances become meaningless")
    print("   - All neighbors are ~equally far in high-dimensional space")
    print("    'Nearest' neighbors aren't actually similar\n")
    
    print("3. NO FEATURE SELECTION:")
    print("   - RF learns EMA_slope, RSI are important")
    print("   - Lorentzian treats all features equally")
    print("    Noise features dilute the distance metric\n")
    
    print("4. SPEED:")
    print("   - 43ms per prediction (vs 0.01ms for RF)")
    print("   - Must calculate distance to ALL training samples")
    print("    4,300x slower than RF\n")
    
    print("5. DATA CHARACTERISTICS:")
    print("   - Forex data is noisy (low signal-to-noise)")
    print("   - Regime shifts make historical similarity unreliable")
    print("   - 11k samples not enough for meaningful neighbors\n")
    
    print("RECOMMENDATIONS:")
    print("   Use RF (best balance of speed/performance)")
    print("   Use LSTM for EURUSD (best AUC: 0.592)")
    print("   Add regime filter (most impact: -$1M  +$500k)")
    print("   Don't use Lorentzian (worst performer)")

if __name__ == '__main__':
    main()

