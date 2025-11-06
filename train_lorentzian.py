"""
Lorentzian Distance-based k-Nearest Neighbors Classifier
More robust to outliers than Euclidean distance.
Popular in TradingView ML indicators.
"""
import pandas as pd
import numpy as np
import argparse
import yaml
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import time

from lorentzian_classifier import LorentzianClassifier

def time_split(df, train_ratio=0.7, val_ratio=0.15):
    """Time-aware train/val/test split."""
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_val]
    test = df.iloc[n_val:]
    
    return train, val, test

def main():
    parser = argparse.ArgumentParser(description='Train Lorentzian Classifier')
    parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., EURUSD.sim)')
    parser.add_argument('--k', type=int, default=8, help='Number of neighbors (default: 8)')
    parser.add_argument('--no-weights', action='store_true', help='Disable distance weighting')
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open('config.yaml'))
    feat_dir = Path(cfg['feature_dir'])
    model_dir = Path(cfg['model_dir'])
    model_dir.mkdir(exist_ok=True)
    
    symbol = args.symbol
    print(f"\n{'='*80}")
    print(f"Training Lorentzian Classifier: {symbol}")
    print(f"{'='*80}\n")
    print(f"Parameters:")
    print(f"  k neighbors: {args.k}")
    print(f"  Distance weighting: {not args.no_weights}")
    
    # Load features
    feat_path = feat_dir / f'{symbol}_features.csv'
    if not feat_path.exists():
        print(f"ERROR: {feat_path} not found!")
        print("Run: python build_features.py --symbol", symbol)
        return
    
    df = pd.read_csv(feat_path)
    print(f"\nLoaded {len(df)} samples from {feat_path}")
    
    # Drop rows with NaN
    df_clean = df.dropna()
    print(f"After dropping NaN: {len(df_clean)} samples")
    
    # Feature columns (exclude label and metadata)
    feature_cols = [c for c in df_clean.columns 
                   if c not in ['y', 'time', 'fwd_ret', 'fwd_ret_net']]
    
    X = df_clean[feature_cols].values
    y = df_clean['y'].values
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Class distribution: {np.bincount(y)} (0={np.sum(y==0)}, 1={np.sum(y==1)})")
    
    # Time-based split
    train, val, test = time_split(df_clean)
    
    X_train = train[feature_cols].values
    y_train = train['y'].values
    X_val = val[feature_cols].values
    y_val = val['y'].values
    X_test = test[feature_cols].values
    y_test = test['y'].values
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(df_clean)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(df_clean)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(df_clean)*100:.1f}%)")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Lorentzian classifier
    print(f"\nTraining Lorentzian Classifier (k={args.k})...")
    start_time = time.time()
    
    clf = LorentzianClassifier(
        k=args.k,
        weight_by_distance=not args.no_weights
    )
    clf.fit(X_train_scaled, y_train)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")
    
    # Evaluate on train set (sample for speed)
    print("\nEvaluating on train set (sample 1000 for speed)...")
    train_sample_idx = np.random.choice(len(X_train_scaled), 
                                       min(1000, len(X_train_scaled)), 
                                       replace=False)
    X_train_sample = X_train_scaled[train_sample_idx]
    y_train_sample = y_train[train_sample_idx]
    
    start_time = time.time()
    y_train_pred_proba = clf.predict_proba(X_train_sample)[:, 1]
    pred_time = time.time() - start_time
    
    y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
    train_auc = roc_auc_score(y_train_sample, y_train_pred_proba)
    train_acc = accuracy_score(y_train_sample, y_train_pred)
    
    print(f"Train AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
    print(f"Prediction time: {pred_time:.2f}s for {len(X_train_sample)} samples")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    start_time = time.time()
    y_val_pred_proba = clf.predict_proba(X_val_scaled)[:, 1]
    pred_time = time.time() - start_time
    
    y_val_pred = (y_val_pred_proba >= 0.5).astype(int)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"Val AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
    print(f"Prediction time: {pred_time:.2f}s for {len(X_val_scaled)} samples")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    start_time = time.time()
    y_test_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    pred_time = time.time() - start_time
    
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"Test AUC: {test_auc:.4f}, Acc: {test_acc:.4f}")
    print(f"Prediction time: {pred_time:.2f}s for {len(X_test_scaled)} samples")
    
    print("\nDetailed classification report (Test set):")
    print(classification_report(y_test, y_test_pred, digits=4))
    
    # Save model
    model_path = model_dir / f'{symbol}_lorentzian.pkl'
    scaler_path = model_dir / f'{symbol}_lorentzian_scaler.pkl'
    
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n{'='*80}")
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"{'='*80}\n")
    
    # Summary
    print("SUMMARY:")
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Val AUC:   {val_auc:.4f}")
    print(f"  Test AUC:  {test_auc:.4f}")
    print(f"  k:         {args.k}")
    print(f"  Weighted:  {not args.no_weights}")
    
    # Speed comparison
    avg_pred_time = pred_time / len(X_test_scaled) * 1000
    print(f"\nSpeed: {avg_pred_time:.2f}ms per prediction")
    if avg_pred_time > 100:
        print("  WARNING: Lorentzian is slow for large datasets!")
        print("   Consider using smaller k or sampling training data.")

if __name__ == '__main__':
    main()

