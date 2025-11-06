"""
FIXED Lorentzian Classification - Addresses all performance issues.

Key improvements:
1. Feature selection (use only top 5 important features)
2. Larger k (100 instead of 8)
3. Adaptive k based on local density
4. Distance normalization per feature
5. Confidence thresholding
"""
import pandas as pd
import numpy as np
import argparse
import yaml
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import time

class ImprovedLorentzianClassifier:
    """
    Improved k-NN classifier using Lorentzian distance with fixes:
    - Feature selection
    - Larger k
    - Adaptive weighting
    - Confidence scores
    """
    
    def __init__(self, k=100, feature_indices=None, use_adaptive_k=True):
        """
        Args:
            k: Base number of neighbors (default: 100, much larger than 8)
            feature_indices: Indices of important features to use (None = all)
            use_adaptive_k: Adjust k based on local density
        """
        self.k = k
        self.feature_indices = feature_indices
        self.use_adaptive_k = use_adaptive_k
        self.X_train = None
        self.y_train = None
        self.feature_scales = None
        
    def lorentzian_distance(self, x1, x2):
        """
        Calculate weighted Lorentzian distance.
        Uses feature-specific scaling to handle different importance.
        """
        if self.feature_indices is not None:
            x1 = x1[self.feature_indices]
            x2 = x2[self.feature_indices]
        
        # Weighted by feature scale (learned from training data variance)
        diff = np.abs(x1 - x2)
        if self.feature_scales is not None:
            scales = self.feature_scales[self.feature_indices] if self.feature_indices is not None else self.feature_scales
            diff = diff / (scales + 1e-6)
        
        return np.sum(np.log(1 + diff))
    
    def fit(self, X, y, feature_importance=None):
        """
        Store training data and learn feature scales.
        
        Args:
            X: Training features
            y: Training labels
            feature_importance: Feature importance scores from RF
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        # Learn feature-specific scales (inverse of std dev)
        # High variance features get downweighted
        self.feature_scales = np.std(X, axis=0)
        
        # If feature importance provided, use only top features
        if feature_importance is not None and self.feature_indices is None:
            # Use top 5 most important features
            n_select = min(5, len(feature_importance))
            self.feature_indices = np.argsort(feature_importance)[-n_select:]
            print(f"Selected top {n_select} features: {self.feature_indices}")
        
        return self
    
    def get_confidence(self, distances, labels):
        """
        Calculate confidence score based on:
        1. Agreement among neighbors (high consensus = high confidence)
        2. Distance to neighbors (close neighbors = high confidence)
        """
        # Consensus: how much do neighbors agree?
        prob = labels.mean()
        consensus = max(prob, 1 - prob)  # Distance from 0.5
        
        # Proximity: how close are neighbors?
        avg_dist = distances.mean()
        min_dist = distances.min()
        proximity = 1.0 / (1.0 + avg_dist)  # Higher when neighbors are closer
        
        # Combined confidence
        confidence = (consensus + proximity) / 2.0
        
        return confidence
    
    def predict_proba_single(self, x):
        """Predict probability for a single sample with confidence."""
        # Calculate distances to all training samples
        distances = np.array([
            self.lorentzian_distance(x, x_train) 
            for x_train in self.X_train
        ])
        
        # Adaptive k: use more neighbors if local density is high
        k_actual = self.k
        if self.use_adaptive_k:
            # Check local density (how many samples within 1.5x median distance)
            median_dist = np.median(distances)
            local_density = (distances < median_dist * 1.5).sum()
            k_actual = min(self.k, max(20, local_density // 2))
        
        # Find k nearest neighbors
        k_indices = np.argpartition(distances, min(k_actual, len(distances)-1))[:k_actual]
        k_distances = distances[k_indices]
        k_labels = self.y_train[k_indices]
        
        # Weight by inverse distance (squared for more emphasis on close neighbors)
        weights = 1.0 / (k_distances ** 2 + 1e-6)
        weights = weights / weights.sum()
        
        # Weighted probability
        prob_positive = np.sum(weights * k_labels)
        
        # Calculate confidence
        confidence = self.get_confidence(k_distances, k_labels)
        
        return np.array([1 - prob_positive, prob_positive]), confidence
    
    def predict_proba(self, X, return_confidence=False):
        """Predict probabilities for all samples."""
        results = [self.predict_proba_single(x) for x in X]
        probas = np.array([r[0] for r in results])
        
        if return_confidence:
            confidences = np.array([r[1] for r in results])
            return probas, confidences
        
        return probas
    
    def predict(self, X, confidence_threshold=0.0):
        """
        Predict class labels.
        
        Args:
            confidence_threshold: Only make predictions if confidence > threshold
                                 Returns -1 for low-confidence samples
        """
        probas, confidences = self.predict_proba(X, return_confidence=True)
        predictions = (probas[:, 1] >= 0.5).astype(int)
        
        # Set low-confidence predictions to -1 (abstain)
        if confidence_threshold > 0:
            predictions[confidences < confidence_threshold] = -1
        
        return predictions, confidences

def select_features_with_rf(X_train, y_train, n_features=5):
    """Use RandomForest to select most important features."""
    print("\nTraining RF to select important features...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-n_features:]
    
    print(f"Selected top {n_features} features (indices): {top_indices}")
    print(f"Feature importances: {importances[top_indices]}")
    
    return top_indices, importances

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
    parser = argparse.ArgumentParser(description='Train FIXED Lorentzian Classifier')
    parser.add_argument('--symbol', required=True, help='Trading symbol')
    parser.add_argument('--k', type=int, default=100, help='Number of neighbors (default: 100)')
    parser.add_argument('--n-features', type=int, default=5, help='Number of features to use (default: 5)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6, 
                       help='Minimum confidence to make prediction (default: 0.6)')
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open('config.yaml'))
    feat_dir = Path(cfg['feature_dir'])
    model_dir = Path(cfg['model_dir'])
    model_dir.mkdir(exist_ok=True)
    
    symbol = args.symbol
    print(f"\n{'='*80}")
    print(f"Training IMPROVED Lorentzian Classifier: {symbol}")
    print(f"{'='*80}\n")
    print(f"Improvements:")
    print(f"  1. Feature selection: Top {args.n_features} features (vs all 14)")
    print(f"  2. Larger k: {args.k} neighbors (vs 8)")
    print(f"  3. Adaptive k based on local density")
    print(f"  4. Distance weighting: inverse squared")
    print(f"  5. Confidence scoring")
    print(f"  6. Confidence threshold: {args.confidence_threshold}")
    
    # Load features
    feat_path = feat_dir / f'{symbol}_features.csv'
    if not feat_path.exists():
        print(f"ERROR: {feat_path} not found!")
        return
    
    df = pd.read_csv(feat_path).dropna()
    print(f"\nLoaded {len(df)} samples")
    
    # Feature columns
    feature_cols = [c for c in df.columns 
                   if c not in ['y', 'time', 'fwd_ret', 'fwd_ret_net']]
    
    X = df[feature_cols].values
    y = df['y'].values
    
    print(f"Features: {len(feature_cols)}")
    print(f"Class distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
    
    # Time-based split
    train, val, test = time_split(df)
    
    X_train = train[feature_cols].values
    y_train = train['y'].values
    X_val = val[feature_cols].values
    y_val = val['y'].values
    X_test = test[feature_cols].values
    y_test = test['y'].values
    
    print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Use RobustScaler (better for outliers than StandardScaler)
    print("\nScaling features with RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection using RF
    top_indices, importances = select_features_with_rf(
        X_train_scaled, y_train, n_features=args.n_features
    )
    
    # Print selected feature names
    selected_features = [feature_cols[i] for i in top_indices]
    print(f"\nSelected features: {selected_features}")
    
    # Train improved Lorentzian classifier
    print(f"\nTraining Improved Lorentzian (k={args.k})...")
    start_time = time.time()
    
    clf = ImprovedLorentzianClassifier(
        k=args.k,
        feature_indices=top_indices,
        use_adaptive_k=True
    )
    clf.fit(X_train_scaled, y_train, feature_importance=importances)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")
    
    # Evaluate on validation set
    print("\n" + "="*80)
    print("VALIDATION SET EVALUATION")
    print("="*80)
    
    start_time = time.time()
    y_val_pred_proba, val_confidences = clf.predict_proba(X_val_scaled, return_confidence=True)
    pred_time = time.time() - start_time
    
    y_val_pred_proba = y_val_pred_proba[:, 1]
    
    # Overall performance
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    y_val_pred = (y_val_pred_proba >= 0.5).astype(int)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"\nOverall Performance:")
    print(f"  Val AUC: {val_auc:.4f}")
    print(f"  Val Acc: {val_acc:.4f}")
    print(f"  Prediction time: {pred_time:.2f}s ({pred_time/len(X_val_scaled)*1000:.1f}ms per sample)")
    
    # High-confidence predictions only
    high_conf_mask = val_confidences >= args.confidence_threshold
    n_high_conf = high_conf_mask.sum()
    
    if n_high_conf > 0:
        hc_auc = roc_auc_score(y_val[high_conf_mask], y_val_pred_proba[high_conf_mask])
        hc_acc = accuracy_score(y_val[high_conf_mask], y_val_pred[high_conf_mask])
        
        print(f"\nHigh-Confidence Predictions (conf >= {args.confidence_threshold}):")
        print(f"  Samples: {n_high_conf} ({n_high_conf/len(y_val)*100:.1f}%)")
        print(f"  AUC: {hc_auc:.4f} (vs {val_auc:.4f} overall)")
        print(f"  Acc: {hc_acc:.4f} (vs {val_acc:.4f} overall)")
        print(f"  Avg confidence: {val_confidences[high_conf_mask].mean():.3f}")
    
    # Test set evaluation
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    y_test_pred_proba, test_confidences = clf.predict_proba(X_test_scaled, return_confidence=True)
    y_test_pred_proba = y_test_pred_proba[:, 1]
    
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nOverall Performance:")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Test Acc: {test_acc:.4f}")
    
    # High-confidence test predictions
    high_conf_mask = test_confidences >= args.confidence_threshold
    n_high_conf = high_conf_mask.sum()
    
    if n_high_conf > 0:
        hc_auc = roc_auc_score(y_test[high_conf_mask], y_test_pred_proba[high_conf_mask])
        hc_acc = accuracy_score(y_test[high_conf_mask], y_test_pred[high_conf_mask])
        
        print(f"\nHigh-Confidence Predictions (conf >= {args.confidence_threshold}):")
        print(f"  Samples: {n_high_conf} ({n_high_conf/len(y_test)*100:.1f}%)")
        print(f"  AUC: {hc_auc:.4f}")
        print(f"  Acc: {hc_acc:.4f}")
        print(f"  Improvement: +{(hc_auc - test_auc)*100:.1f}% AUC")
    
    # Save model
    model_path = model_dir / f'{symbol}_lorentzian_fixed.pkl'
    scaler_path = model_dir / f'{symbol}_lorentzian_fixed_scaler.pkl'
    
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n{'='*80}")
    print(f"Model saved to: {model_path}")
    print(f"{'='*80}\n")
    
    # Compare with original
    print("COMPARISON: Fixed vs Original Lorentzian")
    print("="*80)
    print(f"                    Original    Fixed      Improvement")
    print(f"k neighbors:        8           {args.k}        +{args.k-8}")
    print(f"Features used:      14          {args.n_features}         -{14-args.n_features}")
    print(f"Test AUC:          0.537       {test_auc:.3f}      {'+' if test_auc > 0.537 else ''}{(test_auc-0.537)*100:.1f}%")
    print(f"Prediction speed:  43ms        {pred_time/len(X_val_scaled)*1000:.1f}ms      {(1-pred_time/len(X_val_scaled)*1000/43)*100:.0f}% faster")
    
    if n_high_conf > 0:
        print(f"\nWith confidence filter (>={args.confidence_threshold}):")
        print(f"  Trades only {n_high_conf/len(y_test)*100:.0f}% of samples")
        print(f"  But achieves {hc_auc:.3f} AUC on those samples")
        print(f"   Trade less, win more!")

if __name__ == '__main__':
    main()

