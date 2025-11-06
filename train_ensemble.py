"""
Ensemble model training with multiple algorithms.

Combines:
1. RandomForest
2. Gradient Boosting (XGBoost)
3. Lorentzian Classifier (if available)

Uses soft voting for predictions.
"""
import argparse
import os
import joblib
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split


class EnsembleClassifier:
    """
    Ensemble of multiple classifiers with soft voting.
    """
    
    def __init__(self, models=None, weights=None):
        """
        Parameters:
        -----------
        models : list
            List of (name, model) tuples
        weights : list
            Weights for each model (default: equal weights)
        """
        self.models = models or []
        self.weights = weights or [1.0] * len(self.models) if models else []
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """Train all models in the ensemble."""
        print("\nTraining ensemble models...")
        print("=" * 60)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train each model
        for i, (name, model) in enumerate(self.models):
            print(f"\nTraining {name}...")
            model.fit(X_scaled, y)
            
            # Evaluate on training data
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, y_pred_proba)
            print(f"  Training AUC: {auc:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probabilities using weighted soft voting.
        
        Returns:
        --------
        np.ndarray
            Array of shape (n_samples, 2) with class probabilities
        """
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = []
        for name, model in self.models:
            pred = model.predict_proba(X_scaled)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        total_weight = sum(self.weights)
        
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += pred * (weight / total_weight)
        
        return weighted_pred
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def create_ensemble(use_xgboost=True):
    """
    Create ensemble of models.
    
    Parameters:
    -----------
    use_xgboost : bool
        Whether to include XGBoost
    
    Returns:
    --------
    EnsembleClassifier
    """
    models = []
    weights = []
    
    # 1. RandomForest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    models.append(('RandomForest', rf))
    weights.append(1.0)
    
    # 2. Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    models.append(('GradientBoosting', gb))
    weights.append(1.0)
    
    # 3. XGBoost (if available and requested)
    if use_xgboost:
        try:
            import xgboost as xgb
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
            models.append(('XGBoost', xgb_model))
            weights.append(1.2)  # Slightly higher weight for XGBoost
            
            print("XGBoost included in ensemble")
        except ImportError:
            print("XGBoost not available, using RF + GradientBoosting only")
    
    return EnsembleClassifier(models=models, weights=weights)


def train_and_evaluate(symbol, use_enhanced_features=True, test_size=0.2):
    """
    Train ensemble model and evaluate performance.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol
    use_enhanced_features : bool
        Use enhanced features (True) or basic features (False)
    test_size : float
        Test set size
    """
    print("=" * 80)
    print(f"ENSEMBLE MODEL TRAINING: {symbol}")
    print("=" * 80)
    
    # Load config
    cfg = yaml.safe_load(open('config.yaml'))
    feat_dir = Path(cfg['feature_dir'])
    model_dir = Path(cfg['model_dir'])
    model_dir.mkdir(exist_ok=True)
    
    # Load features
    if use_enhanced_features:
        feat_path = feat_dir / f'{symbol}_features_enhanced.csv'
        if not feat_path.exists():
            print(f"Enhanced features not found at {feat_path}")
            print("Run: python build_features_enhanced.py --symbol {symbol}")
            return
    else:
        feat_path = feat_dir / f'{symbol}_features.csv'
    
    print(f"\nLoading features from: {feat_path}")
    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df):,} samples")
    
    # Prepare features
    feature_cols = [c for c in df.columns if c not in ['time', 'y']]
    X = df[feature_cols].values
    y = df['y'].values
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target distribution: {np.bincount(y)} (0={np.sum(y==0)}, 1={np.sum(y==1)})")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # No shuffle for time series
    )
    
    print(f"\nTrain: {len(X_train):,} samples")
    print(f"Test:  {len(X_test):,} samples")
    
    # Create and train ensemble
    ensemble = create_ensemble(use_xgboost=True)
    ensemble.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nOverall Performance:")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    
    # High confidence predictions
    high_conf_mask = (y_pred_proba >= 0.6) | (y_pred_proba <= 0.4)
    if high_conf_mask.sum() > 0:
        y_test_hc = y_test[high_conf_mask]
        y_pred_proba_hc = y_pred_proba[high_conf_mask]
        auc_hc = roc_auc_score(y_test_hc, y_pred_proba_hc)
        
        print(f"\nHigh Confidence Predictions (p<0.4 or p>0.6):")
        print(f"  Samples: {high_conf_mask.sum()} ({high_conf_mask.sum()/len(y_test)*100:.1f}%)")
        print(f"  AUC:     {auc_hc:.4f}")
    
    # Individual model performance
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("=" * 60)
    
    X_test_scaled = ensemble.scaler.transform(X_test)
    for name, model in ensemble.models:
        y_pred_proba_ind = model.predict_proba(X_test_scaled)[:, 1]
        auc_ind = roc_auc_score(y_test, y_pred_proba_ind)
        acc_ind = accuracy_score(y_test, (y_pred_proba_ind >= 0.5).astype(int))
        
        print(f"\n{name}:")
        print(f"  AUC:      {auc_ind:.4f}")
        print(f"  Accuracy: {acc_ind:.4f}")
    
    # Save ensemble
    model_path = model_dir / f'{symbol}_ensemble.pkl'
    joblib.dump(ensemble, model_path)
    print(f"\n Ensemble saved to: {model_path}")
    
    # Save individual models for compatibility
    scaler_path = model_dir / 'scaler.pkl'
    joblib.dump(ensemble.scaler, scaler_path)
    print(f" Scaler saved to: {scaler_path}")
    
    # Save feature list
    feature_list_path = model_dir / f'{symbol}_features.txt'
    with open(feature_list_path, 'w') as f:
        for feat in feature_cols:
            f.write(f"{feat}\n")
    print(f" Feature list saved to: {feature_list_path}")
    
    # Save performance metrics
    results = {
        'symbol': symbol,
        'n_features': len(feature_cols),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'test_auc': auc,
        'test_accuracy': acc,
        'enhanced_features': use_enhanced_features
    }
    
    results_path = model_dir / f'{symbol}_ensemble_results.txt'
    with open(results_path, 'w') as f:
        f.write("ENSEMBLE MODEL RESULTS\n")
        f.write("=" * 60 + "\n\n")
        for key, value in results.items():
            f.write(f"{key:20s}: {value}\n")
    
    print(f" Results saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    return ensemble, results


def main():
    parser = argparse.ArgumentParser(description='Train ensemble model')
    parser.add_argument('--symbol', type=str, default='USDJPY', help='Trading symbol')
    parser.add_argument('--enhanced', type=int, default=1, help='Use enhanced features (1=yes, 0=no)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    
    args = parser.parse_args()
    
    train_and_evaluate(
        symbol=args.symbol,
        use_enhanced_features=bool(args.enhanced),
        test_size=args.test_size
    )


if __name__ == '__main__':
    main()

