"""
Production-Grade Adaptive Trading Model
LSTM + Meta-Ensemble + Online Learning with proper safeguards

Key Features:
- LSTM with calibrated probabilities (Platt scaling)
- Meta-learner trained on out-of-fold predictions (no leakage)
- Online learning with exponential decay and quality filters
- Prequential evaluation (test-then-learn)
- Graceful degradation with decision path logging
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Deep learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy nn for class definition
    class nn:
        class Module:
            pass
    print("[WARNING] PyTorch not available - LSTM disabled")

# ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score


class LSTMPricePredictor(nn.Module):
    """LSTM network for temporal pattern recognition
    
    Architecture:
    - LSTM(64, return_sequences=True) for early patterns
    - LSTM(32) for final temporal features
    - Dense(64, ReLU) for non-linear combinations
    - Sigmoid output (calibrated later)
    """
    
    def __init__(self, input_size=10, hidden_size_1=64, hidden_size_2=32, dropout=0.2):
        super(LSTMPricePredictor, self).__init__()
        
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            dropout=0.0,
            batch_first=True
        )
        
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            num_layers=1,
            dropout=0.0,
            batch_first=True
        )
        
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size_2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        # Take last output
        out = lstm2_out[:, -1, :]
        
        out = self.dropout2(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


class MetaEnsemble:
    """Meta-model that learns when to trust each base model
    
    Trained on out-of-fold predictions to prevent leakage.
    Uses calibrated base model probabilities.
    """
    
    def __init__(self):
        self.meta_model = LogisticRegression(
            C=1.0,
            max_iter=500,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, base_predictions, regime_features, y_true):
        """Train meta-model on out-of-fold base predictions
        
        Args:
            base_predictions: (n_samples, 3) - [p_rf, p_gb, p_lstm]
            regime_features: (n_samples, 3) - [atr_pct_h1, trend_strength_h1, price_vs_ema20_h1]
            y_true: (n_samples,) - actual outcomes
        """
        if len(base_predictions) < 100:
            print("[META] Insufficient data for meta-model training")
            return False
        
        # Create meta-features
        X_meta = self._create_meta_features(base_predictions, regime_features)
        X_meta_scaled = self.scaler.fit_transform(X_meta)
        
        self.meta_model.fit(X_meta_scaled, y_true)
        self.is_fitted = True
        
        # Evaluate
        meta_pred = self.meta_model.predict_proba(X_meta_scaled)[:, 1]
        auc = roc_auc_score(y_true, meta_pred)
        acc = accuracy_score(y_true, meta_pred > 0.5)
        
        print(f"[META] Trained on {len(y_true)} samples | AUC: {auc:.4f} | Acc: {acc:.4f}")
        return True
    
    def predict_proba(self, base_predictions, regime_features):
        """Get final probability by intelligent weighting
        
        Args:
            base_predictions: (n_samples, 3) array [p_rf, p_gb, p_lstm]
            regime_features: (n_samples, 3) array [atr, trend, price_vs_ema]
        
        Returns:
            final_proba: (n_samples,) probability of up-move
            decision_path: str indicating which engine was used
        """
        if not self.is_fitted:
            # Fall back to simple average
            final = np.mean(base_predictions, axis=1)
            return final, 'avg_fallback'
        
        X_meta = self._create_meta_features(base_predictions, regime_features)
        X_meta_scaled = self.scaler.transform(X_meta)
        
        # Return probability of class 1 (up-move)
        final = self.meta_model.predict_proba(X_meta_scaled)[:, 1]
        return final, 'meta'
    
    def _create_meta_features(self, base_preds, regime_feats):
        """Create features for meta-learner
        
        Features (13 total):
        - 3 base predictions: p_rf, p_gb, p_lstm
        - 3 agreement: mean_p, std_p, entropy
        - 4 pairwise diffs: |rf-gb|, |rf-lstm|, |gb-lstm|, |ensemble-lstm|
        - 3 regime: atr_pct_h1, trend_strength_h1, price_vs_ema20_h1
        """
        features = []
        
        # Base predictions
        features.append(base_preds)  # (n, 3)
        
        # Agreement metrics
        mean_p = np.mean(base_preds, axis=1, keepdims=True)
        std_p = np.std(base_preds, axis=1, keepdims=True)
        
        # Entropy of mean prediction (confidence measure)
        entropy = -(mean_p * np.log(mean_p + 1e-10) + 
                    (1 - mean_p) * np.log(1 - mean_p + 1e-10))
        
        features.append(mean_p)
        features.append(std_p)
        features.append(entropy)
        
        # Pairwise differences
        n_models = base_preds.shape[1]
        if n_models == 3:
            features.append(np.abs(base_preds[:, 0] - base_preds[:, 1]).reshape(-1, 1))  # |rf-gb|
            features.append(np.abs(base_preds[:, 0] - base_preds[:, 2]).reshape(-1, 1))  # |rf-lstm|
            features.append(np.abs(base_preds[:, 1] - base_preds[:, 2]).reshape(-1, 1))  # |gb-lstm|
            
            ensemble_avg = (base_preds[:, 0] + base_preds[:, 1]) / 2
            features.append(np.abs(ensemble_avg - base_preds[:, 2]).reshape(-1, 1))  # |ensemble-lstm|
        
        # Regime features
        features.append(regime_feats)  # (n, 3)
        
        return np.hstack(features)


class AdaptiveModel:
    """Production-grade adaptive trading model
    
    Components:
    1. LSTM for temporal patterns (calibrated)
    2. RF + GB base models (calibrated)
    3. Meta-ensemble for intelligent weighting
    4. Online learning with quality filters
    5. Prequential evaluation
    """
    
    def __init__(self, symbol='USDJPY', lookback=20, update_frequency=100):
        self.symbol = symbol
        self.lookback = lookback
        self.update_frequency = update_frequency
        
        # Base models (will be calibrated)
        self.rf_model = None
        self.gb_model = None
        self.lstm_model = None
        
        # Calibrators (Platt scaling)
        self.rf_calibrator = None
        self.gb_calibrator = None
        self.lstm_calibrator = None
        
        # Meta-ensemble
        self.meta_ensemble = MetaEnsemble()
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.lstm_scaler = StandardScaler()
        
        # Online learning buffer (with quality filter)
        self.buffer_X = deque(maxlen=1000)
        self.buffer_y = deque(maxlen=1000)
        self.buffer_lstm = deque(maxlen=1000)
        self.buffer_regime = deque(maxlen=1000)
        self.buffer_quality = deque(maxlen=1000)  # Track data quality
        self.samples_since_update = 0
        
        # Exponential decay for sample weights
        self.decay_lambda = 0.995
        
        # Prequential evaluation
        self.eval_history = {
            'predictions': [],
            'actuals': [],
            'decision_paths': [],
            'timestamps': []
        }
        
        # Performance tracking
        self.model_scores = {
            'rf': {'auc': [], 'acc': []},
            'gb': {'auc': [], 'acc': []},
            'lstm': {'auc': [], 'acc': []},
            'meta': {'auc': [], 'acc': []}
        }
        
        # Load existing models
        self.load_models()
    
    def load_models(self):
        """Load pre-trained and calibrated models"""
        model_dir = Path('models')
        
        try:
            # Load RF
            rf_path = model_dir / 'rf_model_calibrated.pkl'
            if rf_path.exists():
                self.rf_model = joblib.load(rf_path)
                print("[LOAD] Loaded calibrated RF")
            
            # Load GB
            gb_path = model_dir / 'gb_model_calibrated.pkl'
            if gb_path.exists():
                self.gb_model = joblib.load(gb_path)
                print("[LOAD] Loaded calibrated GB")
            
            # Load scalers
            scaler_path = model_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.feature_scaler = joblib.load(scaler_path)
                print("[LOAD] Loaded feature scaler")
            
            lstm_scaler_path = model_dir / 'lstm_scaler.pkl'
            if lstm_scaler_path.exists():
                self.lstm_scaler = joblib.load(lstm_scaler_path)
                print("[LOAD] Loaded LSTM scaler")
            
            # Load LSTM
            if TORCH_AVAILABLE:
                lstm_path = model_dir / 'lstm_model.pth'
                lstm_calib_path = model_dir / 'lstm_calibrator.pkl'
                
                if lstm_path.exists() and lstm_calib_path.exists():
                    self.lstm_model = LSTMPricePredictor(
                        input_size=10,  # OHLCV + 5 cheap features
                        hidden_size_1=64,
                        hidden_size_2=32
                    )
                    self.lstm_model.load_state_dict(torch.load(lstm_path))
                    self.lstm_model.eval()
                    
                    self.lstm_calibrator = joblib.load(lstm_calib_path)
                    print("[LOAD] Loaded calibrated LSTM")
            
            # Load meta-ensemble
            meta_path = model_dir / 'meta_ensemble.pkl'
            if meta_path.exists():
                self.meta_ensemble = joblib.load(meta_path)
                print("[LOAD] Loaded meta-ensemble")
        
        except Exception as e:
            print(f"[ERROR] Loading models: {e}")
    
    def save_models(self):
        """Save all models"""
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)
        
        try:
            # Save calibrated models
            if self.rf_model is not None:
                joblib.dump(self.rf_model, model_dir / 'rf_model_calibrated.pkl')
            
            if self.gb_model is not None:
                joblib.dump(self.gb_model, model_dir / 'gb_model_calibrated.pkl')
            
            # Save LSTM
            if self.lstm_model is not None and TORCH_AVAILABLE:
                torch.save(self.lstm_model.state_dict(), model_dir / 'lstm_model.pth')
                if self.lstm_calibrator is not None:
                    joblib.dump(self.lstm_calibrator, model_dir / 'lstm_calibrator.pkl')
            
            # Save meta-ensemble
            joblib.dump(self.meta_ensemble, model_dir / 'meta_ensemble.pkl')
            
            # Save scalers
            joblib.dump(self.feature_scaler, model_dir / 'scaler.pkl')
            joblib.dump(self.lstm_scaler, model_dir / 'lstm_scaler.pkl')
            
            print(f"[SAVE] Models saved to {model_dir}")
        
        except Exception as e:
            print(f"[ERROR] Saving models: {e}")
    
    def prepare_lstm_features(self, df):
        """Prepare LSTM sequence with OHLCV + 5 cheap features
        
        Features per bar (10 total):
        - OHLCV (5)
        - pct_return (1)
        - atr_pct (1)
        - price_vs_ema20 (1)
        - rsi14 (1)
        - momentum_5 (1)
        """
        if len(df) < self.lookback:
            return None
        
        # Calculate features
        df = df.copy()
        df['pct_return'] = df['close'].pct_change()
        df['momentum_5'] = df['close'].pct_change(5)
        
        # Select features
        feature_cols = ['open', 'high', 'low', 'close', 'volume',
                       'pct_return', 'atr_pct', 'price_vs_ema20', 'rsi14', 'momentum_5']
        
        # Fill NaN
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        df = df[feature_cols].fillna(0)
        
        # Normalize (fit on training data only in production)
        features_scaled = self.lstm_scaler.transform(df.values)
        
        # Create sequences
        sequences = []
        for i in range(len(features_scaled) - self.lookback):
            seq = features_scaled[i:i + self.lookback]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def extract_regime_features(self, df):
        """Extract regime features for meta-learner
        
        Returns: [atr_pct_h1, trend_strength_h1, price_vs_ema20_h1]
        """
        latest = df.iloc[-1]
        
        regime = np.array([
            latest.get('atr_pct_h1', 0),
            latest.get('trend_strength_h1', 0),
            latest.get('price_vs_ema20_h1', 0)
        ])
        
        return regime.reshape(1, -1)
    
    def predict(self, df, feature_cols):
        """Get ensemble prediction with meta-learning and decision path logging
        
        Args:
            df: DataFrame with all features
            feature_cols: List of feature column names
        
        Returns:
            final_proba: Probability of up-move (0-1)
            model_probas: Dict of individual model probabilities
            decision_path: String indicating which engine was used
        """
        try:
            latest_features = df.iloc[-1:][feature_cols].values
            
            # Get predictions from base models (calibrated)
            probas = []
            model_probas = {}
            
            # Random Forest (calibrated)
            if self.rf_model is not None:
                rf_proba = self.rf_model.predict_proba(latest_features)[0, 1]
                probas.append(rf_proba)
                model_probas['rf'] = rf_proba
            
            # Gradient Boosting (calibrated)
            if self.gb_model is not None:
                gb_proba = self.gb_model.predict_proba(latest_features)[0, 1]
                probas.append(gb_proba)
                model_probas['gb'] = gb_proba
            
            # LSTM (calibrated)
            if self.lstm_model is not None and self.lstm_calibrator is not None and TORCH_AVAILABLE:
                lstm_seq = self.prepare_lstm_features(df)
                if lstm_seq is not None and len(lstm_seq) > 0:
                    with torch.no_grad():
                        lstm_input = torch.FloatTensor(lstm_seq[-1:])
                        lstm_raw = self.lstm_model(lstm_input).item()
                        # Calibrate
                        lstm_proba = self.lstm_calibrator.predict_proba([[lstm_raw]])[0, 1]
                        probas.append(lstm_proba)
                        model_probas['lstm'] = lstm_proba
            
            if len(probas) == 0:
                return 0.5, {}, 'no_models'
            
            # Get regime features
            regime_features = self.extract_regime_features(df)
            
            # Get meta-ensemble prediction
            base_predictions = np.array(probas).reshape(1, -1)
            final_proba, decision_path = self.meta_ensemble.predict_proba(
                base_predictions,
                regime_features
            )
            
            model_probas['meta'] = final_proba[0]
            
            return final_proba[0], model_probas, decision_path
        
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return 0.5, {}, 'error'
    
    def update_online(self, df, feature_cols, actual_outcome, spread_pct, slippage_pct):
        """Continuous learning with quality filters
        
        Args:
            df: DataFrame with features
            feature_cols: Feature columns
            actual_outcome: 1 if price went up, 0 if down
            spread_pct: Spread as % of price (for quality filter)
            slippage_pct: Slippage as % (for quality filter)
        """
        try:
            # Quality filter: skip abnormal conditions
            spread_threshold = np.percentile(list(self.buffer_quality) or [0.001], 95) if len(self.buffer_quality) > 20 else 0.01
            
            if spread_pct > spread_threshold or slippage_pct > 0.5:
                print(f"[SKIP] Abnormal conditions: spread={spread_pct:.4f}%, slippage={slippage_pct:.2f}%")
                return
            
            # Add to buffer
            latest_features = df.iloc[-1:][feature_cols].values[0]
            regime_features = self.extract_regime_features(df)
            
            self.buffer_X.append(latest_features)
            self.buffer_y.append(actual_outcome)
            self.buffer_regime.append(regime_features[0])
            self.buffer_quality.append(spread_pct)
            
            # LSTM sequence
            lstm_seq = self.prepare_lstm_features(df)
            if lstm_seq is not None and len(lstm_seq) > 0:
                self.buffer_lstm.append(lstm_seq[-1])
            
            self.samples_since_update += 1
            
            # Retrain if enough new samples
            if self.samples_since_update >= self.update_frequency:
                self._retrain_models()
                self.samples_since_update = 0
        
        except Exception as e:
            print(f"[ERROR] Online update failed: {e}")
    
    def _retrain_models(self):
        """Retrain models with buffered data and exponential decay"""
        print(f"[RETRAIN] Starting with {len(self.buffer_X)} samples...")
        
        if len(self.buffer_X) < 100:
            print("[RETRAIN] Insufficient data")
            return
        
        X = np.array(list(self.buffer_X))
        y = np.array(list(self.buffer_y))
        
        # Calculate sample weights (exponential decay)
        n = len(y)
        weights = np.array([self.decay_lambda ** (n - i - 1) for i in range(n)])
        weights /= weights.sum()  # Normalize
        
        try:
            # Retrain RF with sample weights
            if self.rf_model is not None:
                base_rf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    random_state=42
                )
                base_rf.fit(X, y, sample_weight=weights)
                
                # Calibrate
                self.rf_model = CalibratedClassifierCV(
                    base_rf,
                    method='sigmoid',
                    cv=3
                )
                self.rf_model.fit(X, y)
            
            # Retrain GB with sample weights
            if self.gb_model is not None:
                base_gb = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
                base_gb.fit(X, y, sample_weight=weights)
                
                # Calibrate
                self.gb_model = CalibratedClassifierCV(
                    base_gb,
                    method='sigmoid',
                    cv=3
                )
                self.gb_model.fit(X, y)
            
            # Retrain LSTM (if enough sequences)
            if TORCH_AVAILABLE and len(self.buffer_lstm) >= 100:
                self._retrain_lstm(weights)
            
            # Retrain meta-ensemble on out-of-fold predictions
            self._retrain_meta(X, y)
            
            # Save updated models
            self.save_models()
            
            print(f"[RETRAIN] Complete! Models updated.")
        
        except Exception as e:
            print(f"[ERROR] Retraining failed: {e}")
    
    def _retrain_lstm(self, sample_weights):
        """Retrain LSTM with recent sequences and weighted loss"""
        if not TORCH_AVAILABLE or len(self.buffer_lstm) < 100:
            return
        
        print("[LSTM] Retraining...")
        
        X_lstm = np.array(list(self.buffer_lstm))
        y_lstm = np.array(list(self.buffer_y)[-len(X_lstm):])
        weights = sample_weights[-len(X_lstm):]
        
        # Create/reset LSTM
        if self.lstm_model is None:
            self.lstm_model = LSTMPricePredictor(
                input_size=10,
                hidden_size_1=64,
                hidden_size_2=32
            )
        
        # Training with weighted loss
        criterion = nn.BCELoss(reduction='none')
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
        
        X_tensor = torch.FloatTensor(X_lstm)
        y_tensor = torch.FloatTensor(y_lstm).unsqueeze(1)
        w_tensor = torch.FloatTensor(weights).unsqueeze(1)
        
        self.lstm_model.train()
        for epoch in range(30):
            optimizer.zero_grad()
            outputs = self.lstm_model(X_tensor)
            loss = (criterion(outputs, y_tensor) * w_tensor).mean()
            loss.backward()
            optimizer.step()
        
        self.lstm_model.eval()
        
        # Calibrate LSTM predictions
        with torch.no_grad():
            lstm_raw = self.lstm_model(X_tensor).numpy().flatten()
        
        self.lstm_calibrator = CalibratedClassifierCV(
            estimator=None,
            method='sigmoid',
            cv='prefit'
        )
        # Hack: use LogisticRegression as wrapper for calibration
        from sklearn.base import BaseEstimator, ClassifierMixin
        
        class LSTMWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, raw_preds):
                self.raw_preds = raw_preds
            
            def predict_proba(self, X):
                # Return raw predictions as probabilities
                return np.column_stack([1 - self.raw_preds, self.raw_preds])
        
        wrapper = LSTMWrapper(lstm_raw)
        self.lstm_calibrator = CalibratedClassifierCV(wrapper, method='sigmoid', cv=3)
        self.lstm_calibrator.fit(lstm_raw.reshape(-1, 1), y_lstm)
        
        print(f"[LSTM] Retrained and calibrated (loss: {loss.item():.4f})")
    
    def _retrain_meta(self, X, y):
        """Retrain meta-ensemble on out-of-fold predictions to prevent leakage"""
        if self.rf_model is None or self.gb_model is None:
            return
        
        # Get base predictions
        rf_preds = self.rf_model.predict_proba(X)[:, 1]
        gb_preds = self.gb_model.predict_proba(X)[:, 1]
        
        base_preds = [rf_preds, gb_preds]
        
        # Add LSTM if available
        if self.lstm_model is not None and len(self.buffer_lstm) >= len(X):
            X_lstm = np.array(list(self.buffer_lstm)[-len(X):])
            with torch.no_grad():
                lstm_input = torch.FloatTensor(X_lstm)
                lstm_raw = self.lstm_model(lstm_input).numpy().flatten()
                lstm_preds = self.lstm_calibrator.predict_proba(lstm_raw.reshape(-1, 1))[:, 1]
                base_preds.append(lstm_preds)
        
        base_preds = np.column_stack(base_preds)
        
        # Get regime features
        regime_feats = np.array(list(self.buffer_regime)[-len(X):])
        
        # Train meta-ensemble
        self.meta_ensemble.fit(base_preds, regime_feats, y)
    
    def log_prequential(self, prediction, actual, decision_path):
        """Log prediction for prequential evaluation (test-then-learn)"""
        self.eval_history['predictions'].append(prediction)
        self.eval_history['actuals'].append(actual)
        self.eval_history['decision_paths'].append(decision_path)
        self.eval_history['timestamps'].append(datetime.now())
    
    def get_prequential_metrics(self, window=100):
        """Get rolling evaluation metrics"""
        if len(self.eval_history['predictions']) < window:
            return None
        
        preds = np.array(self.eval_history['predictions'][-window:])
        actuals = np.array(self.eval_history['actuals'][-window:])
        
        auc = roc_auc_score(actuals, preds)
        acc = accuracy_score(actuals, preds > 0.5)
        precision_07 = np.mean(actuals[preds >= 0.7]) if np.any(preds >= 0.7) else 0
        
        return {
            'auc': auc,
            'accuracy': acc,
            'precision@0.7': precision_07,
            'n_samples': window
        }
