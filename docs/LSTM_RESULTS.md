# LSTM Implementation - Complete 

## Summary

Successfully implemented and trained LSTM (Long Short-Term Memory) neural network models for forex prediction using TensorFlow/Keras.

---

## Model Architecture

```
Input: (lookback=60, features=14)
    
LSTM(64 units, return_sequences=True)
    
Dropout(0.2)
    
LSTM(32 units)
    
Dropout(0.2)
    
Dense(32, relu)
    
Dropout(0.1)
    
Dense(1, sigmoid)  probability
```

**Parameters**: 33,729 (131.75 KB)

---

## Training Results

### USDJPY.sim LSTM
```
Train:  AUC=0.6221  ACC=58.8%
Val:    AUC=0.5577  ACC=61.3%
Test:   AUC=0.5789  ACC=62.0%

Epochs: 13 (early stopping)
Best Val AUC: 0.558 (epoch 3)
```

### EURUSD.sim LSTM
```
Train:  AUC=0.6376  ACC=61.2%
Val:    AUC=0.6091  ACC=67.0%
Test:   AUC=0.5923  ACC=57.9%

Epochs: 13 (early stopping)
Best Val AUC: 0.609 (epoch 3)
```

---

## RandomForest vs LSTM Comparison

### USDJPY.sim

| Metric | RandomForest | LSTM | Winner |
|--------|-------------|------|--------|
| **Val AUC** | 0.525 | 0.558 |  **LSTM** |
| **Test AUC** | 0.582 | 0.579 | RF (marginal) |
| **Val ACC** | 62.6% | 61.3% | RF |
| **Test ACC** | 71.1% | 62.0% | RF |
| **Training Time** | ~1 sec | ~30 sec | RF |
| **Model Size** | ~15 MB | 132 KB | LSTM |

**Verdict**: LSTM shows slight improvement in validation AUC, but RF is faster and simpler for baseline.

### EURUSD.sim

| Metric | RandomForest | LSTM | Winner |
|--------|-------------|------|--------|
| **Val AUC** | 0.592 | 0.609 |  **LSTM** |
| **Test AUC** | 0.614 | 0.592 | RF |
| **Val ACC** | 70.4% | 67.0% | RF |
| **Test ACC** | 63.4% | 57.9% | RF |
| **Training Time** | ~1 sec | ~30 sec | RF |

**Verdict**: LSTM shows better validation AUC for EURUSD. Mixed results overall.

---

## Key Features

###  Implemented
- [x] LSTM with 2 layers (64  32 units)
- [x] Dropout regularization (0.2)
- [x] Sequence modeling (60-bar lookback)
- [x] Class-weighted training
- [x] Early stopping (patience=10)
- [x] Learning rate reduction on plateau
- [x] Model checkpointing (best model saved)
- [x] Proper time-series train/val/test split
- [x] Feature scaling (StandardScaler)
- [x] Evaluation metrics (AUC, ACC)

###  Training Features
- **Optimizer**: Adam (lr=0.001  auto-reduced)
- **Loss**: Binary crossentropy
- **Batch Size**: 64
- **Max Epochs**: 50 (stopped early ~13)
- **Class Weights**: Balanced

---

## Files Created

### Models
- `models/EURUSD.sim_lstm.keras` - Final EURUSD LSTM model
- `models/EURUSD.sim_lstm_best.keras` - Best EURUSD model (epoch 3)
- `models/EURUSD.sim_lstm_scaler.pkl` - Feature scaler
- `models/EURUSD.sim_lstm_metadata.json` - Model metadata

- `models/USDJPY.sim_lstm.keras` - Final USDJPY LSTM model
- `models/USDJPY.sim_lstm_best.keras` - Best USDJPY model (epoch 3)
- `models/USDJPY.sim_lstm_scaler.pkl` - Feature scaler
- `models/USDJPY.sim_lstm_metadata.json` - Model metadata

### Scripts
- `train_lstm.py` - LSTM training script

---

## Usage

### Train LSTM Model
```powershell
# Basic training
python train_lstm.py --symbol USDJPY.sim

# With custom parameters
python train_lstm.py --symbol EURUSD.sim \
  --lookback 60 \
  --epochs 50 \
  --batch_size 64 \
  --lstm_units 64 \
  --dropout 0.2 \
  --lr 0.001
```

### Load and Use LSTM Model
```python
import tensorflow as tf
import joblib
import numpy as np

# Load model and scaler
model = tf.keras.models.load_model('models/USDJPY.sim_lstm_best.keras')
scaler = joblib.load('models/USDJPY.sim_lstm_scaler.pkl')

# Prepare sequence (60 bars  14 features)
X = ...  # Your features
X_scaled = scaler.transform(X)
X_seq = X_scaled[-60:].reshape(1, 60, 14)

# Predict
prob = model.predict(X_seq)[0, 0]
print(f"Probability of up move: {prob:.3f}")
```

---

## Observations

### Strengths of LSTM
 Better at capturing temporal dependencies
 Can learn sequential patterns
 Slightly better validation AUC on EURUSD
 Smaller model size

### Weaknesses of LSTM
 Overfits more (train AUC >> val AUC)
 Much slower to train (~30x slower)
 Requires more data for good performance
 More hyperparameters to tune
 Needs sequences (loses first 60 bars)

### Strengths of RandomForest
 Fast training
 More stable (less overfitting)
 Works well with limited data
 No sequence requirement
 Feature importance available

### Weaknesses of RandomForest
 Can't capture temporal dependencies
 Treats each bar independently
 Larger model file size

---

## Recommendations

### For Production Use

**Use RandomForest if**:
- You want fast iteration
- You have limited data
- You need interpretability
- Simplicity is priority

**Use LSTM if**:
- You have lots of data (50k+ bars)
- You believe temporal patterns exist
- You can afford longer training
- You want to try advanced architectures

**Best Approach**:
1. Start with RandomForest (baseline)
2. Try LSTM for improvement
3. Create ensemble (RF + LSTM)
4. Use walk-forward validation for both

---

## Next Steps for LSTM

### To Improve Performance
1.  More data (100k+ bars)
2.  Longer sequences (lookback=120 or 240)
3.  Add GRU layers (faster than LSTM)
4.  Bidirectional LSTM
5.  Attention mechanism
6.  More features (order flow, tick data)
7.  Hyperparameter tuning (grid search)
8.  Regularization (L1/L2, higher dropout)
9.  Walk-forward training
10.  Ensemble with RF

### Advanced Architectures to Try
- Transformer (self-attention)
- Conv1D + LSTM (CNN-LSTM hybrid)
- Temporal Convolutional Network (TCN)
- Multi-task learning (predict multiple horizons)

---

## Performance Summary

| Model | Symbol | Val AUC | Test AUC | Status |
|-------|--------|---------|----------|--------|
| RF | USDJPY.sim | 0.525 | 0.582 |  Baseline |
| **LSTM** | **USDJPY.sim** | **0.558** | **0.579** |  **Slight improvement** |
| RF | EURUSD.sim | 0.592 | 0.614 |  Baseline |
| **LSTM** | **EURUSD.sim** | **0.609** | **0.592** |  **Better val AUC** |

**Conclusion**: LSTM shows promise but needs more tuning. RF remains solid baseline.

---

## Dependencies Added
- `tensorflow>=2.15.0` (331 MB installed)
  - Includes Keras 3.12
  - GPU support (if available)
  - TensorBoard for visualization

---

**Status**:  LSTM implementation complete and working
**Recommendation**: Use RF for now, explore LSTM with more data/tuning
**Next**: Create ensemble or implement walk-forward validation

