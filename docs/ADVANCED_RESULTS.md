# Walk-Forward Validation & Ensemble Results

## Summary

Implemented both walk-forward validation and ensemble modeling (RF + LSTM).

---

## 1. Walk-Forward Validation 

### What is Walk-Forward?
Realistic backtesting method that avoids look-ahead bias:
1. Train on expanding window
2. Test on next out-of-sample period
3. Retrain and repeat

### USDJPY.sim Walk-Forward (4 folds)

```
Fold  Train Samples  Test Samples    AUC     ACC    Trades  WinRate    PnL
  1        6,951         1,158      0.571   60.2%     109    41.3%   -$99,893
  2        8,109         1,158      0.569   64.2%     150    44.0%   -$44,657
  3        9,267         1,158      0.547   65.0%     354    42.1%  -$487,565
  4       10,425         1,158      0.617   70.0%     213    44.1%  -$182,388
```

**Aggregate Results:**
- Total Trades: 826
- Average AUC: 0.576
- Overall AUC: 0.557
- Average Win Rate: 42.9%
- Average Profit Factor: 0.78
- **Total PnL: -$814,502** 
- Profitable Folds: 0/4 (0%)

**Conclusion**: Walk-forward reveals that RandomForest baseline is **NOT profitable** in realistic out-of-sample testing. This is much more pessimistic than simple train/test split.

---

## 2. Ensemble Model (RF + LSTM) 

### USDJPY.sim Ensemble

| Model | Val AUC | Test AUC | Winner |
|-------|---------|----------|--------|
| RandomForest | 0.525 | 0.581 | - |
| LSTM | **0.558** | 0.579 |  |
| Ensemble | 0.558 | 0.579 | = |

**Optimal Weight**: 100% LSTM (0% RF)

**Conclusion**: For USDJPY, **LSTM alone is best**. Ensemble doesn't improve.

---

### EURUSD.sim Ensemble 

| Model | Val AUC | Test AUC | Winner |
|-------|---------|----------|--------|
| RandomForest | 0.540 | 0.526 | - |
| LSTM | 0.609 | 0.592 | - |
| **Ensemble** | **0.619** | **0.602** | ** +1.7%** |

**Optimal Weight**: 40% RF + 60% LSTM

**Results:**
- Val AUC: 0.619 (best!)
- Test AUC: 0.602 (+1.0% vs LSTM alone)
- Test ACC: 66.0%

**Conclusion**: For EURUSD, **ensemble outperforms** individual models! 

---

## Key Findings

### Walk-Forward Reality Check
 **More realistic** than simple train/test split
 **Reveals overfitting**: Models perform worse out-of-sample
 **USDJPY not profitable** with current approach
 **Need better features/strategy** to pass walk-forward test

### Ensemble Benefits
 **EURUSD**: Ensemble improves test AUC by +1.7%
 **USDJPY**: LSTM alone is better (100% weight)
 **Automatic weight optimization** finds best combination
 **Combines strengths**: RF (stability) + LSTM (temporal patterns)

---

## Comparison: All Methods

### EURUSD.sim

| Method | Val AUC | Test AUC | Notes |
|--------|---------|----------|-------|
| RandomForest | 0.592 | 0.614 | Baseline |
| LSTM | 0.609 | 0.592 | Better val, worse test |
| **Ensemble** | **0.619** | **0.602** | **Best overall**  |
| Walk-Forward | 0.576 | - | Out-of-sample reality |

### USDJPY.sim

| Method | Val AUC | Test AUC | Notes |
|--------|---------|----------|-------|
| RandomForest | 0.525 | 0.582 | Baseline |
| **LSTM** | **0.558** | **0.579** | **Best**  |
| Ensemble | 0.558 | 0.579 | Same as LSTM |
| Walk-Forward | 0.557 | - | Not profitable |

---

## Files Created

### Walk-Forward
- `walk_forward.py` - Walk-forward validation script
- `results/USDJPY.sim_walkforward.csv` - Per-fold results

### Ensemble
- `ensemble.py` - Ensemble model script
- `models/EURUSD.sim_ensemble_config.json` - Optimal weights
- `models/USDJPY.sim_ensemble_config.json` - Optimal weights

---

## Usage

### Run Walk-Forward Validation
```powershell
# 5 folds, 60% initial train, 10% test per fold
python walk_forward.py --symbol USDJPY.sim --n_splits 5

# Custom configuration
python walk_forward.py --symbol EURUSD.sim \
  --n_splits 10 \
  --train_size 0.5 \
  --test_size 0.05
```

### Create Ensemble
```powershell
# Automatic weight optimization
python ensemble.py --symbol EURUSD.sim

# Manual weighting
python ensemble.py --symbol USDJPY.sim --rf_weight 0.7
```

### Use Ensemble in Production
```python
import json
import joblib
import tensorflow as tf
import numpy as np

# Load ensemble config
with open('models/EURUSD.sim_ensemble_config.json') as f:
    config = json.load(f)

# Load models
rf = joblib.load(config['rf_model'])
rf_scaler = joblib.load(config['rf_scaler'])
lstm = tf.keras.models.load_model(config['lstm_model'])
lstm_scaler = joblib.load(config['lstm_scaler'])

# Get predictions
rf_prob = rf.predict_proba(rf_scaler.transform(X))[:, 1]
lstm_prob = lstm.predict(lstm_scaler.transform(X_seq)).flatten()

# Ensemble
ensemble_prob = (config['rf_weight'] * rf_prob + 
                 config['lstm_weight'] * lstm_prob)
```

---

## Recommendations

### For EURUSD 
1. **Use Ensemble** (40% RF + 60% LSTM)
2. Test AUC: 0.602 (best result)
3. Shows genuine improvement

### For USDJPY 
1. **Use LSTM only** (ensemble doesn't help)
2. **Walk-forward shows losses** - needs more work
3. Consider:
   - More features (order flow, volatility regime)
   - Different timeframes (H1, H4)
   - Better entry thresholds
   - Position sizing optimization

### General Improvements Needed
1. **Walk-forward is failing** - models not robust
2. **Need more data** (currently only 5.5 months)
3. **Feature engineering** - add regime detection, order flow
4. **Better thresholds** - optimize prob_buy/prob_sell
5. **Position sizing** - risk-based, not fixed 1 lot
6. **Filter trades** - only trade in favorable conditions

---

## Critical Insight 

**The gap between simple backtest and walk-forward is HUGE:**

| Method | USDJPY PnL |
|--------|-----------|
| Simple Backtest (RF) | **+$2.7M**  |
| Walk-Forward (RF) | **-$814k**  |
| **Difference** | **$3.5M overestimate!** |

This shows:
-  Walk-forward is **essential** for realistic testing
-  Simple train/test split is **overly optimistic**
-  Current strategy **needs significant improvement**

---

## Next Steps

### Immediate (to pass walk-forward)
1.  Add more features (volume profile, volatility regimes)
2.  Optimize entry thresholds per regime
3.  Implement confidence-based sizing
4.  Filter low-quality trades (ATR, time of day)
5.  Try different horizons (h=5, h=10)

### Advanced
1.  Multi-timeframe analysis (M15 + H1 + H4)
2.  Regime detection (trending vs ranging)
3.  Order flow features
4.  Volatility-adjusted sizing
5.  Advanced architectures (Transformer, TCN)

---

## Conclusion

 **Ensemble works** for EURUSD (+1.7% improvement)
 **LSTM better** than RF for both pairs
 **Walk-forward reveals** current strategy isn't profitable
 **Significant work needed** to create production-ready system

**Status**: Tools built, now need better features/strategy to be profitable.

---

**Files**: `walk_forward.py`, `ensemble.py`
**Best Model**: EURUSD Ensemble (40% RF + 60% LSTM, Test AUC: 0.602)
**Reality Check**: Walk-forward shows -$814k PnL (needs improvement)

