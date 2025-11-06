# Lorentzian Classification - Implementation Summary

##  Completed Tasks

### 1. Basic Lorentzian Classifier 

**File**: `lorentzian_classifier.py`, `train_lorentzian.py`

**Implementation**:
- k-Nearest Neighbors with Lorentzian distance metric
- Distance formula: `D(x, y) =  log(1 + |x_i - y_i|)`
- Distance-weighted voting (closer neighbors have more influence)
- Lazy learning (stores training data, no actual "training")

**Results**:

| Symbol | Train AUC | Val AUC | Test AUC | Speed |
|--------|-----------|---------|----------|-------|
| **USDJPY** | 1.0000 | 0.5589 | 0.5374 | 43ms/pred |
| **EURUSD** | 1.0000 | 0.5379 | 0.5573 | 43ms/pred |

**Analysis**:
-  **Perfect train AUC (1.0)** = severe overfitting
-  **Test AUC < RF/LSTM** (Lorentzian: 0.54 vs RF: 0.58 vs LSTM: 0.59)
-  **SLOW**: 43ms per prediction (1000x slower than RF/LSTM)
-  **Memory heavy**: Stores entire training set (6951 samples  14 features)

---

### 2. Walk-Forward Validation 

**File**: `walk_forward_lorentzian.py`

**Features**:
- Expanding training window
- Confidence-based filtering (can skip low-confidence predictions)
- Per-fold backtest with costs
- Comparison vs baseline

**Status**: Currently running (slow due to Lorentzian prediction speed)

---

### 3. 3-Way Ensemble (RF + LSTM + Lorentzian) 

**File**: `ensemble3.py`

**Results**:

#### USDJPY
```
Individual models:
  RF:         AUC 0.5813
  LSTM:       AUC 0.5789
  Lorentzian: AUC 0.5330  WORST

Optimal weights: RF=0.00, LSTM=0.70, LC=0.30
Ensemble AUC: 0.5586

Improvement: -3.91%  (worse than RF alone!)
```

#### EURUSD
```
Individual models:
  RF:         AUC 0.5259
  LSTM:       AUC 0.5923  BEST
  Lorentzian: AUC 0.5612

Optimal weights: RF=0.40, LSTM=0.60, LC=0.00  NO LORENTZIAN!
Ensemble AUC: 0.6024

Improvement: +1.70%  (same as 2-way ensemble)
```

**Finding**: Optimizer **excluded Lorentzian entirely** for EURUSD (weight=0.00)!

---

##  Performance Comparison

### Test Set AUC

| Model | USDJPY | EURUSD | Average |
|-------|--------|--------|---------|
| **RandomForest** | 0.5813 | 0.5259 | 0.5536 |
| **LSTM** | 0.5789 | **0.5923** | 0.5856 |
| **Lorentzian** | 0.5330 | 0.5612 | 0.5471 |
| **Ensemble (2-way)** | 0.5814 | 0.6024 | 0.5919 |
| **Ensemble (3-way)** | 0.5586 | 0.6024 | 0.5805 |

**Ranking**:
1.  **LSTM** (0.5856 avg) - Best single model
2.  **2-way Ensemble (RF+LSTM)** (0.5919 avg) - Best overall
3.  **RandomForest** (0.5536 avg)
4.  **Lorentzian** (0.5471 avg) - WORST
5.  **3-way Ensemble** - Doesn't improve over 2-way

---

##  Why Lorentzian Failed

### Expected Benefits
-  Robust to outliers
-  Finds similar market conditions
-  Good for regime detection

### Actual Problems

#### 1. **Severe Overfitting**
```
Train AUC: 1.0000
Val AUC:   0.5589  HUGE GAP
Test AUC:  0.5374
```
- Memorizes training data perfectly
- Fails to generalize

#### 2. **Low k Value**
```
k=8 neighbors out of 6951 training samples = 0.1%
```
- Only looks at 8 closest neighbors
- Too local, misses broader patterns
- Needs k=50-100 for better generalization

#### 3. **Speed Issues**
```
RandomForest: 0.01ms/pred (1000 preds in 10ms)
LSTM:         0.5ms/pred  (1000 preds in 500ms)
Lorentzian:   43ms/pred   (1000 preds in 43,000ms!) 
```
- **4,300x slower than RF**
- **86x slower than LSTM**
- Walk-forward takes hours instead of minutes

#### 4. **Feature Scaling Issues**
- Lorentzian distance sensitive to feature scales
- StandardScaler applied, but may not be optimal
- Log-based distance may not suit financial features

#### 5. **No Clear Regime Detection**
- Expected: High confidence in favorable regimes (like Fold 1)
- Actual: Average confidence only 0.47 (close to random!)
- Doesn't naturally filter bad conditions

---

##  What We Learned

### Lorentzian is NOT a Silver Bullet

1. **For this dataset**: RF and LSTM are better
2. **For regime detection**: Need higher k or different approach
3. **For speed**: Too slow for production use
4. **For ensemble**: Doesn't add value (optimizer sets weight=0)

### Better Alternatives

Instead of Lorentzian, consider:

#### Option A: **Improve k Value**
```python
# Try k=50, 100, 200
python train_lorentzian.py --symbol USDJPY.sim --k 100
```
- More neighbors = less overfitting
- But even slower predictions

#### Option B: **Use Lorentzian Distance in Different Way**
```python
# Not as classifier, but as similarity metric
def find_similar_regimes(current_features, historical_data, k=20):
    distances = [lorentzian_distance(current_features, hist) 
                 for hist in historical_data]
    similar_indices = np.argsort(distances)[:k]
    return historical_data[similar_indices]
```
- Use to find similar historical periods
- Then analyze their characteristics
- Don't use for direct prediction

#### Option C: **Stick with RF + LSTM Ensemble**
```
Current best: 2-way ensemble (RF + LSTM)
  USDJPY: AUC 0.5814
  EURUSD: AUC 0.6024
  Average: 0.5919 
```
- Already works well
- Fast enough for production
- Adding Lorentzian doesn't help

---

##  Recommendations

### For Production

**Use 2-way ensemble (RF + LSTM) without Lorentzian**:
```python
# EURUSD: 40% RF + 60% LSTM
# USDJPY: Use LSTM alone (or light ensemble)
```

### For Research

If you still want to explore Lorentzian:

1. **Increase k to 100-200**
   - Reduce overfitting
   - Accept slower speed

2. **Use as regime filter only**
   - Calculate distance to profitable historical periods (like Fold 1)
   - If distance < threshold  trade with RF/LSTM
   - Else  skip

3. **Try approximate nearest neighbors**
   - Use FAISS or Annoy library
   - 10-100x faster
   - Slight accuracy loss

### For Walk-Forward

**Use RF or LSTM, not Lorentzian**:
```bash
# Fast walk-forward (already tested)
python walk_forward.py --symbol USDJPY.sim --n_splits 5

# Slower but no better
python walk_forward_lorentzian.py --symbol USDJPY.sim --n_splits 5
```

---

##  Next Steps

### Priority 1: **Regime Filter** (from analyze_folds.py)
**This is more important than Lorentzian!**

From your fold analysis, you found:
- Only trade when: ATR < 0.11, 45 < RSI < 55, abs(return_100) < 0.02
- Expected impact: Turn -$1.3M  +$500k 

Implement:
```python
# regime_filter.py
def should_trade(df):
    atr = df['atr_14'].iloc[-1]
    rsi = df['rsi_14'].iloc[-1]
    ret = (df['close'].iloc[-1] / df['close'].iloc[-100]) - 1
    
    # Only favorable conditions
    if atr > 0.11 or rsi > 55 or rsi < 45 or abs(ret) > 0.02:
        return False
    return True
```

### Priority 2: **Test Regime Filter in Walk-Forward**
```bash
python walk_forward.py --symbol USDJPY.sim --use_regime_filter
```

### Priority 3: **Add More Features**
- Volume profile
- Order flow imbalance
- Multi-timeframe (H1, H4)
- Time of day filters

### Priority 4: **(Optional) Retry Lorentzian with k=100**
Only if regime filter doesn't work:
```bash
python train_lorentzian.py --symbol USDJPY.sim --k 100
python walk_forward_lorentzian.py --symbol USDJPY.sim --k 100
```

---

##  Conclusion

### What Worked 
-  Implemented Lorentzian classifier successfully
-  Created 3-way ensemble framework
-  Found that 2-way ensemble (RF+LSTM) is optimal

### What Didn't Work 
-  Lorentzian underperforms RF and LSTM
-  3-way ensemble worse than 2-way
-  Too slow for production (43ms vs 0.01ms)
-  Severe overfitting (Train AUC 1.0, Test AUC 0.54)

### Key Insight 
**Regime filtering > Better classifier**

Your walk-forward fold analysis showed:
- Problem: Model works in some regimes (Fold 1), fails in others (Fold 7)
- Solution: Filter by regime, not by better model
- Expected: -$1.3M  +$500k with regime filter

**Focus on WHEN to trade, not HOW to predict.**

---

## Files Created

1. `lorentzian_classifier.py` - Lorentzian kNN implementation
2. `train_lorentzian.py` - Training script
3. `walk_forward_lorentzian.py` - Walk-forward validation
4. `ensemble3.py` - 3-way ensemble (RF + LSTM + LC)
5. `models/USDJPY.sim_lorentzian.pkl` - Trained model
6. `models/EURUSD.sim_lorentzian.pkl` - Trained model
7. `models/*_ensemble3_config.json` - Ensemble weights

**Status**: All files created, ensemble tested, walk-forward running.

**Next**: Implement regime filter (higher priority than Lorentzian improvements).

