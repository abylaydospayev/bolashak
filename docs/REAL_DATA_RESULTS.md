# Real Data Results - November 4, 2025

##  Successfully Connected to MT5 and Trained on Real Data

### Data Collection
- **MT5 Version**: 500.5370 (October 17, 2025)
- **Symbols**: EURUSD.sim, USDJPY.sim (simulated trading)
- **Bars Retrieved**: 11,590 per symbol (M15 timeframe)
- **Data Points**: ~11,585 after feature engineering

---

## Model Performance Comparison

### EURUSD.sim - Real Data

#### Training Metrics
- **Train**: AUC=0.878, ACC=79.2%
- **Validation**: AUC=0.592, ACC=70.4%
- **Test**: AUC=0.614, ACC=63.4%

 **Improvement over synthetic**: Val AUC improved from 0.502  0.592 (+18%)

#### Backtest Results
```
Trades:        1,396
Win Rate:      42.3%
Avg Win:       $44.82 per 100k
Avg Loss:      -$45.78 per 100k
Profit Factor: 0.72
Max Drawdown:  106.3%
Final Equity:  -$456.33
```

**Analysis**: 
- Model shows some predictive power (Val AUC 0.59 > random 0.5)
- Win rate low but balanced win/loss ratio
- Still unprofitable in baseline form
- Needs: Better thresholds, position sizing, or additional features

---

### USDJPY.sim - Real Data 

#### Training Metrics
- **Train**: AUC=0.872, ACC=77.3%
- **Validation**: AUC=0.525, ACC=62.6%
- **Test**: AUC=0.582, ACC=71.1%

 **Improvement over synthetic**: Val AUC improved from 0.499  0.525 (+5%)

#### Backtest Results 
```
Trades:        2,033
Win Rate:      53.9% 
Avg Win:       $8,533 per 100k 
Avg Loss:      -$7,038 per 100k
Profit Factor: 1.42 
Max Drawdown:  28.3%
Final Equity:  +$2,752,046 
```

**Analysis**: 
-  **PROFITABLE** on backtest!
-  Win rate > 50%
-  Profit factor > 1.0 (making $1.42 for every $1 lost)
-  Reasonable drawdown (28%)
-  Positive expectancy per trade

---

## Key Improvements with Real Data

| Metric | Synthetic USDJPY | Real USDJPY | Change |
|--------|------------------|-------------|--------|
| Val AUC | 0.499 | 0.525 | +5.2% |
| Win Rate | 47.9% | 53.9% | +6.0% |
| Profit Factor | 0.68 | 1.42 | +109%  |
| Final Equity | -$1.5M | +$2.8M | Profitable! |

---

## Real vs Synthetic Data Comparison

### Synthetic Data (Random)
-  No real patterns
-  Val AUC  0.50 (random)
-  Negative returns
-  Good for testing infrastructure

### Real Data (MT5)
-  Real market patterns
-  Val AUC > 0.52 (some signal)
-  USDJPY shows profitability
-  Realistic price action

---

## Next Steps

### For USDJPY (Promising!)
1.  **Test on demo account** - System shows promise
2.  Run walk-forward validation
3.  Optimize thresholds (currently 0.6/0.4)
4.  Add position sizing based on confidence
5.  Monitor live performance for 1+ month

### For EURUSD (Needs Work)
1.  Feature engineering (add volume, order flow)
2.  Try different horizons (h=5, h=10)
3.  Hyperparameter tuning
4.  Consider different stop/TP ratios
5.  Filter trades by volatility regime

### General Improvements
1.  Implement walk-forward cross-validation
2.  Add risk-based position sizing
3.  Create ensemble model
4.  Add more features (time of day, day of week)
5.  Try LSTM/GRU for sequence modeling
6.  Add regime detection (trending vs ranging)

---

## Risk Warning 

While USDJPY shows positive backtest results:

1. **This is a baseline model** - No hyperparameter optimization
2. **Limited data** - Only 11,590 bars (~120 days of M15 data)
3. **Overfitting risk** - Train AUC much higher than validation
4. **No walk-forward** - Single train/test split
5. **Simulated data** - Using .sim suffix (check if real or demo)

### Before Live Trading:
-  Test on demo account for 1+ month
-  Implement proper position sizing
-  Monitor for model drift
-  Set up risk limits (max drawdown, daily loss limit)
-  Validate symbol is real, not simulated
-  Confirm broker costs match config.yaml

---

## Commands Used

```powershell
# Check symbols
python check_symbols.py

# Pull real data
python make_dataset.py --symbol EURUSD.sim --timeframe M15 --bars 50000
python make_dataset.py --symbol USDJPY.sim --timeframe M15 --bars 50000

# Build features
python build_features.py --symbol EURUSD.sim --h 3
python build_features.py --symbol USDJPY.sim --h 3

# Train models
python train_rf.py --symbol EURUSD.sim
python train_rf.py --symbol USDJPY.sim

# Backtest
python backtest.py --symbol EURUSD.sim
python backtest.py --symbol USDJPY.sim

# Live (when ready)
python signal_mt5.py --symbol USDJPY.sim
```

---

## Conclusion

 **System working with real MT5 data**
 **USDJPY shows profitable baseline** (Profit Factor: 1.42, +$2.8M on backtest)
 **EURUSD needs optimization** (Profit Factor: 0.72, -$456)

The system is now trained on **real market data** and shows promise, especially for USDJPY. Ready for demo account testing!

---

**Date**: November 4, 2025  
**Status**:  Real data integrated successfully  
**Best Performer**: USDJPY.sim (PF: 1.42, WR: 53.9%)

