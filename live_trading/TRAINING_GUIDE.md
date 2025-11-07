# Adaptive Model Training & Testing Guide

## Overview
Complete guide for training, testing, and deploying the adaptive model with LSTM + Meta-Ensemble + Online Learning.

---

## 🎯 Quick Start

### 1. **Full Test Suite** (Recommended First Run)
Runs all tests: training, prequential evaluation, ablation study
```bash
python live_trading/test_adaptive_model.py --mode full --days 90
```

**What it does:**
- ✅ Fetches 90 days of USDJPY M5 data
- ✅ Trains RF + GB + LSTM + Meta-Ensemble
- ✅ Evaluates on 30% holdout test set
- ✅ Runs prequential (test-then-learn) evaluation
- ✅ Runs ablation study (Baseline → +LSTM → +Online → +Meta)
- ✅ Saves trained model and results JSON

**Expected output:**
```
ADAPTIVE MODEL TEST SUITE
==============================================================
Symbol: USDJPY
Lookback: 90 days

PHASE 1: INITIAL TRAINING
----------------------------------------------------------
Train: 18,000 samples | Test: 7,700 samples
Accuracy:          0.6250
Precision:         0.6180
Recall:            0.6310
AUC:               0.6720
Precision @ 0.7:   0.7240 (892 samples)

PHASE 2: PREQUENTIAL EVALUATION (Test-Then-Learn)
----------------------------------------------------------
Sample  50: Accuracy=0.6200, AUC=0.6680
Sample 100: Accuracy=0.6340, AUC=0.6790
...

PHASE 3: ABLATION STUDY
----------------------------------------------------------
Baseline (RF+GB):        Acc=0.6150, AUC=0.6580
+LSTM:                   Acc=0.6280, AUC=0.6710 (+0.0130)
+Online Learning:        Acc=0.6410, AUC=0.6850 (+0.0130)
+Meta-Ensemble:          Acc=0.6520, AUC=0.6950 (+0.0110)

Total Improvement:       +0.0370 accuracy, +0.0370 AUC
```

**Output files:**
- `adaptive_model_results_20251107_143022.json` - Test metrics
- `adaptive_model_20251107_143022.pkl` - Trained model

---

### 2. **Train Only** (Quick Training)
Just train and save model without extensive testing
```bash
python live_trading/test_adaptive_model.py --mode train_only --days 60
```

**Use when:**
- Quick model update needed
- Limited time for testing
- You already validated the architecture

---

### 3. **Shadow Mode** (Live Testing Without Trading)
Test model on live data WITHOUT placing real trades
```bash
python live_trading/test_adaptive_model.py --mode shadow --shadow-duration 120
```

**What it does:**
- ✅ Loads latest trained model (or trains new one)
- ✅ Monitors live market for 120 minutes
- ✅ Logs BUY/SELL signals WITHOUT trading
- ✅ Saves signals to `shadow_mode_log.txt`

**Expected output:**
```
SHADOW MODE TEST
==============================================================
Duration: 120 minutes
Will log signals WITHOUT placing trades

[SHADOW 14:30:15] Signal #1: BUY @ 149.245 | Probability: 0.723
[SHADOW 14:42:30] Signal #2: SELL @ 149.189 | Probability: 0.285
[SHADOW 15:01:45] Signal #3: BUY @ 149.312 | Probability: 0.741
...

[SHADOW] Completed at 16:30:15
[SHADOW] Total signals generated: 12
[SHADOW] Log saved to: shadow_mode_log.txt
```

**Shadow log format:**
```
14:30:15,BUY,149.245,0.723
14:42:30,SELL,149.189,0.285
15:01:45,BUY,149.312,0.741
```

---

## 📊 Understanding the Metrics

### Training Metrics
- **Accuracy**: % of correct predictions (target: >60%)
- **Precision**: When model predicts BUY/SELL, how often is it right? (target: >65%)
- **Recall**: Of all actual UP moves, how many did we catch? (target: >60%)
- **AUC**: Overall ranking quality (target: >0.65)
- **Precision @ 0.7**: High-confidence BUY signals (target: >70%)
- **Precision @ 0.3**: High-confidence SELL signals (target: >70%)

### Prequential Evaluation
Tests model's ability to learn from new data in real-time (test-then-learn approach)
- Should see gradual improvement over time
- AUC should stabilize around final value
- If AUC drops, online learning may be overfitting

### Ablation Study
Shows contribution of each component:
1. **Baseline (RF + GB)**: Core ensemble (target: 61-62% accuracy)
2. **+LSTM**: Adds temporal patterns (+1-2% accuracy)
3. **+Online**: Adapts to recent data (+1-2% accuracy)
4. **+Meta**: Learns when to trust each model (+1-2% accuracy)

**Total improvement target**: +3-5% accuracy over baseline

---

## 🔧 Advanced Configuration

### Custom Data Range
```bash
# Test on 180 days (more data = better LSTM training)
python live_trading/test_adaptive_model.py --mode full --days 180

# Quick test on 30 days (faster but less reliable)
python live_trading/test_adaptive_model.py --mode full --days 30
```

### Extended Shadow Mode
```bash
# Run overnight (480 minutes = 8 hours)
python live_trading/test_adaptive_model.py --mode shadow --shadow-duration 480

# Run full week (10,080 minutes = 7 days)
python live_trading/test_adaptive_model.py --mode shadow --shadow-duration 10080
```

---

## 🚀 Deployment Workflow

### Step 1: Initial Training (Local PC)
```bash
# Run full test suite
python live_trading/test_adaptive_model.py --mode full --days 90

# Review results
# - Check accuracy > 60%
# - Check AUC > 0.65
# - Verify ablation improvements
```

### Step 2: Shadow Mode Testing (1-2 weeks)
```bash
# Start shadow mode
python live_trading/test_adaptive_model.py --mode shadow --shadow-duration 10080

# Monitor shadow_mode_log.txt
# - Count signals per day
# - Check probability distribution
# - Verify no crashes/errors
```

### Step 3: Analyze Shadow Results
```python
# Analyze shadow mode signals
import pandas as pd

df = pd.read_csv('shadow_mode_log.txt', 
                 names=['time', 'signal', 'price', 'probability'])

print(f"Total signals: {len(df)}")
print(f"BUY signals: {len(df[df['signal'] == 'BUY'])}")
print(f"SELL signals: {len(df[df['signal'] == 'SELL'])}")
print(f"Avg probability: {df['probability'].mean():.3f}")

# Check distribution
print("\nProbability distribution:")
print(df['probability'].describe())
```

### Step 4: Deploy to Production
```bash
# On Azure VM:
cd C:\Users\abylay_dos\Desktop\bolashak

# Pull latest code
git pull origin main

# Copy trained model from local PC
# (use RDP or git to transfer .pkl file)

# Update .env
echo "ADAPTIVE_MODEL_ENABLED=true" >> .env
echo "ADAPTIVE_MODEL_PATH=adaptive_model_20251107_143022.pkl" >> .env

# Start bot with adaptive model
python live_trading/demo_bot_with_risk.py
```

---

## 📈 Performance Expectations

### Good Results (Ready for Production)
- ✅ Accuracy: 62-68%
- ✅ AUC: 0.67-0.75
- ✅ Precision @ 0.7: >70%
- ✅ Ablation improvement: +3-5%
- ✅ Prequential AUC stable or improving

### Warning Signs (Need More Training)
- ⚠️ Accuracy: <60%
- ⚠️ AUC: <0.65
- ⚠️ Precision @ 0.7: <65%
- ⚠️ Ablation improvement: <2%
- ⚠️ Prequential AUC declining

### Critical Issues (Do NOT Deploy)
- ❌ Accuracy: <55% (worse than coin flip)
- ❌ AUC: <0.60
- ❌ Training errors/crashes
- ❌ Prequential AUC drops >5% over time
- ❌ Shadow mode crashes or produces invalid signals

---

## 🛠️ Troubleshooting

### "No data fetched"
```bash
# Check MT5 connection
python -c "import MetaTrader5 as mt5; print(mt5.initialize())"

# Verify symbol exists
python -c "import MetaTrader5 as mt5; mt5.initialize(); print(mt5.symbol_info('USDJPY'))"
```

### "PyTorch not found"
```bash
# Install PyTorch
pip install torch

# Or use CPU-only version (lighter)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Low accuracy (<60%)
- Increase `--days` to 180 or more (more training data)
- Check feature engineering (ensure no NaN values)
- Verify target calculation (next bar direction)
- Consider different symbol or timeframe

### Shadow mode no signals
- Check probability thresholds (0.7 for BUY, 0.3 for SELL)
- Verify market is open (signals only during trading hours)
- Lower thresholds temporarily for testing (0.6/0.4)

---

## 📝 Next Steps After Training

1. **Review Results**: Check metrics vs targets above
2. **Run Shadow Mode**: Test for 1-2 weeks without trading
3. **Analyze Signals**: Review shadow_mode_log.txt for signal quality
4. **Backtest**: Use historical shadow signals to estimate P/L
5. **Deploy Gradually**: 
   - Start with 1-2 trades/day limit
   - Monitor closely for first week
   - Scale up if performance matches expectations

---

## 🎓 Tips for Best Results

### Training Data
- **Minimum**: 60 days (~17,280 M5 bars)
- **Recommended**: 90 days (~25,920 M5 bars)
- **Optimal**: 180 days (~51,840 M5 bars)
- More data = better LSTM pattern learning

### Shadow Mode Duration
- **Minimum**: 24 hours (verify no crashes)
- **Recommended**: 1-2 weeks (statistical validity)
- **Optimal**: 1 month (covers different market regimes)

### Model Retraining
- **Initial**: Train on 90-180 days
- **Update**: Retrain weekly with latest data
- **Online**: Model self-updates every 100 trades automatically

### Quality Checks
- ✅ Accuracy should improve slightly in prequential evaluation
- ✅ Ablation study shows positive contribution from each component
- ✅ Shadow mode generates reasonable signal frequency (5-15 per day)
- ✅ Probability distribution centered around 0.5 with tails at extremes
