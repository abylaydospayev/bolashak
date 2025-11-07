# How to Test and Train the Adaptive Model

## 📋 Summary

You have **3 ways** to test and train the adaptive model:

---

## 1. **Quick Start with Real Data** (RECOMMENDED)

### Train the model on historical USDJPY data:

```bash
# Full test suite (90 days of data)
python live_trading/test_adaptive_model.py --mode full --days 90
```

**What you'll get:**
- ✅ Model trained on ~26,000 M5 bars
- ✅ Test accuracy, precision, AUC metrics
- ✅ Prequential evaluation (test-then-learn)
- ✅ Ablation study showing contribution of each component
- ✅ Saved model file: `adaptive_model_TIMESTAMP.pkl`
- ✅ Results JSON: `adaptive_model_results_TIMESTAMP.json`

**Expected results:**
```
TRAINING RESULTS
--------------------------------------------------------------
Accuracy:              0.6250  (✓ GOOD)
Precision:             0.6180  (✓ GOOD)
AUC:                   0.6720  (✓ GOOD)

High Confidence Signals:
  BUY (p >= 0.7):      0.7240  (892 signals)
  SELL (p <= 0.3):     0.7180  (745 signals)

ABLATION STUDY
--------------------------------------------------------------
Baseline (RF+GB):      Acc=0.6150, AUC=0.6580
+LSTM:                 Acc=0.6280, AUC=0.6710 (+0.0130)
+Online Learning:      Acc=0.6410, AUC=0.6850 (+0.0130)
+Meta-Ensemble:        Acc=0.6520, AUC=0.6950 (+0.0110)

Total Improvement:     +0.0370 accuracy, +0.0370 AUC
```

---

## 2. **Shadow Mode Testing** (Test on Live Data WITHOUT Trading)

### Test the model on live market data without placing real trades:

```bash
# Run for 2 hours (120 minutes)
python live_trading/test_adaptive_model.py --mode shadow --shadow-duration 120
```

**What it does:**
- Monitors live USDJPY market
- Generates BUY/SELL signals based on model predictions
- Logs signals WITHOUT placing trades
- Saves to `shadow_mode_log.txt`

**Example output:**
```
SHADOW MODE TEST
--------------------------------------------------------------
Duration: 120 minutes
Will log signals WITHOUT placing trades

[SHADOW 14:30:15] Signal #1: BUY @ 149.245 | Probability: 0.723
[SHADOW 14:42:30] Signal #2: SELL @ 149.189 | Probability: 0.285
[SHADOW 15:01:45] Signal #3: BUY @ 149.312 | Probability: 0.741

[SHADOW] Completed at 16:30:15
[SHADOW] Total signals generated: 12
[SHADOW] Log saved to: shadow_mode_log.txt
```

**Analyze shadow results:**
```python
import pandas as pd

df = pd.read_csv('shadow_mode_log.txt', names=['time', 'signal', 'price', 'probability'])
print(f"Total signals: {len(df)}")
print(f"BUY: {len(df[df['signal']=='BUY'])}, SELL: {len(df[df['signal']=='SELL'])}")
print(f"Avg probability: {df['probability'].mean():.3f}")
```

---

## 3. **Deploy to Production**

### After shadow mode looks good, deploy to Azure VM:

```bash
# On your LOCAL PC:
# 1. Push trained model to GitHub
git add adaptive_model_*.pkl
git commit -m "feat: Add trained adaptive model"
git push origin main

# 2. RDP to Azure VM
mstsc /v:20.9.129.81

# On AZURE VM:
# 3. Pull latest code
cd C:\Users\abylay_dos\Desktop\bolashak
git pull origin main

# 4. Activate Python environment
.\.venv\Scripts\Activate.ps1

# 5. Install PyTorch (if using LSTM)
pip install torch

# 6. Update .env with adaptive model settings
notepad .env
```

**Add to .env:**
```bash
# Adaptive Model Settings
ADAPTIVE_MODEL_ENABLED=true
ADAPTIVE_MODEL_PATH=adaptive_model_20251107_143022.pkl
```

**Start the bot:**
```bash
python live_trading/demo_bot_with_risk.py
```

---

## 📊 Understanding the Results

### Good Results (Ready for Production)
- ✅ **Accuracy: 62-68%** - Model is correct 6-7 times out of 10
- ✅ **AUC: 0.67-0.75** - Good ranking of predictions
- ✅ **Precision @ 0.7: >70%** - High confidence BUYs are accurate
- ✅ **Ablation: +3-5%** - Each component adds value

### Warning Signs (Need More Data/Tuning)
- ⚠️ **Accuracy: <60%** - Below target
- ⚠️ **AUC: <0.65** - Weak predictions
- ⚠️ **Precision @ 0.7: <65%** - High confidence signals unreliable
- ⚠️ **Ablation: <2%** - Components not helping

### Do NOT Deploy If:
- ❌ **Accuracy: <55%** - Worse than random
- ❌ **Training crashes** - Model unstable
- ❌ **Prequential AUC declining** - Overfitting to recent data
- ❌ **Shadow mode crashes** - Not production-ready

---

## 🔧 Quick Fixes

### Problem: "No data fetched"
```bash
# Solution: Ensure MT5 is logged in
# Check MT5 terminal on your PC, verify OANDA-Demo-1 is connected
```

### Problem: "PyTorch not available"
```bash
# Solution: Install PyTorch
pip install torch

# Or CPU-only version (lighter, faster install)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Problem: Low accuracy (<60%)
```bash
# Solution: Use more training data
python live_trading/test_adaptive_model.py --mode full --days 180
```

### Problem: No shadow mode signals
```bash
# Solution: Market might be closed or low volatility
# Check MT5 terminal - verify live quotes are updating
# Try running during London/NY session overlap (13:00-17:00 UTC)
```

---

## 🎯 Recommended Workflow

### Week 1: Initial Training
```bash
# Day 1: Train model on 90 days
python live_trading/test_adaptive_model.py --mode full --days 90

# Review results - should have Acc>60%, AUC>0.65
# If not, try 180 days: --days 180
```

### Week 2-3: Shadow Mode Testing
```bash
# Run shadow mode for 1-2 weeks
python live_trading/test_adaptive_model.py --mode shadow --shadow-duration 10080

# Analyze results after 1 week:
# - Count signals/day (expect 5-15)
# - Check probability distribution
# - Verify no crashes/errors
```

### Week 4: Production Deployment
```bash
# If shadow mode looks good:
# 1. Deploy to Azure VM
# 2. Set ADAPTIVE_MODEL_ENABLED=true
# 3. Monitor closely for first 3-5 days
# 4. Check logs daily for unusual behavior
```

---

## 📈 Performance Monitoring

### Daily Checks (First Week)
- Check bot logs for errors
- Verify signals match shadow mode frequency
- Monitor P/L (should improve gradually)
- Check prequential AUC (should be stable)

### Weekly Checks
- Compare win rate vs historical
- Check calibration (predicted probability vs actual outcomes)
- Review model confidence distribution
- Retrain if accuracy drops >5%

### Monthly Maintenance
- Retrain model on latest 90 days
- Run ablation study to verify components still helping
- Update to latest market regime data
- Review and optimize hyperparameters if needed

---

## ✅ Success Criteria

Your model is ready for production when:

1. ✅ **Training accuracy: 62-68%**
2. ✅ **Test AUC: 0.67-0.75**
3. ✅ **Precision @ 0.7: >70%**
4. ✅ **Shadow mode: 1-2 weeks without crashes**
5. ✅ **Signal frequency: 5-15 per day**
6. ✅ **Prequential AUC: Stable or improving**
7. ✅ **Ablation: Each component adds 1-2%**

Once all criteria met → **Deploy to production!** 🚀
