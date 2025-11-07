# ATR-Based Volatility Management

## 🎯 What Changed

Your bot now uses **professional adaptive risk management** based on ATR (Average True Range):

### Key Features

✅ **Dynamic Stop Loss/Take Profit**
- Calculated from ATR(14) on H1 timeframe
- SL range: 20–80 pips (widens in volatility)
- TP = SL × 1.6 (maintains consistent R:R)

✅ **Fixed % Risk Per Trade**
- Always risk 0.5% of account balance
- Lot size automatically adjusts
- Smaller positions in volatile markets

✅ **Extreme Volatility Protection**
- Skip trades if ATR > 2.5× baseline (75+ pips)
- Halve position size if ATR > 2.0× (60+ pips)

✅ **FTMO-Safe Limits**
- 3% daily soft stop (pause trading)
- $4,948 daily hard stop (5% of $100k)
- Max 3.0 lots total exposure
- 15-minute minimum trade spacing

---

## 📊 How It Works

### Volatility Regimes (Example: $100k account)

| Market | ATR | vol_factor | SL | TP | Lot Size |
|--------|-----|------------|----|----|----------|
| **Calm** | 25 pips | 0.83× | 25 pips | 40 pips | ~0.54 lots |
| **Normal** | 30 pips | 1.0× | 30 pips | 48 pips | ~0.54 lots |
| **Active** | 45 pips | 1.5× | 45 pips | 72 pips | ~0.36 lots |
| **High Vol** | 60 pips | 2.0× | 60 pips | 96 pips | ~0.27 lots (halved) |
| **Extreme** | 80+ pips | 2.5×+ | — | — | **Skip trade** |

### Formula

```python
# 1) Measure volatility
vol_factor = ATR_H1 / 30  # baseline = 30 pips

# 2) Calculate SL/TP
SL_pips = clamp(30 × vol_factor, 20, 80)
TP_pips = SL_pips × 1.6

# 3) Position size from fixed risk
risk_$ = balance × 0.5%  # $100k × 0.5% = $500
lot_size = risk_$ / (SL_pips × $9.17/pip/lot)
lot_size = clamp(lot_size, 0.10, 1.50)

# 4) Extreme vol protection
if vol_factor ≥ 2.5: skip trade
if vol_factor ≥ 2.0: lot_size × 0.5
```

---

## 🚀 Deploy to VM

### 1. Pull latest code:
```powershell
cd C:\trading\bolashak
git pull origin main
```

### 2. Update `.env`:
```bash
# Add these ATR settings:
ATR_VOLATILITY_ADJUSTMENT=true
ATR_PERIOD=14
ATR_BASELINE_PIPS=30
MIN_SL_PIPS=20
MAX_SL_PIPS=80
RISK_REWARD_RATIO=1.6
RISK_PERCENT_PER_TRADE=0.5
PIP_VALUE_PER_LOT=9.17
MIN_LOT_SIZE=0.10
MAX_LOT_SIZE=1.50
MAX_TOTAL_LOTS=3.0
EXTREME_VOL_FACTOR=2.5
HIGH_VOL_FACTOR=2.0
MAX_POSITIONS=3
MIN_INTERVAL_SECONDS=900
MAX_DAILY_LOSS=4948
DAILY_LOSS_SOFT_STOP_PCT=3.0
```

### 3. Update main.py to use new API:
```python
# Before placing trade:
result = risk_manager.adjust_for_volatility(account_balance, symbol='USDJPY')

if result is None:
    print("🚨 Extreme volatility - skipping trade")
    continue

lot_size, sl_pips, tp_pips, atr_pips, vol_factor = result

# Use these values for the trade
```

### 4. Restart bot:
```powershell
.\stop_bot.ps1
.\start_bot.ps1
.\view_logs.ps1
```

---

## 📈 Expected Behavior

**Normal Market (ATR ≈ 30 pips):**
```
📊 Volatility Analysis:
   Status: 🟡 NORMAL | ATR: 30.5 pips (factor: 1.02x)
   🛑 Stop Loss: 31 pips
   🎯 Take Profit: 50 pips (R:R = 1:1.61)
   💼 Position Size: 0.54 lots (risk: $500.00 = 0.5%)
```

**High Volatility (ATR ≈ 60 pips):**
```
⚠️ HIGH VOLATILITY: ATR 62.3 pips (2.08x) - Halving position size
📊 Volatility Analysis:
   Status: 🔴 HIGH | ATR: 62.3 pips (factor: 2.08x)
   🛑 Stop Loss: 62 pips
   🎯 Take Profit: 99 pips (R:R = 1:1.60)
   💼 Position Size: 0.13 lots (risk: $250.00 = 0.25%)
```

**Extreme Volatility (ATR > 75 pips):**
```
🚨 EXTREME VOLATILITY: ATR 78.4 pips (2.61x baseline)
   Skipping new trades until volatility normalizes
```

---

## ⚠️ About H1: -1

Your logs show `H1: -1` which means your H1 trend indicator is showing SELL signal.

Currently you have **conflicting signals**:
- Model probability: 0.65–0.68 (predicts BUY)
- H1 trend: -1 (indicates SELL)

### Recommendation:

Either:
1. **Ignore H1** - trade only on model predictions
2. **Require agreement** - only trade when both agree
3. **Use H1 as filter** - no BUY on H1=-1, no SELL on H1=+1

Which approach do you prefer?

---

## 🎓 Benefits

✅ **Consistent risk** - always 0.5% per trade regardless of volatility  
✅ **Adaptive sizing** - smaller lots in wild markets  
✅ **Wider stops** - avoid premature exits during volatility spikes  
✅ **Automatic protection** - skips extreme conditions  
✅ **FTMO-compliant** - hard caps on daily loss and exposure  
✅ **Professional-grade** - used by institutional traders  

Your bot now trades like a professional risk manager! 🚀
