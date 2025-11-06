
# FX ML Starter (EURUSD & USDJPY, MT5)

Minimal scaffold to:
1) Pull data from MT5.
2) Build features/labels (RSI, EMA, ATR; horizon h=3).
3) Train a simple RandomForest predictor.
4) Backtest with ATR SL/TP and fees/slippage.
5) Run a lightweight live signal loop on MT5 with bracket orders.

> Start simple (RF); later swap in LSTM with the same feature pipeline.

## Quickstart

```bash
# 0) Create venv and install (Windows PowerShell)
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

1) **Export data** (pull from MT5 and save CSVs):
```bash
python make_dataset.py --symbol EURUSD --timeframe M15 --bars 50000
python make_dataset.py --symbol USDJPY --timeframe M15 --bars 50000
```

2) **Build features + labels**:
```bash
python build_features.py --symbol EURUSD --h 3
python build_features.py --symbol USDJPY --h 3
```

3) **Train RandomForest model**:
```bash
python train_rf.py --symbol EURUSD
python train_rf.py --symbol USDJPY
```

4) **Backtest**:
```bash
python backtest.py --symbol EURUSD
python backtest.py --symbol USDJPY
```

5) **Go live (paper/demo)**:
```bash
python signal_mt5.py --symbol EURUSD
# Run in a separate terminal for USDJPY if desired
```

## File overview
- `indicators.py` — EMA, RSI, ATR (pure numpy/pandas; no extra deps).
- `make_dataset.py` — Fetches rates from MT5 via `MetaTrader5` package.
- `build_features.py` — Adds indicators, time features, labels (with costs).
- `train_rf.py` — Fits a RandomForest; saves `models/<symbol>_rf.pkl` and `scaler.pkl`.
- `backtest.py` — Simple bracket-order backtest with SL/TP and slippage.
- `signal_mt5.py` — Live loop: computes latest features → predicts → places bracket order in MT5.
- `config.yaml` — Tunables (risk, thresholds, costs, SL/TP multiples, etc.).

> Note: Live trading is risky. Use demo first. Double‑check broker rules, lot sizes, and symbol suffixes (e.g., `EURUSD.a`).

## Next steps (optional)
- Add walk-forward cross-validation in backtest.
- Add confidence-weighted sizing.
- Swap `train_rf.py` with an LSTM (Keras) using 60-bar windows.
