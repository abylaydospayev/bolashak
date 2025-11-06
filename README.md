# Algorithmic Trading System - USDJPY & EURUSD

# FX ML Starter (EURUSD & USDJPY, MT5)

Machine learning-based trading system for forex markets using MetaTrader 5. Features ensemble models, multi-timeframe analysis, comprehensive risk management, and Monte Carlo validation.

Minimal scaffold to:

## Overview1) Pull data from MT5.

2) Build features/labels (RSI, EMA, ATR; horizon h=3).

This project implements a production-ready algorithmic trading system with:3) Train a simple RandomForest predictor.

4) Backtest with ATR SL/TP and fees/slippage.

- **Ensemble ML Models**: Random Forest, Lorentzian Classification, LSTM networks5) Run a lightweight live signal loop on MT5 with bracket orders.

- **Multi-Timeframe Analysis**: M15, M30, H1, H4 aggregated features

- **Professional Risk Management**: Position limits, stop-loss/take-profit, daily loss limits> Start simple (RF); later swap in LSTM with the same feature pipeline.

- **Monte Carlo Validation**: Statistical validation with 100+ random train/test splits

- **Live Trading**: Automated execution on MetaTrader 5 with Azure VM deployment## Quickstart



## Performance Summary```bash

# 0) Create venv and install (Windows PowerShell)

### USDJPY (Production Model)python -m venv .venv

- **Accuracy**: 61.5% (95% CI: 59.9%, 63.3%). .venv/Scripts/Activate.ps1

- **ROC AUC**: 68.1% (95% CI: 66.2%, 70.3%)pip install -r requirements.txt

- **Win Rate @ Threshold 0.7**: 54.2% (1,658 trades)```

- **Win Rate @ Threshold 0.8**: 85.8% (232 trades)

- **Status**: Currently deployed on Azure VM1) **Export data** (pull from MT5 and save CSVs):

```bash

### EURUSD (Experimental Model)python make_dataset.py --symbol EURUSD --timeframe M15 --bars 50000

- **Accuracy**: 14.5% (requires retraining)python make_dataset.py --symbol USDJPY --timeframe M15 --bars 50000

- **Win Rate @ Threshold 0.8**: 61.5% (13 trades only)```

- **Status**: Paper trading only, not production-ready

2) **Build features + labels**:

## Project Structure```bash

python build_features.py --symbol EURUSD --h 3

```python build_features.py --symbol USDJPY --h 3

bolashak/```

├── models/                  # Trained ML models (.pkl files)

├── data/                    # Historical price data (CSV)3) **Train RandomForest model**:

├── results/                 # Backtest results and analysis charts```bash

├── live_trading/           # Production trading scriptspython train_rf.py --symbol EURUSD

│   ├── demo_bot_with_risk.py   # Main trading botpython train_rf.py --symbol USDJPY

│   ├── risk_manager.py         # Risk management system```

│   └── .env                    # Configuration (not in repo)

├── tests/                   # Unit tests4) **Backtest**:

├── validation/             # Validation reports```bash

├── make_dataset.py         # Data collection from MT5python backtest.py --symbol EURUSD

├── build_features_enhanced.py  # Feature engineeringpython backtest.py --symbol USDJPY

├── train_rf.py             # Random Forest training```

├── backtest.py             # Strategy backtesting

├── test_usdjpy_monte_carlo.py  # Monte Carlo simulation5) **Go live (paper/demo)**:

├── analyze_usdjpy_predictions.py  # Model analysis```bash

└── requirements.txt        # Python dependenciespython signal_mt5.py --symbol EURUSD

```# Run in a separate terminal for USDJPY if desired

```

## Installation

## File overview

### Prerequisites- `indicators.py` — EMA, RSI, ATR (pure numpy/pandas; no extra deps).

- Python 3.10+- `make_dataset.py` — Fetches rates from MT5 via `MetaTrader5` package.

- MetaTrader 5 terminal- `build_features.py` — Adds indicators, time features, labels (with costs).

- Trading account (demo or live)- `train_rf.py` — Fits a RandomForest; saves `models/<symbol>_rf.pkl` and `scaler.pkl`.

- `backtest.py` — Simple bracket-order backtest with SL/TP and slippage.

### Setup- `signal_mt5.py` — Live loop: computes latest features → predicts → places bracket order in MT5.

- `config.yaml` — Tunables (risk, thresholds, costs, SL/TP multiples, etc.).

1. Clone the repository:

```bash> Note: Live trading is risky. Use demo first. Double‑check broker rules, lot sizes, and symbol suffixes (e.g., `EURUSD.a`).

git clone https://github.com/abylaydospayev/bolashak.git

cd bolashak## Next steps (optional)

```- Add walk-forward cross-validation in backtest.

- Add confidence-weighted sizing.

2. Create virtual environment:- Swap `train_rf.py` with an LSTM (Keras) using 60-bar windows.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure settings:
```bash
cp live_trading/.env.example live_trading/.env
# Edit .env with your MT5 credentials and risk parameters
```

## Usage

### Data Collection

Collect historical data from MetaTrader 5:

```bash
python make_dataset.py --symbol USDJPY.sim --timeframe M15 --bars 100000
python make_dataset.py --symbol EURUSD.sim --timeframe M15 --bars 100000
```

### Model Training

Train ensemble models:

```bash
python train_rf.py --symbol USDJPY
python train_rf.py --symbol EURUSD
```

### Backtesting

Test strategy performance:

```bash
python backtest.py --symbol USDJPY
python backtest.py --symbol EURUSD
```

### Monte Carlo Validation

Run statistical validation:

```bash
python test_usdjpy_monte_carlo.py
python analyze_usdjpy_predictions.py
```

### Live Trading

Run the trading bot (demo account recommended):

```bash
cd live_trading
python demo_bot_with_risk.py
```

## Configuration

Key parameters in `live_trading/.env`:

```bash
# Trading Parameters
MT5_SYMBOL=USDJPY.sim
MT5_LOT_SIZE=0.5

# Risk Management
MAX_POSITIONS=5              # Maximum concurrent positions
MIN_INTERVAL_SECONDS=180     # Cooldown between trades
STOP_LOSS_PIPS=30           # Stop loss distance
TAKE_PROFIT_PIPS=50         # Take profit distance
MAX_DAILY_LOSS=4948         # Daily loss limit (5% of $100k)

# Signal Thresholds
BUY_THRESHOLD=0.70          # Minimum probability for BUY
SELL_THRESHOLD=0.30         # Maximum probability for SELL
```

## Risk Management Features

- **Position Limits**: Prevents overexposure (max 5 concurrent positions)
- **Trade Cooldown**: Minimum 180 seconds between trades
- **Automatic Stop Loss/Take Profit**: 30 pips SL, 50 pips TP
- **Daily Loss Limit**: Circuit breaker at $4,948 loss (FTMO-compliant)
- **Signal Strength Validation**: Only trades high-confidence signals

## Monte Carlo Analysis

Comprehensive validation with 100 random train/test splits:

- **USDJPY**: 61.5% accuracy, well-calibrated predictions
- **EURUSD**: Severe UP bias, requires retraining
- **Threshold Optimization**: Optimal at 0.7 (54.2% win rate) or 0.8 (85.8% win rate)

See `docs/USDJPY_VS_EURUSD_COMPARISON.md` for detailed analysis.

## Deployment

### Azure VM Deployment

The system is designed for 24/7 operation on Azure Virtual Machines:

1. Provision Windows Server VM (Standard_B2ats_v2 or higher)
2. Install MetaTrader 5 and Python
3. Clone repository and configure
4. Set up Windows Task Scheduler for automatic startup

See `docs/DEPLOYMENT_GUIDE.md` for detailed instructions.

## Testing

Run unit tests:

```bash
pytest tests/
```

Run integration tests:

```bash
python tests/test_eurusd_model.py
```

## Documentation

- `docs/DEPLOYMENT_GUIDE.md` - Complete deployment instructions
- `docs/USDJPY_VS_EURUSD_COMPARISON.md` - Model comparison analysis
- `docs/EURUSD_MONTE_CARLO_REPORT.md` - EURUSD validation results

## Contributing

This is a personal trading project. Please open an issue for bug reports or feature suggestions.

## Disclaimer

**This software is for educational purposes only.** Trading forex carries substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

- Never trade with money you cannot afford to lose
- Always test on demo accounts first
- Understand the risks before deploying to live trading
- Monitor positions and system health regularly

## License

MIT License - See LICENSE file for details

## Author

Abylay Dospayev
- GitHub: [@abylaydospayev](https://github.com/abylaydospayev)

## Acknowledgments

- MetaTrader 5 for trading platform
- Scikit-learn for machine learning frameworks
- Azure for cloud infrastructure
