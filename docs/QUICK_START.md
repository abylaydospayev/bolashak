# Quick Start Guide

Get up and running with the trading system in 10 minutes.

## Prerequisites

- Windows 10/11 or Linux
- Python 3.10 or higher
- MetaTrader 5 installed
- Demo trading account (OANDA or any MT5 broker)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/abylaydospayev/bolashak.git
cd bolashak
```

### 2. Set Up Python Environment

Windows:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure MetaTrader 5

1. Open MetaTrader 5
2. Login to your demo account
3. Enable AutoTrading (Ctrl+E or Tools > Options > Expert Advisors > Allow Algorithmic Trading)
4. Add USDJPY.sim symbol to Market Watch

### 4. Configure Environment

```bash
cd live_trading
cp .env.example .env
```

Edit `.env` with your settings:
```bash
MT5_SYMBOL=USDJPY.sim
MT5_LOT_SIZE=0.5
MAX_POSITIONS=5
BUY_THRESHOLD=0.70
```

## Usage

### Option 1: Use Pre-trained Models

The repository includes pre-trained models. Start trading immediately:

```bash
cd live_trading
python demo_bot_with_risk.py
```

### Option 2: Train Your Own Models

1. Collect data:
```bash
python make_dataset.py --symbol USDJPY.sim --timeframe M15 --bars 100000
```

2. Train model:
```bash
python train_rf.py --symbol USDJPY
```

3. Run backtest:
```bash
python backtest.py --symbol USDJPY
```

4. Start trading:
```bash
cd live_trading
python demo_bot_with_risk.py
```

## Verify Installation

Run system tests:
```bash
python test_system.py
```

Expected output:
- All imports successful
- Data files present
- Models loaded correctly

## Monitor Trading

The bot will display real-time information:
```
[Iteration] Price: 153.074, Probability: 0.495, H1 Trend: -1, Positions: 0
```

Press Ctrl+C to stop the bot safely.

## Safety Features

The system includes automatic risk management:

- Maximum 5 concurrent positions
- 30 pip stop loss on every trade
- 50 pip take profit target
- Daily loss limit: $4,948
- Minimum 3 minutes between trades

## Troubleshooting

### MT5 Connection Failed

1. Check MetaTrader 5 is running
2. Verify AutoTrading is enabled
3. Ensure correct symbol name (USDJPY.sim vs USDJPY)

### No Trades Executing

1. Check BUY_THRESHOLD (0.70 recommended)
2. Verify AutoTrading is enabled in MT5
3. Check daily loss limit not reached
4. Verify minimum interval (180 seconds) passed

### Model Not Found

1. Check models/ directory has .pkl files
2. Run training scripts if needed
3. Verify MODEL_PATH in .env

## Performance Expectations

With USDJPY at threshold 0.70:

- Win rate: ~54%
- Trades per day: 8-10
- Average trade duration: 2-4 hours
- Drawdown: <5% (with risk management)

## Next Steps

- Read full documentation in docs/
- Review Monte Carlo analysis results
- Customize risk parameters
- Deploy to Azure VM for 24/7 operation

## Getting Help

- Check docs/DEPLOYMENT_GUIDE.md for detailed instructions
- Review docs/USDJPY_VS_EURUSD_COMPARISON.md for model analysis
- Open an issue on GitHub for bugs or questions

## Important Reminders

1. Always test on demo account first
2. Never trade with money you can't afford to lose
3. Monitor positions regularly
4. Understand the risks before live trading
5. Past performance does not guarantee future results
