# MetaTrader 5 Live Trading Setup

## Prerequisites

1. **MetaTrader 5 Terminal** installed on Windows
   - Download from: https://www.metatrader5.com/en/download
   - Default path: `C:\Program Files\MetaTrader 5\terminal64.exe`

2. **MT5 Demo or Live Account** credentials
   - Account number (login)
   - Password
   - Broker server name

3. **Python packages** (already installed if you ran the setup):
   ```powershell
   pip install MetaTrader5 python-dotenv
   ```

---

## Quick Start

### 1. Create `.env` file

Copy `.env.example` to `.env`:
```powershell
Copy-Item live_trading\.env.example live_trading\.env
```

Edit `live_trading\.env` with your credentials:
```ini
MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_LOGIN=1600037272
MT5_PASSWORD=your_actual_password
MT5_SERVER=OANDA-Demo-1
MT5_SYMBOL=USDJPY
MT5_LOT_SIZE=0.1
```

**⚠️ IMPORTANT:** Add `.env` to `.gitignore` to prevent committing credentials!

### 2. Test Connection

```powershell
.\.venv\Scripts\python.exe live_trading\test_mt5_connection.py
```

Expected output:
```
✅ MT5 initialized
Fetching 100 M15 bars for USDJPY...
[DataFrame with OHLC data]
Account info: [balance, equity, etc.]
✅ MT5 shutdown
```

### 3. Run Demo Trading Bot

```powershell
.\.venv\Scripts\python.exe live_trading\demo_bot.py
```

This will:
- Load your validated USDJPY ensemble model
- Connect to MT5
- Monitor M15 bars in real-time
- Generate trading signals
- Execute trades automatically with proper SL/TP
- Log all activity to `live_trading/logs/`

---

## Alternative: Environment Variables (PowerShell)

If you don't want to use `.env` file:

```powershell
$env:MT5_TERMINAL_PATH = "C:\Program Files\MetaTrader 5\terminal64.exe"
$env:MT5_LOGIN = "1600037272"
$env:MT5_PASSWORD = "your_password"
$env:MT5_SERVER = "OANDA-Demo-1"
```

Then run scripts normally.

---

## Risk Management Settings

Default guardrails (configured in `demo_bot.py`):
- **Risk per trade:** 1.0% of equity
- **Max open positions:** 3
- **Max daily loss:** 3.0% (trading halts)
- **Max drawdown:** 10.0% (emergency stop)
- **Min model confidence:** 0.80 (prob_buy/sell threshold)

To modify, edit `live_trading/demo_bot.py` line ~40.

---

## File Structure

```
live_trading/
├── .env.example          # Template for credentials
├── .env                  # Your actual credentials (git-ignored)
├── mt5_client.py         # MT5 API wrapper
├── demo_bot.py           # USDJPY trading bot
├── test_mt5_connection.py # Connection test
├── monitor.py            # Real-time monitoring dashboard
└── logs/                 # Trading logs (auto-created)
```

---

## Monitoring

Real-time dashboard:
```powershell
.\.venv\Scripts\python.exe live_trading\monitor.py
```

Shows:
- Current equity and P&L
- Open positions
- Recent trades
- Win rate and drawdown
- Model predictions

---

## Troubleshooting

### "Failed to initialize MT5"
- Ensure MT5 terminal is installed
- Check terminal path in `.env`
- Try launching MT5 terminal manually first
- Verify login/password/server are correct

### "Symbol USDJPY not found"
- Open MT5 terminal
- Go to Market Watch (Ctrl+M)
- Right-click → Symbols → Find "USDJPY" → Show

### "Order failed, retcode=XXX"
- Check account has sufficient margin
- Verify symbol is tradable (not closed market)
- Check lot size is within broker limits (min 0.01)

### "Connection timeout"
- Ensure internet connection is stable
- Check MT5 terminal is connected (bottom-right shows connection)
- Try different broker server if available

---

## Safety Checklist

Before going live:

- [ ] Tested on demo account for 4-8 weeks
- [ ] Live results match backtest within ±5%
- [ ] Win rate > 75% on demo
- [ ] Max DD < 10% on demo
- [ ] No system crashes or missed trades
- [ ] Slippage < 1 pip average
- [ ] Reviewed all logs and trades
- [ ] Started with small capital ($10-20k)
- [ ] Using conservative risk (0.5% per trade initially)

---

## Expected Performance (USDJPY Demo)

Based on validation results:
- **Monthly ROI:** 20-25% (on $100k account)
- **Win Rate:** 77-80%
- **Max Drawdown:** 6-8%
- **Trades/Month:** ~350
- **Monthly Profit:** $20-25k (with $100k capital)

**Note:** Live trading will have higher variance. Start small!

---

## Support

- Check logs in `live_trading/logs/` for detailed error messages
- MT5 API docs: https://www.mql5.com/en/docs/python_metatrader5
- For issues, review the validation reports in `validation/`

---

**Ready to start!** Run the test connection first, then launch the demo bot. 🚀
