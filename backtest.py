
import argparse, os, yaml, joblib
import pandas as pd
import numpy as np
from position_sizing import PositionSizer

def calculate_trade_costs(entry_price, pip_size, cfg, symbol=''):
    """Calculate total trading costs for a round-turn trade (entry + exit).
    
    Returns cost in dollars for 1 standard lot (100k units).
    
    For EURUSD: 1 pip = $0.0001 price = $10/lot (100k * 0.0001)
    For USDJPY: 1 pip = $0.01 price = $10/lot at 100 rate (100k * 0.01 / rate)
    """
    # Spread: paid on both entry and exit
    spread_pips = cfg['spread_pips'] * 2  # entry + exit
    # Slippage: assumed on both entry and exit
    slippage_pips = cfg['slippage_pips'] * 2
    total_pips = spread_pips + slippage_pips
    
    # Calculate pip value in dollars per standard lot
    if 'JPY' in symbol.upper():
        # For JPY pairs: pip value = (pip_size / exchange_rate) * 100000
        # At 150 USDJPY: (0.01 / 150) * 100000 = $6.67 per pip
        pip_value_usd = (pip_size / entry_price) * 100000 if entry_price > 0 else 10.0
    else:
        # For EUR/USD etc: pip value = pip_size * 100000
        # 0.0001 * 100000 = $10 per pip
        pip_value_usd = pip_size * 100000
    
    cost_from_pips = total_pips * pip_value_usd
    
    # Commission (already in dollars per lot)
    commission = cfg.get('commission_per_lot', 7.0)
    
    return cost_from_pips + commission

def backtest(df: pd.DataFrame, proba: np.ndarray, cfg, use_position_sizing=True):
    prob_buy = cfg['prob_buy']
    prob_sell = cfg['prob_sell']
    stop_k = cfg['stop_atr_mult']
    tp_k = cfg['tp_atr_mult']

    cash = 100000.0  # Increased from 10k to 100k for realistic forex account
    equity = cash
    pos = 0  # +1 long, -1 short, 0 flat
    entry_price = None
    peak_equity = equity
    max_dd = 0.0
    trades = []

    prices = df['close'].values
    atr = df['atr14'].values
    times = df['time'].values
    
    # Determine pip size based on symbol
    symbol = df.attrs.get('symbol', '')
    pip_values = {
        'USDJPY': 0.01,
        'EURJPY': 0.01,
        'GBPJPY': 0.01,
        'EURUSD': 0.0001,
        'GBPUSD': 0.0001,
        'AUDUSD': 0.0001,
        'NZDUSD': 0.0001,
        'USDCAD': 0.0001,
        'USDCHF': 0.0001
    }
    
    # Extract base symbol (remove .sim suffix if present)
    base_symbol = symbol.replace('.sim', '').upper()
    pip = pip_values.get(base_symbol, 0.01)  # Default to 0.01 for JPY pairs

    entry_bar = None
    position_size = 1.0  # Default to 1 lot if position sizing disabled
    
    # Initialize position sizer
    position_sizer = PositionSizer(strategy='fixed_fractional', risk_pct=0.01, max_risk_pct=0.02)
    
    for i in range(len(df)-1):
        p = proba[i]
        price = prices[i+1]  # next bar open assumption is close[i+1] for simplicity here
        bar_atr = atr[i]

        # exit logic if in position
        if pos != 0 and entry_price is not None and entry_bar is not None:
            stop = entry_price - stop_k*bar_atr if pos>0 else entry_price + stop_k*bar_atr
            tp   = entry_price + tp_k*bar_atr if pos>0 else entry_price - tp_k*bar_atr
            
            # Check using high/low for stop/tp hits
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            
            pnl = 0.0
            exit_now = False
            exit_price = prices[i]
            
            # Check if stop or TP hit
            if pos > 0:
                if low <= stop:
                    exit_now = True
                    exit_price = stop
                    pnl = stop - entry_price
                elif high >= tp:
                    exit_now = True
                    exit_price = tp
                    pnl = tp - entry_price
            else:  # pos < 0
                if high >= stop:
                    exit_now = True
                    exit_price = stop
                    pnl = entry_price - stop
                elif low <= tp:
                    exit_now = True
                    exit_price = tp
                    pnl = entry_price - tp
            
            # Time-based exit after horizon bars
            if not exit_now and (i - entry_bar) >= 3:  # horizon = 3 bars
                exit_now = True
                exit_price = prices[i]
                pnl = (exit_price - entry_price) if pos > 0 else (entry_price - exit_price)

            if exit_now:
                # Calculate gross PnL (scaled by position size)
                pnl_gross = pnl * 100000 * position_size  # position_size lots
                
                # Deduct trading costs (scaled by position size)
                costs = calculate_trade_costs(entry_price, pip, cfg, df.attrs.get('symbol', '')) * position_size
                pnl_net = pnl_gross - costs
                
                equity += pnl_net
                trades.append(pnl / abs(entry_price) if entry_price else pnl)  # Store as % return for metrics
                pos = 0
                entry_price = None
                entry_bar = None
                position_size = 1.0  # Reset
                peak_equity = max(peak_equity, equity)
                if peak_equity > 0:
                    max_dd = max(max_dd, (peak_equity - equity)/peak_equity)

        # entry if flat
        if pos == 0:
            if p >= prob_buy:
                pos = +1
                entry_price = price
                entry_bar = i
                
                # Calculate position size based on stop loss distance
                if use_position_sizing:
                    stop_loss_pips = stop_k * bar_atr / pip
                    
                    # Calculate pip value in USD per standard lot
                    if 'JPY' in base_symbol:
                        pip_value_usd = (pip / price) * 100000 if price > 0 else 10.0
                    else:
                        pip_value_usd = pip * 100000
                    
                    position_size = position_sizer.calculate_size(
                        equity=equity,
                        stop_loss_pips=stop_loss_pips,
                        pip_value=pip_value_usd
                    )
                else:
                    position_size = 1.0
                    
            elif p <= prob_sell:
                pos = -1
                entry_price = price
                entry_bar = i
                
                # Calculate position size based on stop loss distance
                if use_position_sizing:
                    stop_loss_pips = stop_k * bar_atr / pip
                    
                    # Calculate pip value in USD per standard lot
                    if 'JPY' in base_symbol:
                        pip_value_usd = (pip / price) * 100000 if price > 0 else 10.0
                    else:
                        pip_value_usd = pip * 100000
                    
                    position_size = position_sizer.calculate_size(
                        equity=equity,
                        stop_loss_pips=stop_loss_pips,
                        pip_value=pip_value_usd
                    )
                else:
                    position_size = 1.0

    # Metrics
    if trades:
        wins = [t for t in trades if t>0]
        losses = [t for t in trades if t<=0]
        winrate = len(wins)/len(trades)
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        pf = (sum(wins)/abs(sum(losses))) if losses else np.inf
    else:
        winrate=avg_win=avg_loss=pf=0.0

    return {
        'trades': len(trades),
        'winrate': winrate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': pf,
        'max_dd': max_dd,
        'equity': equity
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open('config.yaml'))

    feat_path = os.path.join(cfg['feature_dir'], f"{args.symbol}_features.csv")
    df = pd.read_csv(feat_path, parse_dates=['time'])
    df.attrs['symbol'] = args.symbol

    feature_cols = [c for c in df.columns if c not in ('time','y')]
    from sklearn.preprocessing import StandardScaler
    scaler = joblib.load(os.path.join(cfg['model_dir'], "scaler.pkl"))
    model = joblib.load(os.path.join(cfg['model_dir'], f"{args.symbol}_rf.pkl"))
    Xs = scaler.transform(df[feature_cols].values)
    proba = model.predict_proba(Xs)[:,1]

    report = backtest(df, proba, cfg)
    print(f"Backtest {args.symbol}: {report}")

if __name__ == '__main__':
    main()
