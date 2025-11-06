
import argparse, time, os, yaml, joblib, sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import MetaTrader5 as mt5
from indicators import ema, rsi, atr, sincos_time

def timeframe_str_to_mt5(tf_str: str):
    mapping = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }
    return mapping[tf_str]

def build_latest_features(symbol, tf_str, bars=200):
    tf = timeframe_str_to_mt5(tf_str)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.rename(columns={'tick_volume':'volume'})
    df = df[['time','open','high','low','close','volume']].dropna()
    # indicators
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['rsi14'] = rsi(df['close'], 14)
    df['atr14'] = atr(df, 14)
    df['ema50_slope'] = df['ema50'].diff(5)
    sin_h, cos_h = sincos_time(df['time'])
    df['sin_hour'] = sin_h.values
    df['cos_hour'] = cos_h.values
    df['ret1'] = df['close'].pct_change(1)
    df['atr_pct'] = df['atr14'] / df['close']
    return df.dropna()

def place_bracket(symbol, lot, direction, stop_price, tp_price, deviation_points, magic):
    # direction: +1 buy, -1 sell
    price = mt5.symbol_info_tick(symbol).ask if direction>0 else mt5.symbol_info_tick(symbol).bid
    order_type = mt5.ORDER_TYPE_BUY if direction>0 else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": stop_price,
        "tp": tp_price,
        "deviation": deviation_points,
        "magic": magic,
        "comment": "rf-signal",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open('config.yaml'))

    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    model_path = os.path.join(cfg['model_dir'], f"{args.symbol}_rf.pkl")
    scaler_path = os.path.join(cfg['model_dir'], "scaler.pkl")
    model = joblib.load(model_path)
    from sklearn.preprocessing import StandardScaler
    scaler = joblib.load(scaler_path)

    tf_str = cfg['timeframe']
    prob_buy = cfg['prob_buy']; prob_sell = cfg['prob_sell']
    stop_k = cfg['stop_atr_mult']; tp_k = cfg['tp_atr_mult']
    deviation = cfg['deviation_points']; magic = cfg['magic']

    print(f"Live loop for {args.symbol} on {tf_str}. Ctrl+C to stop.")
    last_time = None
    try:
        while True:
            df = build_latest_features(args.symbol, tf_str, bars=300)
            if df.empty: 
                time.sleep(1); continue
            # new bar check
            cur_time = df['time'].iloc[-1]
            if last_time is not None and cur_time == last_time:
                time.sleep(1); continue
            last_time = cur_time

            feature_cols = [c for c in df.columns if c not in ('time')]
            x_now = df[feature_cols].iloc[-1:].values
            xs = scaler.transform(x_now)
            p_up = model.predict_proba(xs)[:,1][0]

            price = df['close'].iloc[-1]
            atr = df['atr14'].iloc[-1]
            # simple ATR band gating (optional)
            atr_pct = atr / price
            if not (cfg['min_atr_pct'] <= atr_pct <= cfg['max_atr_pct']):
                print(f"{cur_time} | {args.symbol} | p_up={p_up:.3f} | ATR% out of band -> no trade")
                time.sleep(1); continue

            if p_up >= prob_buy or p_up <= prob_sell:
                direction = +1 if p_up >= prob_buy else -1
                sl = price - stop_k*atr if direction>0 else price + stop_k*atr
                tp = price + tp_k*atr if direction>0 else price - tp_k*atr

                # naive fixed lot; replace with risk-based sizing if desired
                lot = min(max(cfg['min_lot'], 0.01), cfg['max_lot'])
                res = place_bracket(args.symbol, lot, direction, sl, tp, deviation, magic)
                print(f"{cur_time} | {args.symbol} | p_up={p_up:.3f} | dir={direction} | lot={lot} | SL={sl:.5f} TP={tp:.5f} | result={res.retcode}")
            else:
                print(f"{cur_time} | {args.symbol} | p_up={p_up:.3f} | hold")

            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        mt5.shutdown()

if __name__ == '__main__':
    main()
