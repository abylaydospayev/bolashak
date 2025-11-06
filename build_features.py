
import argparse, os
import pandas as pd
import numpy as np
import yaml
from indicators import ema, rsi, atr, pct_change, sincos_time

def pip_value(symbol:str)->float:
    # For labeling/costs we need pips-to-price conversion.
    # EURUSD pip = 0.0001, USDJPY pip = 0.01
    return 0.0001 if symbol.upper().startswith('EURUSD') else 0.01

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--h', type=int, default=None, help='horizon bars ahead')
    args = parser.parse_args()

    cfg = yaml.safe_load(open('config.yaml'))
    h = args.h or cfg['horizon']
    data_dir = cfg['data_dir']
    feat_dir = cfg['feature_dir']
    os.makedirs(feat_dir, exist_ok=True)

    # Find CSV by prefix
    candidates = [f for f in os.listdir(data_dir) if f.startswith(args.symbol+"_")]
    if not candidates:
        raise FileNotFoundError(f"No CSV found for {args.symbol} in {data_dir}. Run make_dataset.py first.")
    csv_path = os.path.join(data_dir, sorted(candidates)[-1])
    df = pd.read_csv(csv_path, parse_dates=['time'])

    # Indicators
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['rsi14'] = rsi(df['close'], 14)
    df['atr14'] = atr(df, 14)
    df['ema50_slope'] = df['ema50'].diff(5)
    # Ensure timezone-aware datetime
    time_col = df['time'] if df['time'].dt.tz is not None else df['time'].dt.tz_localize('UTC')
    sin_h, cos_h = sincos_time(time_col)
    df['sin_hour'] = sin_h.values
    df['cos_hour'] = cos_h.values
    df['ret1'] = pct_change(df['close'], 1)
    df['atr_pct'] = df['atr14'] / df['close']

    # Forward return (net of fees/slippage)
    pip = pip_value(args.symbol)
    spread = cfg['spread_pips'] * pip
    slippage = cfg['slippage_pips'] * pip
    # Assume round-turn commission roughly equal to spread in price terms for simplicity here (adjust if needed)
    fees = spread + slippage

    df['fwd_ret'] = df['close'].shift(-h) / df['close'] - 1.0
    df['fwd_ret_net'] = df['fwd_ret'] - fees / df['close']

    # Label: up if > +0.03% = 0.0003
    df['y'] = (df['fwd_ret_net'] > 0.0003).astype(int)

    # Drop NaNs
    feature_cols = ['open','high','low','close','volume','ema20','ema50','rsi14','atr14',
                    'ema50_slope','sin_hour','cos_hour','ret1','atr_pct']
    out = df[['time'] + feature_cols + ['y']].dropna().reset_index(drop=True)
    os.makedirs(feat_dir, exist_ok=True)
    out_path = os.path.join(feat_dir, f"{args.symbol}_features.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved features to {out_path} with {len(out)} rows.")

if __name__ == '__main__':
    main()
