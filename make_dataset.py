
import argparse, os
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
import yaml

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--timeframe', default=None, help="Defaults to config.yaml timeframe")
    parser.add_argument('--bars', type=int, default=100000, help="Default 100k bars = ~2.8 years at M15")
    args = parser.parse_args()

    cfg = yaml.safe_load(open('config.yaml'))
    tf_str = args.timeframe or cfg['timeframe']
    tf = timeframe_str_to_mt5(tf_str)

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    # Pull last N bars up to now (UTC)
    rates = mt5.copy_rates_from_pos(args.symbol, tf, 0, args.bars)
    if rates is None:
        raise RuntimeError(f"Failed to fetch rates for {args.symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.rename(columns={'tick_volume':'volume'})
    df = df[['time','open','high','low','close','volume']].dropna()
    os.makedirs(cfg['data_dir'], exist_ok=True)
    out = os.path.join(cfg['data_dir'], f"{args.symbol}_{tf_str}.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} bars to {out}")

    mt5.shutdown()

if __name__ == '__main__':
    main()
