
import numpy as np
import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods)

def sincos_time(time_series):
    # hour-of-day cyclical encoding (UTC)
    # Accept either DatetimeIndex or Series
    if isinstance(time_series, pd.Series):
        dt = time_series.dt
        h = dt.hour + dt.minute/60.0
        sin_h = np.sin(2*np.pi*h/24.0)
        cos_h = np.cos(2*np.pi*h/24.0)
        return pd.Series(sin_h.values, index=time_series.index, name='sin_hour'), pd.Series(cos_h.values, index=time_series.index, name='cos_hour')
    else:  # DatetimeIndex
        h = time_series.hour + time_series.minute/60.0
        sin_h = np.sin(2*np.pi*h/24.0)
        cos_h = np.cos(2*np.pi*h/24.0)
        return pd.Series(sin_h, index=time_series, name='sin_hour'), pd.Series(cos_h, index=time_series, name='cos_hour')
