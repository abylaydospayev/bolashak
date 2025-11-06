"""Create sample forex data for testing without MT5 connection."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_forex_data(symbol, bars=10000):
    """Generate realistic-looking forex data."""
    np.random.seed(42)
    
    # Starting price
    if 'EURUSD' in symbol:
        base_price = 1.0800
        pip = 0.0001
    else:  # USDJPY
        base_price = 149.50
        pip = 0.01
    
    # Generate timestamps (M15 = 15 minutes)
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=15 * bars)
    times = pd.date_range(start=start_time, end=end_time, freq='15min')[:bars]
    
    # Generate price movements with realistic characteristics
    returns = np.random.normal(0, 0.0002, bars)  # ~2 pips std dev
    # Add some trending behavior
    trend = np.linspace(0, 0.002, bars) + np.sin(np.linspace(0, 4*np.pi, bars)) * 0.001
    returns += trend / bars
    
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    highs = close_prices * (1 + np.abs(np.random.normal(0, 0.0003, bars)))
    lows = close_prices * (1 - np.abs(np.random.normal(0, 0.0003, bars)))
    opens = close_prices * (1 + np.random.normal(0, 0.0001, bars))
    
    # Ensure OHLC relationships hold
    highs = np.maximum(highs, np.maximum(opens, close_prices))
    lows = np.minimum(lows, np.minimum(opens, close_prices))
    
    # Generate volume
    volumes = np.random.randint(50, 500, bars)
    
    df = pd.DataFrame({
        'time': times,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': volumes
    })
    
    return df

def main():
    os.makedirs('data', exist_ok=True)
    
    for symbol in ['EURUSD', 'USDJPY']:
        df = create_sample_forex_data(symbol, bars=50000)
        output_path = f'data/{symbol}_M15.csv'
        df.to_csv(output_path, index=False)
        print(f"Created {output_path} with {len(df)} bars")
        print(f"  Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

if __name__ == '__main__':
    main()
