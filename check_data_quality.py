"""Verify real data quality."""
import pandas as pd

print("=" * 70)
print("Real Data Quality Check")
print("=" * 70)

for symbol in ['EURUSD.sim', 'USDJPY.sim']:
    print(f"\n{symbol}:")
    print("-" * 70)
    
    df = pd.read_csv(f'data/{symbol}_M15.csv', parse_dates=['time'])
    
    print(f"Total bars: {len(df)}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
    print(f"\nFirst 5 bars:")
    print(df.head(5)[['time', 'open', 'high', 'low', 'close', 'volume']])
    print(f"\nLast 5 bars:")
    print(df.tail(5)[['time', 'open', 'high', 'low', 'close', 'volume']])
    
    # Check for data quality issues
    print(f"\nData Quality:")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicate times: {df['time'].duplicated().sum()}")
    print(f"  OHLC valid: {(df['high'] >= df['low']).all()}")
    print(f"  Close in range: {((df['close'] >= df['low']) & (df['close'] <= df['high'])).all()}")
