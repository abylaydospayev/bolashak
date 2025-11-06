"""Debug EURUSD backtest to see what's causing massive losses."""
import pandas as pd
import yaml

# Load EURUSD data
df = pd.read_csv('features/EURUSD_features_enhanced.csv')
df.attrs['symbol'] = 'EURUSD'

print("EURUSD Data:")
print(f"  Rows: {len(df)}")
print(f"  Close price range: {df['close'].min():.5f} to {df['close'].max():.5f}")
print(f"  ATR14 mean: {df['atr14'].mean():.6f}")
print(f"  ATR14 range: {df['atr14'].min():.6f} to {df['atr14'].max():.6f}")

# Load config
cfg = yaml.safe_load(open('config.yaml'))

# Check pip value
pip = 0.0001
symbol = 'EURUSD'

# Simulate one trade
entry_price = 1.0500  # Typical EURUSD price
stop_k = cfg['stop_atr_mult']  # 1.5
tp_k = cfg['tp_atr_mult']  # 2.5
atr = 0.0005  # Typical ATR for EURUSD (50 pips)

# Calculate SL/TP distances
stop_distance = stop_k * atr  # 1.5 * 0.0005 = 0.00075 (7.5 pips)
tp_distance = tp_k * atr      # 2.5 * 0.0005 = 0.00125 (12.5 pips)

print(f"\nSample Trade:")
print(f"  Entry: {entry_price}")
print(f"  ATR: {atr} ({atr/pip:.1f} pips)")
print(f"  Stop distance: {stop_distance} ({stop_distance/pip:.1f} pips)")
print(f"  TP distance: {tp_distance} ({tp_distance/pip:.1f} pips)")

# If we hit TP on a long trade
pnl_price = tp_distance  # Price move
pnl_gross = pnl_price * 100000  # Convert to USD
print(f"\nIf TP hit (LONG):")
print(f"  PnL (price): {pnl_price}")
print(f"  PnL (gross): ${pnl_gross:.2f}")

# Calculate costs
spread_pips = cfg['spread_pips'] * 2  # 0.8 * 2 = 1.6
slippage_pips = cfg['slippage_pips'] * 2  # 0.4 * 2 = 0.8
total_pips = spread_pips + slippage_pips  # 2.4 pips
pip_value_usd = pip * 100000  # $10 per pip
cost_from_pips = total_pips * pip_value_usd
commission = cfg.get('commission_per_lot', 7.0)
total_cost = cost_from_pips + commission

print(f"\nCosts:")
print(f"  Spread: {spread_pips} pips")
print(f"  Slippage: {slippage_pips} pips")
print(f"  Total pips: {total_pips}")
print(f"  Pip value: ${pip_value_usd:.2f}")
print(f"  Cost from pips: ${cost_from_pips:.2f}")
print(f"  Commission: ${commission:.2f}")
print(f"  Total cost: ${total_cost:.2f}")

pnl_net = pnl_gross - total_cost
print(f"\nNet P&L: ${pnl_net:.2f}")
print(f"  This should be positive for a TP hit!")

# Now check the actual EURUSD ATR values
print(f"\nActual EURUSD ATR Statistics:")
print(f"  Mean: {df['atr14'].mean():.6f} ({df['atr14'].mean()/pip:.1f} pips)")
print(f"  Median: {df['atr14'].median():.6f} ({df['atr14'].median()/pip:.1f} pips)")
print(f"  Min: {df['atr14'].min():.6f} ({df['atr14'].min()/pip:.1f} pips)")
print(f"  Max: {df['atr14'].max():.6f} ({df['atr14'].max()/pip:.1f} pips)")

# Calculate typical SL/TP for EURUSD
typical_atr = df['atr14'].median()
typical_sl_pips = (stop_k * typical_atr) / pip
typical_tp_pips = (tp_k * typical_atr) / pip

print(f"\nTypical SL/TP for EURUSD:")
print(f"  SL: {typical_sl_pips:.1f} pips")
print(f"  TP: {typical_tp_pips:.1f} pips")
print(f"  Cost: {total_pips:.1f} pips")
print(f"  R:R after costs: {(typical_tp_pips - total_pips) / (typical_sl_pips + total_pips):.2f}:1")
