"""Test cost calculations."""
import yaml

cfg = yaml.safe_load(open('config.yaml'))

pip_eurusd = 0.0001
pip_usdjpy = 0.01

# Total pips per round-turn
spread_pips = cfg['spread_pips'] * 2  # entry + exit
slippage_pips = cfg['slippage_pips'] * 2
total_pips = spread_pips + slippage_pips

print("=" * 70)
print("Trading Cost Calculation")
print("=" * 70)
print(f"\nSpread: {cfg['spread_pips']} pips  2 = {spread_pips} pips")
print(f"Slippage: {cfg['slippage_pips']} pips  2 = {slippage_pips} pips")
print(f"Total pips per round-turn: {total_pips} pips")
print(f"Commission: ${cfg['commission_per_lot']}")

print("\n" + "=" * 70)
print("EURUSD (1 pip = $0.0001 price, 1 lot = 100k units)")
print("=" * 70)
cost_from_pips_eur = total_pips * pip_eurusd * 100000
cost_total_eur = cost_from_pips_eur + cfg['commission_per_lot']
print(f"Cost from pips: {total_pips}  {pip_eurusd}  100,000 = ${cost_from_pips_eur:.2f}")
print(f"Commission: ${cfg['commission_per_lot']:.2f}")
print(f"TOTAL per trade: ${cost_total_eur:.2f}")

print("\n" + "=" * 70)
print("USDJPY (1 pip = $0.01 price, 1 lot = 100k units)")
print("=" * 70)
cost_from_pips_jpy = total_pips * pip_usdjpy * 100000
cost_total_jpy = cost_from_pips_jpy + cfg['commission_per_lot']
print(f"Cost from pips: {total_pips}  {pip_usdjpy}  100,000 = ${cost_from_pips_jpy:.2f}")
print(f"Commission: ${cfg['commission_per_lot']:.2f}")
print(f"TOTAL per trade: ${cost_total_jpy:.2f}")

# Check impact on recent results
print("\n" + "=" * 70)
print("Impact on Backtest Results")
print("=" * 70)
eurusd_trades = 1396
usdjpy_trades = 2033

eurusd_cost_total = eurusd_trades * cost_total_eur
usdjpy_cost_total = usdjpy_trades * cost_total_jpy

print(f"\nEURUSD: {eurusd_trades} trades  ${cost_total_eur:.2f} = ${eurusd_cost_total:,.2f}")
print(f"USDJPY: {usdjpy_trades} trades  ${cost_total_jpy:.2f} = ${usdjpy_cost_total:,.2f}")

