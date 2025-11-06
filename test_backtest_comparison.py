"""Quick test of backtest with EURUSD vs USDJPY to see P&L calculations."""
import pandas as pd
import numpy as np
import yaml
from backtest import backtest

# Load config
cfg = yaml.safe_load(open('config.yaml'))

# Test with EURUSD
print("=" * 80)
print("TESTING EURUSD")
print("=" * 80)

df_eur = pd.read_csv('features/EURUSD_features_enhanced.csv')
df_eur.attrs['symbol'] = 'EURUSD'

# Create simple predictions: always confident BUY
proba_eur = np.full(len(df_eur), 0.85)

result_eur = backtest(df_eur, proba_eur, cfg, use_position_sizing=True)
print(f"\nEURUSD Results:")
print(f"  Trades: {result_eur['trades']}")
print(f"  Win Rate: {result_eur['winrate']:.1%}")
print(f"  Final Equity: ${result_eur['equity']:,.0f}")
print(f"  P&L: ${result_eur['equity'] - 100000:,.0f}")
print(f"  Max DD: {result_eur['max_dd']:.2%}")

# Test with USDJPY
print("\n" + "=" * 80)
print("TESTING USDJPY")
print("=" * 80)

df_usd = pd.read_csv('features/USDJPY_features_enhanced.csv')
df_usd.attrs['symbol'] = 'USDJPY'

# Create simple predictions: always confident BUY
proba_usd = np.full(len(df_usd), 0.85)

result_usd = backtest(df_usd, proba_usd, cfg, use_position_sizing=True)
print(f"\nUSDJPY Results:")
print(f"  Trades: {result_usd['trades']}")
print(f"  Win Rate: {result_usd['winrate']:.1%}")
print(f"  Final Equity: ${result_usd['equity']:,.0f}")
print(f"  P&L: ${result_usd['equity'] - 100000:,.0f}")
print(f"  Max DD: {result_usd['max_dd']:.2%}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Both use same predictions (0.85 probability)")
print(f"EURUSD P&L: ${result_eur['equity'] - 100000:,.0f}")
print(f"USDJPY P&L: ${result_usd['equity'] - 100000:,.0f}")
