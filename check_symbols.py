"""Check available MT5 symbols to find correct symbol names."""
import MetaTrader5 as mt5

if not mt5.initialize():
    print(f"MT5 initialization failed: {mt5.last_error()}")
    exit(1)

print("Searching for EUR and JPY symbols...")
print("=" * 70)

# Get all symbols
symbols = mt5.symbols_get()
print(f"Total symbols available: {len(symbols)}\n")

# Filter for EUR and JPY pairs
eur_symbols = [s for s in symbols if 'EUR' in s.name and 'USD' in s.name]
jpy_symbols = [s for s in symbols if 'USD' in s.name and 'JPY' in s.name]

print("EURUSD variants:")
for s in eur_symbols[:10]:  # Show first 10
    visible = "" if s.visible else ""
    print(f"  {visible} {s.name:<20} - {s.description}")

print(f"\nUSDJPY variants:")
for s in jpy_symbols[:10]:  # Show first 10
    visible = "" if s.visible else ""
    print(f"  {visible} {s.name:<20} - {s.description}")

# Try to find the most likely candidates
print("\n" + "=" * 70)
print("Recommended symbols (visible in Market Watch):")
print("=" * 70)

recommended_eur = [s.name for s in eur_symbols if s.visible]
recommended_jpy = [s.name for s in jpy_symbols if s.visible]

if recommended_eur:
    print(f"EURUSD: {recommended_eur[0]}")
else:
    print(f"EURUSD: {eur_symbols[0].name if eur_symbols else 'Not found'} (not visible - may need to enable)")

if recommended_jpy:
    print(f"USDJPY: {recommended_jpy[0]}")
else:
    print(f"USDJPY: {jpy_symbols[0].name if jpy_symbols else 'Not found'} (not visible - may need to enable)")

mt5.shutdown()

