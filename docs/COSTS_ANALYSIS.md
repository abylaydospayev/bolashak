# Trading Costs Analysis

## Current Implementation

###  Costs INCLUDED in Feature Engineering (Labels)
**File**: `build_features.py`

```python
spread = cfg['spread_pips'] * pip      # 0.8 pips for EURUSD/USDJPY
slippage = cfg['slippage_pips'] * pip  # 0.4 pips
fees = spread + slippage                # Total: 1.2 pips per round-turn

df['fwd_ret_net'] = df['fwd_ret'] - fees / df['close']
df['y'] = (df['fwd_ret_net'] > 0.0003).astype(int)
```

**Impact**: 
- Model learns to predict trades that are profitable AFTER costs
- This is GOOD - the model is cost-aware during training

###  Costs NOT INCLUDED in Backtest (Current)
**File**: `backtest.py`

Current backtest only calculates:
```python
pnl = (exit_price - entry_price) if pos > 0 else (entry_price - exit_price)
equity += pnl * 100000  # Raw PnL without costs
```

**Missing**:
-  Spread (entry + exit)
-  Slippage (entry + exit)
-  Commission ($7 per lot round-turn)

**Impact**: Backtest results are OPTIMISTIC (overstate profitability)

---

## Cost Breakdown per Trade

### EURUSD
- **Spread**: 0.8 pips  2 (entry + exit) = 1.6 pips = $16 per lot
- **Slippage**: 0.4 pips  2 = 0.8 pips = $8 per lot
- **Commission**: $7 round-turn
- **Total**: ~$31 per lot per round-turn

### USDJPY  
- **Spread**: 0.8 pips  2 = 1.6 pips = $16 per lot (assuming 1 pip = $10 @ 150 rate)
- **Slippage**: 0.4 pips  2 = 0.8 pips = $8 per lot
- **Commission**: $7 round-turn
- **Total**: ~$31 per lot per round-turn

---

## Impact on Current Results

### USDJPY Backtest (Current - WITHOUT costs)
```
Trades: 2,033
Final Equity: +$2,752,046
```

**Estimated cost impact**:
- 2,033 trades  $31 = **-$63,023**

**Adjusted estimate**: $2,752,046 - $63,023 = **~$2,689,023**  Still profitable!

### EURUSD Backtest (Current)
```
Trades: 1,396
Final Equity: -$456
```

**Estimated cost impact**:
- 1,396 trades  $31 = **-$43,276**

**Adjusted estimate**: -$456 - $43,276 = **~-$43,732**  More negative

---

## Recommendation: Fix Required

The backtest should deduct costs for realistic results. Here's what needs to be added:

```python
def calculate_costs(entry_price, pip_size, cfg):
    """Calculate total trading costs for a round-turn trade."""
    spread_cost = cfg['spread_pips'] * pip_size * 2  # entry + exit
    slippage_cost = cfg['slippage_pips'] * pip_size * 2
    commission = cfg['commission_per_lot']
    
    # Convert pip costs to price
    spread_price = spread_cost
    slippage_price = slippage_cost
    
    # Total in dollars for 1 lot (100k units)
    total_cost = (spread_price + slippage_price) * 100000 + commission
    return total_cost
```

Then in backtest:
```python
if exit_now:
    pnl_gross = pnl * 100000
    costs = calculate_costs(entry_price, pip, cfg)
    pnl_net = pnl_gross - costs  # Deduct costs
    equity += pnl_net
```

---

## Status

-  **Labels**: Costs properly included
-  **Backtest**: Costs NOT deducted (needs fix)
-  **Live Trading**: Costs will be real (broker will charge)

**Conclusion**: Backtest overstates profitability. USDJPY likely still profitable after costs, but EURUSD will be more negative.

