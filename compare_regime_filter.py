"""Compare regime filter results."""
import pandas as pd

no_filter = pd.read_csv('results/USDJPY.sim_walkforward_no_filter.csv')
with_filter = pd.read_csv('results/USDJPY.sim_walkforward_regime_filtered.csv')

print('\n' + '='*60)
print('REGIME FILTER IMPACT - USDJPY Walk-Forward')
print('='*60)
print(f'\nWITHOUT Filter:')
print(f'  Total PnL:     ${no_filter["total_pnl"].sum():>12,.0f}')
print(f'  Trades:        {no_filter["n_trades"].sum():>12,}')
print(f'  Avg PnL/Trade: ${no_filter["total_pnl"].sum()/no_filter["n_trades"].sum():>12,.0f}')

print(f'\nWITH Regime Filter:')
print(f'  Total PnL:     ${with_filter["total_pnl"].sum():>12,.0f}')
print(f'  Trades:        {with_filter["n_trades"].sum():>12,}')
print(f'  Avg PnL/Trade: ${with_filter["total_pnl"].sum()/with_filter["n_trades"].sum():>12,.0f}')

improvement = no_filter['total_pnl'].sum() - with_filter['total_pnl'].sum()
pct_improvement = improvement / abs(no_filter['total_pnl'].sum()) * 100
trades_reduction = (1 - with_filter['n_trades'].sum() / no_filter['n_trades'].sum()) * 100

print(f'\n{"-"*60}')
print(f'IMPROVEMENT:')
print(f'  PnL Impact:    ${improvement:>12,.0f} ({pct_improvement:+.1f}%)')
print(f'  Trades Reduced: {trades_reduction:>11.0f}%')
print(f'  Status:         {" BETTER" if improvement > 0 else " WORSE"}')
print('='*60)

print('\n\nPer-Fold Comparison:')
print(f'{"Fold":<6} {"No Filter PnL":<18} {"With Filter PnL":<18} {"Improvement":<15}')
print('-'*60)
for i in range(len(no_filter)):
    nf_pnl = no_filter.iloc[i]['total_pnl']
    wf_pnl = with_filter.iloc[i]['total_pnl']
    imp = nf_pnl - wf_pnl
    print(f'{int(no_filter.iloc[i]["fold"]):<6} ${nf_pnl:>15,.0f}  ${wf_pnl:>15,.0f}  ${imp:>12,.0f}')

print('\n')

