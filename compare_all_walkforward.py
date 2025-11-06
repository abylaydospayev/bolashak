"""Compare all walk-forward approaches."""
import pandas as pd

print('\n' + '='*80)
print('WALK-FORWARD COMPARISON - USDJPY')
print('='*80)

# Load results
rf_no_filter = pd.read_csv('results/USDJPY.sim_walkforward_no_filter.csv')
rf_with_filter = pd.read_csv('results/USDJPY.sim_walkforward_regime_filtered.csv')
lc_with_filters = pd.read_csv('results/USDJPY.sim_walkforward_lorentzian_regime.csv')

approaches = [
    ('RF (No Filter)', rf_no_filter),
    ('RF + Regime Filter', rf_with_filter),
    ('LC (k=100) + Confidence + Regime', lc_with_filters)
]

print(f"\n{'Approach':<35} {'Total PnL':<15} {'Trades':<10} {'Traded%':<10} {'Avg AUC':<10}")
print(''*80)

for name, df in approaches:
    total_pnl = df['total_pnl'].sum()
    total_trades = df['n_trades'].sum()
    avg_auc = df['auc'].mean()
    
    if 'pct_traded' in df.columns:
        pct_traded = df['pct_traded'].mean()
    elif 'pct_allowed' in df.columns:
        pct_traded = df['pct_allowed'].mean()
    else:
        pct_traded = 100.0
    
    print(f"{name:<35} ${total_pnl:>13,.0f} {total_trades:>8} {pct_traded:>8.1f}%  {avg_auc:>8.3f}")

print(''*80)

# Calculate improvements
baseline_pnl = rf_no_filter['total_pnl'].sum()
rf_filter_improvement = baseline_pnl - rf_with_filter['total_pnl'].sum()
lc_improvement = baseline_pnl - lc_with_filters['total_pnl'].sum()

print(f"\nIMPROVEMENTS vs RF (No Filter):")
print(f"  RF + Regime:                ${rf_filter_improvement:>13,.0f} ({rf_filter_improvement/abs(baseline_pnl)*100:+.1f}%)")
print(f"  LC + Confidence + Regime:   ${lc_improvement:>13,.0f} ({lc_improvement/abs(baseline_pnl)*100:+.1f}%)")

improvement_lc_vs_rf_filter = rf_with_filter['total_pnl'].sum() - lc_with_filters['total_pnl'].sum()
print(f"\nLC vs RF+Regime:")
print(f"  Additional improvement:     ${improvement_lc_vs_rf_filter:>13,.0f} ({improvement_lc_vs_rf_filter/abs(rf_with_filter['total_pnl'].sum())*100:+.1f}%)")

print('\n' + '='*80)
print('PER-FOLD COMPARISON')
print('='*80)

print(f"\n{'Fold':<6} {'RF (No Filter)':<18} {'RF + Regime':<18} {'LC + Conf + Regime':<20}")
print(''*80)

for i in range(len(rf_no_filter)):
    fold = int(rf_no_filter.iloc[i]['fold'])
    pnl1 = rf_no_filter.iloc[i]['total_pnl']
    pnl2 = rf_with_filter.iloc[i]['total_pnl']
    pnl3 = lc_with_filters.iloc[i]['total_pnl']
    
    print(f"{fold:<6} ${pnl1:>15,.0f}  ${pnl2:>15,.0f}  ${pnl3:>17,.0f}")

print(''*80)

# Best fold analysis
print('\n' + '='*80)
print('KEY INSIGHTS')
print('='*80)

lc_best_fold = lc_with_filters.loc[lc_with_filters['total_pnl'].idxmax()]
print(f"\n BEST FOLD (Lorentzian):")
print(f"   Fold {int(lc_best_fold['fold'])}: ${lc_best_fold['total_pnl']:,.0f}")
print(f"   AUC: {lc_best_fold['auc']:.3f}, Win Rate: {lc_best_fold['win_rate']*100:.1f}%")
print(f"   Traded: {lc_best_fold['pct_traded']:.1f}% of bars")

print(f"\n OVERALL PERFORMANCE:")
print(f"   Lorentzian is {abs(improvement_lc_vs_rf_filter)/abs(rf_with_filter['total_pnl'].sum())*100:.0f}% better than RF+Regime")
print(f"   Total improvement: ${lc_improvement:,.0f} vs baseline")
print(f"   Trade reduction: {100-lc_with_filters['pct_traded'].mean():.0f}% fewer signals")

print('\n' + '='*80 + '\n')

