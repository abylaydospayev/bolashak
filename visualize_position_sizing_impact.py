"""
Visualize the impact of position sizing on trading system performance.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create results comparison
results = {
    'Configuration': [
        'OLD: Fixed 1 lot\n+ Regime Filter',
        'NEW: 1% Position Sizing\n+ Regime Filter', 
        'NEW: 1% Position Sizing\nNo Filter'
    ],
    'Total_PnL': [-900000, -15301, -44603],
    'Total_Trades': [207, 393, 1158],
    'Max_DD_Pct': [100, 15.14, 39.49],
    'Win_Rate_Pct': [45, 43.8, 43.9]
}

df = pd.DataFrame(results)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Position Sizing Impact Analysis', fontsize=16, fontweight='bold')

# 1. Total PnL comparison (log scale for better visualization)
ax1 = axes[0, 0]
colors = ['red', 'orange', 'darkred']
bars = ax1.bar(range(len(df)), np.abs(df['Total_PnL']), color=colors, alpha=0.7, edgecolor='black')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, df['Total_PnL'])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'${val:,.0f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_yscale('log')
ax1.set_ylabel('Total Loss (USD, log scale)', fontsize=12)
ax1.set_title('Total PnL Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels(df['Configuration'], fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add improvement annotations
improvement_1 = (df['Total_PnL'][0] - df['Total_PnL'][1]) / abs(df['Total_PnL'][0]) * 100
ax1.annotate(f'98.3% improvement', xy=(0.5, 100000), 
            xytext=(0.5, 500000),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, color='green', fontweight='bold',
            ha='center')

# 2. Max Drawdown comparison
ax2 = axes[0, 1]
colors_dd = ['darkred', 'orange', 'red']
bars_dd = ax2.bar(range(len(df)), df['Max_DD_Pct'], color=colors_dd, alpha=0.7, edgecolor='black')

# Add value labels
for bar, val in zip(bars_dd, df['Max_DD_Pct']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('Maximum Drawdown (%)', fontsize=12)
ax2.set_title('Risk Control (Max Drawdown)', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(df)))
ax2.set_xticklabels(df['Configuration'], fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=20, color='orange', linestyle='--', linewidth=1, label='20% threshold')
ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, label='50% danger zone')
ax2.legend(fontsize=9)

# 3. Number of trades
ax3 = axes[1, 0]
colors_trades = ['blue', 'green', 'purple']
bars_trades = ax3.bar(range(len(df)), df['Total_Trades'], color=colors_trades, alpha=0.7, edgecolor='black')

# Add value labels
for bar, val in zip(bars_trades, df['Total_Trades']):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:,}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_ylabel('Number of Trades', fontsize=12)
ax3.set_title('Trading Activity', fontsize=13, fontweight='bold')
ax3.set_xticks(range(len(df)))
ax3.set_xticklabels(df['Configuration'], fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# 4. Performance summary table
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

# Create summary table
summary_data = []
for i in range(len(df)):
    improvement = ((df['Total_PnL'][0] - df['Total_PnL'][i]) / abs(df['Total_PnL'][0]) * 100) if i > 0 else 0
    summary_data.append([
        df['Configuration'][i].replace('\n', ' '),
        f"${df['Total_PnL'][i]:,.0f}",
        f"{df['Max_DD_Pct'][i]:.1f}%",
        f"{df['Total_Trades'][i]:,}",
        f"+{improvement:.1f}%" if i > 0 else "Baseline"
    ])

table = ax4.table(cellText=summary_data,
                 colLabels=['Configuration', 'Total PnL', 'Max DD', 'Trades', 'vs Baseline'],
                 cellLoc='left',
                 loc='center',
                 colWidths=[0.35, 0.15, 0.12, 0.12, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color code the cells
for i in range(len(summary_data)):
    for j in range(len(summary_data[i])):
        cell = table[(i+1, j)]
        if j == 1:  # PnL column
            cell.set_facecolor('#ffcccc' if i == 0 else '#ffe6cc')
        elif j == 2:  # Max DD column
            if df['Max_DD_Pct'][i] > 50:
                cell.set_facecolor('#ff9999')
            elif df['Max_DD_Pct'][i] > 20:
                cell.set_facecolor('#ffcc99')
            else:
                cell.set_facecolor('#ccffcc')
        elif j == 4 and i > 0:  # Improvement column
            cell.set_facecolor('#ccffcc')

# Header styling
for j in range(5):
    table[(0, j)].set_facecolor('#4CAF50')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax4.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()

# Save
output_path = Path('results/position_sizing_impact.png')
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

plt.show()

# Create detailed comparison
print("\n" + "="*80)
print("POSITION SIZING IMPACT ANALYSIS")
print("="*80)

print("\nConfiguration Comparison:")
print("-" * 80)
for i in range(len(df)):
    print(f"\n{i+1}. {df['Configuration'][i].replace(chr(10), ' ')}")
    print(f"   Total PnL:     ${df['Total_PnL'][i]:>12,}")
    print(f"   Total Trades:  {df['Total_Trades'][i]:>12,}")
    print(f"   Max DD:        {df['Max_DD_Pct'][i]:>12.2f}%")
    print(f"   Win Rate:      {df['Win_Rate_Pct'][i]:>12.1f}%")
    
    if i > 0:
        improvement = ((df['Total_PnL'][0] - df['Total_PnL'][i]) / abs(df['Total_PnL'][0]) * 100)
        print(f"   Improvement:   {improvement:>12.1f}% vs baseline")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print("\n1. POSITION SIZING IMPACT:")
print(f"    Fixed 1 lot:        ${df['Total_PnL'][0]:>12,}  (catastrophic)")
print(f"    1% position sizing: ${df['Total_PnL'][1]:>12,}  (controlled)")
print(f"    Improvement:        98.3%")

print("\n2. REGIME FILTER IMPACT (with proper sizing):")
print(f"    No filter:          ${df['Total_PnL'][2]:>12,}")
print(f"    With filter:        ${df['Total_PnL'][1]:>12,}")
print(f"    Improvement:        {((df['Total_PnL'][2] - df['Total_PnL'][1]) / abs(df['Total_PnL'][2]) * 100):.1f}%")

print("\n3. RISK CONTROL:")
print(f"    Fixed 1 lot:        {df['Max_DD_Pct'][0]:.1f}% max DD  (account blown)")
print(f"    1% sizing:          {df['Max_DD_Pct'][1]:.1f}% max DD  (controlled)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\n Position sizing is THE critical factor (98% impact)")
print(" Regime filter adds significant value (66% improvement)")
print(" System still lacks edge (losing money despite good risk management)")
print("\nNext step: Improve predictive model to AUC > 0.55")
print("="*80 + "\n")

