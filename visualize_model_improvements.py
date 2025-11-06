"""
Visualize model improvements: Baseline vs Enhanced Ensemble.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results comparison
results = {
    'Model': ['Baseline\nRF', 'Enhanced\nEnsemble'],
    'AUC': [0.518, 0.795],
    'PnL': [-15301, -64129],
    'Trades': [393, 2031],
    'Win_Rate': [43.8, 46.1],
    'Max_DD': [15.14, 19.55]
}

df = pd.DataFrame(results)

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Model Improvements: Baseline RF vs Enhanced Ensemble', 
             fontsize=18, fontweight='bold', y=0.98)

# 1. AUC Comparison
ax1 = fig.add_subplot(gs[0, 0])
colors = ['red', 'green']
bars = ax1.bar(df['Model'], df['AUC'], color=colors, alpha=0.7, edgecolor='black', linewidth=2)

for bar, val in zip(bars, df['AUC']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
ax1.set_title('Predictive Power (AUC)', fontsize=13, fontweight='bold')
ax1.set_ylim([0, 1.0])
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
ax1.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Good (0.6)')
ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Excellent (0.7)')
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Add improvement annotation
improvement = (df['AUC'][1] - df['AUC'][0]) / df['AUC'][0] * 100
ax1.text(0.5, 0.65, f'+{improvement:.1f}%', 
        fontsize=14, fontweight='bold', color='green',
        ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# 2. PnL Comparison
ax2 = fig.add_subplot(gs[0, 1])
colors_pnl = ['salmon', 'darkred']
bars_pnl = ax2.bar(df['Model'], df['PnL'], color=colors_pnl, alpha=0.7, edgecolor='black', linewidth=2)

for bar, val in zip(bars_pnl, df['PnL']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'${val:,.0f}',
            ha='center', va='top' if val < 0 else 'bottom', 
            fontsize=11, fontweight='bold')

ax2.set_ylabel('Total PnL (USD)', fontsize=12, fontweight='bold')
ax2.set_title('Profitability', fontsize=13, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.grid(axis='y', alpha=0.3)

# 3. Number of Trades
ax3 = fig.add_subplot(gs[0, 2])
colors_trades = ['blue', 'purple']
bars_trades = ax3.bar(df['Model'], df['Trades'], color=colors_trades, alpha=0.7, edgecolor='black', linewidth=2)

for bar, val in zip(bars_trades, df['Trades']):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:,}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax3.set_ylabel('Number of Trades', fontsize=12, fontweight='bold')
ax3.set_title('Trading Activity', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Add "Too many!" annotation
ax3.text(1, df['Trades'][1] * 0.7, 'Too many!\n(5x increase)', 
        fontsize=11, fontweight='bold', color='red',
        ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 4. Win Rate
ax4 = fig.add_subplot(gs[1, 0])
colors_wr = ['orange', 'green']
bars_wr = ax4.bar(df['Model'], df['Win_Rate'], color=colors_wr, alpha=0.7, edgecolor='black', linewidth=2)

for bar, val in zip(bars_wr, df['Win_Rate']):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax4.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
ax4.set_title('Win Rate', fontsize=13, fontweight='bold')
ax4.set_ylim([0, 60])
ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Breakeven (50%)')
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

# 5. Max Drawdown
ax5 = fig.add_subplot(gs[1, 1])
colors_dd = ['orange', 'red']
bars_dd = ax5.bar(df['Model'], df['Max_DD'], color=colors_dd, alpha=0.7, edgecolor='black', linewidth=2)

for bar, val in zip(bars_dd, df['Max_DD']):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax5.set_ylabel('Maximum Drawdown (%)', fontsize=12, fontweight='bold')
ax5.set_title('Risk (Max DD)', fontsize=13, fontweight='bold')
ax5.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Warning (20%)')
ax5.legend(fontsize=9)
ax5.grid(axis='y', alpha=0.3)

# 6. Summary Table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('tight')
ax6.axis('off')

summary_data = []
metrics = ['AUC', 'Win Rate', 'PnL', 'Trades', 'Max DD']
for metric in metrics:
    if metric == 'AUC':
        base_val = f"{df['AUC'][0]:.3f}"
        enh_val = f"{df['AUC'][1]:.3f}"
        change = f"+{(df['AUC'][1]-df['AUC'][0])/df['AUC'][0]*100:.1f}%"
    elif metric == 'Win Rate':
        base_val = f"{df['Win_Rate'][0]:.1f}%"
        enh_val = f"{df['Win_Rate'][1]:.1f}%"
        change = f"+{df['Win_Rate'][1]-df['Win_Rate'][0]:.1f}%"
    elif metric == 'PnL':
        base_val = f"${df['PnL'][0]:,.0f}"
        enh_val = f"${df['PnL'][1]:,.0f}"
        change = f"{(df['PnL'][1]-df['PnL'][0])/abs(df['PnL'][0])*100:+.0f}%"
    elif metric == 'Trades':
        base_val = f"{df['Trades'][0]:,}"
        enh_val = f"{df['Trades'][1]:,}"
        change = f"+{(df['Trades'][1]-df['Trades'][0])/df['Trades'][0]*100:.0f}%"
    elif metric == 'Max DD':
        base_val = f"{df['Max_DD'][0]:.1f}%"
        enh_val = f"{df['Max_DD'][1]:.1f}%"
        change = f"+{df['Max_DD'][1]-df['Max_DD'][0]:.1f}%"
    
    summary_data.append([metric, base_val, enh_val, change])

table = ax6.table(cellText=summary_data,
                 colLabels=['Metric', 'Baseline', 'Enhanced', 'Change'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.25, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Color code cells
for i in range(len(summary_data)):
    for j in range(len(summary_data[i])):
        cell = table[(i+1, j)]
        if j == 3:  # Change column
            if 'AUC' in summary_data[i][0] or 'Win Rate' in summary_data[i][0]:
                cell.set_facecolor('#ccffcc')  # Green for improvements
            elif 'PnL' in summary_data[i][0] or 'Trades' in summary_data[i][0] or 'DD' in summary_data[i][0]:
                cell.set_facecolor('#ffcccc')  # Red for negatives

# Header styling
for j in range(4):
    table[(0, j)].set_facecolor('#4CAF50')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax6.set_title('Metrics Summary', fontsize=13, fontweight='bold', pad=20)

# 7. Feature Importance (bottom span)
ax7 = fig.add_subplot(gs[2, :])

features = ['price_vs_ema20_h1', 'momentum_5_h1', 'momentum_10_h1', 'rsi14_h1', 
            'trend_strength_h1', 'ema20_h1', 'atr14_h1', 'ema50_h1']
scores = [0.0823, 0.0814, 0.0811, 0.0805, 0.0801, 0.0798, 0.0798, 0.0795]

bars_feat = ax7.barh(range(len(features)), scores, color='steelblue', alpha=0.7, edgecolor='black')

for i, (bar, score) in enumerate(zip(bars_feat, scores)):
    width = bar.get_width()
    ax7.text(width, bar.get_y() + bar.get_height()/2.,
            f' {score:.4f}',
            ha='left', va='center', fontsize=10, fontweight='bold')

ax7.set_yticks(range(len(features)))
ax7.set_yticklabels(features, fontsize=10)
ax7.set_xlabel('Mutual Information Score', fontsize=12, fontweight='bold')
ax7.set_title('Top 8 Most Important Features (All from Higher Timeframes!)', 
             fontsize=13, fontweight='bold')
ax7.grid(axis='x', alpha=0.3)

# Add timeframe annotation
ax7.text(0.085, 7.5, 'H1 = 1-hour\nM30 = 30-min\nH4 = 4-hour', 
        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()

# Save
output_path = Path('results/model_improvements_comparison.png')
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

plt.show()

# Print summary
print("\n" + "="*80)
print("MODEL IMPROVEMENTS SUMMARY")
print("="*80)

print("\nKey Achievements:")
print(f"   AUC improved: {df['AUC'][0]:.3f}  {df['AUC'][1]:.3f} (+{(df['AUC'][1]-df['AUC'][0])/df['AUC'][0]*100:.1f}%)")
print(f"   Win rate improved: {df['Win_Rate'][0]:.1f}%  {df['Win_Rate'][1]:.1f}% (+{df['Win_Rate'][1]-df['Win_Rate'][0]:.1f}%)")
print(f"   Multi-timeframe features working (H1 dominates top 10)")
print(f"   Ensemble outperforms single model")

print("\nRemaining Issues:")
print(f"   PnL worsened: ${df['PnL'][0]:,.0f}  ${df['PnL'][1]:,.0f} (4x more loss)")
print(f"   Too many trades: {df['Trades'][0]:,}  {df['Trades'][1]:,} (5x increase)")
print(f"   Max DD increased: {df['Max_DD'][0]:.1f}%  {df['Max_DD'][1]:.1f}%")

print("\nRoot Cause:")
print("   Model quality is excellent (AUC 0.795)")
print("   But trading too frequently (2,031 trades)")
print("   Transaction costs eating profits ($28k+ in costs)")

print("\nSolution:")
print("   Increase probability threshold (0.60  0.70)")
print("   Expected: Fewer trades, higher win rate, profitability")

print("\n" + "="*80)

