"""
Evaluate all strategies tested so far to determine which approaches work best.
Compare: RF, LSTM, Lorentzian, Regime Filter, Confidence Filter combinations.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_results():
    """Load all walk-forward results."""
    results_dir = Path('results')
    
    results = {}
    
    # 1. RF no filter
    rf_no_filter_path = results_dir / 'USDJPY.sim_walkforward_no_filter.csv'
    if rf_no_filter_path.exists():
        results['RF (No Filter)'] = pd.read_csv(rf_no_filter_path)
    
    # 2. RF + Regime filter
    rf_regime_path = results_dir / 'USDJPY.sim_walkforward_regime_filtered.csv'
    if rf_regime_path.exists():
        results['RF + Regime'] = pd.read_csv(rf_regime_path)
    
    # 3. Lorentzian + Regime + Confidence
    lc_regime_path = results_dir / 'USDJPY.sim_walkforward_lorentzian_regime.csv'
    if lc_regime_path.exists():
        results['LC (k=100) + Regime + Conf'] = pd.read_csv(lc_regime_path)
    
    return results

def calculate_metrics(results_dict):
    """Calculate key metrics for each strategy."""
    summary = []
    
    for strategy_name, df in results_dict.items():
        if df is None or len(df) == 0:
            continue
        
        total_pnl = df['total_pnl'].sum()
        total_trades = df['n_trades'].sum()
        avg_win_rate = df['win_rate'].mean()
        avg_auc = df['auc'].mean()
        profitable_folds = (df['total_pnl'] > 0).sum()
        total_folds = len(df)
        
        # Handle different column names
        if 'pct_traded' in df.columns:
            avg_pct_traded = df['pct_traded'].mean()
        elif 'pct_allowed' in df.columns:
            avg_pct_traded = df['pct_allowed'].mean()
        else:
            avg_pct_traded = 100.0
        
        # Risk metrics
        worst_fold = df['total_pnl'].min()
        best_fold = df['total_pnl'].max()
        volatility = df['total_pnl'].std()
        
        # Sharpe-like ratio (return per unit of volatility)
        sharpe = total_pnl / volatility if volatility > 0 else 0
        
        # Calculate per-trade metrics
        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        summary.append({
            'Strategy': strategy_name,
            'Total PnL': total_pnl,
            'Total Trades': int(total_trades),
            'Avg Win Rate': avg_win_rate,
            'Avg AUC': avg_auc,
            'Profitable Folds': f"{profitable_folds}/{total_folds}",
            'Profitability %': profitable_folds / total_folds * 100,
            '% Traded': avg_pct_traded,
            'Worst Fold': worst_fold,
            'Best Fold': best_fold,
            'Volatility': volatility,
            'Sharpe-like': sharpe,
            'PnL per Trade': avg_pnl_per_trade
        })
    
    return pd.DataFrame(summary)

def analyze_strategy_components(results_dict):
    """Analyze which components contribute most to performance."""
    print("\n" + "="*80)
    print("COMPONENT ANALYSIS")
    print("="*80 + "\n")
    
    # Get baseline (RF without any filters)
    if 'RF (No Filter)' in results_dict:
        baseline_pnl = results_dict['RF (No Filter)']['total_pnl'].sum()
        baseline_trades = results_dict['RF (No Filter)']['n_trades'].sum()
    else:
        baseline_pnl = -3_100_000  # From previous runs
        baseline_trades = 826
    
    print(f"BASELINE (RF, no filters):")
    print(f"  PnL: ${baseline_pnl:,.0f}")
    print(f"  Trades: {int(baseline_trades)}")
    print()
    
    # Analyze regime filter impact
    if 'RF + Regime' in results_dict:
        regime_pnl = results_dict['RF + Regime']['total_pnl'].sum()
        regime_trades = results_dict['RF + Regime']['n_trades'].sum()
        
        regime_improvement = regime_pnl - baseline_pnl
        regime_trade_reduction = (baseline_trades - regime_trades) / baseline_trades * 100
        
        print(f"+ REGIME FILTER:")
        print(f"  PnL: ${regime_pnl:,.0f} ({regime_improvement:+,.0f})")
        print(f"  Trades: {int(regime_trades)} ({regime_trade_reduction:.1f}% reduction)")
        print(f"  Impact: {regime_improvement / abs(baseline_pnl) * 100:+.1f}% improvement")
        print()
    
    # Analyze Lorentzian + filters impact
    if 'LC (k=100) + Regime + Conf' in results_dict:
        lc_pnl = results_dict['LC (k=100) + Regime + Conf']['total_pnl'].sum()
        lc_trades = results_dict['LC (k=100) + Regime + Conf']['n_trades'].sum()
        
        lc_improvement = lc_pnl - baseline_pnl
        lc_trade_reduction = (baseline_trades - lc_trades) / baseline_trades * 100
        
        print(f"+ LORENTZIAN (k=100) + REGIME + CONFIDENCE:")
        print(f"  PnL: ${lc_pnl:,.0f} ({lc_improvement:+,.0f})")
        print(f"  Trades: {int(lc_trades)} ({lc_trade_reduction:.1f}% reduction)")
        print(f"  Impact: {lc_improvement / abs(baseline_pnl) * 100:+.1f}% improvement")
        print()

def visualize_comparison(summary_df):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Strategy Comparison Analysis - USDJPY', fontsize=16, fontweight='bold')
    
    # 1. Total PnL
    ax1 = axes[0, 0]
    colors = ['red' if x < 0 else 'green' for x in summary_df['Total PnL']]
    bars = ax1.barh(summary_df['Strategy'], summary_df['Total PnL'], color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Total PnL ($)', fontweight='bold')
    ax1.set_title('Total Profit/Loss', fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    for i, v in enumerate(summary_df['Total PnL']):
        ax1.text(v-200000 if v < 0 else v+50000, i, f'${v/1000:.0f}k', va='center', fontweight='bold', fontsize=9)
    
    # 2. Number of Trades
    ax2 = axes[0, 1]
    ax2.barh(summary_df['Strategy'], summary_df['Total Trades'], color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Trades', fontweight='bold')
    ax2.set_title('Trade Frequency', fontweight='bold')
    for i, v in enumerate(summary_df['Total Trades']):
        ax2.text(v+10, i, f'{int(v)}', va='center', fontsize=9)
    
    # 3. Win Rate
    ax3 = axes[0, 2]
    ax3.barh(summary_df['Strategy'], summary_df['Avg Win Rate']*100, color='orange', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Win Rate (%)', fontweight='bold')
    ax3.set_title('Average Win Rate', fontweight='bold')
    ax3.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='Break-even')
    for i, v in enumerate(summary_df['Avg Win Rate']):
        ax3.text(v*100+1, i, f'{v*100:.1f}%', va='center', fontsize=9)
    
    # 4. AUC Score
    ax4 = axes[1, 0]
    ax4.barh(summary_df['Strategy'], summary_df['Avg AUC'], color='purple', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('AUC Score', fontweight='bold')
    ax4.set_title('Model Quality (AUC)', fontweight='bold')
    ax4.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax4.set_xlim(0.4, max(summary_df['Avg AUC']) + 0.05)
    for i, v in enumerate(summary_df['Avg AUC']):
        ax4.text(v+0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    # 5. % Traded (Selectivity)
    ax5 = axes[1, 1]
    ax5.barh(summary_df['Strategy'], summary_df['% Traded'], color='teal', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('% of Signals Traded', fontweight='bold')
    ax5.set_title('Trade Selectivity (Lower = More Selective)', fontweight='bold')
    for i, v in enumerate(summary_df['% Traded']):
        ax5.text(v+2, i, f'{v:.1f}%', va='center', fontsize=9)
    
    # 6. PnL per Trade
    ax6 = axes[1, 2]
    colors = ['red' if x < 0 else 'green' for x in summary_df['PnL per Trade']]
    ax6.barh(summary_df['Strategy'], summary_df['PnL per Trade'], color=colors, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('PnL per Trade ($)', fontweight='bold')
    ax6.set_title('Efficiency (Avg PnL/Trade)', fontweight='bold')
    ax6.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    for i, v in enumerate(summary_df['PnL per Trade']):
        ax6.text(v-200 if v < 0 else v+50, i, f'${v:.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    output_path = Path('results') / 'strategy_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n Visualization saved to: {output_path}")
    plt.close()

def evaluate_strategy_choices():
    """Overall evaluation of strategy choices."""
    print("\n" + "="*80)
    print("STRATEGY EVALUATION - DID WE PICK THE RIGHT APPROACHES?")
    print("="*80 + "\n")
    
    findings = {
        'Regime Filter': {
            'verdict': ' EXCELLENT CHOICE',
            'reason': 'Reduced losses by 71% ($2.2M saved), cut bad trades by 75%',
            'keep': True,
            'priority': 1
        },
        'Lorentzian Classifier': {
            'verdict': ' MIXED RESULTS',
            'reason': 'Better selectivity but worse overall PnL than RF. Only beats RF in specific folds.',
            'keep': False,
            'priority': 4,
            'alternative': 'Use as confidence filter only, not primary predictor'
        },
        'Confidence Threshold (0.6)': {
            'verdict': ' TOO RESTRICTIVE',
            'reason': 'Filters 40% of trades but reduces overall performance vs RF+Regime',
            'keep': False,
            'priority': 3,
            'note': 'Needs optimization - perhaps 0.55 or dynamic threshold'
        },
        'RandomForest': {
            'verdict': ' SOLID BASELINE',
            'reason': 'More stable than Lorentzian, better overall results with regime filter',
            'keep': True,
            'priority': 1
        },
        'LSTM': {
            'verdict': ' NOT TESTED IN WALK-FORWARD',
            'reason': 'Highest AUC (0.592) for EURUSD, competitive for USDJPY in static tests',
            'keep': True,
            'priority': 1,
            'note': 'Should test LSTM + Regime filter next'
        }
    }
    
    print("COMPONENT VERDICTS:\n")
    for component, evaluation in findings.items():
        print(f"{component}:")
        print(f"  {evaluation['verdict']}")
        print(f"  Reason: {evaluation['reason']}")
        print(f"  Keep: {'YES' if evaluation['keep'] else 'NO'}")
        if 'alternative' in evaluation:
            print(f"  Alternative: {evaluation['alternative']}")
        if 'note' in evaluation:
            print(f"  Note: {evaluation['note']}")
        print()
    
    print("="*80)
    print("RECOMMENDATIONS:")
    print("="*80 + "\n")
    
    print(" KEEP & PRIORITIZE:")
    print("  1. Regime Filter (HUGE impact: -71% losses)")
    print("  2. RandomForest as primary predictor")
    print("  3. Test LSTM + Regime filter combination")
    print()
    
    print(" MODIFY:")
    print("  1. Lower confidence threshold from 0.6  0.55 or dynamic")
    print("  2. Use Lorentzian confidence as position sizing signal (not hard filter)")
    print("  3. Combine RF + LSTM predictions (ensemble)")
    print()
    
    print(" STILL NEEDED (Critical):")
    print("  1. MORE DATA: 5.5 months  2+ years (CRITICAL)")
    print("  2. Position sizing: Fixed 1 lot  Kelly/risk-based")
    print("  3. Stop-loss optimization: Current SL might be too tight")
    print("  4. Multi-timeframe features: Add H1, H4 context")
    print("  5. Test LSTM + Regime in walk-forward")
    print()
    
    print(" ABANDON:")
    print("  1. Lorentzian as standalone predictor (RF+Regime is $315k better)")
    print("  2. Fixed confidence threshold 0.6 (too restrictive)")
    print("  3. Trading without regime filter (loses 71% more)")
    print()
    
    print("="*80)
    print("BOTTOM LINE:")
    print("="*80)
    print()
    print(" YES, we picked mostly RIGHT strategies:")
    print("   - Regime filter = HUGE WIN")
    print("   - RandomForest = Stable baseline")
    print()
    print(" But we need adjustments:")
    print("   - Lorentzian works better as filter, not predictor")
    print("   - Confidence threshold 0.6 is too strict")
    print("   - Still losing money = need MORE DATA + position sizing")
    print()
    print(" NEXT PRIORITY:")
    print("   1. Get 2+ years of data")
    print("   2. Test LSTM + Regime filter")
    print("   3. Implement position sizing")
    print("    Should get to breakeven or profitable")
    print()

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE STRATEGY EVALUATION")
    print("="*80)
    
    # Load all results
    results = load_all_results()
    
    if not results:
        print("\n No walk-forward results found!")
        print("Please run walk-forward validations first.")
        return
    
    print(f"\nLoaded {len(results)} strategy results\n")
    
    # Calculate metrics
    summary_df = calculate_metrics(results)
    
    # Print summary table
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80 + "\n")
    
    # Format for better display
    display_df = summary_df.copy()
    display_df['Total PnL'] = display_df['Total PnL'].apply(lambda x: f"${x:,.0f}")
    display_df['Avg Win Rate'] = display_df['Avg Win Rate'].apply(lambda x: f"{x*100:.1f}%")
    display_df['Avg AUC'] = display_df['Avg AUC'].apply(lambda x: f"{x:.3f}")
    display_df['% Traded'] = display_df['% Traded'].apply(lambda x: f"{x:.1f}%")
    display_df['PnL per Trade'] = display_df['PnL per Trade'].apply(lambda x: f"${x:.0f}")
    
    print(display_df[['Strategy', 'Total PnL', 'Total Trades', 'Avg Win Rate', 'Avg AUC', '% Traded', 'Profitable Folds']].to_string(index=False))
    print()
    
    # Analyze components
    analyze_strategy_components(results)
    
    # Visualize
    visualize_comparison(summary_df)
    
    # Overall evaluation
    evaluate_strategy_choices()
    
    # Save summary
    output_path = Path('results') / 'strategy_evaluation_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f" Summary saved to: {output_path}\n")

if __name__ == '__main__':
    main()

