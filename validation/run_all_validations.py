"""
Comprehensive Multi-Pair Validation Script

Runs all validation tests (Walk-Forward, OOS, Monte Carlo) for multiple currency pairs.
Fixes:
1. Font warning - use compatible fonts
2. Honest Monte Carlo CI - show actual distribution, not just mean
3. Realistic transaction costs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from walk_forward_h1_filter import run_walk_forward_with_filters
from validation.out_of_sample_test import test_out_of_sample
from validation.monte_carlo_simulation import run_full_monte_carlo_analysis

# Set font to avoid warnings
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style('whitegrid')


def run_comprehensive_validation(symbols, stop_mult=1.5, tp_mult=2.5, n_simulations=10000):
    """Run complete validation suite for multiple pairs."""
    
    print("="*80)
    print("COMPREHENSIVE MULTI-PAIR VALIDATION")
    print("="*80)
    print(f"\nTesting: {', '.join(symbols)}")
    print(f"Configuration: SL={stop_mult}ATR, TP={tp_mult}ATR")
    print(f"Monte Carlo Simulations: {n_simulations:,}\n")
    
    all_results = []
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING: {symbol}")
        print(f"{'='*80}\n")
        
        results = {
            'symbol': symbol,
            'stop_mult': stop_mult,
            'tp_mult': tp_mult
        }
        
        # 1. Check/build enhanced features
        feat_path = Path('features') / f'{symbol}_features_enhanced.csv'
        if not feat_path.exists():
            print(f"Building enhanced features for {symbol}...")
            import subprocess
            subprocess.run([
                sys.executable, 'build_features_enhanced.py', 
                '--symbol', symbol
            ])
        
        # 2. Walk-Forward Validation
        print(f"\n{''*80}")
        print("STEP 1: WALK-FORWARD VALIDATION")
        print(f"{''*80}\n")
        
        try:
            wf_results = run_walk_forward_with_filters(
                symbol=symbol,
                n_splits=4,
                use_h1_filter=True,
                stop_mult=stop_mult,
                tp_mult=tp_mult,
                max_features=30
            )
            
            if wf_results is not None:
                results['wf_total_pnl'] = wf_results['pnl'].sum()
                results['wf_total_trades'] = wf_results['trades'].sum()
                results['wf_avg_win_rate'] = wf_results['win_rate'].mean()
                results['wf_profitable_folds'] = (wf_results['pnl'] > 0).sum()
                results['wf_total_folds'] = len(wf_results)
        except Exception as e:
            print(f"ERROR in walk-forward: {e}")
            results['wf_total_pnl'] = None
        
        # 3. Out-of-Sample Test
        print(f"\n{''*80}")
        print("STEP 2: OUT-OF-SAMPLE VALIDATION")
        print(f"{''*80}\n")
        
        try:
            oos_results = test_out_of_sample(
                symbol=symbol,
                start_date=None,
                stop_mult=stop_mult,
                tp_mult=tp_mult
            )
            
            if oos_results is not None:
                results['oos_pnl'] = oos_results['pnl']
                results['oos_win_rate'] = oos_results['win_rate']
                results['oos_auc'] = oos_results['auc']
                results['oos_trades'] = oos_results['trades']
                results['oos_max_dd'] = oos_results['max_dd_pct']
        except Exception as e:
            print(f"ERROR in OOS test: {e}")
            results['oos_pnl'] = None
        
        # 4. Monte Carlo Simulation
        print(f"\n{''*80}")
        print("STEP 3: MONTE CARLO SIMULATION")
        print(f"{''*80}\n")
        
        try:
            mc_results = run_full_monte_carlo_analysis(
                symbol=symbol,
                config_name=f'sl{stop_mult}_tp{tp_mult}',
                n_simulations=n_simulations
            )
            
            if mc_results is not None:
                results['mc_mean_pnl'] = mc_results['mean_pnl']
                results['mc_ci_95_lower'] = mc_results['ci_95_lower']
                results['mc_ci_95_upper'] = mc_results['ci_95_upper']
                results['mc_prob_profit'] = mc_results['prob_profit']
                results['mc_risk_of_ruin'] = mc_results['risk_of_ruin']
                results['mc_mean_max_dd'] = mc_results['mean_max_dd']
        except Exception as e:
            print(f"ERROR in Monte Carlo: {e}")
            results['mc_mean_pnl'] = None
        
        all_results.append(results)
    
    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_file = Path('validation/results') / f'multi_pair_summary_sl{stop_mult}_tp{tp_mult}.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("MULTI-PAIR VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Pair':<10} {'WF PnL':<12} {'OOS PnL':<12} {'MC Mean':<12} {'WF WR%':<8} {'Status':<10}")
    print(""*80)
    
    for _, row in results_df.iterrows():
        wf_pnl = row['wf_total_pnl'] if pd.notna(row['wf_total_pnl']) else 0
        oos_pnl = row['oos_pnl'] if pd.notna(row['oos_pnl']) else 0
        mc_pnl = row['mc_mean_pnl'] if pd.notna(row['mc_mean_pnl']) else 0
        wf_wr = row['wf_avg_win_rate'] if pd.notna(row['wf_avg_win_rate']) else 0
        
        status = "" if wf_pnl > 0 and oos_pnl > 0 else ""
        
        print(f"{row['symbol']:<10} ${wf_pnl:>10,.0f} ${oos_pnl:>10,.0f} ${mc_pnl:>10,.0f} {wf_wr:>6.1f}% {status:<10}")
    
    print(""*80)
    
    # Grand totals
    total_wf = results_df['wf_total_pnl'].sum()
    total_oos = results_df['oos_pnl'].sum()
    total_mc = results_df['mc_mean_pnl'].sum()
    avg_wr = results_df['wf_avg_win_rate'].mean()
    
    print(f"{'TOTAL':<10} ${total_wf:>10,.0f} ${total_oos:>10,.0f} ${total_mc:>10,.0f} {avg_wr:>6.1f}%")
    
    profitable_pairs = ((results_df['wf_total_pnl'] > 0) & (results_df['oos_pnl'] > 0)).sum()
    print(f"\nProfitable Pairs: {profitable_pairs}/{len(symbols)}")
    
    print(f"\nResults saved to: {output_file}")
    
    # Create comparison visualization
    create_comparison_visualization(results_df)
    
    return results_df


def create_comparison_visualization(results_df):
    """Create multi-pair comparison charts."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Pair Validation Comparison', fontsize=16, fontweight='bold')
    
    symbols = results_df['symbol'].values
    
    # 1. P&L Comparison
    ax = axes[0, 0]
    x = np.arange(len(symbols))
    width = 0.25
    
    wf_pnl = results_df['wf_total_pnl'].fillna(0)
    oos_pnl = results_df['oos_pnl'].fillna(0)
    mc_pnl = results_df['mc_mean_pnl'].fillna(0)
    
    ax.bar(x - width, wf_pnl, width, label='Walk-Forward', color='steelblue', alpha=0.8)
    ax.bar(x, oos_pnl, width, label='Out-of-Sample', color='green', alpha=0.8)
    ax.bar(x + width, mc_pnl, width, label='Monte Carlo', color='coral', alpha=0.8)
    
    ax.set_xlabel('Currency Pair', fontweight='bold')
    ax.set_ylabel('P&L ($)', fontweight='bold')
    ax.set_title('P&L Comparison Across Tests', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(symbols)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 2. Win Rate Comparison
    ax = axes[0, 1]
    win_rates = results_df['wf_avg_win_rate'].fillna(0)
    oos_wr = results_df['oos_win_rate'].fillna(0)
    
    x_pos = np.arange(len(symbols))
    ax.bar(x_pos - 0.2, win_rates, 0.4, label='Walk-Forward', color='steelblue', alpha=0.8)
    ax.bar(x_pos + 0.2, oos_wr, 0.4, label='Out-of-Sample', color='green', alpha=0.8)
    
    ax.axhline(y=70, color='red', linestyle='--', linewidth=1.5, label='70% Target')
    ax.set_xlabel('Currency Pair', fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontweight='bold')
    ax.set_title('Win Rate Comparison', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(symbols)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Monte Carlo Confidence Intervals
    ax = axes[1, 0]
    mc_mean = results_df['mc_mean_pnl'].fillna(0)
    mc_lower = results_df['mc_ci_95_lower'].fillna(0)
    mc_upper = results_df['mc_ci_95_upper'].fillna(0)
    
    y_pos = np.arange(len(symbols))
    ax.barh(y_pos, mc_mean, color='steelblue', alpha=0.7, height=0.4)
    
    # Add error bars for CI
    errors_lower = mc_mean - mc_lower
    errors_upper = mc_upper - mc_mean
    ax.errorbar(mc_mean, y_pos, xerr=[errors_lower, errors_upper], 
                fmt='none', ecolor='black', capsize=5, capthick=2)
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(symbols)
    ax.set_xlabel('P&L ($)', fontweight='bold')
    ax.set_title('Monte Carlo: Mean P&L with 95% CI', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Risk Metrics
    ax = axes[1, 1]
    risk_data = {
        'Prob Profit': results_df['mc_prob_profit'].fillna(0),
        'Risk of Ruin': results_df['mc_risk_of_ruin'].fillna(0),
        'Mean Max DD': results_df['mc_mean_max_dd'].fillna(0)
    }
    
    x_pos = np.arange(len(symbols))
    width = 0.25
    
    for i, (metric, values) in enumerate(risk_data.items()):
        offset = (i - 1) * width
        color = ['green', 'red', 'orange'][i]
        ax.bar(x_pos + offset, values, width, label=metric, color=color, alpha=0.7)
    
    ax.set_xlabel('Currency Pair', fontweight='bold')
    ax.set_ylabel('Value (%)', fontweight='bold')
    ax.set_title('Risk Metrics Comparison', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(symbols)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path('validation/visualizations') / 'multi_pair_comparison.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nComparison visualization saved to: {output_file}")
    
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive multi-pair validation')
    parser.add_argument('--symbols', nargs='+', default=['USDJPY', 'EURUSD', 'GBPUSD'],
                        help='Currency pairs to test')
    parser.add_argument('--stop_mult', type=float, default=1.5, help='Stop loss multiplier')
    parser.add_argument('--tp_mult', type=float, default=2.5, help='Take profit multiplier')
    parser.add_argument('--n_simulations', type=int, default=10000, help='Monte Carlo simulations')
    
    args = parser.parse_args()
    
    results = run_comprehensive_validation(
        symbols=args.symbols,
        stop_mult=args.stop_mult,
        tp_mult=args.tp_mult,
        n_simulations=args.n_simulations
    )

