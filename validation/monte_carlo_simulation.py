"""
Monte Carlo Simulation for Trading System

Runs thousands of simulations with randomized trade order to:
1. Calculate probability distributions of returns
2. Estimate maximum drawdown statistics
3. Compute risk of ruin
4. Generate confidence intervals for expected performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_trade_history(symbol, config_name='sl1.5_tp2.5'):
    """Load trades from walk-forward results."""
    result_file = Path('results') / f'{symbol}_walkforward_h1filter_{config_name}.csv'
    
    if not result_file.exists():
        print(f"ERROR: Results not found at {result_file}")
        return None
    
    results_df = pd.read_csv(result_file)
    return results_df


def extract_individual_trades(symbol, config_name='sl1.5_tp2.5'):
    """
    Extract individual trade outcomes from backtest results.
    Since we don't have individual trades, we'll simulate based on aggregate stats.
    """
    results_df = load_trade_history(symbol, config_name)
    
    if results_df is None:
        return None
    
    all_trades = []
    
    for _, row in results_df.iterrows():
        n_trades = int(row['trades'])
        win_rate = row['win_rate'] / 100.0
        pnl = row['pnl']
        
        if n_trades == 0:
            continue
        
        n_wins = int(row.get('wins', n_trades * win_rate))
        n_losses = n_trades - n_wins
        
        # Estimate avg win and avg loss from PnL
        # PnL = (n_wins * avg_win) - (n_losses * avg_loss)
        # Assume R:R ratio of 1:1.67 (TP=2.5, SL=1.5)
        # avg_win = 1.67 * avg_loss
        
        if n_losses > 0:
            # PnL = n_wins * 1.67 * X - n_losses * X
            # PnL = X * (n_wins * 1.67 - n_losses)
            X = pnl / (n_wins * 1.67 - n_losses) if (n_wins * 1.67 - n_losses) != 0 else 0
            avg_loss = abs(X) if X < 0 else abs(X)
            avg_win = 1.67 * avg_loss
        else:
            avg_win = pnl / n_wins if n_wins > 0 else 0
            avg_loss = avg_win / 1.67
        
        # Create individual trade records
        for _ in range(n_wins):
            all_trades.append(avg_win)
        for _ in range(n_losses):
            all_trades.append(-avg_loss)
    
    return np.array(all_trades)


def run_monte_carlo_simulation(trades, initial_capital=100000, n_simulations=10000, 
                               risk_of_ruin_threshold=0.5):
    """
    Run Monte Carlo simulation by randomizing trade order.
    
    Parameters:
    -----------
    trades : array
        Array of individual trade P&Ls
    initial_capital : float
        Starting capital
    n_simulations : int
        Number of simulations to run
    risk_of_ruin_threshold : float
        Threshold for considering account "ruined" (0.5 = 50% loss)
    
    Returns:
    --------
    dict with simulation results
    """
    
    print("="*80)
    print("MONTE CARLO SIMULATION")
    print("="*80)
    print(f"\nTrades: {len(trades):,}")
    print(f"Initial Capital: ${initial_capital:,}")
    print(f"Simulations: {n_simulations:,}")
    print(f"Risk of Ruin Threshold: {risk_of_ruin_threshold*100:.0f}%\n")
    
    # Statistics on trades
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    
    print(f"Trade Statistics:")
    print(f"  Win Rate: {len(wins)/len(trades)*100:.1f}%")
    print(f"  Avg Win: ${wins.mean():.2f}" if len(wins) > 0 else "  Avg Win: N/A")
    print(f"  Avg Loss: ${losses.mean():.2f}" if len(losses) > 0 else "  Avg Loss: N/A")
    print(f"  Profit Factor: {abs(wins.sum()/losses.sum()):.2f}" if len(losses) > 0 and losses.sum() != 0 else "  Profit Factor: N/A")
    print(f"  Expectancy: ${trades.mean():.2f} per trade\n")
    
    # Run simulations
    final_balances = []
    max_drawdowns = []
    ruin_count = 0
    
    print(f"Running {n_simulations:,} simulations...")
    
    for sim in range(n_simulations):
        # Randomize trade order
        shuffled_trades = np.random.permutation(trades)
        
        # Calculate equity curve
        equity = initial_capital
        peak_equity = equity
        max_dd = 0.0
        
        for trade in shuffled_trades:
            equity += trade
            
            # Track peak and drawdown
            if equity > peak_equity:
                peak_equity = equity
            
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            max_dd = max(max_dd, dd)
            
            # Check for ruin
            if equity < initial_capital * (1 - risk_of_ruin_threshold):
                ruin_count += 1
                break
        
        final_balances.append(equity)
        max_drawdowns.append(max_dd)
        
        if (sim + 1) % 1000 == 0:
            print(f"  Completed {sim + 1:,}/{n_simulations:,} simulations...")
    
    final_balances = np.array(final_balances)
    max_drawdowns = np.array(max_drawdowns)
    
    # Calculate statistics
    final_pnl = final_balances - initial_capital
    
    results = {
        'n_trades': len(trades),
        'n_simulations': n_simulations,
        'initial_capital': initial_capital,
        
        # Final P&L statistics
        'mean_pnl': final_pnl.mean(),
        'median_pnl': np.median(final_pnl),
        'std_pnl': final_pnl.std(),
        'min_pnl': final_pnl.min(),
        'max_pnl': final_pnl.max(),
        
        # Confidence intervals (95%)
        'ci_95_lower': np.percentile(final_pnl, 2.5),
        'ci_95_upper': np.percentile(final_pnl, 97.5),
        'ci_90_lower': np.percentile(final_pnl, 5),
        'ci_90_upper': np.percentile(final_pnl, 95),
        
        # Drawdown statistics
        'mean_max_dd': max_drawdowns.mean() * 100,
        'median_max_dd': np.median(max_drawdowns) * 100,
        'max_max_dd': max_drawdowns.max() * 100,
        'dd_95_percentile': np.percentile(max_drawdowns, 95) * 100,
        
        # Risk metrics
        'prob_profit': (final_pnl > 0).sum() / n_simulations * 100,
        'prob_loss_10pct': (final_pnl < -initial_capital * 0.10).sum() / n_simulations * 100,
        'prob_loss_20pct': (final_pnl < -initial_capital * 0.20).sum() / n_simulations * 100,
        'risk_of_ruin': ruin_count / n_simulations * 100,
        
        # Return distribution
        'final_balances': final_balances,
        'max_drawdowns': max_drawdowns
    }
    
    return results


def print_monte_carlo_results(results):
    """Print formatted Monte Carlo results."""
    
    print("\n" + "="*80)
    print("MONTE CARLO RESULTS")
    print("="*80)
    
    print(f"\n FINAL P&L DISTRIBUTION:")
    print(f"  Mean:              ${results['mean_pnl']:>12,.0f}")
    print(f"  Median:            ${results['median_pnl']:>12,.0f}")
    print(f"  Std Deviation:     ${results['std_pnl']:>12,.0f}")
    print(f"  Min:               ${results['min_pnl']:>12,.0f}")
    print(f"  Max:               ${results['max_pnl']:>12,.0f}")
    
    print(f"\n CONFIDENCE INTERVALS:")
    print(f"  95% CI:            ${results['ci_95_lower']:>12,.0f} to ${results['ci_95_upper']:>12,.0f}")
    print(f"  90% CI:            ${results['ci_90_lower']:>12,.0f} to ${results['ci_90_upper']:>12,.0f}")
    
    print(f"\n DRAWDOWN STATISTICS:")
    print(f"  Mean Max DD:       {results['mean_max_dd']:>11.2f}%")
    print(f"  Median Max DD:     {results['median_max_dd']:>11.2f}%")
    print(f"  Worst Case DD:     {results['max_max_dd']:>11.2f}%")
    print(f"  95th Percentile:   {results['dd_95_percentile']:>11.2f}%")
    
    print(f"\n  RISK METRICS:")
    print(f"  Probability of Profit:      {results['prob_profit']:>6.1f}%")
    print(f"  Probability of -10% Loss:   {results['prob_loss_10pct']:>6.1f}%")
    print(f"  Probability of -20% Loss:   {results['prob_loss_20pct']:>6.1f}%")
    print(f"  Risk of Ruin (>50% loss):   {results['risk_of_ruin']:>6.1f}%")
    
    # Assessment
    print(f"\n{'='*80}")
    print("ASSESSMENT:")
    print("="*80)
    
    if results['ci_95_lower'] > 0:
        print(" 95% CONFIDENCE: System is profitable")
    elif results['ci_90_lower'] > 0:
        print("  90% CONFIDENCE: System is profitable (moderate confidence)")
    else:
        print(" RISK: System may not be consistently profitable")
    
    if results['risk_of_ruin'] < 1.0:
        print(" SAFETY: Risk of ruin is very low (<1%)")
    elif results['risk_of_ruin'] < 5.0:
        print("  CAUTION: Risk of ruin is moderate (1-5%)")
    else:
        print(" DANGER: Risk of ruin is high (>5%)")
    
    if results['dd_95_percentile'] < 10.0:
        print(" DRAWDOWN: Expected drawdown is acceptable (<10%)")
    elif results['dd_95_percentile'] < 20.0:
        print("  DRAWDOWN: Expected drawdown is moderate (10-20%)")
    else:
        print(" DRAWDOWN: Expected drawdown is high (>20%)")
    
    print("="*80 + "\n")


def visualize_monte_carlo(results, symbol, output_dir='validation/visualizations'):
    """Create visualizations of Monte Carlo results."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Monte Carlo Simulation Results - {symbol}', fontsize=16, fontweight='bold')
    
    # 1. Final P&L Distribution
    ax = axes[0, 0]
    final_pnl = results['final_balances'] - results['initial_capital']
    ax.hist(final_pnl, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(results['mean_pnl'], color='red', linestyle='--', linewidth=2, label=f"Mean: ${results['mean_pnl']:,.0f}")
    ax.axvline(results['ci_95_lower'], color='orange', linestyle='--', linewidth=1.5, label=f"95% CI Lower: ${results['ci_95_lower']:,.0f}")
    ax.axvline(results['ci_95_upper'], color='orange', linestyle='--', linewidth=1.5, label=f"95% CI Upper: ${results['ci_95_upper']:,.0f}")
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Final P&L ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Final P&L', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Maximum Drawdown Distribution
    ax = axes[0, 1]
    ax.hist(results['max_drawdowns'] * 100, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(results['mean_max_dd'], color='red', linestyle='--', linewidth=2, label=f"Mean: {results['mean_max_dd']:.2f}%")
    ax.axvline(results['dd_95_percentile'], color='darkred', linestyle='--', linewidth=1.5, label=f"95th %ile: {results['dd_95_percentile']:.2f}%")
    ax.set_xlabel('Maximum Drawdown (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Maximum Drawdown', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Cumulative Probability of P&L
    ax = axes[1, 0]
    sorted_pnl = np.sort(final_pnl)
    cumulative_prob = np.arange(1, len(sorted_pnl) + 1) / len(sorted_pnl) * 100
    ax.plot(sorted_pnl, cumulative_prob, linewidth=2, color='darkblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Breakeven')
    ax.axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Final P&L ($)', fontsize=12)
    ax.set_ylabel('Cumulative Probability (%)', fontsize=12)
    ax.set_title('Cumulative Distribution of Returns', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text for key probabilities
    prob_profit = results['prob_profit']
    ax.text(0.05, 0.95, f"Prob(Profit) = {prob_profit:.1f}%", 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Risk Metrics Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
MONTE CARLO SUMMARY
{'='*40}

Simulations: {results['n_simulations']:,}
Trades per simulation: {results['n_trades']:,}

EXPECTED RETURN:
  Mean P&L: ${results['mean_pnl']:,.0f}
  Median P&L: ${results['median_pnl']:,.0f}
  
CONFIDENCE INTERVALS (95%):
  Lower: ${results['ci_95_lower']:,.0f}
  Upper: ${results['ci_95_upper']:,.0f}
  
DRAWDOWN RISK:
  Mean Max DD: {results['mean_max_dd']:.2f}%
  95th Percentile: {results['dd_95_percentile']:.2f}%
  Worst Case: {results['max_max_dd']:.2f}%
  
PROBABILITY ANALYSIS:
  Profit: {results['prob_profit']:.1f}%
  Loss >10%: {results['prob_loss_10pct']:.1f}%
  Loss >20%: {results['prob_loss_20pct']:.1f}%
  Ruin (>50%): {results['risk_of_ruin']:.1f}%
    """
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_dir) / f'{symbol}_monte_carlo_simulation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    plt.close()


def run_full_monte_carlo_analysis(symbol, config_name='sl1.5_tp2.5', n_simulations=10000):
    """Run complete Monte Carlo analysis."""
    
    print(f"\nRunning Monte Carlo Analysis for {symbol}...")
    print(f"Configuration: {config_name}\n")
    
    # Extract trades
    trades = extract_individual_trades(symbol, config_name)
    
    if trades is None or len(trades) == 0:
        print(f"ERROR: Could not extract trades for {symbol}")
        return None
    
    # Run simulation
    results = run_monte_carlo_simulation(trades, n_simulations=n_simulations)
    
    # Print results
    print_monte_carlo_results(results)
    
    # Visualize
    visualize_monte_carlo(results, symbol)
    
    # Save results to CSV
    results_summary = {
        'symbol': symbol,
        'config': config_name,
        'n_simulations': results['n_simulations'],
        'n_trades': results['n_trades'],
        'mean_pnl': results['mean_pnl'],
        'median_pnl': results['median_pnl'],
        'ci_95_lower': results['ci_95_lower'],
        'ci_95_upper': results['ci_95_upper'],
        'mean_max_dd_pct': results['mean_max_dd'],
        'dd_95_percentile': results['dd_95_percentile'],
        'prob_profit': results['prob_profit'],
        'risk_of_ruin': results['risk_of_ruin']
    }
    
    results_df = pd.DataFrame([results_summary])
    output_file = f"validation/results/{symbol}_monte_carlo_{config_name}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Monte Carlo simulation')
    parser.add_argument('--symbol', type=str, default='USDJPY', help='Symbol to test')
    parser.add_argument('--config', type=str, default='sl1.5_tp2.5', help='Config name')
    parser.add_argument('--n_simulations', type=int, default=10000, help='Number of simulations')
    
    args = parser.parse_args()
    
    run_full_monte_carlo_analysis(
        symbol=args.symbol,
        config_name=args.config,
        n_simulations=args.n_simulations
    )

