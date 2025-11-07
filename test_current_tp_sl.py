"""
Monte Carlo simulation with current TP/SL settings
SL: 30 pips, TP: 50 pips
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from make_dataset import fetch_mt5_data
import MetaTrader5 as mt5
from indicators import add_indicators

# Configuration matching your current bot
STOP_LOSS_PIPS = 30
TAKE_PROFIT_PIPS = 50
BUY_THRESHOLD = 0.70
SELL_THRESHOLD = 0.30
SYMBOL = "USDJPY.sim"

def calculate_pnl(entry_price, exit_price, position_type, volume=0.5):
    """Calculate P&L for a trade"""
    if position_type == 'BUY':
        pips = (exit_price - entry_price) * 10000
    else:  # SELL
        pips = (entry_price - exit_price) * 10000
    
    # For USDJPY: 1 pip = ~$10 per lot
    pnl = pips * 10 * volume
    return pnl, pips

def simulate_trade_outcome(entry_price, position_type, future_prices):
    """
    Simulate if trade hits SL or TP
    Returns: ('SL'|'TP'|'OPEN', pnl, pips)
    """
    sl_distance = STOP_LOSS_PIPS / 10000
    tp_distance = TAKE_PROFIT_PIPS / 10000
    
    if position_type == 'BUY':
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
        
        for price in future_prices:
            if price <= sl_price:
                return 'SL', -STOP_LOSS_PIPS * 10 * 0.5, -STOP_LOSS_PIPS
            elif price >= tp_price:
                return 'TP', TAKE_PROFIT_PIPS * 10 * 0.5, TAKE_PROFIT_PIPS
    else:  # SELL
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance
        
        for price in future_prices:
            if price >= sl_price:
                return 'SL', -STOP_LOSS_PIPS * 10 * 0.5, -STOP_LOSS_PIPS
            elif price <= tp_price:
                return 'TP', TAKE_PROFIT_PIPS * 10 * 0.5, TAKE_PROFIT_PIPS
    
    # Trade still open
    final_pnl, final_pips = calculate_pnl(entry_price, future_prices[-1], position_type)
    return 'OPEN', final_pnl, final_pips

def run_monte_carlo(n_simulations=100):
    """Run Monte Carlo simulation with current settings"""
    
    print("\n" + "="*70)
    print("MONTE CARLO SIMULATION - CURRENT TP/SL SETTINGS")
    print("="*70)
    print(f"Symbol: {SYMBOL}")
    print(f"Stop Loss: {STOP_LOSS_PIPS} pips")
    print(f"Take Profit: {TAKE_PROFIT_PIPS} pips")
    print(f"Buy Threshold: {BUY_THRESHOLD}")
    print(f"Sell Threshold: {SELL_THRESHOLD}")
    print(f"Simulations: {n_simulations}")
    print("="*70 + "\n")
    
    # Load model
    model_path = Path(f"models/USDJPY_ensemble_oos.pkl")
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    model = joblib.load(model_path)
    print(f"Loaded model: {model_path.name}\n")
    
    # Load data
    print("Loading data...")
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
    
    df = fetch_mt5_data(SYMBOL, timeframe=mt5.TIMEFRAME_M5, bars=15000)
    mt5.shutdown()
    
    if df is None or len(df) == 0:
        print("No data loaded")
        return
    
    print(f"Loaded {len(df)} bars\n")
    
    # Build features
    print("Building features...")
    df = add_indicators(df)
    
    # Create target
    df['target'] = df['close'].shift(-5) - df['close']
    df['target_binary'] = (df['target'] > 0.001).astype(int)
    
    df = df.dropna()
    print(f"Clean data: {len(df)} rows\n")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['target', 'target_binary']]
    X = df[feature_cols]
    
    # Run simulations
    all_results = []
    
    for sim in range(n_simulations):
        # Random train/test split
        split_idx = np.random.randint(int(len(df)*0.6), int(len(df)*0.9))
        X_test = X.iloc[split_idx:]
        
        # Predict
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # Simulate trades
        trades = []
        for i in range(len(X_test) - 100):  # Leave room for future prices
            prob = probabilities[i]
            entry_price = df.iloc[split_idx + i]['close']
            future_prices = df.iloc[split_idx + i + 1:split_idx + i + 100]['close'].values
            
            if prob > BUY_THRESHOLD:
                outcome, pnl, pips = simulate_trade_outcome(entry_price, 'BUY', future_prices)
                trades.append({
                    'type': 'BUY',
                    'prob': prob,
                    'entry': entry_price,
                    'outcome': outcome,
                    'pnl': pnl,
                    'pips': pips
                })
            elif prob < SELL_THRESHOLD:
                outcome, pnl, pips = simulate_trade_outcome(entry_price, 'SELL', future_prices)
                trades.append({
                    'type': 'SELL',
                    'prob': prob,
                    'entry': entry_price,
                    'outcome': outcome,
                    'pnl': pnl,
                    'pips': pips
                })
        
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Calculate metrics
            total_pnl = trades_df['pnl'].sum()
            tp_trades = (trades_df['outcome'] == 'TP').sum()
            sl_trades = (trades_df['outcome'] == 'SL').sum()
            open_trades = (trades_df['outcome'] == 'OPEN').sum()
            
            win_rate = tp_trades / (tp_trades + sl_trades) * 100 if (tp_trades + sl_trades) > 0 else 0
            
            all_results.append({
                'simulation': sim + 1,
                'total_trades': len(trades_df),
                'tp_trades': tp_trades,
                'sl_trades': sl_trades,
                'open_trades': open_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl_per_trade': total_pnl / len(trades_df)
            })
        
        if (sim + 1) % 10 == 0:
            print(f"Completed {sim + 1}/{n_simulations} simulations...")
    
    # Analyze results
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70 + "\n")
    
    print(f"Average Trades per Simulation: {results_df['total_trades'].mean():.1f}")
    print(f"Average TP Hits: {results_df['tp_trades'].mean():.1f}")
    print(f"Average SL Hits: {results_df['sl_trades'].mean():.1f}")
    print(f"Average Open Trades: {results_df['open_trades'].mean():.1f}")
    print(f"\nWin Rate: {results_df['win_rate'].mean():.2f}% (+/- {results_df['win_rate'].std():.2f}%)")
    print(f"\nTotal P&L: ${results_df['total_pnl'].mean():.2f} (+/- ${results_df['total_pnl'].std():.2f})")
    print(f"Avg P&L per Trade: ${results_df['avg_pnl_per_trade'].mean():.2f}")
    
    print("\n" + "="*70)
    print("EXPECTANCY ANALYSIS")
    print("="*70 + "\n")
    
    avg_tp = results_df['tp_trades'].mean()
    avg_sl = results_df['sl_trades'].mean()
    total_closed = avg_tp + avg_sl
    
    if total_closed > 0:
        win_rate_actual = avg_tp / total_closed * 100
        loss_rate = avg_sl / total_closed * 100
        
        avg_win = TAKE_PROFIT_PIPS * 10 * 0.5  # $250
        avg_loss = STOP_LOSS_PIPS * 10 * 0.5   # $150
        
        expectancy = (win_rate_actual/100 * avg_win) - (loss_rate/100 * avg_loss)
        
        print(f"Win Rate (closed trades only): {win_rate_actual:.2f}%")
        print(f"Loss Rate: {loss_rate:.2f}%")
        print(f"\nAverage Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"\nExpectancy per Trade: ${expectancy:.2f}")
        
        if expectancy > 0:
            print(f"\nVERDICT: PROFITABLE SYSTEM ✓")
            print(f"Expected to make ${expectancy:.2f} per trade on average")
        else:
            print(f"\nVERDICT: UNPROFITABLE SYSTEM ✗")
            print(f"Expected to lose ${abs(expectancy):.2f} per trade on average")
            print(f"Recommendation: Reduce TP or increase win rate")
    
    # Save results
    results_df.to_csv('results/current_tp_sl_monte_carlo.csv', index=False)
    print(f"\nResults saved to: results/current_tp_sl_monte_carlo.csv")
    
    print("\n" + "="*70)
    
    return results_df

if __name__ == "__main__":
    results = run_monte_carlo(100)
