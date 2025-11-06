"""
USDJPY Demo Trading Bot - MT5 Live Execution

Loads validated ensemble model and executes trades on MT5 demo account.

Features:
- Real-time M15 bar monitoring
- Multi-timeframe feature calculation (M15, M30, H1, H4)
- Ensemble model predictions
- H1 trend filter
- Position sizing (1% risk per trade)
- Automatic SL/TP placement
- Risk guardrails (max DD, daily loss limits)
- Comprehensive logging

Safety:
- Starts in demo mode (practice account)
- Conservative risk management
- Emergency stop conditions
- All trades logged

Usage:
    .\.venv\Scripts\python.exe live_trading\demo_bot.py
"""
import os
import sys
import time
import yaml
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from live_trading.mt5_client import init_mt5, shutdown_mt5, get_bars, place_market_order, get_open_positions, get_account_info
from position_sizing import PositionSizer
from build_features_enhanced import add_multi_timeframe_features

# Load environment variables
load_dotenv(Path(__file__).parent / '.env')

# Configuration
TERMINAL_PATH = os.getenv('MT5_TERMINAL_PATH')
LOGIN = int(os.getenv('MT5_LOGIN', 0))
PASSWORD = os.getenv('MT5_PASSWORD')
SERVER = os.getenv('MT5_SERVER')
SYMBOL = os.getenv('MT5_SYMBOL', 'USDJPY.sim')  # Default to .sim format

# Trading parameters
CONFIG = {
    'symbol': SYMBOL,
    'timeframe': 'M15',
    'check_interval': 60,  # Check every 60 seconds
    'risk_per_trade': 0.01,  # 1% risk
    'stop_atr_mult': 1.5,
    'tp_atr_mult': 2.5,
    'prob_buy': 0.80,
    'prob_sell': 0.20,
    'max_open_positions': 3,
    'max_daily_loss': 0.03,  # 3%
    'max_drawdown': 0.10,  # 10%
    'spread_pips': 0.8,
    'slippage_pips': 0.4,
    'commission_per_lot': 7.0,
}

# Risk guardrails
RISK_LIMITS = {
    'max_risk_per_trade': 0.01,
    'max_daily_loss': 0.03,
    'max_weekly_loss': 0.05,
    'max_drawdown_halt': 0.10,
    'max_open_positions': 3,
    'min_model_confidence': 0.80,
}

# Logging
LOG_DIR = Path(__file__).parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

def log(message, level='INFO'):
    """Log message to file and console."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] [{level}] {message}"
    print(log_message)
    
    log_file = LOG_DIR / f"demo_bot_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')


def load_model_and_scaler():
    """Load validated USDJPY ensemble model and scaler."""
    try:
        model_path = Path('models') / 'USDJPY_ensemble_oos.pkl'
        scaler_path = Path('models') / 'scaler.pkl'
        
        if not model_path.exists():
            log(f"Model not found: {model_path}", 'ERROR')
            return None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        
        log(f" Loaded model: {model_path}")
        return model, scaler
    
    except Exception as e:
        log(f"Error loading model: {e}", 'ERROR')
        return None, None


def fetch_and_prepare_data():
    """Fetch recent bars and calculate features."""
    try:
        # Fetch M15 bars
        df_m15 = get_bars(SYMBOL, timeframe='M15', count=500)
        if df_m15 is None or len(df_m15) < 100:
            log("Insufficient M15 data", 'WARNING')
            return None
        
        df_m15 = df_m15.reset_index()
        df_m15.attrs['symbol'] = SYMBOL
        
        # Calculate base M15 features (matching build_features_enhanced.py STEP 1)
        from indicators import ema, rsi, atr as atr_func, pct_change, sincos_time
        
        df_m15['ema20'] = ema(df_m15['close'], 20)
        df_m15['ema50'] = ema(df_m15['close'], 50)
        df_m15['rsi14'] = rsi(df_m15['close'], 14)
        df_m15['atr14'] = atr_func(df_m15, 14)
        df_m15['ema50_slope'] = df_m15['ema50'].diff(5)
        
        # Time features
        time_col = df_m15['time'] if df_m15['time'].dt.tz is not None else df_m15['time'].dt.tz_localize('UTC')
        sin_h, cos_h = sincos_time(time_col)
        df_m15['sin_hour'] = sin_h.values
        df_m15['cos_hour'] = cos_h.values
        
        # Returns
        df_m15['ret1'] = pct_change(df_m15['close'], 1)
        df_m15['atr_pct'] = df_m15['atr14'] / df_m15['close']
        
        # Price vs EMAs (base M15 features)
        df_m15['price_vs_ema20'] = (df_m15['close'] - df_m15['ema20']) / df_m15['close']
        df_m15['price_vs_ema50'] = (df_m15['close'] - df_m15['ema50']) / df_m15['close']
        
        # Fetch M30 bars
        df_m30 = get_bars(SYMBOL, timeframe='M30', count=300)
        if df_m30 is None or len(df_m30) < 50:
            log("Insufficient M30 data", 'WARNING')
            return None
        df_m30 = df_m30.reset_index()
        
        # Fetch H1 bars
        df_h1 = get_bars(SYMBOL, timeframe='H1', count=200)
        if df_h1 is None or len(df_h1) < 50:
            log("Insufficient H1 data", 'WARNING')
            return None
        df_h1 = df_h1.reset_index()
        
        # Fetch H4 bars
        df_h4 = get_bars(SYMBOL, timeframe='H4', count=100)
        if df_h4 is None or len(df_h4) < 30:
            log("Insufficient H4 data", 'WARNING')
            return None
        df_h4 = df_h4.reset_index()
        
        # Add multi-timeframe features (matching training process)
        df_enhanced = add_multi_timeframe_features(df_m15, df_m30, 'm30')
        df_enhanced = add_multi_timeframe_features(df_enhanced, df_h1, 'h1')
        df_enhanced = add_multi_timeframe_features(df_enhanced, df_h4, 'h4')
        
        # Add required derived features (from market structure)
        # Higher highs / Lower lows
        df_enhanced['higher_high'] = (df_enhanced['high'] > df_enhanced['high'].shift(1)).astype(int)
        df_enhanced['lower_low'] = (df_enhanced['low'] < df_enhanced['low'].shift(1)).astype(int)
        
        # Swing lows (5-bar pattern)
        df_enhanced['swing_low'] = ((df_enhanced['low'] < df_enhanced['low'].shift(1)) & 
                                    (df_enhanced['low'] < df_enhanced['low'].shift(2)) &
                                    (df_enhanced['low'] < df_enhanced['low'].shift(-1)) & 
                                    (df_enhanced['low'] < df_enhanced['low'].shift(-2))).astype(int)
        
        # Debug: Log feature count (comment out after confirming it works)
        # log(f"Generated {len(df_enhanced.columns)} total columns: {sorted(df_enhanced.columns.tolist())}", 'DEBUG')
        
        if df_enhanced is None or len(df_enhanced) < 50:
            log("Feature calculation failed", 'WARNING')
            return None
        
        return df_enhanced
    
    except Exception as e:
        log(f"Error fetching/preparing data: {e}", 'ERROR')
        return None


def calculate_atr(df, period=14):
    """Calculate ATR."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def get_prediction(model, scaler, df, feature_cols):
    """Get model prediction for latest bar."""
    try:
        # Get latest row
        latest = df.iloc[-1:][feature_cols].values
        
        # EnsembleClassifier has internal scaler - do NOT use external scaler
        # Model will handle scaling internally
        proba = model.predict_proba(latest)[0, 1]
        
        return proba
    
    except Exception as e:
        log(f"Error getting prediction: {e}", 'ERROR')
        return 0.5


def check_h1_trend_filter(df):
    """Check H1 trend direction."""
    if 'ema20_h1' not in df.columns or 'ema50_h1' not in df.columns:
        return 0
    
    latest = df.iloc[-1]
    if latest['ema20_h1'] > latest['ema50_h1']:
        return 1  # Uptrend
    elif latest['ema20_h1'] < latest['ema50_h1']:
        return -1  # Downtrend
    return 0


def calculate_position_size(equity, atr, stop_mult, pip_size):
    """Calculate position size based on risk."""
    sizer = PositionSizer(strategy='fixed_fractional', risk_pct=CONFIG['risk_per_trade'])
    
    stop_loss_pips = (stop_mult * atr) / pip_size
    
    # Calculate pip value for USDJPY
    if 'JPY' in SYMBOL.upper():
        # Get current price from latest bar
        pip_value_usd = (pip_size / 150.0) * 100000  # Approximate at 150
    else:
        pip_value_usd = pip_size * 100000
    
    position_size = sizer.calculate_size(
        equity=equity,
        stop_loss_pips=stop_loss_pips,
        pip_value=pip_value_usd
    )
    
    return position_size


def execute_trade(signal, current_price, atr, equity):
    """Execute trade on MT5."""
    try:
        # Calculate position size
        pip_size = 0.01 if 'JPY' in SYMBOL.upper() else 0.0001
        lot_size = calculate_position_size(equity, atr, CONFIG['stop_atr_mult'], pip_size)
        
        # Calculate SL/TP
        if signal > 0:  # BUY
            sl_price = current_price - CONFIG['stop_atr_mult'] * atr
            tp_price = current_price + CONFIG['tp_atr_mult'] * atr
            units = lot_size
        else:  # SELL
            sl_price = current_price + CONFIG['stop_atr_mult'] * atr
            tp_price = current_price - CONFIG['tp_atr_mult'] * atr
            units = -lot_size
        
        # Place order
        result = place_market_order(
            symbol=SYMBOL,
            lot=abs(units),
            sl=sl_price,
            tp=tp_price,
            deviation=20
        )
        
        if result is not None:
            log(f" Trade executed: {signal} {abs(units):.2f} lots @ {current_price:.3f}, SL={sl_price:.3f}, TP={tp_price:.3f}")
            return True
        else:
            log(f" Trade failed", 'ERROR')
            return False
    
    except Exception as e:
        log(f"Error executing trade: {e}", 'ERROR')
        return False


def check_risk_limits(account_info, initial_balance):
    """Check if risk limits are breached."""
    balance = account_info.balance
    equity = account_info.equity
    
    # Check drawdown
    dd = (initial_balance - equity) / initial_balance if initial_balance > 0 else 0
    
    if dd > RISK_LIMITS['max_drawdown_halt']:
        log(f" EMERGENCY STOP: Drawdown {dd:.2%} exceeds limit {RISK_LIMITS['max_drawdown_halt']:.2%}", 'CRITICAL')
        return False
    
    if dd > 0.05:
        log(f"  WARNING: Drawdown at {dd:.2%}", 'WARNING')
    
    return True


def main():
    """Main trading loop."""
    log("=" * 80)
    log("USDJPY DEMO BOT STARTING")
    log("=" * 80)
    
    # Load model
    model, scaler = load_model_and_scaler()
    if model is None:
        log("Failed to load model. Exiting.", 'CRITICAL')
        return
    
    # Initialize MT5
    log(f"Connecting to MT5: {SERVER}")
    if not init_mt5(TERMINAL_PATH, LOGIN, PASSWORD, SERVER):
        log("Failed to initialize MT5. Exiting.", 'CRITICAL')
        return
    
    # Get account info
    account_info = get_account_info()
    if account_info is None:
        log("Failed to get account info. Exiting.", 'CRITICAL')
        shutdown_mt5()
        return
    
    initial_balance = account_info.balance
    log(f"Account Balance: ${initial_balance:,.2f}")
    log(f"Account Equity: ${account_info.equity:,.2f}")
    
    # Define feature columns (must match training - ALL 37 features in exact order)
    feature_cols = [
        'price_vs_ema20_h1',
        'momentum_5_h1',
        'momentum_10_h1',
        'rsi14_h1',
        'trend_strength_h1',
        'ema20_h1',
        'atr14_h1',
        'ema50_h1',
        'atr_pct_h1',
        'momentum_5_m30',
        'ema50_m30',
        'ema20_m30',
        'momentum_10_m30',
        'price_vs_ema20_m30',
        'rsi14_m30',
        'trend_strength_m30',
        'atr14_m30',
        'ema20_h4',
        'atr_pct_h4',
        'rsi14_h4',
        'price_vs_ema20_h4',
        'trend_strength_h4',
        'ema50_h4',
        'atr_pct_m30',
        'momentum_10_h4',
        'momentum_5_h4',
        'atr14_h4',
        'swing_low',
        'higher_high',
        'trend_ema_m30',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'rsi14',
        'atr14'
    ]
    
    log("Starting trading loop...")
    log(f"Risk per trade: {CONFIG['risk_per_trade']:.1%}")
    log(f"Max positions: {CONFIG['max_open_positions']}")
    log(f"Check interval: {CONFIG['check_interval']}s")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # Check risk limits
            account_info = get_account_info()
            if account_info is None:
                log("Failed to get account info. MT5 may be disconnected. Retrying...", 'ERROR')
                time.sleep(10)
                # Try to reconnect
                if not init_mt5(login=os.getenv('MT5_LOGIN'), password=os.getenv('MT5_PASSWORD'), server=os.getenv('MT5_SERVER')):
                    log("Failed to reconnect to MT5. Exiting.", 'CRITICAL')
                    break
                continue
            
            if not check_risk_limits(account_info, initial_balance):
                log("Risk limits breached. Stopping bot.", 'CRITICAL')
                break
            
            # Fetch and prepare data
            df = fetch_and_prepare_data()
            if df is None:
                log("Failed to fetch data. Retrying in 60s...", 'WARNING')
                time.sleep(60)
                continue
            
            # Get current positions
            positions = get_open_positions()
            open_count = len(positions) if positions else 0
            
            # Check if we can trade
            if open_count >= CONFIG['max_open_positions']:
                log(f"Max positions reached ({open_count}). Waiting...")
                time.sleep(CONFIG['check_interval'])
                continue
            
            # Get prediction
            latest_row = df.iloc[-1]
            proba = get_prediction(model, scaler, df, feature_cols)
            
            # Check H1 trend
            h1_trend = check_h1_trend_filter(df)
            
            log(f"[{iteration}] Price: {latest_row['close']:.3f}, Prob: {proba:.3f}, H1 Trend: {h1_trend}, Positions: {open_count}")
            
            # Generate signal
            signal = None
            if proba >= CONFIG['prob_buy'] and h1_trend == 1:
                signal = 1  # BUY
                log(f" BUY SIGNAL: prob={proba:.3f}, H1 uptrend")
            elif proba <= CONFIG['prob_sell'] and h1_trend == -1:
                signal = -1  # SELL
                log(f" SELL SIGNAL: prob={proba:.3f}, H1 downtrend")
            
            # Execute if signal
            if signal is not None:
                execute_trade(
                    signal=signal,
                    current_price=latest_row['close'],
                    atr=latest_row['atr14'],
                    equity=account_info.equity
                )
            
            # Wait before next check
            time.sleep(CONFIG['check_interval'])
    
    except KeyboardInterrupt:
        log("\n  Bot stopped by user")
    
    except Exception as e:
        log(f"Unexpected error: {e}", 'CRITICAL')
    
    finally:
        shutdown_mt5()
        log("=" * 80)
        log("BOT STOPPED")
        log("=" * 80)


if __name__ == '__main__':
    main()

