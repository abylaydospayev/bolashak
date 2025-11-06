"""
USDJPY Demo Trading Bot - MT5 Live Execution with Advanced Risk Management

Features:
- Real-time M15 bar monitoring
- Multi-timeframe feature calculation (M15, M30, H1, H4)
- Ensemble model predictions
- H1 trend filter
- ADVANCED RISK MANAGEMENT:
  * Maximum 3 concurrent positions
  * 5-minute cooldown between trades
  * Stop loss/take profit on every trade
  * Daily loss limits
  * Signal strength validation
- Comprehensive logging

Usage:
    python live_trading\demo_bot_with_risk.py
"""
import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from live_trading.mt5_client import init_mt5, shutdown_mt5, get_bars, place_market_order, get_open_positions, get_account_info
from live_trading.risk_manager import RiskManager
from position_sizing import PositionSizer
from build_features_enhanced import add_multi_timeframe_features

# Load environment variables
load_dotenv(Path(__file__).parent / '.env')

# Configuration
TERMINAL_PATH = os.getenv('MT5_TERMINAL_PATH')
LOGIN = int(os.getenv('MT5_LOGIN', 0))
PASSWORD = os.getenv('MT5_PASSWORD')
SERVER = os.getenv('MT5_SERVER')
SYMBOL = os.getenv('MT5_SYMBOL', 'USDJPY.sim')
LOT_SIZE = float(os.getenv('MT5_LOT_SIZE', 0.1))

# Trading parameters
CONFIG = {
    'symbol': SYMBOL,
    'timeframe': 'M15',
    'check_interval': 60,  # Check every 60 seconds
    'risk_per_trade': 0.01,  # 1% risk
    'stop_atr_mult': 1.5,
    'tp_atr_mult': 2.5,
    'max_drawdown': 0.10,  # 10%
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
        
        log(f"✅ Loaded model: {model_path}")
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
        
        # Calculate base M15 features
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
        
        # Price vs EMAs
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
        
        # Add multi-timeframe features
        df_enhanced = add_multi_timeframe_features(df_m15, df_m30, 'm30')
        df_enhanced = add_multi_timeframe_features(df_enhanced, df_h1, 'h1')
        df_enhanced = add_multi_timeframe_features(df_enhanced, df_h4, 'h4')
        
        # Add required derived features
        df_enhanced['higher_high'] = (df_enhanced['high'] > df_enhanced['high'].shift(1)).astype(int)
        df_enhanced['lower_low'] = (df_enhanced['low'] < df_enhanced['low'].shift(1)).astype(int)
        
        # Swing lows
        df_enhanced['swing_low'] = ((df_enhanced['low'] < df_enhanced['low'].shift(1)) & 
                                    (df_enhanced['low'] < df_enhanced['low'].shift(2)) &
                                    (df_enhanced['low'] < df_enhanced['low'].shift(-1)) & 
                                    (df_enhanced['low'] < df_enhanced['low'].shift(-2))).astype(int)
        
        if df_enhanced is None or len(df_enhanced) < 50:
            log("Feature calculation failed", 'WARNING')
            return None
        
        return df_enhanced
    
    except Exception as e:
        log(f"Error fetching/preparing data: {e}", 'ERROR')
        return None


def get_prediction(model, scaler, df, feature_cols):
    """Get model prediction for latest bar."""
    try:
        latest = df.iloc[-1:][feature_cols].values
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


def execute_trade(signal_type, current_price, risk_manager):
    """Execute trade on MT5 with risk management."""
    try:
        # Calculate SL/TP using risk manager
        sl_price, tp_price = risk_manager.calculate_sl_tp(current_price, signal_type)
        
        # Determine direction
        if signal_type == 'BUY':
            order_type = 0  # Buy
        else:
            order_type = 1  # Sell
        
        # Place order
        result = place_market_order(
            symbol=SYMBOL,
            lot=LOT_SIZE,
            order_type=order_type,
            sl=sl_price,
            tp=tp_price,
            deviation=20
        )
        
        if result is not None:
            # Update last trade time
            risk_manager.update_last_trade_time()
            log(f"✅ {signal_type} executed: {LOT_SIZE} lots @ {current_price:.3f}, SL={sl_price:.3f}, TP={tp_price:.3f}")
            return True
        else:
            log(f"❌ {signal_type} failed", 'ERROR')
            return False
    
    except Exception as e:
        log(f"Error executing trade: {e}", 'ERROR')
        return False


def check_drawdown_limit(account_info, initial_balance):
    """Check if drawdown exceeds limit."""
    balance = account_info.balance
    equity = account_info.equity
    
    dd = (initial_balance - equity) / initial_balance if initial_balance > 0 else 0
    
    if dd > CONFIG['max_drawdown']:
        log(f"⛔ EMERGENCY STOP: Drawdown {dd:.2%} exceeds limit {CONFIG['max_drawdown']:.2%}", 'CRITICAL')
        return False
    
    if dd > 0.05:
        log(f"⚠️  WARNING: Drawdown at {dd:.2%}", 'WARNING')
    
    return True


def main():
    """Main trading loop with risk management."""
    log("=" * 80)
    log("USDJPY DEMO BOT STARTING (WITH RISK MANAGEMENT)")
    log("=" * 80)
    
    # Load model
    model, scaler = load_model_and_scaler()
    if model is None:
        log("Failed to load model. Exiting.", 'CRITICAL')
        return
    
    # Initialize Risk Manager
    risk_manager = RiskManager()
    log(f"✅ Risk Manager initialized:")
    log(f"   - Max positions: {risk_manager.max_positions}")
    log(f"   - Min interval: {risk_manager.min_interval}s")
    log(f"   - Stop loss: {risk_manager.stop_loss_pips} pips")
    log(f"   - Take profit: {risk_manager.take_profit_pips} pips")
    log(f"   - Max daily loss: ${risk_manager.max_daily_loss}")
    log(f"   - Buy threshold: {risk_manager.buy_threshold}")
    log(f"   - Sell threshold: {risk_manager.sell_threshold}")
    
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
        'price_vs_ema20_h1', 'momentum_5_h1', 'momentum_10_h1', 'rsi14_h1', 'trend_strength_h1',
        'ema20_h1', 'atr14_h1', 'ema50_h1', 'atr_pct_h1', 'momentum_5_m30', 'ema50_m30',
        'ema20_m30', 'momentum_10_m30', 'price_vs_ema20_m30', 'rsi14_m30', 'trend_strength_m30',
        'atr14_m30', 'ema20_h4', 'atr_pct_h4', 'rsi14_h4', 'price_vs_ema20_h4', 'trend_strength_h4',
        'ema50_h4', 'atr_pct_m30', 'momentum_10_h4', 'momentum_5_h4', 'atr14_h4', 'swing_low',
        'higher_high', 'trend_ema_m30', 'open', 'high', 'low', 'close', 'volume', 'rsi14', 'atr14'
    ]
    
    log("Starting trading loop...")
    log(f"Lot size: {LOT_SIZE}")
    log(f"Check interval: {CONFIG['check_interval']}s")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # Check drawdown limit
            account_info = get_account_info()
            if account_info is None:
                log("Failed to get account info. Retrying...", 'ERROR')
                time.sleep(10)
                continue
            
            if not check_drawdown_limit(account_info, initial_balance):
                log("Drawdown limit breached. Stopping bot.", 'CRITICAL')
                break
            
            # Fetch and prepare data
            df = fetch_and_prepare_data()
            if df is None:
                log("Failed to fetch data. Retrying in 60s...", 'WARNING')
                time.sleep(60)
                continue
            
            # Get prediction
            latest_row = df.iloc[-1]
            proba = get_prediction(model, scaler, df, feature_cols)
            
            # Check H1 trend
            h1_trend = check_h1_trend_filter(df)
            
            # Get position summary from risk manager
            import MetaTrader5 as mt5
            position_summary = risk_manager.get_position_summary(mt5)
            open_count = len(mt5.positions_get(symbol=SYMBOL)) if mt5.positions_get(symbol=SYMBOL) else 0
            
            log(f"[{iteration}] Price: {latest_row['close']:.3f}, Prob: {proba:.3f}, H1: {h1_trend}, Pos: {open_count}")
            if iteration % 10 == 0:  # Log position summary every 10 iterations
                log(position_summary)
            
            # Check if we can trade (risk management gate)
            if not risk_manager.can_open_trade(mt5):
                time.sleep(CONFIG['check_interval'])
                continue
            
            # Generate signal
            signal = None
            signal_type = None
            
            if proba >= risk_manager.buy_threshold and h1_trend == 1:
                signal_type = 'BUY'
                # Validate signal strength
                if risk_manager.is_signal_strong(proba, signal_type):
                    signal = 1
                    log(f"🔵 BUY SIGNAL: prob={proba:.3f}, H1 uptrend, STRONG")
                else:
                    log(f"⚪ BUY signal weak: prob={proba:.3f}, skipping")
            
            elif proba <= risk_manager.sell_threshold and h1_trend == -1:
                signal_type = 'SELL'
                # Validate signal strength
                if risk_manager.is_signal_strong(proba, signal_type):
                    signal = -1
                    log(f"🔴 SELL SIGNAL: prob={proba:.3f}, H1 downtrend, STRONG")
                else:
                    log(f"⚪ SELL signal weak: prob={proba:.3f}, skipping")
            
            # Execute if signal is strong enough
            if signal is not None and signal_type is not None:
                execute_trade(
                    signal_type=signal_type,
                    current_price=latest_row['close'],
                    risk_manager=risk_manager
                )
            
            # Wait before next check
            time.sleep(CONFIG['check_interval'])
    
    except KeyboardInterrupt:
        log("\n⏹️  Bot stopped by user")
    
    except Exception as e:
        log(f"Unexpected error: {e}", 'CRITICAL')
        import traceback
        log(traceback.format_exc(), 'CRITICAL')
    
    finally:
        shutdown_mt5()
        log("=" * 80)
        log("BOT STOPPED")
        log("=" * 80)


if __name__ == '__main__':
    main()
