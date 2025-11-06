"""
USDJPY Trading Bot for Ubuntu/Linux using Oanda API
Runs 24/7 without MT5 - optimized for Azure VM "Sabyr"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.oanda_client import OandaClient
from build_features_enhanced import add_multi_timeframe_features
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_dir / 'ubuntu_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


class TradingBot:
    def __init__(self):
        self.client = OandaClient()
        self.symbol = os.getenv('SYMBOL', 'USDJPY')
        self.model = None
        self.running = False
        
        # Trading parameters
        self.probability_threshold = 0.80  # 80% confidence to enter
        self.max_positions = 3
        self.position_size_lots = 0.01  # Start small
        
        # Risk management
        self.risk_per_trade = 0.01  # 1% of account
        self.max_daily_loss = 0.03  # 3% daily loss limit
        self.max_drawdown = 0.10  # 10% max drawdown
        self.stop_loss_atr_multiplier = 2.0
        self.take_profit_atr_multiplier = 3.0
        
        # State tracking
        self.initial_balance = None
        self.peak_balance = None
        self.daily_start_balance = None
        self.daily_pnl = 0.0
        self.last_day = None
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load ensemble model"""
        try:
            model_path = Path(__file__).parent.parent / 'models' / 'USDJPY_ensemble_oos.pkl'
            self.model = joblib.load(model_path)
            logging.info(f" Model loaded from {model_path}")
        except Exception as e:
            logging.error(f" Failed to load model: {e}")
            raise
    
    def fetch_and_prepare_data(self):
        """Fetch multi-timeframe data and calculate features"""
        try:
            # Fetch data from all timeframes
            df_m15 = self.client.get_bars(self.symbol, 'M15', 300)
            df_m30 = self.client.get_bars(self.symbol, 'M30', 300)
            df_h1 = self.client.get_bars(self.symbol, 'H1', 300)
            df_h4 = self.client.get_bars(self.symbol, 'H4', 300)
            
            if any(df is None for df in [df_m15, df_m30, df_h1, df_h4]):
                logging.warning("Failed to fetch one or more timeframes")
                return None
            
            # Rename columns to match feature engineering expectations
            for df in [df_m15, df_m30, df_h1, df_h4]:
                if 'tick_volume' in df.columns:
                    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            # Add multi-timeframe features
            df_m15 = add_multi_timeframe_features(df_m15, df_m30, df_h1, df_h4)
            
            if df_m15 is None or len(df_m15) == 0:
                logging.warning("Feature calculation returned empty DataFrame")
                return None
            
            # Drop NaN rows
            df_m15.dropna(inplace=True)
            
            if len(df_m15) == 0:
                logging.warning("No data after dropping NaN")
                return None
            
            logging.info(f" Fetched and prepared {len(df_m15)} M15 bars with {len(df_m15.columns)} features")
            return df_m15
            
        except Exception as e:
            logging.error(f" Error fetching/preparing data: {e}")
            return None
    
    def get_prediction(self, df):
        """Get model prediction for the latest bar"""
        try:
            # Get the latest row
            latest = df.iloc[[-1]].copy()
            
            # Select features expected by model
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi_14', 'atr_14', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'macd', 'macd_signal', 'macd_hist',
                'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'stoch_k', 'stoch_d',
                'adx', 'cci',
                'price_change', 'volume_change',
                # M30 features
                'M30_rsi_14', 'M30_atr_14', 'M30_adx', 'M30_macd',
                'M30_bb_width', 'M30_sma_20',
                # H1 features
                'H1_rsi_14', 'H1_atr_14', 'H1_adx', 'H1_macd',
                'H1_bb_width', 'H1_sma_20', 'H1_trend',
                # H4 features
                'H4_rsi_14', 'H4_atr_14', 'H4_adx', 'H4_macd',
                'H4_bb_width', 'H4_sma_20'
            ]
            
            # Filter only available features
            available_features = [col for col in feature_columns if col in latest.columns]
            X = latest[available_features]
            
            # Get prediction probability
            y_prob = self.model.predict_proba(X)[0]
            
            return {
                'buy_probability': y_prob[1],
                'sell_probability': y_prob[0],
                'features': X.iloc[0].to_dict(),
                'prediction': 'buy' if y_prob[1] > y_prob[0] else 'sell'
            }
            
        except Exception as e:
            logging.error(f" Prediction error: {e}")
            return None
    
    def calculate_position_size(self, account_balance, atr, stop_loss_pips):
        """Calculate position size based on risk"""
        try:
            # Risk amount in dollars
            risk_amount = account_balance * self.risk_per_trade
            
            # Position size in lots
            # For USDJPY: 1 lot = 100,000 units, 1 pip = 0.01
            pip_value = 10  # For 1 standard lot of USDJPY
            
            # Calculate lots based on risk
            lots = risk_amount / (stop_loss_pips * pip_value)
            
            # Round to 2 decimals (0.01 lot minimum)
            lots = max(0.01, round(lots, 2))
            
            # Cap at max position size
            lots = min(lots, self.position_size_lots * 2)  # Max 2x base size
            
            return lots
            
        except Exception as e:
            logging.error(f" Position size calculation error: {e}")
            return self.position_size_lots
    
    def check_risk_limits(self):
        """Check if risk limits are breached"""
        try:
            account = self.client.get_account_info()
            if account is None:
                return False
            
            balance = account['balance']
            
            # Initialize tracking on first run
            if self.initial_balance is None:
                self.initial_balance = balance
                self.peak_balance = balance
                self.daily_start_balance = balance
                self.last_day = datetime.now().day
            
            # Update peak balance
            if balance > self.peak_balance:
                self.peak_balance = balance
            
            # Reset daily tracking
            current_day = datetime.now().day
            if current_day != self.last_day:
                self.daily_start_balance = balance
                self.daily_pnl = 0.0
                self.last_day = current_day
                logging.info(f" New trading day started. Starting balance: ${balance:,.2f}")
            
            # Calculate daily P&L
            self.daily_pnl = balance - self.daily_start_balance
            daily_pnl_pct = (self.daily_pnl / self.daily_start_balance) * 100
            
            # Check daily loss limit
            if self.daily_pnl < -(self.daily_start_balance * self.max_daily_loss):
                logging.warning(f" Daily loss limit breached: {daily_pnl_pct:.2f}% (limit: {self.max_daily_loss*100:.1f}%)")
                return False
            
            # Check max drawdown
            drawdown = (self.peak_balance - balance) / self.peak_balance
            if drawdown > self.max_drawdown:
                logging.warning(f" Max drawdown breached: {drawdown*100:.2f}% (limit: {self.max_drawdown*100:.1f}%)")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f" Risk check error: {e}")
            return True  # Allow trading if check fails
    
    def run(self):
        """Main trading loop"""
        logging.info("=" * 80)
        logging.info("USDJPY TRADING BOT STARTED (Ubuntu/Oanda API)")
        logging.info(f"VM: Sabyr (westus3) | Symbol: {self.symbol}")
        logging.info(f"Threshold: {self.probability_threshold} | Max Positions: {self.max_positions}")
        logging.info("=" * 80)
        
        # Initialize Oanda connection
        if not self.client.initialize():
            logging.error(" Failed to connect to Oanda")
            return
        
        # Get initial account info
        account = self.client.get_account_info()
        if account:
            logging.info(f" Account Balance: ${account['balance']:,.2f}")
            logging.info(f" Account Equity: ${account['equity']:,.2f}")
        
        self.running = True
        iteration = 0
        
        try:
            while self.running:
                iteration += 1
                
                # Check risk limits
                if not self.check_risk_limits():
                    logging.warning(" Risk limits breached - stopping trading")
                    break
                
                # Fetch and prepare data
                df = self.fetch_and_prepare_data()
                if df is None:
                    logging.warning(" Failed to fetch data, retrying in 60s...")
                    time.sleep(60)
                    continue
                
                # Get prediction
                prediction = self.get_prediction(df)
                if prediction is None:
                    logging.warning(" Failed to get prediction, retrying in 60s...")
                    time.sleep(60)
                    continue
                
                # Current price and features
                current_price = df['close'].iloc[-1]
                atr = df['atr_14'].iloc[-1]
                buy_prob = prediction['buy_probability']
                h1_trend = prediction['features'].get('H1_trend', 0)
                
                # Get current positions
                positions = self.client.get_open_positions(self.symbol)
                position_count = len(positions)
                
                # Log current state
                logging.info(
                    f"[{iteration}] Price: {current_price:.3f} | "
                    f"BuyProb: {buy_prob:.3f} | "
                    f"H1Trend: {h1_trend:.0f} | "
                    f"ATR: {atr:.4f} | "
                    f"Positions: {position_count}"
                )
                
                # Trading logic
                if position_count < self.max_positions:
                    # Enter long if high buy probability and uptrend
                    if buy_prob >= self.probability_threshold and h1_trend > 0:
                        # Calculate position size
                        account = self.client.get_account_info()
                        if account:
                            stop_loss_pips = atr * self.stop_loss_atr_multiplier * 100  # Convert to pips
                            lots = self.calculate_position_size(
                                account['balance'],
                                atr,
                                stop_loss_pips
                            )
                            
                            # Calculate SL and TP
                            sl_price = current_price - (atr * self.stop_loss_atr_multiplier)
                            tp_price = current_price + (atr * self.take_profit_atr_multiplier)
                            
                            logging.info(f" BUY SIGNAL: Prob={buy_prob:.3f}, Lots={lots}, SL={sl_price:.3f}, TP={tp_price:.3f}")
                            
                            # Place order
                            result = self.client.place_market_order(
                                symbol=self.symbol,
                                order_type='buy',
                                volume=lots,
                                sl=sl_price,
                                tp=tp_price,
                                comment=f"Bot_Buy_P{buy_prob:.2f}"
                            )
                            
                            if result.get('success'):
                                logging.info(f" BUY order executed at {result.get('price', 0):.3f}")
                            else:
                                logging.error(f" BUY order failed: {result.get('error')}")
                    
                    # Enter short if low buy probability (high sell probability) and downtrend
                    elif buy_prob <= (1 - self.probability_threshold) and h1_trend < 0:
                        account = self.client.get_account_info()
                        if account:
                            stop_loss_pips = atr * self.stop_loss_atr_multiplier * 100
                            lots = self.calculate_position_size(
                                account['balance'],
                                atr,
                                stop_loss_pips
                            )
                            
                            # Calculate SL and TP
                            sl_price = current_price + (atr * self.stop_loss_atr_multiplier)
                            tp_price = current_price - (atr * self.take_profit_atr_multiplier)
                            
                            logging.info(f" SELL SIGNAL: Prob={1-buy_prob:.3f}, Lots={lots}, SL={sl_price:.3f}, TP={tp_price:.3f}")
                            
                            # Place order
                            result = self.client.place_market_order(
                                symbol=self.symbol,
                                order_type='sell',
                                volume=lots,
                                sl=sl_price,
                                tp=tp_price,
                                comment=f"Bot_Sell_P{1-buy_prob:.2f}"
                            )
                            
                            if result.get('success'):
                                logging.info(f" SELL order executed at {result.get('price', 0):.3f}")
                            else:
                                logging.error(f" SELL order failed: {result.get('error')}")
                
                # Show position status if any
                if positions:
                    total_profit = sum(p['profit'] for p in positions)
                    logging.info(f" Open positions: {position_count} | Total P&L: ${total_profit:.2f}")
                
                # Sleep 60 seconds before next iteration
                time.sleep(60)
                
        except KeyboardInterrupt:
            logging.info(" Bot stopped by user (Ctrl+C)")
        except Exception as e:
            logging.error(f" Unexpected error: {e}", exc_info=True)
        finally:
            # Cleanup
            self.client.shutdown()
            logging.info("=" * 80)
            logging.info("BOT STOPPED")
            logging.info("=" * 80)


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()

