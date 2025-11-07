"""
Test and Train Adaptive Model
=============================
Comprehensive script for:
1. Training the adaptive model on historical data
2. Running ablation studies (Baseline -> +LSTM -> +Online -> +Meta)
3. Prequential evaluation (test-then-learn)
4. Shadow mode testing
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from dotenv import load_dotenv
import pickle
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.adaptive_model import AdaptiveModel
from live_trading.mt5_functions import initialize_mt5


class AdaptiveModelTester:
    """Test and evaluate the adaptive model"""
    
    def __init__(self, symbol='USDJPY', lookback_days=90):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.model = AdaptiveModel()
        self.results = {
            'train': {},
            'prequential': {},
            'ablation': {}
        }
        
    def fetch_historical_data(self, days=90):
        """Fetch historical M5 data for training"""
        print(f"\n[FETCH] Getting {days} days of {self.symbol} M5 data...")
        
        if not mt5.initialize():
            print(f"[ERROR] MT5 initialization failed: {mt5.last_error()}")
            return None
            
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # Fetch M5 bars
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M5, from_date, to_date)
        mt5.shutdown()
        
        if rates is None or len(rates) == 0:
            print(f"[ERROR] No data fetched")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        print(f"[SUCCESS] Fetched {len(df)} M5 bars from {df['time'].min()} to {df['time'].max()}")
        return df
        
    def prepare_features(self, df):
        """Prepare features from OHLCV data"""
        print("\n[PREP] Calculating features...")
        
        df = df.copy()
        
        # Price features
        df['pct_return'] = (df['close'] - df['open']) / df['open']
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        
        # Technical indicators
        # EMA
        for period in [20, 50]:
            df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'price_vs_ema{period}'] = (df['close'] - df[f'ema{period}']) / df[f'ema{period}']
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr14'] = df['tr'].rolling(window=14).mean()
        df['atr_pct'] = df['atr14'] / df['close']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume
        df['volume_sma20'] = df['tick_volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_sma20']
        
        # Target: Next bar direction (1 = up, 0 = down)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop NaN rows
        df = df.dropna()
        
        print(f"[SUCCESS] Prepared {len(df)} samples with {len(df.columns)} features")
        return df
        
    def train_initial_model(self, df, train_size=0.7):
        """Train initial model on historical data"""
        print("\n" + "="*60)
        print("PHASE 1: INITIAL TRAINING")
        print("="*60)
        
        # Split train/test
        split_idx = int(len(df) * train_size)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"\n[SPLIT] Train: {len(train_df)} samples | Test: {len(test_df)} samples")
        
        # Select feature columns
        feature_cols = [
            'open', 'high', 'low', 'close', 'tick_volume',
            'pct_return', 'high_low_range',
            'ema20', 'ema50', 'price_vs_ema20', 'price_vs_ema50',
            'atr14', 'atr_pct', 'rsi14',
            'macd', 'macd_signal', 'macd_hist',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_position',
            'volume_sma20', 'volume_ratio'
        ]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values
        
        print(f"\n[TRAIN] Training on {len(X_train)} samples...")
        
        # Train model
        self.model.train(X_train, y_train)
        
        # Evaluate on test set
        print(f"\n[TEST] Evaluating on {len(X_test)} samples...")
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        
        predictions = []
        probabilities = []
        
        for i in range(len(X_test)):
            pred, proba = self.model.predict(X_test[i:i+1])
            predictions.append(pred)
            probabilities.append(proba)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        auc = roc_auc_score(y_test, probabilities)
        
        # Precision at different thresholds
        precision_at_07 = precision_score(y_test, (probabilities >= 0.7).astype(int), zero_division=0)
        precision_at_03 = precision_score(y_test, (probabilities <= 0.3).astype(int), zero_division=0)
        
        self.results['train'] = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'precision_at_0.7': precision_at_07,
            'precision_at_0.3': precision_at_03
        }
        
        print("\n" + "-"*60)
        print("INITIAL TRAINING RESULTS")
        print("-"*60)
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Precision:         {precision:.4f}")
        print(f"Recall:            {recall:.4f}")
        print(f"AUC:               {auc:.4f}")
        print(f"Precision @ 0.7:   {precision_at_07:.4f} ({np.sum(probabilities >= 0.7)} samples)")
        print(f"Precision @ 0.3:   {precision_at_03:.4f} ({np.sum(probabilities <= 0.3)} samples)")
        print("-"*60)
        
        return test_df, feature_cols
        
    def run_prequential_evaluation(self, test_df, feature_cols, update_every=50):
        """Test-then-learn evaluation"""
        print("\n" + "="*60)
        print("PHASE 2: PREQUENTIAL EVALUATION (Test-Then-Learn)")
        print("="*60)
        
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values
        
        predictions = []
        probabilities = []
        accuracies = []
        aucs = []
        
        print(f"\n[PREQUENTIAL] Processing {len(X_test)} samples...")
        print(f"[INFO] Updating model every {update_every} samples\n")
        
        for i in range(len(X_test)):
            # TEST: Predict on current sample
            pred, proba = self.model.predict(X_test[i:i+1])
            predictions.append(pred)
            probabilities.append(proba)
            
            # LEARN: Update model with true label
            self.model.update_online(X_test[i:i+1], y_test[i:i+1])
            
            # Calculate running metrics every 50 samples
            if (i + 1) % update_every == 0:
                from sklearn.metrics import accuracy_score, roc_auc_score
                
                preds_so_far = np.array(predictions)
                probas_so_far = np.array(probabilities)
                y_so_far = y_test[:i+1]
                
                acc = accuracy_score(y_so_far, preds_so_far)
                auc = roc_auc_score(y_so_far, probas_so_far)
                
                accuracies.append(acc)
                aucs.append(auc)
                
                print(f"  Sample {i+1:4d}/{len(X_test)}: Accuracy={acc:.4f}, AUC={auc:.4f}")
        
        # Final metrics
        from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        final_accuracy = accuracy_score(y_test, predictions)
        final_precision = precision_score(y_test, predictions, zero_division=0)
        final_auc = roc_auc_score(y_test, probabilities)
        
        self.results['prequential'] = {
            'samples': len(X_test),
            'accuracy': final_accuracy,
            'precision': final_precision,
            'auc': final_auc,
            'accuracy_over_time': accuracies,
            'auc_over_time': aucs
        }
        
        print("\n" + "-"*60)
        print("PREQUENTIAL EVALUATION RESULTS")
        print("-"*60)
        print(f"Final Accuracy:    {final_accuracy:.4f}")
        print(f"Final Precision:   {final_precision:.4f}")
        print(f"Final AUC:         {final_auc:.4f}")
        print(f"Accuracy trend:    {accuracies[0]:.4f} -> {accuracies[-1]:.4f}")
        print(f"AUC trend:         {aucs[0]:.4f} -> {aucs[-1]:.4f}")
        print("-"*60)
        
    def run_ablation_study(self, df, feature_cols):
        """Test each component: Baseline -> +LSTM -> +Online -> +Meta"""
        print("\n" + "="*60)
        print("PHASE 3: ABLATION STUDY")
        print("="*60)
        
        # Split data
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values
        
        ablation_results = {}
        
        # 1. BASELINE: RF + GB only
        print("\n[ABLATION 1] Testing Baseline (RF + GB only)...")
        model1 = AdaptiveModel(use_lstm=False, use_meta=False, enable_online=False)
        model1.train(X_train, y_train)
        preds1, probas1 = [], []
        for i in range(len(X_test)):
            pred, proba = model1.predict(X_test[i:i+1])
            preds1.append(pred)
            probas1.append(proba)
        
        from sklearn.metrics import accuracy_score, roc_auc_score
        acc1 = accuracy_score(y_test, preds1)
        auc1 = roc_auc_score(y_test, probas1)
        ablation_results['baseline'] = {'accuracy': acc1, 'auc': auc1}
        print(f"  Baseline: Accuracy={acc1:.4f}, AUC={auc1:.4f}")
        
        # 2. +LSTM
        print("\n[ABLATION 2] Testing +LSTM...")
        model2 = AdaptiveModel(use_lstm=True, use_meta=False, enable_online=False)
        model2.train(X_train, y_train)
        preds2, probas2 = [], []
        for i in range(len(X_test)):
            pred, proba = model2.predict(X_test[i:i+1])
            preds2.append(pred)
            probas2.append(proba)
        
        acc2 = accuracy_score(y_test, preds2)
        auc2 = roc_auc_score(y_test, probas2)
        ablation_results['lstm'] = {'accuracy': acc2, 'auc': auc2}
        print(f"  +LSTM: Accuracy={acc2:.4f}, AUC={auc2:.4f} (Δ={acc2-acc1:+.4f})")
        
        # 3. +Online Learning
        print("\n[ABLATION 3] Testing +Online Learning...")
        model3 = AdaptiveModel(use_lstm=True, use_meta=False, enable_online=True)
        model3.train(X_train, y_train)
        preds3, probas3 = [], []
        for i in range(len(X_test)):
            pred, proba = model3.predict(X_test[i:i+1])
            preds3.append(pred)
            probas3.append(proba)
            model3.update_online(X_test[i:i+1], y_test[i:i+1])
        
        acc3 = accuracy_score(y_test, preds3)
        auc3 = roc_auc_score(y_test, probas3)
        ablation_results['online'] = {'accuracy': acc3, 'auc': auc3}
        print(f"  +Online: Accuracy={acc3:.4f}, AUC={auc3:.4f} (Δ={acc3-acc2:+.4f})")
        
        # 4. +Meta-Ensemble
        print("\n[ABLATION 4] Testing +Meta-Ensemble...")
        model4 = AdaptiveModel(use_lstm=True, use_meta=True, enable_online=True)
        model4.train(X_train, y_train)
        preds4, probas4 = [], []
        for i in range(len(X_test)):
            pred, proba = model4.predict(X_test[i:i+1])
            preds4.append(pred)
            probas4.append(proba)
            model4.update_online(X_test[i:i+1], y_test[i:i+1])
        
        acc4 = accuracy_score(y_test, preds4)
        auc4 = roc_auc_score(y_test, probas4)
        ablation_results['meta'] = {'accuracy': acc4, 'auc': auc4}
        print(f"  +Meta: Accuracy={acc4:.4f}, AUC={auc4:.4f} (Δ={acc4-acc3:+.4f})")
        
        self.results['ablation'] = ablation_results
        
        print("\n" + "-"*60)
        print("ABLATION STUDY SUMMARY")
        print("-"*60)
        print(f"Baseline (RF+GB):        Acc={acc1:.4f}, AUC={auc1:.4f}")
        print(f"+LSTM:                   Acc={acc2:.4f}, AUC={auc2:.4f} ({acc2-acc1:+.4f})")
        print(f"+Online Learning:        Acc={acc3:.4f}, AUC={auc3:.4f} ({acc3-acc2:+.4f})")
        print(f"+Meta-Ensemble:          Acc={acc4:.4f}, AUC={auc4:.4f} ({acc4-acc3:+.4f})")
        print(f"\nTotal Improvement:       {acc4-acc1:+.4f} accuracy, {auc4-auc1:+.4f} AUC")
        print("-"*60)
        
    def save_results(self):
        """Save test results and trained model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results JSON
        results_file = f"adaptive_model_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n[SAVE] Results saved to {results_file}")
        
        # Save trained model
        model_file = f"adaptive_model_{timestamp}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[SAVE] Model saved to {model_file}")
        
        return results_file, model_file
        
    def run_full_test_suite(self):
        """Run complete testing pipeline"""
        print("\n" + "="*60)
        print("ADAPTIVE MODEL TEST SUITE")
        print("="*60)
        print(f"Symbol: {self.symbol}")
        print(f"Lookback: {self.lookback_days} days")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # 1. Fetch data
        df = self.fetch_historical_data(self.lookback_days)
        if df is None:
            print("[ERROR] Failed to fetch data")
            return
            
        # 2. Prepare features
        df = self.prepare_features(df)
        
        # 3. Initial training
        test_df, feature_cols = self.train_initial_model(df)
        
        # 4. Prequential evaluation
        self.run_prequential_evaluation(test_df, feature_cols)
        
        # 5. Ablation study
        self.run_ablation_study(df, feature_cols)
        
        # 6. Save results
        results_file, model_file = self.save_results()
        
        print("\n" + "="*60)
        print("TEST SUITE COMPLETE")
        print("="*60)
        print(f"Results: {results_file}")
        print(f"Model: {model_file}")
        print("="*60)


def run_shadow_mode(model, duration_minutes=60):
    """Run model in shadow mode (log signals without trading)"""
    print("\n" + "="*60)
    print("SHADOW MODE TEST")
    print("="*60)
    print(f"Duration: {duration_minutes} minutes")
    print("Will log signals WITHOUT placing trades")
    print("="*60)
    
    if not initialize_mt5():
        print("[ERROR] Failed to initialize MT5")
        return
    
    from time import sleep
    
    symbol = 'USDJPY'
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    signal_count = 0
    
    print(f"\n[SHADOW] Started at {start_time.strftime('%H:%M:%S')}")
    print(f"[SHADOW] Will run until {end_time.strftime('%H:%M:%S')}\n")
    
    while datetime.now() < end_time:
        # Fetch current market data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
        
        if rates is not None and len(rates) > 0:
            # Prepare features (same as training)
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Get last bar features
            # (simplified - in production, calculate all features)
            current_features = df[['open', 'high', 'low', 'close', 'tick_volume']].iloc[-1].values.reshape(1, -1)
            
            # Get prediction
            pred, proba = model.predict(current_features)
            
            signal = None
            if proba >= 0.70:
                signal = "BUY"
            elif proba <= 0.30:
                signal = "SELL"
            
            if signal:
                signal_count += 1
                timestamp = datetime.now().strftime('%H:%M:%S')
                price = df['close'].iloc[-1]
                
                print(f"[SHADOW {timestamp}] Signal #{signal_count}: {signal} @ {price:.3f} | Probability: {proba:.3f}")
                
                # Log to file
                with open('shadow_mode_log.txt', 'a') as f:
                    f.write(f"{timestamp},{signal},{price:.3f},{proba:.3f}\n")
        
        # Wait 30 seconds before next check
        sleep(30)
    
    mt5.shutdown()
    
    print(f"\n[SHADOW] Completed at {datetime.now().strftime('%H:%M:%S')}")
    print(f"[SHADOW] Total signals generated: {signal_count}")
    print(f"[SHADOW] Log saved to: shadow_mode_log.txt")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test and train adaptive model')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'shadow', 'train_only'],
                       help='Test mode: full (all tests), shadow (live testing), train_only')
    parser.add_argument('--days', type=int, default=90,
                       help='Days of historical data (default: 90)')
    parser.add_argument('--shadow-duration', type=int, default=60,
                       help='Shadow mode duration in minutes (default: 60)')
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    if args.mode == 'full':
        # Run complete test suite
        tester = AdaptiveModelTester(lookback_days=args.days)
        tester.run_full_test_suite()
        
    elif args.mode == 'train_only':
        # Train only
        tester = AdaptiveModelTester(lookback_days=args.days)
        df = tester.fetch_historical_data(args.days)
        if df is not None:
            df = tester.prepare_features(df)
            test_df, feature_cols = tester.train_initial_model(df)
            tester.save_results()
            
    elif args.mode == 'shadow':
        # Load existing model or create new
        import glob
        model_files = glob.glob('adaptive_model_*.pkl')
        
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            print(f"[LOAD] Loading model from {latest_model}")
            with open(latest_model, 'rb') as f:
                model = pickle.load(f)
        else:
            print("[INFO] No existing model found, training new model...")
            tester = AdaptiveModelTester(lookback_days=args.days)
            df = tester.fetch_historical_data(args.days)
            if df is not None:
                df = tester.prepare_features(df)
                tester.train_initial_model(df)
                model = tester.model
            else:
                print("[ERROR] Failed to train model")
                sys.exit(1)
        
        # Run shadow mode
        run_shadow_mode(model, duration_minutes=args.shadow_duration)
