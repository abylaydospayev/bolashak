"""
Quick test script to verify the entire FX ML MT5 pipeline.
Run this to ensure everything is working correctly.
"""
import os
import sys

def check_file(path, description):
    """Check if a file exists and print status."""
    exists = os.path.exists(path)
    status = "" if exists else ""
    print(f"{status} {description}: {path}")
    return exists

def run_tests():
    print("=" * 70)
    print("FX ML MT5 Starter - System Test")
    print("=" * 70)
    
    all_passed = True
    
    # Check Python files
    print("\n Checking source files...")
    files = [
        'indicators.py',
        'make_dataset.py',
        'build_features.py',
        'train_rf.py',
        'backtest.py',
        'signal_mt5.py',
        'config.yaml',
        'requirements.txt'
    ]
    for f in files:
        if not check_file(f, f"Source: {f}"):
            all_passed = False
    
    # Check data files
    print("\n Checking data files...")
    data_files = [
        'data/EURUSD_M15.csv',
        'data/USDJPY_M15.csv'
    ]
    for f in data_files:
        if not check_file(f, f"Data: {f}"):
            all_passed = False
    
    # Check feature files
    print("\n Checking feature files...")
    feature_files = [
        'features/EURUSD_features.csv',
        'features/USDJPY_features.csv'
    ]
    for f in feature_files:
        if not check_file(f, f"Features: {f}"):
            all_passed = False
    
    # Check model files
    print("\n Checking model files...")
    model_files = [
        'models/EURUSD_rf.pkl',
        'models/USDJPY_rf.pkl',
        'models/scaler.pkl'
    ]
    for f in model_files:
        if not check_file(f, f"Model: {f}"):
            all_passed = False
    
    # Test imports
    print("\n Testing imports...")
    try:
        import pandas
        print(" pandas imported successfully")
    except ImportError as e:
        print(f" pandas import failed: {e}")
        all_passed = False
    
    try:
        import numpy
        print(" numpy imported successfully")
    except ImportError as e:
        print(f" numpy import failed: {e}")
        all_passed = False
    
    try:
        import sklearn
        print(" scikit-learn imported successfully")
    except ImportError as e:
        print(f" scikit-learn import failed: {e}")
        all_passed = False
    
    try:
        import MetaTrader5
        print(" MetaTrader5 imported successfully")
    except ImportError as e:
        print(f" MetaTrader5 import failed: {e}")
        all_passed = False
    
    try:
        import yaml
        print(" yaml imported successfully")
    except ImportError as e:
        print(f" yaml import failed: {e}")
        all_passed = False
    
    try:
        import joblib
        print(" joblib imported successfully")
    except ImportError as e:
        print(f" joblib import failed: {e}")
        all_passed = False
    
    # Test indicator functions
    print("\n Testing indicator functions...")
    try:
        from indicators import ema, rsi, atr, sincos_time
        import pandas as pd
        import numpy as np
        
        # Create sample data
        test_data = pd.Series(np.random.randn(100).cumsum() + 100)
        test_df = pd.DataFrame({
            'high': test_data + 1,
            'low': test_data - 1,
            'close': test_data
        })
        test_time = pd.date_range('2024-01-01', periods=100, freq='15min')
        
        # Test EMA
        ema_result = ema(test_data, 20)
        print(f" EMA calculated: {len(ema_result)} values")
        
        # Test RSI
        rsi_result = rsi(test_data, 14)
        print(f" RSI calculated: {len(rsi_result)} values")
        
        # Test ATR
        atr_result = atr(test_df, 14)
        print(f" ATR calculated: {len(atr_result)} values")
        
        # Test sincos_time
        sin_h, cos_h = sincos_time(test_time)
        print(f" Time features calculated: {len(sin_h)} values")
        
    except Exception as e:
        print(f" Indicator test failed: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print(" ALL TESTS PASSED - System is ready!")
        print("\nNext steps:")
        print("1. Connect to MT5 and pull real data:")
        print("   python make_dataset.py --symbol EURUSD --timeframe M15 --bars 50000")
        print("\n2. Or continue with sample data for testing:")
        print("   python signal_mt5.py --symbol EURUSD  (requires MT5)")
    else:
        print(" SOME TESTS FAILED - Please review errors above")
        return 1
    print("=" * 70)
    
    return 0

if __name__ == '__main__':
    sys.exit(run_tests())

