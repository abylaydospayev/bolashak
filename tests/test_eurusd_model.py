"""
EURUSD Model Testing Suite
Tests data loading, feature engineering, model training and evaluation
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from make_dataset import load_ohlcv
from build_features_enhanced import add_multi_timeframe_features
from indicators import ema, rsi, atr
from position_sizing import PositionSizer

class TestEURUSDDataLoading:
    """Test EURUSD data loading and validation"""
    
    def test_load_eurusd_data(self):
        """Test loading EURUSD data"""
        symbol = 'EURUSD.sim'
        df = load_ohlcv(symbol, timeframe='M15')
        
        assert df is not None, "Failed to load EURUSD data"
        assert len(df) > 0, "EURUSD dataframe is empty"
        assert 'close' in df.columns, "Missing 'close' column"
        assert 'high' in df.columns, "Missing 'high' column"
        assert 'low' in df.columns, "Missing 'low' column"
        assert 'volume' in df.columns, "Missing 'volume' column"
        
        print(f"\n✅ Loaded {len(df)} EURUSD bars")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
    
    def test_eurusd_data_quality(self):
        """Test EURUSD data quality"""
        symbol = 'EURUSD.sim'
        df = load_ohlcv(symbol, timeframe='M15')
        
        # Check for missing values
        missing = df.isnull().sum()
        assert missing.sum() == 0, f"Found missing values: {missing[missing > 0]}"
        
        # Check for duplicates
        duplicates = df.index.duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate timestamps"
        
        # Check price validity (EURUSD typically 0.8 - 1.6)
        assert df['close'].min() > 0.5, "EURUSD price too low"
        assert df['close'].max() < 2.0, "EURUSD price too high"
        
        # Check for outliers (15-min returns > 1% are suspicious)
        returns = df['close'].pct_change()
        outliers = (returns.abs() > 0.01).sum()
        print(f"\n   Found {outliers} potential outliers (>1% 15-min move)")
        
        print("✅ EURUSD data quality checks passed")


class TestEURUSDFeatureEngineering:
    """Test EURUSD feature engineering"""
    
    def test_build_features_eurusd(self):
        """Test feature building for EURUSD"""
        symbol = 'EURUSD.sim'
        
        # Load M15 data
        df_m15 = load_ohlcv(symbol, timeframe='M15')
        
        # Add base indicators
        df_m15['ema20'] = ema(df_m15['close'], 20)
        df_m15['ema50'] = ema(df_m15['close'], 50)
        df_m15['rsi14'] = rsi(df_m15['close'], 14)
        df_m15['atr14'] = atr(df_m15, 14)
        
        assert 'ema20' in df_m15.columns
        assert 'rsi14' in df_m15.columns
        assert 'atr14' in df_m15.columns
        
        # Check values are reasonable
        assert df_m15['rsi14'].min() >= 0
        assert df_m15['rsi14'].max() <= 100
        assert df_m15['atr14'].min() >= 0
        
        print(f"\n✅ Built base features for EURUSD")
        print(f"   RSI range: {df_m15['rsi14'].min():.1f} - {df_m15['rsi14'].max():.1f}")
        print(f"   ATR range: {df_m15['atr14'].min():.5f} - {df_m15['atr14'].max():.5f}")
    
    def test_multi_timeframe_features_eurusd(self):
        """Test multi-timeframe feature engineering for EURUSD"""
        symbol = 'EURUSD.sim'
        
        # Load all timeframes
        df_m15 = load_ohlcv(symbol, timeframe='M15')
        df_m30 = load_ohlcv(symbol, timeframe='M30')
        df_h1 = load_ohlcv(symbol, timeframe='H1')
        
        # Add M30 features
        df_enhanced = add_multi_timeframe_features(df_m15, df_m30, 'm30')
        
        # Check M30 features were added
        m30_features = [col for col in df_enhanced.columns if '_m30' in col]
        assert len(m30_features) > 0, "No M30 features added"
        
        print(f"\n✅ Multi-timeframe features for EURUSD")
        print(f"   M30 features added: {len(m30_features)}")
        print(f"   Sample features: {m30_features[:5]}")


class TestEURUSDPositionSizing:
    """Test position sizing for EURUSD"""
    
    def test_position_sizing_eurusd(self):
        """Test position size calculation for EURUSD"""
        
        sizer = PositionSizer(strategy='fixed_fractional', risk_pct=0.01)
        
        # EURUSD typical values
        equity = 100000
        stop_loss_pips = 30
        
        # For EURUSD: 1 pip = 0.0001, 1 lot = $10/pip
        pip_value = 10.0
        
        position_size = sizer.calculate_size(
            equity=equity,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )
        
        assert position_size > 0, "Position size should be positive"
        assert position_size < 10.0, "Position size seems too large for EURUSD"
        
        # Expected: $100k * 1% / (30 pips * $10/pip) = 3.33 lots
        expected = (equity * 0.01) / (stop_loss_pips * pip_value)
        
        print(f"\n✅ EURUSD Position Sizing:")
        print(f"   Equity: ${equity:,.0f}")
        print(f"   Risk: 1%")
        print(f"   Stop Loss: {stop_loss_pips} pips")
        print(f"   Position Size: {position_size:.2f} lots")
        print(f"   Expected: {expected:.2f} lots")
        
        assert abs(position_size - expected) < 0.1, "Position size calculation mismatch"
    
    def test_ftmo_position_sizing_eurusd(self):
        """Test FTMO-style position sizing for EURUSD"""
        
        sizer = PositionSizer(strategy='fixed_fractional', risk_pct=0.01)
        
        equity = 98972.55  # Current account balance
        stop_loss_pips = 30
        pip_value = 10.0  # EURUSD standard lot
        
        position_size = sizer.calculate_size(
            equity=equity,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )
        
        # Calculate risk
        dollar_risk = equity * 0.01
        
        print(f"\n✅ FTMO EURUSD Position Sizing:")
        print(f"   Account: ${equity:,.2f}")
        print(f"   Max Risk per Trade: ${dollar_risk:.2f} (1%)")
        print(f"   Stop Loss: {stop_loss_pips} pips")
        print(f"   Position Size: {position_size:.2f} lots")
        print(f"   Risk per Pip: ${pip_value * position_size:.2f}")
        
        assert position_size > 0
        assert dollar_risk <= 1000  # Should be ~$990


class TestEURUSDModel:
    """Test EURUSD model training and evaluation"""
    
    def test_model_file_exists(self):
        """Test if EURUSD model file exists"""
        model_path = Path('models/EURUSD_ensemble_oos.pkl')
        
        if not model_path.exists():
            print("\n⚠️  EURUSD model not found. Need to train it first.")
            print("   Run: python train_eurusd_model.py")
            pytest.skip("EURUSD model not trained yet")
        else:
            print(f"\n✅ Found EURUSD model: {model_path}")
    
    def test_load_eurusd_model(self):
        """Test loading EURUSD model"""
        import joblib
        
        model_path = Path('models/EURUSD_ensemble_oos.pkl')
        
        if not model_path.exists():
            pytest.skip("EURUSD model not trained yet")
        
        try:
            model = joblib.load(model_path)
            print(f"\n✅ Loaded EURUSD model: {type(model).__name__}")
            
            # Check if model has required methods
            assert hasattr(model, 'predict'), "Model missing predict method"
            assert hasattr(model, 'predict_proba'), "Model missing predict_proba method"
            
        except Exception as e:
            pytest.fail(f"Failed to load model: {e}")
    
    def test_model_prediction_eurusd(self):
        """Test EURUSD model prediction"""
        import joblib
        
        model_path = Path('models/EURUSD_ensemble_oos.pkl')
        scaler_path = Path('models/scaler.pkl')
        
        if not model_path.exists():
            pytest.skip("EURUSD model not found")
        
        # Load model
        model = joblib.load(model_path)
        
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        else:
            scaler = None
        
        # Load data
        symbol = 'EURUSD.sim'
        df = load_ohlcv(symbol, timeframe='M15')
        
        # Add features (simplified for test)
        df['ema20'] = ema(df['close'], 20)
        df['ema50'] = ema(df['close'], 50)
        df['rsi14'] = rsi(df['close'], 14)
        df['atr14'] = atr(df, 14)
        
        df_clean = df.dropna()
        
        # Get latest data
        feature_cols = ['ema20', 'ema50', 'rsi14', 'atr14']
        X = df_clean[feature_cols].tail(1)
        
        # Predict
        try:
            if scaler:
                X_scaled = scaler.transform(X)
                probability = model.predict_proba(X_scaled)[0, 1]
            else:
                probability = model.predict_proba(X)[0, 1]
            
            print(f"\n✅ EURUSD Model Prediction:")
            print(f"   Latest Close: {df_clean['close'].iloc[-1]:.5f}")
            print(f"   Probability (UP): {probability:.3f}")
            
            assert 0 <= probability <= 1, "Probability should be between 0 and 1"
            
        except Exception as e:
            pytest.fail(f"Prediction failed: {e}")


def run_all_eurusd_tests():
    """Run all EURUSD tests"""
    print("\n" + "="*70)
    print("🧪 RUNNING EURUSD MODEL TESTS")
    print("="*70 + "\n")
    
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_all_eurusd_tests()
