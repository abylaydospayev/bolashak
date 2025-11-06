"""
Enhanced feature engineering with multi-timeframe analysis.

Improvements:
1. Multi-timeframe features (M30, H1, H4 trends)
2. Advanced technical indicators
3. Market structure features
4. Volatility regime indicators
"""

import argparse, os
import pandas as pd
import numpy as np
import yaml
from indicators import ema, rsi, atr, pct_change, sincos_time


def pip_value(symbol: str) -> float:
    """Get pip value for symbol."""
    return 0.0001 if symbol.upper().startswith('EURUSD') else 0.01


def resample_to_higher_timeframe(df, timeframe):
    """
    Resample M15 data to higher timeframes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        M15 data with time, open, high, low, close, volume
    timeframe : str
        Target timeframe: '30T' (M30), '1H' (H1), '4H' (H4)
    
    Returns:
    --------
    pd.DataFrame
        Resampled data
    """
    df = df.set_index('time')
    
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled.reset_index()


def add_multi_timeframe_features(df_m15, df_higher, timeframe_name):
    """
    Add features from higher timeframe to M15 data.
    
    Parameters:
    -----------
    df_m15 : pd.DataFrame
        M15 timeframe data
    df_higher : pd.DataFrame
        Higher timeframe data (M30, H1, or H4)
    timeframe_name : str
        Name suffix for features (e.g., 'm30', 'h1', 'h4')
    
    Returns:
    --------
    pd.DataFrame
        M15 data with additional timeframe features
    """
    # Calculate indicators on higher timeframe
    df_higher['ema20'] = ema(df_higher['close'], 20)
    df_higher['ema50'] = ema(df_higher['close'], 50)
    df_higher['rsi14'] = rsi(df_higher['close'], 14)
    df_higher['atr14'] = atr(df_higher, 14)
    
    # Trend indicators
    df_higher[f'trend_ema'] = (df_higher['ema20'] > df_higher['ema50']).astype(int)
    df_higher[f'trend_strength'] = (df_higher['ema20'] - df_higher['ema50']) / df_higher['close']
    df_higher[f'price_vs_ema20'] = (df_higher['close'] - df_higher['ema20']) / df_higher['close']
    
    # Momentum
    df_higher[f'momentum_5'] = df_higher['close'].pct_change(5)
    df_higher[f'momentum_10'] = df_higher['close'].pct_change(10)
    
    # Volatility
    df_higher[f'atr_pct'] = df_higher['atr14'] / df_higher['close']
    
    # Merge back to M15 using forward fill
    df_higher = df_higher.set_index('time')
    df_m15 = df_m15.set_index('time')
    
    # Features to merge
    features_to_merge = [
        'ema20', 'ema50', 'rsi14', 'atr14',
        'trend_ema', 'trend_strength', 'price_vs_ema20',
        'momentum_5', 'momentum_10', 'atr_pct'
    ]
    
    for feature in features_to_merge:
        if feature in df_higher.columns:
            # Rename with timeframe suffix
            new_name = f'{feature}_{timeframe_name}'
            df_m15[new_name] = df_higher[feature].reindex(df_m15.index, method='ffill')
    
    return df_m15.reset_index()


def add_market_structure_features(df):
    """
    Add market structure and pattern features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data
    
    Returns:
    --------
    pd.DataFrame
        Data with additional structure features
    """
    # Price position within range
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # Candle patterns
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    
    # Bullish/Bearish
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    
    # Higher highs / Lower lows
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    
    # Swing highs/lows (5-bar pattern)
    df['swing_high'] = ((df['high'] > df['high'].shift(1)) & 
                        (df['high'] > df['high'].shift(2)) &
                        (df['high'] > df['high'].shift(-1)) & 
                        (df['high'] > df['high'].shift(-2))).astype(int)
    
    df['swing_low'] = ((df['low'] < df['low'].shift(1)) & 
                       (df['low'] < df['low'].shift(2)) &
                       (df['low'] < df['low'].shift(-1)) & 
                       (df['low'] < df['low'].shift(-2))).astype(int)
    
    return df


def add_volatility_regime_features(df):
    """
    Add volatility regime indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with ATR
    
    Returns:
    --------
    pd.DataFrame
        Data with volatility regime features
    """
    # ATR percentile (rolling 100 bars)
    df['atr_percentile'] = df['atr14'].rolling(100).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5
    )
    
    # Volatility expansion/contraction
    df['atr_change'] = df['atr14'].pct_change(5)
    df['vol_expanding'] = (df['atr_change'] > 0.1).astype(int)
    df['vol_contracting'] = (df['atr_change'] < -0.1).astype(int)
    
    # Bollinger Band width
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_width'] = (4 * df['bb_std']) / df['bb_middle']
    df['bb_position'] = (df['close'] - (df['bb_middle'] - 2*df['bb_std'])) / (4*df['bb_std'] + 1e-10)
    
    return df


def add_momentum_features(df):
    """
    Add advanced momentum indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data
    
    Returns:
    --------
    pd.DataFrame
        Data with momentum features
    """
    # Rate of change at multiple periods
    for period in [3, 5, 10, 20]:
        df[f'roc_{period}'] = df['close'].pct_change(period)
    
    # RSI divergence (simple version)
    df['rsi_change'] = df['rsi14'].diff(5)
    df['price_change'] = df['close'].pct_change(5)
    df['rsi_divergence'] = df['rsi_change'] * df['price_change']  # Negative = divergence
    
    # Trend consistency (count of directional moves)
    df['up_bars'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['trend_consistency'] = df['up_bars'].rolling(10).sum() / 10  # % of up bars in last 10
    
    return df


def select_important_features(df, target_col='y', max_features=30):
    """
    Select most important features using mutual information.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with features
    target_col : str
        Target column name
    max_features : int
        Maximum number of features to select
    
    Returns:
    --------
    list
        Selected feature names
    """
    from sklearn.feature_selection import mutual_info_classif
    
    # Get feature columns (exclude time, target, forward returns)
    exclude_cols = ['time', 'y', 'fwd_ret', 'fwd_ret_net']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Drop any remaining NaNs for this analysis
    df_clean = df[feature_cols + [target_col]].dropna()
    
    if len(df_clean) < 100:
        print("Warning: Not enough data for feature selection")
        return feature_cols[:max_features]
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Create scores dataframe
    feature_scores = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # Select top features
    selected_features = feature_scores.head(max_features)['feature'].tolist()
    
    print("\nTop 10 Most Important Features:")
    print("=" * 60)
    for i, row in feature_scores.head(10).iterrows():
        print(f"{row['feature']:30s} {row['mi_score']:.6f}")
    
    print(f"\nSelected {len(selected_features)} features out of {len(feature_cols)}")
    
    return selected_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--h', type=int, default=None, help='horizon bars ahead')
    parser.add_argument('--use-mtf', type=int, default=1, help='Use multi-timeframe features (1=yes, 0=no)')
    parser.add_argument('--feature-selection', type=int, default=1, help='Apply feature selection (1=yes, 0=no)')
    parser.add_argument('--max-features', type=int, default=30, help='Max features after selection')
    args = parser.parse_args()

    cfg = yaml.safe_load(open('config.yaml'))
    h = args.h or cfg['horizon']
    data_dir = cfg['data_dir']
    feat_dir = cfg['feature_dir']
    os.makedirs(feat_dir, exist_ok=True)

    # Find CSV by prefix
    candidates = [f for f in os.listdir(data_dir) if f.startswith(args.symbol+"_")]
    if not candidates:
        raise FileNotFoundError(f"No CSV found for {args.symbol} in {data_dir}. Run make_dataset.py first.")
    csv_path = os.path.join(data_dir, sorted(candidates)[-1])
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['time'])
    print(f"Loaded {len(df):,} bars")

    # ========================================
    # STEP 1: Basic features (M15)
    # ========================================
    print("\nBuilding basic features...")
    
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['rsi14'] = rsi(df['close'], 14)
    df['atr14'] = atr(df, 14)
    df['ema50_slope'] = df['ema50'].diff(5)
    
    # Time features
    time_col = df['time'] if df['time'].dt.tz is not None else df['time'].dt.tz_localize('UTC')
    sin_h, cos_h = sincos_time(time_col)
    df['sin_hour'] = sin_h.values
    df['cos_hour'] = cos_h.values
    
    # Returns
    df['ret1'] = pct_change(df['close'], 1)
    df['atr_pct'] = df['atr14'] / df['close']
    
    # ========================================
    # STEP 2: Multi-timeframe features
    # ========================================
    if args.use_mtf:
        print("\nBuilding multi-timeframe features...")
        
        # M30 (30-minute)
        print("  - M30 features...")
        df_m30 = resample_to_higher_timeframe(df[['time', 'open', 'high', 'low', 'close', 'volume']].copy(), '30T')
        df = add_multi_timeframe_features(df, df_m30, 'm30')
        
        # H1 (1-hour)
        print("  - H1 features...")
        df_h1 = resample_to_higher_timeframe(df[['time', 'open', 'high', 'low', 'close', 'volume']].copy(), '1H')
        df = add_multi_timeframe_features(df, df_h1, 'h1')
        
        # H4 (4-hour)
        print("  - H4 features...")
        df_h4 = resample_to_higher_timeframe(df[['time', 'open', 'high', 'low', 'close', 'volume']].copy(), '4H')
        df = add_multi_timeframe_features(df, df_h4, 'h4')
    
    # ========================================
    # STEP 3: Market structure features
    # ========================================
    print("\nBuilding market structure features...")
    df = add_market_structure_features(df)
    
    # ========================================
    # STEP 4: Volatility regime features
    # ========================================
    print("\nBuilding volatility regime features...")
    df = add_volatility_regime_features(df)
    
    # ========================================
    # STEP 5: Advanced momentum features
    # ========================================
    print("\nBuilding momentum features...")
    df = add_momentum_features(df)
    
    # ========================================
    # STEP 6: Target variable
    # ========================================
    print("\nCreating target variable...")
    
    pip = pip_value(args.symbol)
    spread = cfg['spread_pips'] * pip
    slippage = cfg['slippage_pips'] * pip
    fees = spread + slippage

    df['fwd_ret'] = df['close'].shift(-h) / df['close'] - 1.0
    df['fwd_ret_net'] = df['fwd_ret'] - fees / df['close']
    df['y'] = (df['fwd_ret_net'] > 0.0003).astype(int)
    
    # ========================================
    # STEP 7: Feature selection
    # ========================================
    if args.feature_selection:
        print("\nPerforming feature selection...")
        
        # Get all feature columns
        exclude_cols = ['time', 'y', 'fwd_ret', 'fwd_ret_net', 'rsi14', 'atr14', 'close', 'open', 'high', 'low', 'volume']
        all_features = [c for c in df.columns if c not in exclude_cols]
        
        print(f"Total features before selection: {len(all_features)}")
        
        # Select important features
        selected_features = select_important_features(df, target_col='y', max_features=args.max_features)
        
        # Always keep required columns for regime filter and backtest compatibility
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi14', 'atr14']
        for col in required_cols:
            if col in df.columns and col not in selected_features:
                selected_features.append(col)
        
        # Keep time, target, and selected features
        output_cols = ['time'] + selected_features + ['y']
        df = df[output_cols]
    else:
        # Keep all features
        exclude_cols = ['fwd_ret', 'fwd_ret_net']
        output_cols = [c for c in df.columns if c not in exclude_cols]
        df = df[output_cols]
    
    # ========================================
    # STEP 8: Save
    # ========================================
    df = df.dropna().reset_index(drop=True)
    
    out_path = os.path.join(feat_dir, f"{args.symbol}_features_enhanced.csv")
    df.to_csv(out_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"SUCCESS: Saved enhanced features to {out_path}")
    print(f"Final dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(f"Features: {len(df.columns) - 2} (excluding time, y)")
    print("=" * 60)
    
    # Print feature breakdown
    feature_cols = [c for c in df.columns if c not in ['time', 'y']]
    
    mtf_features = [c for c in feature_cols if any(x in c for x in ['_m30', '_h1', '_h4'])]
    structure_features = [c for c in feature_cols if any(x in c for x in ['body', 'shadow', 'swing', 'price_position'])]
    volatility_features = [c for c in feature_cols if any(x in c for x in ['atr', 'vol_', 'bb_'])]
    momentum_features = [c for c in feature_cols if any(x in c for x in ['roc_', 'rsi', 'momentum', 'trend_'])]
    
    print(f"\nFeature breakdown:")
    print(f"  Multi-timeframe: {len(mtf_features)}")
    print(f"  Market structure: {len(structure_features)}")
    print(f"  Volatility regime: {len(volatility_features)}")
    print(f"  Momentum: {len(momentum_features)}")
    print(f"  Other: {len(feature_cols) - len(mtf_features) - len(structure_features) - len(volatility_features) - len(momentum_features)}")


if __name__ == '__main__':
    main()
