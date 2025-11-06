"""
Train LSTM model for forex prediction.
Uses sequences of lookback bars to predict future direction.
"""
import argparse, os, yaml, joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

def create_sequences(X, y, lookback):
    """Create sequences for LSTM input.
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Labels (n_samples,)
        lookback: Number of timesteps to look back
        
    Returns:
        X_seq: (n_samples - lookback, lookback, n_features)
        y_seq: (n_samples - lookback,)
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def time_split(df, train_ratio=0.7, val_ratio=0.15):
    """Time-based train/val/test split."""
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    return df.iloc[:n_train], df.iloc[n_train:n_val], df.iloc[n_val:]

def build_lstm_model(lookback, n_features, lstm_units=64, dropout=0.2):
    """Build LSTM model architecture.
    
    Args:
        lookback: Number of timesteps
        n_features: Number of features
        lstm_units: LSTM hidden units
        dropout: Dropout rate
    """
    model = keras.Sequential([
        # First LSTM layer with return sequences
        layers.LSTM(lstm_units, return_sequences=True, input_shape=(lookback, n_features)),
        layers.Dropout(dropout),
        
        # Second LSTM layer
        layers.LSTM(lstm_units // 2, return_sequences=False),
        layers.Dropout(dropout),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dropout(dropout / 2),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True, help='Symbol to train (e.g., EURUSD.sim)')
    parser.add_argument('--lookback', type=int, default=None, help='Sequence length (default from config)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lstm_units', type=int, default=64, help='LSTM units')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open('config.yaml'))
    lookback = args.lookback or cfg['lookback']
    feat_dir = cfg['feature_dir']
    model_dir = cfg['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    
    # Load features
    csv_path = os.path.join(feat_dir, f"{args.symbol}_features.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Features not found: {csv_path}. Run build_features.py first.")
    
    df = pd.read_csv(csv_path, parse_dates=['time'])
    print(f"Loaded {len(df)} samples from {csv_path}")
    
    # Prepare features
    feature_cols = [c for c in df.columns if c not in ('time', 'y')]
    X = df[feature_cols].values
    y = df['y'].values
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Lookback: {lookback} bars")
    
    # Time-based split
    train_df, val_df, test_df = time_split(df)
    X_train = train_df[feature_cols].values
    y_train = train_df['y'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['y'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['y'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create sequences
    print("\nCreating sequences...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, lookback)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, lookback)
    
    print(f"Train sequences: {X_train_seq.shape}")
    print(f"Val sequences: {X_val_seq.shape}")
    print(f"Test sequences: {X_test_seq.shape}")
    
    # Class balance
    train_pos = y_train_seq.sum()
    train_neg = len(y_train_seq) - train_pos
    print(f"\nClass balance (train): {train_pos} positive, {train_neg} negative")
    
    # Calculate class weights
    class_weight = {
        0: len(y_train_seq) / (2 * train_neg),
        1: len(y_train_seq) / (2 * train_pos)
    }
    
    # Build model
    print("\nBuilding LSTM model...")
    model = build_lstm_model(
        lookback=lookback,
        n_features=X_train_seq.shape[2],
        lstm_units=args.lstm_units,
        dropout=args.dropout
    )
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    print(model.summary())
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        os.path.join(model_dir, f"{args.symbol}_lstm_best.keras"),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    # Train
    print("\nTraining LSTM model...")
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    
    def evaluate_split(name, X_seq, y_seq):
        proba = model.predict(X_seq, verbose=0).flatten()
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_seq, proba)
        acc = accuracy_score(y_seq, pred)
        print(f"{name:8s}: AUC={auc:.4f}  ACC={acc:.4f}  ({len(y_seq)} samples)")
        return auc, acc
    
    train_auc, train_acc = evaluate_split("Train", X_train_seq, y_train_seq)
    val_auc, val_acc = evaluate_split("Val", X_val_seq, y_val_seq)
    test_auc, test_acc = evaluate_split("Test", X_test_seq, y_test_seq)
    
    # Save model and scaler
    model_path = os.path.join(model_dir, f"{args.symbol}_lstm.keras")
    model.save(model_path)
    
    scaler_path = os.path.join(model_dir, f"{args.symbol}_lstm_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        'symbol': args.symbol,
        'lookback': lookback,
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'lstm_units': args.lstm_units,
        'dropout': args.dropout,
        'train_auc': float(train_auc),
        'val_auc': float(val_auc),
        'test_auc': float(test_auc),
        'train_acc': float(train_acc),
        'val_acc': float(val_acc),
        'test_acc': float(test_acc)
    }
    
    import json
    with open(os.path.join(model_dir, f"{args.symbol}_lstm_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n Model saved to {model_path}")
    print(f" Scaler saved to {scaler_path}")
    print(f" Best model saved to {model_dir}/{args.symbol}_lstm_best.keras")
    
    # Training history summary
    print("\n" + "="*70)
    print("Training History (last 5 epochs)")
    print("="*70)
    for key in ['loss', 'accuracy', 'auc', 'val_loss', 'val_accuracy', 'val_auc']:
        if key in history.history:
            values = history.history[key][-5:]
            print(f"{key:15s}: {' '.join([f'{v:.4f}' for v in values])}")

if __name__ == '__main__':
    main()

