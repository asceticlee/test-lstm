#!/usr/bin/env python3
"""
Regime-Specific LSTM Training for Stock Price Regression

This script trains LSTM models on data from a specific market regime.
It uses the regime classifications to filter training data to only include
weeks that belong to the specified regime, allowing for specialized models
that are optimized for specific market conditions.

Usage:
    python stockprice_lstm_regime_regression.py <regime_number> <labelNumber> [--validation_split 0.2]
    
Example:
    python stockprice_lstm_regime_regression.py 0 10 --validation_split 0.2
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import csv
import argparse

print(f"TensorFlow version: {tf.__version__}")

# Get project paths
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
data_dir = project_root / 'data'
models_dir = project_root / 'models'
regime_dir = project_root / 'regime_analysis'

# Ensure directories exist
models_dir.mkdir(exist_ok=True)

def load_regime_data(regime_number):
    """Load and filter data for a specific paradigm"""
    print(f"Loading data for regime {regime_number}...")
    
    # Load paradigm assignments
    regime_file = regime_dir / 'regime_assignments.csv'
    if not regime_file.exists():
        raise FileNotFoundError(f"Regime assignments file not found: {regime_file}")
    
    regime_df = pd.read_csv(regime_file)
    
    # Filter for specific paradigm
    regime_data = regime_df[regime_df['Regime'] == regime_number]
    
    if len(regime_data) == 0:
        raise ValueError(f"No data found for regime {regime_number}")
    
    print(f"Found {len(regime_data):,} data points for regime {regime_number}")
    
    # Get unique weeks for this paradigm
    regime_weeks = regime_data['Week'].unique()
    print(f"Regime {regime_number} spans {len(regime_weeks)} weeks")
    
    # Load full trading data
    data_file = data_dir / 'trainingData.csv'
    if not data_file.exists():
        raise FileNotFoundError(f"Training data file not found: {data_file}")
    
    full_df = pd.read_csv(data_file)
    
    # Filter full data to only include paradigm weeks
    regime_trading_days = set(regime_data['TradingDay'].unique())
    filtered_df = full_df[full_df['TradingDay'].isin(regime_trading_days)].reset_index(drop=True)
    
    print(f"Filtered to {len(filtered_df):,} rows for regime {regime_number}")
    
    return filtered_df, regime_weeks

def split_regime_data(df, regime_weeks, validation_split=0.2, random_seed=42):
    """Split paradigm data into training and validation sets by weeks"""
    print(f"Splitting data with validation split: {validation_split}")
    
    # Randomly split weeks
    np.random.seed(random_seed)
    shuffled_weeks = np.random.permutation(regime_weeks)
    
    n_val_weeks = int(len(regime_weeks) * validation_split)
    val_weeks = set(shuffled_weeks[:n_val_weeks])
    train_weeks = set(shuffled_weeks[n_val_weeks:])
    
    print(f"Training weeks: {len(train_weeks)}, Validation weeks: {len(val_weeks)}")
    
    # Create week mapping from paradigm assignments
    regime_file = regime_dir / 'regime_assignments.csv'
    regime_df = pd.read_csv(regime_file)
    
    # Create mapping from TradingDay to Week
    day_to_week = dict(zip(regime_df['TradingDay'], regime_df['Week']))
    
    # Split data
    train_days = set()
    val_days = set()
    
    for day in df['TradingDay'].unique():
        week = day_to_week.get(day)
        if week in train_weeks:
            train_days.add(day)
        elif week in val_weeks:
            val_days.add(day)
    
    train_df = df[df['TradingDay'].isin(train_days)].reset_index(drop=True)
    val_df = df[df['TradingDay'].isin(val_days)].reset_index(drop=True)
    
    print(f"Training data: {len(train_df):,} rows")
    print(f"Validation data: {len(val_df):,} rows")
    
    return train_df, val_df

def create_sequences(df, seq_length, feature_cols, target_col):
    """Create sequences for LSTM training, respecting data continuity"""
    df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
    features = df[feature_cols].values
    targets = df[target_col].values
    days = df['TradingDay'].values
    ms = df['TradingMsOfDay'].values
    
    X = []
    y = []
    
    i = 0
    while i <= len(df) - seq_length:
        # Check if the sequence is consecutive (same day, 60000 ms increments)
        is_consecutive = True
        current_day = days[i]
        current_ms = ms[i]
        
        for j in range(1, seq_length):
            if days[i + j] != current_day or ms[i + j] != current_ms + j * 60000:
                is_consecutive = False
                break
        
        if is_consecutive:
            seq = features[i:i + seq_length]
            label = targets[i + seq_length - 1]  # Target of the last timestep
            X.append(seq)
            y.append(label)
            i += 1  # Sliding window, step by 1
        else:
            # Skip to the next potential start (after the gap)
            i += 1
    
    return np.array(X), np.array(y)

def balance_samples(X, y, random_seed=42):
    """Balance positive and negative samples"""
    pos_idx = np.where(y > 0)[0]
    neg_idx = np.where(y <= 0)[0]
    
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        print("Warning: Cannot balance - only one class present")
        return X, y
    
    min_count = min(len(pos_idx), len(neg_idx))
    np.random.seed(random_seed)
    pos_sample = np.random.choice(pos_idx, min_count, replace=False)
    neg_sample = np.random.choice(neg_idx, min_count, replace=False)
    balanced_idx = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(balanced_idx)
    
    X_balanced = X[balanced_idx]
    y_balanced = y[balanced_idx]
    
    print(f"Balanced samples: positive={np.sum(y_balanced > 0)}, negative={np.sum(y_balanced <= 0)}")
    
    return X_balanced, y_balanced

def get_thresholded_direction_accuracies(actual, predicted):
    """Calculate thresholded direction accuracies"""
    actual_up = (actual > 0)
    actual_down = (actual <= 0)
    results = {}
    for t in np.arange(0, 0.81, 0.1):
        pred_up_thr = (predicted > t)
        pred_down_thr = (predicted < -t)
        n_pred_up_thr = np.sum(pred_up_thr)
        n_pred_down_thr = np.sum(pred_down_thr)
        tu_thr = np.sum(pred_up_thr & actual_up)
        td_thr = np.sum(pred_down_thr & actual_down)
        up_acc_thr = tu_thr / n_pred_up_thr if n_pred_up_thr > 0 else 0.0
        down_acc_thr = td_thr / n_pred_down_thr if n_pred_down_thr > 0 else 0.0
        results[f'up_acc_thr_{t:.1f}'] = up_acc_thr
        results[f'down_acc_thr_{t:.1f}'] = down_acc_thr
    return results

def print_direction_accuracy(actual, predicted, label):
    """Print direction accuracy statistics"""
    actual_up = (actual > 0)
    actual_down = (actual <= 0)
    pred_up = (predicted > 0)
    pred_down = (predicted <= 0)

    n_pred_up = np.sum(pred_up)
    n_pred_down = np.sum(pred_down)
    n_actual_up = np.sum(actual_up)
    n_actual_down = np.sum(actual_down)

    tu = np.sum(pred_up & actual_up)
    td = np.sum(pred_down & actual_down)

    up_acc = tu / n_pred_up if n_pred_up > 0 else 0.0
    down_acc = td / n_pred_down if n_pred_down > 0 else 0.0

    print(f"\n{label} set direction prediction:")
    print(f"  Actual up count: {n_actual_up}, down count: {n_actual_down}")
    print(f"  Predicted up count: {n_pred_up}, down count: {n_pred_down}")
    print(f"  Up prediction accuracy: {up_acc:.2%} ({tu}/{n_pred_up})")
    print(f"  Down prediction accuracy: {down_acc:.2%} ({td}/{n_pred_down})")

    print(f"\n{label} set thresholded direction prediction:")
    for t in np.arange(0, 0.81, 0.1):
        pred_up_thr = (predicted > t)
        pred_down_thr = (predicted < -t)
        n_pred_up_thr = np.sum(pred_up_thr)
        n_pred_down_thr = np.sum(pred_down_thr)
        tu_thr = np.sum(pred_up_thr & actual_up)
        td_thr = np.sum(pred_down_thr & actual_down)
        up_acc_thr = tu_thr / n_pred_up_thr if n_pred_up_thr > 0 else 0.0
        down_acc_thr = td_thr / n_pred_down_thr if n_pred_down_thr > 0 else 0.0
        print(f"  Threshold: {t:.1f} | Up acc: {up_acc_thr:.2%} ({tu_thr}/{n_pred_up_thr}), Down acc: {down_acc_thr:.2%} ({td_thr}/{n_pred_down_thr})")

def train_regime_model(regime_number, label_number, validation_split=0.2, 
                        seq_length=15, learning_rate=0.0009, epochs=150, batch_size=32):
    """Train LSTM model for a specific paradigm"""
    print(f"\n{'='*60}")
    print(f"Training LSTM model for Regime {regime_number}, Label {label_number}")
    print(f"{'='*60}")
    
    # Load paradigm data
    df, regime_weeks = load_regime_data(regime_number)
    
    # Split into train/validation by weeks
    train_df, val_df = split_regime_data(df, regime_weeks, validation_split)
    
    # Define feature and target columns
    feature_cols = df.columns[5:49]  # Features: columns 6 to 49 (0-based index 5 to 48)
    target_col = f'Label_{label_number}'
    num_features = len(feature_cols)
    
    print(f"Using {num_features} features for Label_{label_number}")
    
    # Create sequences
    print("Creating training sequences...")
    X_train, y_train = create_sequences(train_df, seq_length, feature_cols, target_col)
    
    print("Creating validation sequences...")
    X_val, y_val = create_sequences(val_df, seq_length, feature_cols, target_col)
    
    print(f'Training sequences shape: {X_train.shape}, Training labels shape: {y_train.shape}')
    print(f'Validation sequences shape: {X_val.shape}, Validation labels shape: {y_val.shape}')
    
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Insufficient data to create sequences for training or validation")
    
    # Balance training samples
    X_train, y_train = balance_samples(X_train, y_train)
    
    # Scale features (fit on train, apply to validation)
    scaler = MinMaxScaler()
    X_train_reshaped = X_train.reshape(-1, num_features)
    X_val_reshaped = X_val.reshape(-1, num_features)
    
    scaler.fit(X_train_reshaped)
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
    
    # Generate model ID
    model_log_path = models_dir / 'regime_model_log.csv'
    
    if model_log_path.exists():
        with open(model_log_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) > 1:
                # Parse existing model IDs to find the highest number for this paradigm
                max_id = 0
                for row in rows[1:]:  # Skip header
                    model_id_field = row[0]
                    if isinstance(model_id_field, str) and '_' in model_id_field:
                        try:
                            # Extract paradigm and numeric part from P00_00001 format
                            parts = model_id_field.split('_')
                            if len(parts) == 2:
                                paradigm_part = int(parts[0][1:])  # Remove 'P' and convert
                                numeric_part = int(parts[1])
                                if paradigm_part == regime_number:
                                    max_id = max(max_id, numeric_part)
                        except (ValueError, IndexError):
                            continue
                model_id = max_id + 1
            else:
                model_id = 1
    else:
        model_id = 1
    
    model_id_str = f"R{regime_number:02d}_{model_id:05d}"
    
    # Save scaler parameters
    scaler_params = {
        "Min": scaler.data_min_.tolist(),
        "Max": scaler.data_max_.tolist()
    }
    scaler_params_path = models_dir / f'scaler_params_{model_id_str}.json'
    with open(scaler_params_path, 'w') as f:
        json.dump(scaler_params, f)
    
    # Build model
    print("Building LSTM model...")
    model = keras.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, num_features)),
        keras.layers.LSTM(50),
        keras.layers.Dense(1)  # Output layer for regression
    ])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    
    # Train model
    print("Training model...")
    early_stop = EarlyStopping(monitor='val_mae', patience=100, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=120, min_lr=1e-6, verbose=1)
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_scaled, y_val),
        callbacks=[reduce_lr],
        verbose=1
    )
    
    # Get final metrics
    train_loss = history.history['loss'][-1] if 'loss' in history.history else None
    train_mae = history.history['mae'][-1] if 'mae' in history.history else None
    val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
    val_mae = history.history['val_mae'][-1] if 'val_mae' in history.history else None
    
    # Evaluate
    val_loss_eval, val_mae_eval = model.evaluate(X_val_scaled, y_val, verbose=0)
    print(f'Validation Loss (MSE): {val_loss_eval:.6f}')
    print(f'Validation MAE: {val_mae_eval:.6f}')
    
    # Save model
    model_path = models_dir / f'lstm_regime_model_{model_id_str}.keras'
    model.save(model_path)
    print(f"Saved model to {model_path}")
    
    # Make predictions for accuracy analysis
    train_pred = model.predict(X_train_scaled, verbose=0).flatten()
    val_pred = model.predict(X_val_scaled, verbose=0).flatten()
    
    # Calculate direction accuracies
    train_thr_accs = get_thresholded_direction_accuracies(y_train, train_pred)
    val_thr_accs = get_thresholded_direction_accuracies(y_val, val_pred)
    
    # Print accuracy results
    print_direction_accuracy(y_train, train_pred, 'Training')
    print_direction_accuracy(y_val, val_pred, 'Validation')
    
    # Log results
    base_header = ['model_id', 'paradigm', 'label_number', 'validation_split', 'seq_length', 
                   'learning_rate', 'epochs', 'batch_size', 'train_weeks', 'val_weeks',
                   'train_samples', 'val_samples', 'train_loss', 'train_mae', 'val_loss', 'val_mae']
    base_row = [model_id_str, regime_number, label_number, validation_split, seq_length,
                learning_rate, epochs, batch_size, len(regime_weeks) - int(len(regime_weeks) * validation_split),
                int(len(regime_weeks) * validation_split), len(y_train), len(y_val),
                train_loss, train_mae, val_loss, val_mae]
    
    # Add thresholded accuracies
    thresholds = [round(t, 1) for t in np.arange(0, 0.81, 0.1)]
    split_names = ['train', 'val']
    split_accs = [train_thr_accs, val_thr_accs]
    
    header = base_header.copy()
    row = base_row.copy()
    for split, accs in zip(split_names, split_accs):
        for t in thresholds:
            header.append(f'{split}_up_acc_thr_{t:.1f}')
            header.append(f'{split}_down_acc_thr_{t:.1f}')
            row.append(accs.get(f'up_acc_thr_{t:.1f}', 0.0))
            row.append(accs.get(f'down_acc_thr_{t:.1f}', 0.0))
    
    # Write to log
    write_header = not model_log_path.exists()
    mode = 'w' if write_header else 'a'
    with open(model_log_path, mode, newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    
    print(f"\n{'='*60}")
    print(f"Model training completed successfully!")
    print(f"Model ID: {model_id_str}")
    print(f"Regime: {regime_number}")
    print(f"Label: {label_number}")
    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Validation MAE: {val_mae_eval:.6f}")
    print(f"{'='*60}")
    
    return model_id_str, val_mae_eval

def main():
    parser = argparse.ArgumentParser(description='Train regime-specific LSTM models')
    parser.add_argument('regime_number', type=int, help='Regime number to train on (0-7, based on your regime analysis)')
    parser.add_argument('label_number', type=int, help='Label number to predict (1-10)')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio (default: 0.2)')
    parser.add_argument('--seq_length', type=int, default=15, help='Sequence length (default: 15)')
    parser.add_argument('--learning_rate', type=float, default=0.0009, help='Learning rate (default: 0.0009)')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs (default: 150)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    
    args = parser.parse_args()
    
    # Validate regime number (check against actual regime analysis)
    regime_file = regime_dir / 'regime_assignments.csv'
    if regime_file.exists():
        regime_df = pd.read_csv(regime_file)
        available_regimes = sorted(regime_df['Regime'].unique())
        max_regime = max(available_regimes)
        min_regime = min(available_regimes)
        
        if args.regime_number < min_regime or args.regime_number > max_regime:
            raise ValueError(f"Regime number must be between {min_regime} and {max_regime}. Available regimes: {available_regimes}")
            raise ValueError(f"Regime number must be between {min_paradigm} and {max_paradigm}. Available regimes: {available_paradigms}")
    else:
        raise FileNotFoundError(f"Regime assignments file not found: {regime_file}")
    
    # Validate label number
    if args.label_number < 1 or args.label_number > 10:
        raise ValueError("Label number must be between 1 and 10")
    
    # Check if regime analysis exists
    if not regime_dir.exists():
        raise FileNotFoundError(f"Regime analysis directory not found: {regime_dir}")
    
    # Train model
    model_id, val_mae = train_regime_model(
        regime_number=args.regime_number,
        label_number=args.label_number,
        validation_split=args.validation_split,
        seq_length=args.seq_length,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print(f"\nTraining completed. Model ID: {model_id}")

if __name__ == "__main__":
    main()
