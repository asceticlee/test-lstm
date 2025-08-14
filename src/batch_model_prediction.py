#!/usr/bin/env python3
"""
Batch Model Prediction Script

This script generates detailed prediction data for LSTM models by running each model
on the entire tradingData.csv dataset and saving the results as CSV files with
TradingDay, TradingMsOfDay, Actual, and Predicted columns.

Output files are saved to test-lstm/model_predictions/ directory with naming pattern:
model_xxxxx_prediction.csv (where xxxxx is the zero-padded model number)

The script supports incremental updates - if a prediction file already exists,
it will only append missing data rather than regenerating the entire file.

Usage:
    python batch_model_prediction.py <start_model_id> <end_model_id>
    
Examples:
    python batch_model_prediction.py 1 10     # Process models 00001 to 00010
    python batch_model_prediction.py 377 377  # Process only model 00377
"""

import sys
import os
import json
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
import tensorflow as tf
from datetime import datetime

# Define asymmetric MSE loss function (needed for loading models that use it)
def asymmetric_mse(y_true, y_pred):
    error = y_true - y_pred
    return tf.reduce_mean(tf.where(error > 0, error**2, 1.5 * error**2))

def create_sequences(df, seq_length, feature_cols, target_col):
    """
    Create sequences for LSTM prediction from DataFrame
    """
    df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
    features = df[feature_cols].values
    targets = df[target_col].values
    days = df['TradingDay'].values
    ms = df['TradingMsOfDay'].values
    
    X, y, day_indices, ms_indices = [], [], [], []
    i = 0
    
    while i <= len(df) - seq_length:
        is_consecutive = True
        current_day = days[i]
        current_ms = ms[i]
        
        # Check if sequence is consecutive within same day
        for j in range(1, seq_length):
            if days[i + j] != current_day or ms[i + j] != current_ms + j * 60000:
                is_consecutive = False
                break
        
        if is_consecutive:
            seq = features[i:i + seq_length]
            label = targets[i + seq_length - 1]
            X.append(seq)
            y.append(label)
            # Store the trading day and ms for the prediction (last element of sequence)
            day_indices.append(days[i + seq_length - 1])
            ms_indices.append(ms[i + seq_length - 1])
            i += 1
        else:
            i += 1
    
    return np.array(X), np.array(y), np.array(day_indices), np.array(ms_indices)

def load_existing_prediction(prediction_file):
    """
    Load existing prediction data and return set of (TradingDay, TradingMsOfDay) tuples
    """
    existing_data = set()
    if os.path.exists(prediction_file):
        try:
            df = pd.read_csv(prediction_file)
            for _, row in df.iterrows():
                existing_data.add((int(row['TradingDay']), int(row['TradingMsOfDay'])))
        except Exception as e:
            print(f"Warning: Could not read existing file {prediction_file}: {e}")
    return existing_data

def append_to_prediction_file(prediction_file, new_data):
    """
    Append new prediction data to CSV file
    """
    file_exists = os.path.exists(prediction_file)
    
    with open(prediction_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(['TradingDay', 'TradingMsOfDay', 'Actual', 'Predicted'])
        
        # Write data rows
        for row in new_data:
            writer.writerow(row)

def process_model_prediction(model_id, data_df, model_dir, prediction_dir):
    """
    Process prediction for a single model
    """
    try:
        # File paths
        model_path = os.path.join(model_dir, f'lstm_stock_model_{model_id}.keras')
        scaler_path = os.path.join(model_dir, f'scaler_params_{model_id}.json')
        prediction_file = os.path.join(prediction_dir, f'model_{model_id}_prediction.csv')
        
        # Check if model files exist
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        
        if not os.path.exists(scaler_path):
            print(f"Scaler file not found: {scaler_path}")
            return False
        
        print(f"Processing model {model_id}...")
        
        # Load existing prediction data to avoid duplication
        existing_data = load_existing_prediction(prediction_file)
        print(f"  Found {len(existing_data)} existing records")
        
        # Load model and scaler params
        model = keras.models.load_model(model_path)
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)
        
        # Get training period and label number from model log
        model_log_path = os.path.join(model_dir, 'model_log.csv')
        label_number = 10  # Default fallback
        train_from = None
        train_to = None
        
        if os.path.exists(model_log_path):
            with open(model_log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('model_id') == model_id:
                        label_number = int(row.get('label_number', 10))
                        train_from = int(row.get('train_from')) if row.get('train_from') else None
                        train_to = int(row.get('train_to')) if row.get('train_to') else None
                        break
        
        # Filter out training period from data
        if train_from is not None and train_to is not None:
            print(f"  Excluding training period: {train_from} to {train_to}")
            # Exclude training period data
            filtered_df = data_df[~((data_df['TradingDay'] >= train_from) & (data_df['TradingDay'] <= train_to))].copy()
            print(f"  Original data: {len(data_df):,} rows, After excluding training: {len(filtered_df):,} rows")
        else:
            print(f"  Warning: Could not find training period for model {model_id}, using all data")
            filtered_df = data_df.copy()
        
        # Feature columns (same as training)
        feature_cols = filtered_df.columns[5:49]  # Assuming columns 5-48 are features
        target_col = f'Label_{label_number}'
        num_features = len(feature_cols)
        seq_length = 30  # Must match training
        
        print(f"  Using {len(feature_cols)} features and target {target_col}")
        
        # Create sequences
        X, y, day_indices, ms_indices = create_sequences(filtered_df, seq_length, feature_cols, target_col)
        
        if len(X) == 0:
            print(f"  No valid sequences found for model {model_id}")
            return False
        
        print(f"  Created {len(X)} sequences")
        
        # Reconstruct scaler from saved params
        if 'Min' in scaler_params and 'Max' in scaler_params:
            # MinMaxScaler
            scaler = MinMaxScaler()
            scaler.data_min_ = np.array(scaler_params['Min'])
            scaler.data_max_ = np.array(scaler_params['Max'])
            scaler.data_range_ = scaler.data_max_ - scaler.data_min_
            scaler.feature_range = (0, 1)
            scaler.scale_ = (scaler.feature_range[1] - scaler.feature_range[0]) / scaler.data_range_
            scaler.min_ = scaler.feature_range[0] - scaler.data_min_ * scaler.scale_
        elif 'Mean' in scaler_params and 'Variance' in scaler_params:
            # StandardScaler
            scaler = StandardScaler()
            scaler.mean_ = np.array(scaler_params['Mean'])
            scaler.var_ = np.array(scaler_params['Variance'])
            scaler.scale_ = np.sqrt(scaler.var_)
        else:
            raise ValueError(f"Unknown scaler parameters: {list(scaler_params.keys())}")
        
        # Scale features
        X_reshaped = X.reshape(-1, num_features)
        X_scaled = scaler.transform(X_reshaped).reshape(X.shape)
        
        # Predict
        print(f"  Running predictions...")
        y_pred = model.predict(X_scaled, verbose=0).flatten()
        
        # Filter out existing data and prepare new data for writing
        new_data = []
        skipped_count = 0
        
        for i in range(len(y)):
            day_ms_tuple = (int(day_indices[i]), int(ms_indices[i]))
            
            if day_ms_tuple not in existing_data:
                new_data.append([
                    int(day_indices[i]),
                    int(ms_indices[i]),
                    float(y[i]),
                    float(y_pred[i])
                ])
            else:
                skipped_count += 1
        
        print(f"  Skipped {skipped_count} existing records")
        print(f"  Adding {len(new_data)} new records")
        
        # Append new data to file
        if new_data:
            append_to_prediction_file(prediction_file, new_data)
            print(f"  Successfully saved to {prediction_file}")
        else:
            print(f"  No new data to add for model {model_id}")
        
        return True
        
    except Exception as e:
        print(f"ERROR processing model {model_id}: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_model_prediction.py <start_model_id> <end_model_id>")
        print("Examples:")
        print("  python batch_model_prediction.py 1 10     # Process models 00001 to 00010")
        print("  python batch_model_prediction.py 377 377  # Process only model 00377")
        sys.exit(1)
    
    try:
        start_model_id = int(sys.argv[1])
        end_model_id = int(sys.argv[2])
    except ValueError:
        print("ERROR: Model IDs must be integers")
        sys.exit(1)
    
    if start_model_id > end_model_id:
        print("ERROR: Start model ID must be <= end model ID")
        sys.exit(1)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_file = os.path.join(project_root, 'data', 'trainingData.csv')
    model_dir = os.path.join(project_root, 'models')
    prediction_dir = os.path.join(project_root, 'model_predictions')
    
    # Create prediction directory if it doesn't exist
    os.makedirs(prediction_dir, exist_ok=True)
    
    # Load trading data
    print("Loading trading data...")
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        sys.exit(1)
    
    try:
        data_df = pd.read_csv(data_file)
        print(f"Loaded {len(data_df):,} rows of trading data")
    except Exception as e:
        print(f"ERROR loading data file: {e}")
        sys.exit(1)
    
    # Process each model
    successful_models = 0
    failed_models = 0
    
    print(f"\nProcessing models {start_model_id:05d} to {end_model_id:05d}")
    print(f"Output directory: {prediction_dir}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    for model_num in range(start_model_id, end_model_id + 1):
        model_id = f"{model_num:05d}"
        
        if process_model_prediction(model_id, data_df, model_dir, prediction_dir):
            successful_models += 1
        else:
            failed_models += 1
        
        print()  # Add blank line between models
    
    # Summary
    print("=" * 60)
    print(f"Batch processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successfully processed: {successful_models} models")
    print(f"Failed to process: {failed_models} models")
    print(f"Total models attempted: {successful_models + failed_models}")
    
    if failed_models > 0:
        print(f"\nWARNING: {failed_models} models failed to process. Check the error messages above.")

if __name__ == "__main__":
    main()
