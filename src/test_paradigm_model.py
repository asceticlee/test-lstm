#!/usr/bin/env python3
"""
Test Paradigm-Specific LSTM Models

This script tests paradigm-specific LSTM models on data from specific date ranges.
It can test models on their own paradigm data or cross-paradigm data to evaluate
specialization effectiveness.

Usage:
    python test_paradigm_model.py <model_id> <test_from> <test_to> [--accuracy-only]
    
Example:
    python test_paradigm_model.py P00_00001 20250701 20250707 --accuracy-only
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import tensorflow as tf
import argparse
import csv

# Define asymmetric MSE loss function (needed for loading models that use it)
def asymmetric_mse(y_true, y_pred):
    error = y_true - y_pred
    return tf.reduce_mean(tf.where(error > 0, error**2, 1.5 * error**2))

def get_thresholded_direction_accuracies(actual, predicted):
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
        results[f'test_up_acc_thr_{t:.1f}'] = up_acc_thr
        results[f'test_down_acc_thr_{t:.1f}'] = down_acc_thr
    return results

def create_sequences(df, seq_length, feature_cols, target_col):
    """Create sequences for LSTM testing, respecting data continuity"""
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
            label = targets[i + seq_length - 1]
            X.append(seq)
            y.append(label)
            i += 1
        else:
            i += 1
    
    return np.array(X), np.array(y)

def get_seq_indices(df, seq_length):
    """Get sequence indices for recovering TradingDay and TradingMsOfDay"""
    days = df['TradingDay'].values
    ms = df['TradingMsOfDay'].values
    seq_idx = []
    for i in range(len(df) - seq_length + 1):
        is_consecutive = True
        current_day = days[i]
        current_ms = ms[i]
        for j in range(1, seq_length):
            if days[i + j] != current_day or ms[i + j] != current_ms + j * 60000:
                is_consecutive = False
                break
        if is_consecutive:
            seq_idx.append(i + seq_length - 1)
    return np.array(seq_idx)

def test_paradigm_model(model_id, test_from, test_to, accuracy_only=False):
    """
    Test a paradigm model on a specific date range and return results.
    
    Args:
        model_id: Paradigm model ID string (e.g., "P00_00001")
        test_from: Start date (yyyymmdd string)
        test_to: End date (yyyymmdd string)
        accuracy_only: If True, return only accuracy metrics
    
    Returns:
        If accuracy_only=True: list of [model_id, test_from, test_to, ...accuracy_values]
        If accuracy_only=False: dict with detailed results
    """
    try:
        # Paths
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent
        data_file = project_root / 'data' / 'trainingData.csv'
        model_dir = project_root / 'models'
        paradigm_dir = project_root / 'paradigm_analysis'
        
        model_path = model_dir / f'lstm_paradigm_model_{model_id}.keras'
        scaler_path = model_dir / f'scaler_params_{model_id}.json'
        
        # Load model and scaler params
        model = keras.models.load_model(model_path)
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)
        
        # Get model parameters from paradigm model log
        paradigm_log_path = model_dir / 'paradigm_model_log.csv'
        label_number = 10  # Default fallback
        seq_length = 15   # Default fallback
        paradigm_number = None
        
        if paradigm_log_path.exists():
            with open(paradigm_log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('model_id') == model_id:
                        label_number = int(row.get('label_number', 10))
                        seq_length = int(row.get('seq_length', 15))
                        paradigm_number = int(row.get('paradigm', 0))
                        break
        
        # Load test data
        df = pd.read_csv(data_file)
        test_df = df[(df['TradingDay'] >= int(test_from)) & (df['TradingDay'] <= int(test_to))].reset_index(drop=True)
        
        if len(test_df) == 0:
            raise ValueError(f"No data found for date range {test_from} to {test_to}")
        
        # Feature columns (same as training)
        feature_cols = test_df.columns[5:49]
        target_col = f'Label_{label_number}'
        num_features = len(feature_cols)
        
        # Create sequences
        X_test, y_test = create_sequences(test_df, seq_length, feature_cols, target_col)
        
        if len(X_test) == 0:
            raise ValueError("No valid sequences could be created from test data")
        
        # Correctly reconstruct scaler from saved params
        if 'Min' in scaler_params and 'Max' in scaler_params:
            # MinMaxScaler
            scaler = MinMaxScaler()
            scaler.data_min_ = np.array(scaler_params['Min'])
            scaler.data_max_ = np.array(scaler_params['Max'])
            scaler.data_range_ = scaler.data_max_ - scaler.data_min_
            scaler.feature_range = (0, 1)
            scaler.scale_ = (scaler.feature_range[1] - scaler.feature_range[0]) / scaler.data_range_
            scaler.min_ = scaler.feature_range[0] - scaler.data_min_ * scaler.scale_
        else:
            raise ValueError(f"Unknown scaler parameters: {list(scaler_params.keys())}")
        
        X_test_reshaped = X_test.reshape(-1, num_features)
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
        
        # Predict
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()
        
        # Get paradigm information for test data
        paradigm_file = paradigm_dir / 'paradigm_assignments.csv'
        test_paradigm_info = ""
        if paradigm_file.exists():
            paradigm_df = pd.read_csv(paradigm_file)
            test_days = set(test_df['TradingDay'].unique())
            test_paradigm_data = paradigm_df[paradigm_df['TradingDay'].isin(test_days)]
            if len(test_paradigm_data) > 0:
                test_paradigms = test_paradigm_data['Paradigm'].value_counts()
                test_paradigm_info = f" (Test data paradigms: {dict(test_paradigms)})"
        
        if accuracy_only:
            accs = get_thresholded_direction_accuracies(y_test, y_pred)
            # Return up accuracies first, then down accuracies to match header order
            up_values = [f'{accs[f"test_up_acc_thr_{t:.1f}"]:.6f}' for t in np.arange(0, 0.81, 0.1)]
            down_values = [f'{accs[f"test_down_acc_thr_{t:.1f}"]:.6f}' for t in np.arange(0, 0.81, 0.1)]
            row = [model_id, test_from, test_to] + up_values + down_values
            return row
        else:
            # Return detailed results
            test_seq_idx = get_seq_indices(test_df, seq_length)
            if len(test_seq_idx) > len(y_test):
                test_seq_idx = test_seq_idx[:len(y_test)]
            
            results_df = pd.DataFrame({
                'TradingDay': test_df['TradingDay'].values[test_seq_idx],
                'TradingMsOfDay': test_df['TradingMsOfDay'].values[test_seq_idx],
                'Actual': y_test,
                'Predicted': y_pred
            })
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            return {
                "model_id": model_id,
                "paradigm_number": paradigm_number,
                "label_number": label_number,
                "seq_length": seq_length,
                "test_from": test_from,
                "test_to": test_to,
                "test_samples": len(y_test),
                "test_mae": mae,
                "test_mse": mse,
                "test_paradigm_info": test_paradigm_info,
                "results_df": results_df,
                "accuracy_metrics": get_thresholded_direction_accuracies(y_test, y_pred)
            }
    
    except Exception as e:
        # Return error row for accuracy_only mode
        if accuracy_only:
            return [model_id, test_from, test_to] + [''] * 18  # 18 accuracy columns
        else:
            return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Test paradigm-specific LSTM models')
    parser.add_argument('model_id', type=str, help='Paradigm Model ID (e.g., P00_00001)')
    parser.add_argument('testFrom', type=str, help='Test from date (yyyymmdd)')
    parser.add_argument('testTo', type=str, help='Test to date (yyyymmdd)')
    parser.add_argument('--accuracy-only', action='store_true', help='Print only thresholded up/down accuracy as CSV row')
    
    args = parser.parse_args()
    
    try:
        if args.accuracy_only:
            result = test_paradigm_model(args.model_id, args.testFrom, args.testTo, accuracy_only=True)
            print(','.join(result))
        else:
            result = test_paradigm_model(args.model_id, args.testFrom, args.testTo, accuracy_only=False)
            
            if "error" in result:
                print(f"ERROR: {result['error']}", file=sys.stderr)
                sys.exit(1)
            
            print(f"Testing Paradigm Model: {result['model_id']}")
            print(f"Trained on Paradigm: {result['paradigm_number']}")
            print(f"Label: {result['label_number']}")
            print(f"Sequence Length: {result['seq_length']}")
            print(f"Test Period: {result['test_from']} to {result['test_to']}")
            print(f"Test Samples: {result['test_samples']}")
            print(f"Test MAE: {result['test_mae']:.6f}")
            print(f"Test MSE: {result['test_mse']:.6f}")
            print(f"Test Data{result['test_paradigm_info']}")
            
            print("\nTradingDay,TradingMsOfDay,Actual,Predicted")
            for row in result['results_df'].itertuples(index=False):
                print(f"{row.TradingDay},{row.TradingMsOfDay},{row.Actual:.6f},{row.Predicted:.6f}")
            
            # Print thresholded accuracies
            print("\nThresholded Direction Accuracies:")
            accs = result['accuracy_metrics']
            for t in np.arange(0, 0.81, 0.1):
                up_acc = accs[f'test_up_acc_thr_{t:.1f}']
                down_acc = accs[f'test_down_acc_thr_{t:.1f}']
                print(f"Threshold {t:.1f}: Up {up_acc:.2%}, Down {down_acc:.2%}")
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
