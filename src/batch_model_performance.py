#!/usr/bin/env python3
"""
Batch Model Performance Script

This script calculates daily performance metrics for LSTM models by analyzing
their prediction files and computing threshold-based accuracies and PnL.

For each model, it generates daily performance data including:
- Upside/downside threshold accuracies (0.0 to 0.8 in 0.1 increments)
- Numerators and denominators for accuracy calculations
- Profit and Loss (PnL) for each threshold level
- Trading day summary statistics

Output files are saved to test-lstm/model_performance/model_daily_performance/ directory
with naming pattern: model_xxxxx_daily_performance.csv

Trading Logic:
- Upside: If prediction >= threshold, go long. PnL = actual_value if actual >= 0, else actual_value
- Downside: If prediction <= -threshold, go short. PnL = -actual_value if actual <= 0, else -actual_value

The script supports incremental updates - if a performance file already exists,
it will only append missing trading days rather than regenerating the entire file.

Usage:
    python batch_model_performance.py <start_model_id> <end_model_id>
    
Examples:
    python batch_model_performance.py 1 10     # Process models 00001 to 00010
    python batch_model_performance.py 377 377  # Process only model 00377
"""

import sys
import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

def load_existing_performance(performance_file):
    """
    Load existing performance data and return set of TradingDays already processed
    """
    existing_days = set()
    if os.path.exists(performance_file):
        try:
            df = pd.read_csv(performance_file)
            for _, row in df.iterrows():
                existing_days.add(int(row['TradingDay']))
        except Exception as e:
            print(f"Warning: Could not read existing file {performance_file}: {e}")
    return existing_days

def append_to_performance_file(performance_file, new_data, thresholds):
    """
    Append new performance data to CSV file
    """
    file_exists = os.path.exists(performance_file)
    
    with open(performance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file doesn't exist
        if not file_exists:
            header = ['TradingDay']
            # Upside accuracy columns
            for t in thresholds:
                header.extend([
                    f'up_acc_thr_{t:.1f}',
                    f'up_num_thr_{t:.1f}',
                    f'up_den_thr_{t:.1f}',
                    f'up_pnl_thr_{t:.1f}'
                ])
            # Downside accuracy columns
            for t in thresholds:
                header.extend([
                    f'down_acc_thr_{t:.1f}',
                    f'down_num_thr_{t:.1f}',
                    f'down_den_thr_{t:.1f}',
                    f'down_pnl_thr_{t:.1f}'
                ])
            writer.writerow(header)
        
        # Write data rows
        for row in new_data:
            writer.writerow(row)

def calculate_daily_performance(prediction_df, trading_day, thresholds):
    """
    Calculate performance metrics for a single trading day
    
    Returns:
        List of performance metrics for the trading day
    """
    # Filter data for the specific trading day
    day_data = prediction_df[prediction_df['TradingDay'] == trading_day].copy()
    
    if len(day_data) == 0:
        return None
    
    results = [trading_day]
    
    # Calculate upside performance for each threshold
    for threshold in thresholds:
        # Upside logic: go long if prediction >= threshold
        long_positions = day_data['Predicted'] >= threshold
        
        if long_positions.sum() == 0:
            # No positions taken
            up_acc = 0.0
            up_num = 0
            up_den = 0
            up_pnl = 0.0
        else:
            # Calculate accuracy: correct if actual >= 0 when we went long
            correct_predictions = (day_data.loc[long_positions, 'Actual'] >= 0).sum()
            total_predictions = long_positions.sum()
            
            up_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            up_num = correct_predictions
            up_den = total_predictions
            
            # Calculate PnL: sum of actual values when we went long
            up_pnl = day_data.loc[long_positions, 'Actual'].sum()
        
        results.extend([up_acc, up_num, up_den, up_pnl])
    
    # Calculate downside performance for each threshold
    for threshold in thresholds:
        # Downside logic: go short if prediction <= -threshold
        short_positions = day_data['Predicted'] <= -threshold
        
        if short_positions.sum() == 0:
            # No positions taken
            down_acc = 0.0
            down_num = 0
            down_den = 0
            down_pnl = 0.0
        else:
            # Calculate accuracy: correct if actual <= 0 when we went short
            correct_predictions = (day_data.loc[short_positions, 'Actual'] <= 0).sum()
            total_predictions = short_positions.sum()
            
            down_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            down_num = correct_predictions
            down_den = total_predictions
            
            # Calculate PnL: negative sum of actual values when we went short
            down_pnl = -day_data.loc[short_positions, 'Actual'].sum()
        
        results.extend([down_acc, down_num, down_den, down_pnl])
    
    return results

def process_model_performance(model_id, prediction_dir, performance_dir):
    """
    Process daily performance for a single model
    """
    try:
        # File paths
        prediction_file = os.path.join(prediction_dir, f'model_{model_id}_prediction.csv')
        performance_file = os.path.join(performance_dir, f'model_{model_id}_daily_performance.csv')
        
        # Check if prediction file exists
        if not os.path.exists(prediction_file):
            print(f"Prediction file not found: {prediction_file}")
            return False
        
        print(f"Processing model {model_id}...")
        
        # Load existing performance data to avoid duplication
        existing_days = load_existing_performance(performance_file)
        print(f"  Found {len(existing_days)} existing trading days")
        
        # Load prediction data
        try:
            prediction_df = pd.read_csv(prediction_file)
        except Exception as e:
            print(f"  ERROR loading prediction file: {e}")
            return False
        
        if len(prediction_df) == 0:
            print(f"  No prediction data found for model {model_id}")
            return False
        
        print(f"  Loaded {len(prediction_df):,} prediction records")
        
        # Get unique trading days
        trading_days = sorted(prediction_df['TradingDay'].unique())
        new_days = [day for day in trading_days if day not in existing_days]
        
        if not new_days:
            print(f"  No new trading days to process for model {model_id}")
            return True
        
        print(f"  Processing {len(new_days)} new trading days")
        
        # Define thresholds (0.0 to 0.8 in 0.1 increments)
        thresholds = np.arange(0.0, 0.81, 0.1)
        
        # Calculate daily performance for new days
        new_performance_data = []
        
        for trading_day in new_days:
            daily_result = calculate_daily_performance(prediction_df, trading_day, thresholds)
            if daily_result is not None:
                new_performance_data.append(daily_result)
        
        # Append new data to file
        if new_performance_data:
            append_to_performance_file(performance_file, new_performance_data, thresholds)
            print(f"  Successfully saved {len(new_performance_data)} days to {performance_file}")
        else:
            print(f"  No valid performance data to add for model {model_id}")
        
        return True
        
    except Exception as e:
        print(f"ERROR processing model {model_id}: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_model_performance.py <start_model_id> <end_model_id>")
        print("Examples:")
        print("  python batch_model_performance.py 1 10     # Process models 00001 to 00010")
        print("  python batch_model_performance.py 377 377  # Process only model 00377")
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
    prediction_dir = os.path.join(project_root, 'model_predictions')
    performance_dir = os.path.join(project_root, 'model_performance', 'model_daily_performance')
    
    # Create performance directory if it doesn't exist
    os.makedirs(performance_dir, exist_ok=True)
    
    # Check if prediction directory exists
    if not os.path.exists(prediction_dir):
        print(f"ERROR: Prediction directory not found: {prediction_dir}")
        print("Please run batch_model_prediction.py first to generate prediction files.")
        sys.exit(1)
    
    # Process each model
    successful_models = 0
    failed_models = 0
    
    print(f"\nProcessing models {start_model_id:05d} to {end_model_id:05d}")
    print(f"Prediction directory: {prediction_dir}")
    print(f"Performance output directory: {performance_dir}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    for model_num in range(start_model_id, end_model_id + 1):
        model_id = f"{model_num:05d}"
        
        if process_model_performance(model_id, prediction_dir, performance_dir):
            successful_models += 1
        else:
            failed_models += 1
        
        print()  # Add blank line between models
    
    # Summary
    print("=" * 70)
    print(f"Batch processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successfully processed: {successful_models} models")
    print(f"Failed to process: {failed_models} models")
    print(f"Total models attempted: {successful_models + failed_models}")
    
    if failed_models > 0:
        print(f"\nWARNING: {failed_models} models failed to process. Check the error messages above.")

if __name__ == "__main__":
    main()