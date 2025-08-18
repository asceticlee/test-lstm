#!/usr/bin/env python3
"""
Batch Model Performance Script (Trading Day Based)

This script calculates daily performance metrics for LSTM models by analyzing
their prediction files and computing threshold-based accuracies and PnL.

For each trading day, it generates performance data for all specified models including:
- Upside/downside threshold accuracies (0.0 to 0.8 in 0.1 increments)
- Numerators and denominators for accuracy calculations
- Profit and Loss (PnL) for each threshold level
- Trading day summary statistics

Output files are saved to test-lstm/model_performance/daily_performance/ directory
with naming pattern: trading_day_YYYYMMDD_performance.csv
Each file contains all models' performance for that specific trading day.

Trading Logic:
- Upside: If prediction >= threshold, go long. PnL = actual_value if actual >= 0, else actual_value
- Downside: If prediction <= -threshold, go short. PnL = -actual_value if actual <= 0, else -actual_value

The script supports incremental updates - if a trading day file already exists,
it will only append missing models rather than regenerating the entire file.

Usage:
    python batch_model_performance.py <start_model_id> <end_model_id>
    
Examples:
    python batch_model_performance.py 1 10     # Process models 00001 to 00010
    python batch_model_performance.py 377 377  # Process only model 00377
"""

import sys
import os
import csv
import time
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

def load_existing_trading_day_performance(performance_file):
    """
    Load existing trading day performance data and return set of ModelIDs already processed
    """
    existing_models = set()
    if os.path.exists(performance_file):
        try:
            df = pd.read_csv(performance_file)
            for _, row in df.iterrows():
                existing_models.add(str(row['ModelID']))
        except Exception as e:
            print(f"Warning: Could not read existing file {performance_file}: {e}")
    return existing_models

def append_to_trading_day_performance_file(performance_file, new_data, thresholds, timeframes):
    """
    Append new performance data to trading day CSV file
    """
    file_exists = os.path.exists(performance_file)
    
    with open(performance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file doesn't exist
        if not file_exists:
            header = ['ModelID', 'TradingDay']
            
            # For each timeframe, add all threshold columns
            for tf_name, tf_spec in timeframes:
                # Upside accuracy columns for this timeframe
                for t in thresholds:
                    header.extend([
                        f'{tf_name}_up_acc_thr_{t:.1f}',
                        f'{tf_name}_up_num_thr_{t:.1f}',
                        f'{tf_name}_up_den_thr_{t:.1f}',
                        f'{tf_name}_up_pnl_thr_{t:.1f}'
                    ])
                # Downside accuracy columns for this timeframe
                for t in thresholds:
                    header.extend([
                        f'{tf_name}_down_acc_thr_{t:.1f}',
                        f'{tf_name}_down_num_thr_{t:.1f}',
                        f'{tf_name}_down_den_thr_{t:.1f}',
                        f'{tf_name}_down_pnl_thr_{t:.1f}'
                    ])
            writer.writerow(header)
        
        # Write data rows
        for row in new_data:
            writer.writerow(row)

def calculate_timeframe_performance(prediction_df, trading_days, current_day_idx, timeframe_spec, thresholds):
    """
    Calculate performance metrics for a specific timeframe ending on current day
    
    Args:
        prediction_df: DataFrame with prediction data
        trading_days: Sorted list of unique trading days
        current_day_idx: Index of current day in trading_days list
        timeframe_spec: Tuple of (type, value) where type is 'trading_days', 'calendar_days', or 'from_begin'
        thresholds: List of threshold values
    
    Returns:
        List of performance metrics for the timeframe
    """
    current_day = trading_days[current_day_idx]
    
    if timeframe_spec[0] == 'from_begin':
        # From beginning: use all data up to current day
        start_day_idx = 0
    elif timeframe_spec[0] == 'trading_days':
        # Look back specified number of trading days
        trading_days_back = timeframe_spec[1]
        start_day_idx = max(0, current_day_idx - trading_days_back + 1)
    elif timeframe_spec[0] == 'calendar_days':
        # Look back specified number of calendar days
        calendar_days_back = timeframe_spec[1]
        current_date = pd.to_datetime(str(current_day), format='%Y%m%d')
        start_date = current_date - pd.Timedelta(days=calendar_days_back - 1)
        start_day_num = int(start_date.strftime('%Y%m%d'))
        
        # Find the earliest trading day that falls within this calendar period
        start_day_idx = 0
        for i, day in enumerate(trading_days[:current_day_idx + 1]):
            if day >= start_day_num:
                start_day_idx = i
                break
    else:
        raise ValueError(f"Unknown timeframe type: {timeframe_spec[0]}")
    
    # Get trading days in the timeframe
    timeframe_trading_days = trading_days[start_day_idx:current_day_idx + 1]
    
    # Filter data for the timeframe
    timeframe_data = prediction_df[prediction_df['TradingDay'].isin(timeframe_trading_days)].copy()
    
    if len(timeframe_data) == 0:
        # Return zeros if no data
        results = []
        for threshold in thresholds:
            results.extend([0.0, 0, 0, 0.0])  # up_acc, up_num, up_den, up_pnl
        for threshold in thresholds:
            results.extend([0.0, 0, 0, 0.0])  # down_acc, down_num, down_den, down_pnl
        return results
    
    results = []
    
    # Calculate upside performance for each threshold
    for threshold in thresholds:
        # Upside logic: go long if prediction >= threshold
        long_positions = timeframe_data['Predicted'] >= threshold
        
        if long_positions.sum() == 0:
            # No positions taken
            up_acc = 0.0
            up_num = 0
            up_den = 0
            up_pnl = 0.0
        else:
            # Calculate accuracy: correct if actual >= 0 when we went long
            correct_predictions = (timeframe_data.loc[long_positions, 'Actual'] >= 0).sum()
            total_predictions = long_positions.sum()
            
            up_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            up_num = correct_predictions
            up_den = total_predictions
            
            # Calculate PnL: sum of actual values when we went long
            up_pnl = timeframe_data.loc[long_positions, 'Actual'].sum()
        
        results.extend([up_acc, up_num, up_den, up_pnl])
    
    # Calculate downside performance for each threshold
    for threshold in thresholds:
        # Downside logic: go short if prediction <= -threshold
        short_positions = timeframe_data['Predicted'] <= -threshold
        
        if short_positions.sum() == 0:
            # No positions taken
            down_acc = 0.0
            down_num = 0
            down_den = 0
            down_pnl = 0.0
        else:
            # Calculate accuracy: correct if actual <= 0 when we went short
            correct_predictions = (timeframe_data.loc[short_positions, 'Actual'] <= 0).sum()
            total_predictions = short_positions.sum()
            
            down_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            down_num = correct_predictions
            down_den = total_predictions
            
            # Calculate PnL: negative sum of actual values when we went short
            down_pnl = -timeframe_data.loc[short_positions, 'Actual'].sum()
        
        results.extend([down_acc, down_num, down_den, down_pnl])
    
    return results

def calculate_multi_timeframe_performance_optimized(trading_day_array, predicted_array, actual_array, trading_day_to_indices, trading_days, current_day_idx, timeframes, thresholds_array, model_id):
    """
    Optimized version using numpy arrays for faster computation
    """
    current_day = trading_days[current_day_idx]
    results = [model_id, current_day]
    
    # Calculate performance for each timeframe
    for tf_name, tf_spec in timeframes:
        tf_results = calculate_timeframe_performance_optimized(
            trading_day_array, predicted_array, actual_array, trading_day_to_indices,
            trading_days, current_day_idx, tf_spec, thresholds_array
        )
        results.extend(tf_results)
    
    return results

def calculate_timeframe_performance_optimized(trading_day_array, predicted_array, actual_array, trading_day_to_indices, trading_days, current_day_idx, timeframe_spec, thresholds_array):
    """
    Optimized timeframe performance calculation using numpy operations
    """
    current_day = trading_days[current_day_idx]
    
    # Determine timeframe trading days
    if timeframe_spec[0] == 'from_begin':
        start_day_idx = 0
    elif timeframe_spec[0] == 'trading_days':
        trading_days_back = timeframe_spec[1]
        start_day_idx = max(0, current_day_idx - trading_days_back + 1)
    elif timeframe_spec[0] == 'calendar_days':
        calendar_days_back = timeframe_spec[1]
        current_date = pd.to_datetime(str(current_day), format='%Y%m%d')
        start_date = current_date - pd.Timedelta(days=calendar_days_back - 1)
        start_day_num = int(start_date.strftime('%Y%m%d'))
        
        start_day_idx = 0
        for i, day in enumerate(trading_days[:current_day_idx + 1]):
            if day >= start_day_num:
                start_day_idx = i
                break
    else:
        raise ValueError(f"Unknown timeframe type: {timeframe_spec[0]}")
    
    # Get indices for timeframe data
    timeframe_trading_days = trading_days[start_day_idx:current_day_idx + 1]
    timeframe_indices = []
    for day in timeframe_trading_days:
        if day in trading_day_to_indices:
            timeframe_indices.extend(trading_day_to_indices[day])
    
    if len(timeframe_indices) == 0:
        # Return zeros if no data
        results = []
        for _ in thresholds_array:
            results.extend([0.0, 0, 0, 0.0])  # up_acc, up_num, up_den, up_pnl
        for _ in thresholds_array:
            results.extend([0.0, 0, 0, 0.0])  # down_acc, down_num, down_den, down_pnl
        return results
    
    # Convert to numpy arrays for vectorized operations
    timeframe_indices = np.array(timeframe_indices)
    timeframe_predicted = predicted_array[timeframe_indices]
    timeframe_actual = actual_array[timeframe_indices]
    
    results = []
    
    # Vectorized upside calculations for all thresholds at once
    long_masks = timeframe_predicted[:, np.newaxis] >= thresholds_array
    up_results = []
    
    for i, threshold in enumerate(thresholds_array):
        long_mask = long_masks[:, i]
        
        if np.sum(long_mask) == 0:
            up_results.extend([0.0, 0, 0, 0.0])
        else:
            correct_predictions = np.sum(timeframe_actual[long_mask] >= 0)
            total_predictions = np.sum(long_mask)
            up_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            up_pnl = np.sum(timeframe_actual[long_mask])
            up_results.extend([up_acc, correct_predictions, total_predictions, up_pnl])
    
    # Vectorized downside calculations for all thresholds at once
    short_masks = timeframe_predicted[:, np.newaxis] <= -thresholds_array
    down_results = []
    
    for i, threshold in enumerate(thresholds_array):
        short_mask = short_masks[:, i]
        
        if np.sum(short_mask) == 0:
            down_results.extend([0.0, 0, 0, 0.0])
        else:
            correct_predictions = np.sum(timeframe_actual[short_mask] <= 0)
            total_predictions = np.sum(short_mask)
            down_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            down_pnl = -np.sum(timeframe_actual[short_mask])
            down_results.extend([down_acc, correct_predictions, total_predictions, down_pnl])
    
    results.extend(up_results)
    results.extend(down_results)
    
    return results

def calculate_multi_timeframe_performance(prediction_df, trading_days, current_day_idx, timeframes, thresholds, model_id):
    """
    Calculate performance metrics for multiple timeframes ending on current day
    
    Args:
        prediction_df: DataFrame with prediction data
        trading_days: Sorted list of unique trading days
        current_day_idx: Index of current day in trading_days list
        timeframes: List of (name, timeframe_spec) tuples
        thresholds: List of threshold values
        model_id: Model ID string
    
    Returns:
        List starting with model_id, trading day followed by all timeframe metrics
    """
    current_day = trading_days[current_day_idx]
    results = [model_id, current_day]
    
    # Calculate performance for each timeframe
    for tf_name, tf_spec in timeframes:
        tf_results = calculate_timeframe_performance(
            prediction_df, trading_days, current_day_idx, tf_spec, thresholds
        )
        results.extend(tf_results)
    
    return results

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

def process_trading_day_performance(trading_day, model_range, prediction_dir, performance_dir):
    """
    Process performance for all models on a single trading day
    """
    try:
        print(f"Processing trading day {trading_day}...")
        
        # Trading day performance file
        performance_file = os.path.join(performance_dir, f'trading_day_{trading_day}_performance.csv')
        
        # Load existing models to avoid duplication
        existing_models = load_existing_trading_day_performance(performance_file)
        print(f"  Found {len(existing_models)} existing models")
        
        # Define thresholds and timeframes
        thresholds = np.arange(0.0, 0.81, 0.1)
        timeframes = [
            ('daily', ('trading_days', 1)),
            ('2day', ('trading_days', 2)),
            ('3day', ('trading_days', 3)),
            ('1week', ('calendar_days', 7)),
            ('2week', ('calendar_days', 14)),
            ('4week', ('calendar_days', 28)),
            ('8week', ('calendar_days', 56)),
            ('13week', ('calendar_days', 91)),
            ('26week', ('calendar_days', 182)),
            ('52week', ('calendar_days', 364)),
            ('from_begin', ('from_begin', None))
        ]
        
        # Collect performance data for all models on this trading day
        new_performance_data = []
        processed_models = 0
        
        start_model_id, end_model_id = model_range
        
        for model_num in range(start_model_id, end_model_id + 1):
            model_id = f"{model_num:05d}"
            
            # Skip if model already processed
            if model_id in existing_models:
                continue
            
            # Check if prediction file exists
            prediction_file = os.path.join(prediction_dir, f'model_{model_id}_prediction.csv')
            if not os.path.exists(prediction_file):
                continue
            
            try:
                # Load prediction data
                prediction_df = pd.read_csv(prediction_file)
                
                # Check if this trading day exists in the model's data
                if trading_day not in prediction_df['TradingDay'].values:
                    continue
                
                # Get sorted trading days for this model
                trading_days = sorted(prediction_df['TradingDay'].unique())
                
                # Find the index of the current trading day
                if trading_day not in trading_days:
                    continue
                
                current_day_idx = trading_days.index(trading_day)
                
                # Pre-compute data structures for optimization
                trading_day_array = prediction_df['TradingDay'].values
                predicted_array = prediction_df['Predicted'].values
                actual_array = prediction_df['Actual'].values
                
                # Create trading day to index mapping
                trading_day_to_indices = {}
                for i, day in enumerate(trading_day_array):
                    if day not in trading_day_to_indices:
                        trading_day_to_indices[day] = []
                    trading_day_to_indices[day].append(i)
                
                # Calculate performance for this model on this trading day
                daily_result = calculate_multi_timeframe_performance_optimized(
                    trading_day_array, predicted_array, actual_array, trading_day_to_indices,
                    trading_days, current_day_idx, timeframes, thresholds, model_id
                )
                
                if daily_result is not None:
                    new_performance_data.append(daily_result)
                    processed_models += 1
                
            except Exception as e:
                print(f"    Error processing model {model_id}: {e}")
                continue
        
        # Append new data to trading day file
        if new_performance_data:
            append_to_trading_day_performance_file(performance_file, new_performance_data, thresholds, timeframes)
            print(f"  Successfully saved {len(new_performance_data)} models to {performance_file}")
        else:
            print(f"  No new models to add for trading day {trading_day}")
        
        return processed_models
        
    except Exception as e:
        print(f"ERROR processing trading day {trading_day}: {e}")
        return 0

def generate_models_alltime_performance(start_model_id, end_model_id, performance_dir):
    """
    Generate models_alltime_performance.csv with all models' from_begin performance
    This file is overwritten each time the script runs
    """
    print("\n" + "=" * 70)
    print("GENERATING MODELS ALL-TIME PERFORMANCE SUMMARY")
    print("=" * 70)
    
    # Define thresholds
    thresholds = np.arange(0.0, 0.81, 0.1)
    
    # Prepare the output file
    output_file = os.path.join(os.path.dirname(performance_dir), 'models_alltime_performance.csv')
    
    # Collect all models' from_begin performance
    all_models_data = []
    processed_models = 0
    
    for model_num in range(start_model_id, end_model_id + 1):
        model_id = f"{model_num:05d}"
        performance_file = os.path.join(performance_dir, f"model_{model_id}_daily_performance.csv")
        
        if not os.path.exists(performance_file):
            print(f"  Warning: Performance file not found for model {model_id}")
            continue
            
        try:
            # Read the performance file
            df = pd.read_csv(performance_file)
            
            if len(df) == 0:
                print(f"  Warning: Empty performance file for model {model_id}")
                continue
            
            # Get the latest (most recent) from_begin performance data
            # This represents the all-time performance up to the latest trading day
            latest_row = df.iloc[-1]
            
            # Create a row for this model's all-time performance
            model_row = {'ModelID': model_id}
            
            # Add upside metrics for each threshold
            for i, threshold in enumerate(thresholds):
                threshold_str = f"{threshold:.1f}"
                model_row[f'alltime_up_acc_{threshold_str}'] = latest_row[f'from_begin_up_acc_thr_{threshold_str}']
                model_row[f'alltime_up_num_{threshold_str}'] = latest_row[f'from_begin_up_num_thr_{threshold_str}']
                model_row[f'alltime_up_den_{threshold_str}'] = latest_row[f'from_begin_up_den_thr_{threshold_str}']
                model_row[f'alltime_up_pnl_{threshold_str}'] = latest_row[f'from_begin_up_pnl_thr_{threshold_str}']
            
            # Add downside metrics for each threshold
            for i, threshold in enumerate(thresholds):
                threshold_str = f"{threshold:.1f}"
                model_row[f'alltime_down_acc_{threshold_str}'] = latest_row[f'from_begin_down_acc_thr_{threshold_str}']
                model_row[f'alltime_down_num_{threshold_str}'] = latest_row[f'from_begin_down_num_thr_{threshold_str}']
                model_row[f'alltime_down_den_{threshold_str}'] = latest_row[f'from_begin_down_den_thr_{threshold_str}']
                model_row[f'alltime_down_pnl_{threshold_str}'] = latest_row[f'from_begin_down_pnl_thr_{threshold_str}']
            
            all_models_data.append(model_row)
            processed_models += 1
            
        except Exception as e:
            print(f"  Error processing model {model_id}: {e}")
            continue
    
    if len(all_models_data) == 0:
        print("  No valid model data found. Skipping all-time performance file generation.")
        return
    
    # Create DataFrame and save to CSV
    try:
        models_df = pd.DataFrame(all_models_data)
        models_df.to_csv(output_file, index=False)
        print(f"  Successfully created {output_file}")
        print(f"  Included {processed_models} models in all-time performance summary")
        
    except Exception as e:
        print(f"  Error creating all-time performance file: {e}")

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
    performance_dir = os.path.join(project_root, 'model_performance', 'daily_performance')
    
    # Create performance directory if it doesn't exist
    os.makedirs(performance_dir, exist_ok=True)
    
    # Check if prediction directory exists
    if not os.path.exists(prediction_dir):
        print(f"ERROR: Prediction directory not found: {prediction_dir}")
        print("Please run batch_model_prediction.py first to generate prediction files.")
        sys.exit(1)
    
    # Get all unique trading days from all model prediction files
    print("Discovering trading days from prediction files...")
    all_trading_days = set()
    
    for model_num in range(start_model_id, end_model_id + 1):
        model_id = f"{model_num:05d}"
        prediction_file = os.path.join(prediction_dir, f'model_{model_id}_prediction.csv')
        
        if os.path.exists(prediction_file):
            try:
                df = pd.read_csv(prediction_file, usecols=['TradingDay'])
                all_trading_days.update(df['TradingDay'].unique())
            except Exception as e:
                print(f"Warning: Could not read {prediction_file}: {e}")
                continue
    
    if not all_trading_days:
        print("ERROR: No trading days found in prediction files")
        sys.exit(1)
    
    # Sort trading days
    sorted_trading_days = sorted(list(all_trading_days))
    print(f"Found {len(sorted_trading_days)} unique trading days from {start_model_id:05d} to {end_model_id:05d}")
    
    # Process each trading day
    successful_days = 0
    total_models_processed = 0
    
    print(f"\nProcessing trading days with models {start_model_id:05d} to {end_model_id:05d}")
    print(f"Prediction directory: {prediction_dir}")
    print(f"Performance output directory: {performance_dir}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    start_time = time.time()
    
    for i, trading_day in enumerate(sorted_trading_days):
        print(f"[{i+1}/{len(sorted_trading_days)}] ", end="")
        models_processed = process_trading_day_performance(
            trading_day, (start_model_id, end_model_id), prediction_dir, performance_dir
        )
        
        if models_processed > 0:
            successful_days += 1
            total_models_processed += models_processed
        
        print()  # Add blank line between trading days
    
    # Generate models_alltime_performance.csv with all models' from_begin performance
    generate_models_alltime_performance(start_model_id, end_model_id, performance_dir)
    
    # Generate performance index for fast lookup
    try:
        sys.path.append(script_dir)
        from performance_index_generator import PerformanceIndexManager
        
        print("\nGenerating daily performance data index...")
        index_manager = PerformanceIndexManager(project_root)
        index_manager.generate_daily_performance_index(start_model_id, end_model_id)
        
    except Exception as e:
        print(f"Warning: Could not generate performance index: {e}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Summary
    print("=" * 70)
    print("Batch processing completed!")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Successfully processed: {successful_days} trading days")
    print(f"Total model records processed: {total_models_processed}")
    print(f"Total trading days attempted: {len(sorted_trading_days)}")
    
    if successful_days < len(sorted_trading_days):
        failed_days = len(sorted_trading_days) - successful_days
        print(f"\nWARNING: {failed_days} trading days had no valid model data. Check the error messages above.")
    
    print(f"\nPerformance files saved to: {performance_dir}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Performance stats
    if total_models_processed > 0:
        avg_models_per_day = total_models_processed / successful_days if successful_days > 0 else 0
        avg_time_per_model = elapsed_time / total_models_processed
        print(f"\nPerformance Statistics:")
        print(f"Average models per trading day: {avg_models_per_day:.1f}")
        print(f"Average time per model record: {avg_time_per_model:.4f} seconds")
    else:
        print("\nNo performance statistics available - no model records were processed successfully.")

if __name__ == "__main__":
    main()