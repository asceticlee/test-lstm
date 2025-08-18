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
import time
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

def append_to_performance_file(performance_file, new_data, thresholds, timeframes):
    """
    Append new performance data to CSV file
    """
    file_exists = os.path.exists(performance_file)
    
    with open(performance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file doesn't exist
        if not file_exists:
            header = ['TradingDay']
            
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

def calculate_multi_timeframe_performance_optimized(trading_day_array, predicted_array, actual_array, trading_day_to_indices, trading_days, current_day_idx, timeframes, thresholds_array):
    """
    Optimized version using numpy arrays for faster computation
    """
    current_day = trading_days[current_day_idx]
    results = [current_day]
    
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

def calculate_multi_timeframe_performance(prediction_df, trading_days, current_day_idx, timeframes, thresholds):
    """
    Calculate performance metrics for multiple timeframes ending on current day
    
    Args:
        prediction_df: DataFrame with prediction data
        trading_days: Sorted list of unique trading days
        current_day_idx: Index of current day in trading_days list
        timeframes: List of (name, timeframe_spec) tuples
        thresholds: List of threshold values
    
    Returns:
        List starting with trading day followed by all timeframe metrics
    """
    current_day = trading_days[current_day_idx]
    results = [current_day]
    
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

def process_model_performance(model_id, prediction_dir, performance_dir):
    """
    Process multi-timeframe performance for a single model
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
        
        # Get unique trading days and filter for new days
        trading_days = sorted(prediction_df['TradingDay'].unique())
        new_days = [day for day in trading_days if day not in existing_days]
        
        if not new_days:
            print(f"  No new trading days to process for model {model_id}")
            return True
        
        print(f"  Processing {len(new_days)} new trading days")
        
        # Define thresholds (0.0 to 0.8 in 0.1 increments)
        thresholds = np.arange(0.0, 0.81, 0.1)
        
        # Define timeframes: (name, (type, value))
        # Type can be 'trading_days', 'calendar_days', or 'from_begin'
        timeframes = [
            ('daily', ('trading_days', 1)),
            ('2day', ('trading_days', 2)),
            ('3day', ('trading_days', 3)),
            ('1week', ('calendar_days', 7)),     # 7 calendar days
            ('2week', ('calendar_days', 14)),    # 14 calendar days
            ('4week', ('calendar_days', 28)),    # 28 calendar days (4 weeks)
            ('8week', ('calendar_days', 56)),    # 56 calendar days (8 weeks)
            ('13week', ('calendar_days', 91)),   # 91 calendar days (13 weeks)
            ('26week', ('calendar_days', 182)),  # 182 calendar days (26 weeks)
            ('52week', ('calendar_days', 364)),  # 364 calendar days (52 weeks)
            ('from_begin', ('from_begin', None)) # All available data from beginning
        ]
        
        # Calculate multi-timeframe performance for new days
        new_performance_data = []
        
        # Pre-compute all data to avoid repeated operations
        print(f"  Pre-computing data structures for optimization...")
        start_time = time.time()
        
        # Convert DataFrame to numpy arrays for faster operations
        trading_day_array = prediction_df['TradingDay'].values
        predicted_array = prediction_df['Predicted'].values  
        actual_array = prediction_df['Actual'].values
        
        # Create trading day to index mapping for faster lookups
        trading_day_to_indices = {}
        for i, day in enumerate(trading_day_array):
            if day not in trading_day_to_indices:
                trading_day_to_indices[day] = []
            trading_day_to_indices[day].append(i)
        
        # Pre-compute threshold arrays for vectorized operations
        thresholds_array = np.array(thresholds)
        
        print(f"  Pre-computation completed in {time.time() - start_time:.2f}s")
        print(f"  Processing {len(new_days)} new trading days with optimized calculations...")
        
        for day in new_days:
            # Find the index of this day in the trading_days list
            day_idx = trading_days.index(day)
            
            # Calculate multi-timeframe performance for this day
            daily_result = calculate_multi_timeframe_performance_optimized(
                trading_day_array, predicted_array, actual_array, trading_day_to_indices,
                trading_days, day_idx, timeframes, thresholds_array
            )
            if daily_result is not None:
                new_performance_data.append(daily_result)
        
        # Append new data to file
        if new_performance_data:
            append_to_performance_file(performance_file, new_performance_data, thresholds, timeframes)
            print(f"  Successfully saved {len(new_performance_data)} days to {performance_file}")
        else:
            print(f"  No valid performance data to add for model {model_id}")
        
        return True
        
    except Exception as e:
        print(f"ERROR processing model {model_id}: {e}")
        return False

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