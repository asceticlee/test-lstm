#!/usr/bin/env python3
"""
Batch Model Regime Performance Script

This script calculates regime-based performance metrics for LSTM models by analyzing
their prediction files in conjunction with market regime clustering results.

For each model and each trading day, it generates regime-specific performance data 
including:
- Upside/downside threshold accuracies (0.0 to 0.8 in 0.1 increments) 
- Numerators and denominators for accuracy calculations
- Profit and Loss (PnL) for each threshold level
- Performance analysis for lookback periods of 1, 2, 3, 4, 5, 10, 20, 30 trading days

The lookback periods consider the latest N trading days that were clustered as that 
specific regime, allowing analysis of model performance during different market regimes.

Output files are saved to test-lstm/model_performance/model_regime_performance/ directory
with naming pattern: model_xxxxx_regime_performance.csv

Trading Logic:
- Upside: If prediction >= threshold, go long. PnL = actual_value if actual >= 0, else actual_value
- Downside: If prediction <= -threshold, go short. PnL = -actual_value if actual <= 0, else -actual_value

Usage:
    python batch_model_regime_performance.py <start_model_id> <end_model_id>
    
Examples:
    python batch_model_regime_performance.py 1 10     # Process models 00001 to 00010
    python batch_model_regime_performance.py 377 377  # Process only model 00377
"""

import sys
import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import time

def load_regime_assignments(regime_file_path):
    """
    Load daily regime assignments from GMM clustering output
    
    Returns:
        tuple: (dict: {trading_day: regime_id}, list: unique_regimes)
    """
    regime_data = {}
    
    if not os.path.exists(regime_file_path):
        raise FileNotFoundError(f"Regime assignments file not found: {regime_file_path}")
    
    df = pd.read_csv(regime_file_path)
    for _, row in df.iterrows():
        regime_data[int(row['trading_day'])] = int(row['Regime'])
    
    # Get unique regimes that actually exist in the data
    unique_regimes = sorted(df['Regime'].unique())
    
    return regime_data, unique_regimes

def get_regime_lookback_days(regime_assignments, trading_days, current_day_idx, target_regime, lookback_days):
    """
    Get the latest N trading days that were assigned to the target regime
    
    Args:
        regime_assignments: Dict mapping trading_day to regime
        trading_days: Sorted list of unique trading days
        current_day_idx: Index of current day in trading_days list
        target_regime: Target regime ID (from available regimes)
        lookback_days: Number of trading days we want to find for this regime
    
    Returns:
        List of trading days that match the target regime (up to lookback_days count)
        Goes back through ALL historical data until finding enough regime days or reaching start
    """
    if current_day_idx < 0:
        return []
    
    regime_days = []
    
    # Look backwards from current day (inclusive) through ALL historical data
    # No artificial limits - only stop when we have enough regime days or reach the beginning
    for i in range(current_day_idx, -1, -1):
        trading_day = trading_days[i]
        
        # Check if this day has regime assignment
        if trading_day in regime_assignments:
            if regime_assignments[trading_day] == target_regime:
                regime_days.append(trading_day)
                
                # Stop when we have enough days of this regime
                if len(regime_days) >= lookback_days:
                    break
    
    return regime_days

def load_existing_regime_performance(performance_file):
    """
    Load existing regime performance data and return set of (TradingDay, Regime) tuples
    """
    existing_data = set()
    if os.path.exists(performance_file):
        try:
            df = pd.read_csv(performance_file)
            for _, row in df.iterrows():
                existing_data.add((int(row['TradingDay']), int(row['Regime'])))
        except Exception as e:
            print(f"Warning: Could not read existing file {performance_file}: {e}")
    return existing_data

def append_to_regime_performance_file(performance_file, new_data, thresholds, lookback_periods):
    """
    Append new regime performance data to CSV file
    """
    file_exists = os.path.exists(performance_file)
    
    with open(performance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file doesn't exist
        if not file_exists:
            header = ['TradingDay', 'Regime']
            
            # Add columns for each lookback period and threshold
            for period in lookback_periods:
                for threshold in thresholds:
                    header.extend([
                        f'{period}day_up_acc_thr_{threshold:.1f}',
                        f'{period}day_up_num_thr_{threshold:.1f}', 
                        f'{period}day_up_den_thr_{threshold:.1f}',
                        f'{period}day_up_pnl_thr_{threshold:.1f}'
                    ])
                
                for threshold in thresholds:
                    header.extend([
                        f'{period}day_down_acc_thr_{threshold:.1f}',
                        f'{period}day_down_num_thr_{threshold:.1f}',
                        f'{period}day_down_den_thr_{threshold:.1f}', 
                        f'{period}day_down_pnl_thr_{threshold:.1f}'
                    ])
            
            writer.writerow(header)
        
        # Write data rows
        for row in new_data:
            writer.writerow(row)

def calculate_regime_performance_optimized(trading_day_array, predicted_array, actual_array, trading_day_to_indices, 
                                         regime_trading_days, thresholds_array):
    """
    Optimized regime performance calculation using numpy operations
    """
    if len(regime_trading_days) == 0:
        # Return zeros if no regime data
        results = []
        for _ in thresholds_array:
            results.extend([0.0, 0, 0, 0.0])  # up_acc, up_num, up_den, up_pnl
        for _ in thresholds_array:
            results.extend([0.0, 0, 0, 0.0])  # down_acc, down_num, down_den, down_pnl
        return results
    
    # Get indices for regime data
    regime_indices = []
    for day in regime_trading_days:
        if day in trading_day_to_indices:
            regime_indices.extend(trading_day_to_indices[day])
    
    if len(regime_indices) == 0:
        # Return zeros if no data
        results = []
        for _ in thresholds_array:
            results.extend([0.0, 0, 0, 0.0])
        for _ in thresholds_array:
            results.extend([0.0, 0, 0, 0.0])
        return results
    
    # Convert to numpy arrays for vectorized operations
    regime_indices = np.array(regime_indices)
    regime_predicted = predicted_array[regime_indices]
    regime_actual = actual_array[regime_indices]
    
    results = []
    
    # Vectorized upside calculations
    long_masks = regime_predicted[:, np.newaxis] >= thresholds_array
    
    for i, threshold in enumerate(thresholds_array):
        long_mask = long_masks[:, i]
        
        if np.sum(long_mask) == 0:
            results.extend([0.0, 0, 0, 0.0])
        else:
            correct_ups = np.sum(long_mask & (regime_actual >= 0))
            total_ups = np.sum(long_mask)
            up_acc = correct_ups / total_ups
            up_pnl = np.sum(regime_actual[long_mask])
            
            results.extend([up_acc, correct_ups, total_ups, up_pnl])
    
    # Vectorized downside calculations
    short_masks = regime_predicted[:, np.newaxis] <= -thresholds_array
    
    for i, threshold in enumerate(thresholds_array):
        short_mask = short_masks[:, i]
        
        if np.sum(short_mask) == 0:
            results.extend([0.0, 0, 0, 0.0])
        else:
            correct_downs = np.sum(short_mask & (regime_actual <= 0))
            total_downs = np.sum(short_mask)
            down_acc = correct_downs / total_downs
            down_pnl = -np.sum(regime_actual[short_mask])
            
            results.extend([down_acc, correct_downs, total_downs, down_pnl])
    
    return results

def calculate_multi_regime_performance_optimized(trading_day_array, predicted_array, actual_array, 
                                               trading_day_to_indices, trading_days, current_day_idx, 
                                               regime_assignments, current_regime, lookback_periods, 
                                               thresholds_array):
    """
    Calculate performance metrics for multiple lookback periods for a specific regime
    """
    results = [trading_days[current_day_idx], current_regime]
    
    # Calculate performance for each lookback period
    for period in lookback_periods:
        regime_days = get_regime_lookback_days(regime_assignments, trading_days, current_day_idx, 
                                             current_regime, period)
        
        period_results = calculate_regime_performance_optimized(
            trading_day_array, predicted_array, actual_array, trading_day_to_indices,
            regime_days, thresholds_array
        )
        results.extend(period_results)
    
    return results

def process_model_regime_performance(model_id, prediction_dir, regime_performance_dir, regime_assignments, unique_regimes):
    """
    Process regime-based performance for a single model
    """
    try:
        # File paths
        prediction_file = os.path.join(prediction_dir, f'model_{model_id}_prediction.csv')
        performance_file = os.path.join(regime_performance_dir, f'model_{model_id}_regime_performance.csv')
        
        # Check if prediction file exists
        if not os.path.exists(prediction_file):
            print(f"Prediction file not found: {prediction_file}")
            return False
        
        print(f"Processing model {model_id}...")
        
        # Load existing performance data to avoid duplication
        existing_data = load_existing_regime_performance(performance_file)
        print(f"  Found {len(existing_data)} existing (day, regime) combinations")
        
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
        
        # Define parameters
        thresholds = np.arange(0.0, 0.81, 0.1)
        lookback_periods = [1, 2, 3, 4, 5, 10, 20, 30]
        regimes = unique_regimes  # Use actual regimes from data
        
        # Get sorted trading days from prediction data
        trading_days = sorted(prediction_df['TradingDay'].unique())
        print(f"  Found {len(trading_days)} unique trading days")
        
        # Pre-compute data structures for optimization
        trading_day_array = prediction_df['TradingDay'].values
        predicted_array = prediction_df['Predicted'].values
        actual_array = prediction_df['Actual'].values
        
        # Create trading day to indices mapping for faster lookups
        trading_day_to_indices = defaultdict(list)
        for idx, day in enumerate(trading_day_array):
            trading_day_to_indices[day].append(idx)
        
        print(f"  Pre-computed data structures for {len(trading_day_array):,} records")
        
        # Process each trading day and regime combination
        new_data = []
        processed_combinations = 0
        skipped_combinations = 0
        
        for current_day_idx, current_day in enumerate(trading_days):
            # Skip if day not in regime assignments
            if current_day not in regime_assignments:
                continue
            
            for regime in regimes:
                # Check if this combination already exists
                if (current_day, regime) in existing_data:
                    skipped_combinations += 1
                    continue
                
                # Calculate performance for this day-regime combination
                regime_results = calculate_multi_regime_performance_optimized(
                    trading_day_array, predicted_array, actual_array,
                    trading_day_to_indices, trading_days, current_day_idx,
                    regime_assignments, regime, lookback_periods, thresholds
                )
                
                new_data.append(regime_results)
                processed_combinations += 1
        
        print(f"  Processed {processed_combinations} new (day, regime) combinations")
        print(f"  Skipped {skipped_combinations} existing combinations")
        
        # Append new data to file
        if new_data:
            append_to_regime_performance_file(performance_file, new_data, thresholds, lookback_periods)
            print(f"  Successfully saved to {performance_file}")
        else:
            print(f"  No new data to add for model {model_id}")
        
        return True
        
    except Exception as e:
        print(f"ERROR processing model {model_id}: {e}")
        return False

def generate_models_alltime_regime_performance(start_model_id, end_model_id, regime_performance_dir, regime_assignments, unique_regimes):
    """
    Generate models_alltime_regime_performance.csv with all models' from_begin regime performance
    This file is overwritten each time the script runs
    """
    print("\n" + "=" * 70)
    print("GENERATING MODELS ALL-TIME REGIME PERFORMANCE SUMMARY")
    print("=" * 70)
    
    # Define parameters
    thresholds = np.arange(0.0, 0.81, 0.1)
    lookback_periods = [1, 2, 3, 4, 5, 10, 20, 30]
    regimes = unique_regimes  # Use actual regimes from data
    
    # Prepare the output file
    output_file = os.path.join(os.path.dirname(regime_performance_dir), 'models_alltime_regime_performance.csv')
    
    # Collect all models' regime performance
    all_models_data = []
    processed_models = 0
    
    for model_num in range(start_model_id, end_model_id + 1):
        model_id = f"{model_num:05d}"
        
        # Path to model's regime performance file
        model_file = os.path.join(regime_performance_dir, f'model_{model_id}_regime_performance.csv')
        
        if not os.path.exists(model_file):
            print(f"  Skipping model {model_id} - regime performance file not found")
            continue
        
        try:
            # Load model's regime performance data
            model_df = pd.read_csv(model_file)
            
            if len(model_df) == 0:
                print(f"  Skipping model {model_id} - no regime performance data")
                continue
            
            # For each regime, calculate alltime performance (from_begin equivalent)
            for regime in regimes:
                regime_data = model_df[model_df['Regime'] == regime]
                
                if len(regime_data) == 0:
                    # If no data for this regime, create row with zeros
                    row = [model_id, regime]
                    
                    # Add zeros for all metrics
                    for period in lookback_periods:
                        for threshold in thresholds:
                            row.extend([0.0, 0, 0, 0.0])  # up_acc, up_num, up_den, up_pnl
                        for threshold in thresholds:
                            row.extend([0.0, 0, 0, 0.0])  # down_acc, down_num, down_den, down_pnl
                    
                    all_models_data.append(row)
                else:
                    # Get the latest trading day's performance for this regime
                    # (represents cumulative performance from beginning)
                    latest_day_data = regime_data.iloc[-1]
                    
                    # Build row with model_id, regime, and all performance metrics
                    row = [model_id, regime]
                    
                    # Add all performance metrics (excluding TradingDay and Regime columns)
                    for col in latest_day_data.index[2:]:  # Skip TradingDay and Regime
                        row.append(latest_day_data[col])
                    
                    all_models_data.append(row)
            
            processed_models += 1
            if processed_models % 10 == 0:
                print(f"  Processed {processed_models} models...")
                
        except Exception as e:
            print(f"  ERROR processing model {model_id}: {e}")
            continue
    
    if len(all_models_data) == 0:
        print("  No regime performance data found for any models")
        return
    
    # Create DataFrame and save to CSV
    try:
        # Build header
        header = ['ModelID', 'Regime']
        for period in lookback_periods:
            for threshold in thresholds:
                header.extend([
                    f'{period}day_up_acc_thr_{threshold:.1f}',
                    f'{period}day_up_num_thr_{threshold:.1f}',
                    f'{period}day_up_den_thr_{threshold:.1f}',
                    f'{period}day_up_pnl_thr_{threshold:.1f}'
                ])
            for threshold in thresholds:
                header.extend([
                    f'{period}day_down_acc_thr_{threshold:.1f}',
                    f'{period}day_down_num_thr_{threshold:.1f}',
                    f'{period}day_down_den_thr_{threshold:.1f}',
                    f'{period}day_down_pnl_thr_{threshold:.1f}'
                ])
        
        # Create DataFrame
        summary_df = pd.DataFrame(all_models_data, columns=header)
        
        # Save to CSV
        summary_df.to_csv(output_file, index=False)
        
        print(f"  Generated {output_file}")
        print(f"  Total rows: {len(summary_df):,} ({processed_models} models × 6 regimes)")
        
    except Exception as e:
        print(f"  ERROR creating summary file: {e}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_model_regime_performance.py <start_model_id> <end_model_id>")
        print("Examples:")
        print("  python batch_model_regime_performance.py 1 10     # Process models 00001 to 00010")
        print("  python batch_model_regime_performance.py 377 377  # Process only model 00377")
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
    regime_performance_dir = os.path.join(project_root, 'model_performance', 'model_regime_performance')
    regime_file = os.path.join(project_root, 'market_regime', 'gmm', 'daily', 'daily_regime_assignments.csv')
    
    # Create regime performance directory if it doesn't exist
    os.makedirs(regime_performance_dir, exist_ok=True)
    
    # Check if prediction directory exists
    if not os.path.exists(prediction_dir):
        print(f"ERROR: Prediction directory not found: {prediction_dir}")
        sys.exit(1)
    
    # Load regime assignments
    print("Loading regime assignments...")
    try:
        regime_assignments, unique_regimes = load_regime_assignments(regime_file)
        print(f"Loaded regime assignments for {len(regime_assignments)} trading days")
        print(f"Found {len(unique_regimes)} unique regimes: {unique_regimes}")
        
        # Print regime distribution
        regime_counts = defaultdict(int)
        for regime in regime_assignments.values():
            regime_counts[regime] += 1
        
        print("Regime distribution:")
        for regime in sorted(regime_counts.keys()):
            print(f"  Regime {regime}: {regime_counts[regime]} days")
        
    except Exception as e:
        print(f"ERROR loading regime assignments: {e}")
        sys.exit(1)
    
    # Process each model
    successful_models = 0
    failed_models = 0
    
    print(f"\nProcessing models {start_model_id:05d} to {end_model_id:05d}")
    print(f"Prediction directory: {prediction_dir}")
    print(f"Regime performance output directory: {regime_performance_dir}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    start_time = time.time()
    
    for model_num in range(start_model_id, end_model_id + 1):
        model_id = f"{model_num:05d}"
        
        if process_model_regime_performance(model_id, prediction_dir, regime_performance_dir, regime_assignments, unique_regimes):
            successful_models += 1
        else:
            failed_models += 1
        
        print()  # Add blank line between models
    
    # Generate models_alltime_regime_performance.csv with all models' regime performance
    generate_models_alltime_regime_performance(start_model_id, end_model_id, regime_performance_dir, regime_assignments, unique_regimes)
    
    # Summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("=" * 70)
    print(f"Batch processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {elapsed_time:.1f} seconds")
    print(f"Successfully processed: {successful_models} models")
    print(f"Failed to process: {failed_models} models")
    print(f"Total models attempted: {successful_models + failed_models}")
    
    if successful_models > 0:
        avg_time_per_model = elapsed_time / successful_models
        print(f"Average time per model: {avg_time_per_model:.1f} seconds")
    
    if failed_models > 0:
        print(f"\nWARNING: {failed_models} models failed to process. Check the error messages above.")

if __name__ == "__main__":
    main()
