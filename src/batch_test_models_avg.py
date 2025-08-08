import csv
import subprocess
import os
import sys
import numpy as np
from datetime import datetime, timedelta

def get_thresholded_direction_accuracies_dual(actual_avg, actual_center, predicted_avg):
    """
    Calculate thresholded direction accuracies for both comparison types:
    1. PredictedAvg vs ActualAvg (like original)
    2. PredictedAvg vs Actual (center label)
    """
    results = {}
    
    # Type 1: PredictedAvg vs ActualAvg
    actual_avg_up = (actual_avg > 0)
    actual_avg_down = (actual_avg <= 0)
    
    # Type 2: PredictedAvg vs Actual (center label)
    actual_center_up = (actual_center > 0)
    actual_center_down = (actual_center <= 0)
    
    for t in np.arange(0, 0.81, 0.1):
        pred_up_thr = (predicted_avg > t)
        pred_down_thr = (predicted_avg < -t)
        n_pred_up_thr = np.sum(pred_up_thr)
        n_pred_down_thr = np.sum(pred_down_thr)
        
        # Type 1: PredictedAvg vs ActualAvg
        tu_avg_thr = np.sum(pred_up_thr & actual_avg_up)
        td_avg_thr = np.sum(pred_down_thr & actual_avg_down)
        up_acc_avg_thr = tu_avg_thr / n_pred_up_thr if n_pred_up_thr > 0 else 0.0
        down_acc_avg_thr = td_avg_thr / n_pred_down_thr if n_pred_down_thr > 0 else 0.0
        
        # Type 2: PredictedAvg vs Actual (center label)
        tu_center_thr = np.sum(pred_up_thr & actual_center_up)
        td_center_thr = np.sum(pred_down_thr & actual_center_down)
        up_acc_center_thr = tu_center_thr / n_pred_up_thr if n_pred_up_thr > 0 else 0.0
        down_acc_center_thr = td_center_thr / n_pred_down_thr if n_pred_down_thr > 0 else 0.0
        
        # Store results with clear naming
        results[f'test_up_acc_avg_thr_{t:.1f}'] = up_acc_avg_thr
        results[f'test_down_acc_avg_thr_{t:.1f}'] = down_acc_avg_thr
        results[f'test_up_acc_center_thr_{t:.1f}'] = up_acc_center_thr
        results[f'test_down_acc_center_thr_{t:.1f}'] = down_acc_center_thr
    
    return results

def test_model_avg_dual(model_id, test_from, test_to):
    """
    Test an averaging model and return both types of accuracy comparisons.
    """
    try:
        # Import here to avoid circular imports
        import json
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from tensorflow import keras
        import tensorflow as tf
        
        # Paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_file = os.path.join(project_root, 'data', 'trainingData.csv')
        model_dir = os.path.join(project_root, 'models')
        model_path = os.path.join(model_dir, f'lstm_stock_model_avg_{model_id}.keras')
        scaler_path = os.path.join(model_dir, f'scaler_params_avg_{model_id}.json')

        # Load model and scaler params
        model = keras.models.load_model(model_path)
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)

        # Get label number and label range from model log
        model_log_path = os.path.join(project_root, 'models', 'model_log_avg.csv')
        label_number = 5  # Default fallback
        label_range = [3, 4, 5, 6, 7]  # Default fallback
        
        if os.path.exists(model_log_path):
            with open(model_log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('model_id') == model_id:
                        label_number = int(row.get('label_number', 5))
                        # Parse label_range from the format "start-end"
                        label_range_str = row.get('label_range', '3-7')
                        start_label, end_label = map(int, label_range_str.split('-'))
                        label_range = list(range(start_label, end_label + 1))
                        break

        # Load data
        df = pd.read_csv(data_file)
        test_df = df[(df['TradingDay'] >= int(test_from)) & (df['TradingDay'] <= int(test_to))].reset_index(drop=True)

        # Feature columns (same as training)
        feature_cols = test_df.columns[5:49]
        target_cols = [f'Label_{i}' for i in label_range]
        num_features = len(feature_cols)
        seq_length = 15  # Must match training

        # Verify that all required label columns exist
        missing_cols = [col for col in target_cols if col not in test_df.columns]
        if missing_cols:
            raise ValueError(f"Missing label columns in data: {missing_cols}")

        # Sequence creation (same as in averaging training script)
        def create_sequences(df, seq_length):
            df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
            features = df[feature_cols].values
            target_values = df[target_cols].values
            days = df['TradingDay'].values
            ms = df['TradingMsOfDay'].values
            
            X = []
            y_center = []  # Center label values
            y_avg = []     # Average values
            
            i = 0
            while i <= len(df) - seq_length:
                is_consecutive = True
                current_day = days[i]
                current_ms = ms[i]
                
                for j in range(1, seq_length):
                    if days[i + j] != current_day or ms[i + j] != current_ms + j * 60000:
                        is_consecutive = False
                        break
                
                if is_consecutive:
                    seq = features[i:i + seq_length]
                    # Get individual labels at the prediction timestep (last timestep)
                    individual_labels = target_values[i + seq_length - 1]
                    # Center label value
                    center_label_value = individual_labels[label_range.index(label_number)]
                    # Average value
                    label_avg = np.mean(individual_labels)
                    X.append(seq)
                    y_center.append(center_label_value)
                    y_avg.append(label_avg)
                    i += 1
                else:
                    i += 1
            
            return np.array(X), np.array(y_center), np.array(y_avg)

        X_test, y_test_center, y_test_avg = create_sequences(test_df, seq_length)

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
        elif 'Mean' in scaler_params and 'Variance' in scaler_params:
            # StandardScaler
            scaler = StandardScaler()
            scaler.mean_ = np.array(scaler_params['Mean'])
            scaler.var_ = np.array(scaler_params['Variance'])
            scaler.scale_ = np.sqrt(scaler.var_)
        else:
            raise ValueError(f"Unknown scaler parameters: {list(scaler_params.keys())}")

        X_test_reshaped = X_test.reshape(-1, num_features)
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

        # Predict
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()

        # Calculate dual accuracies
        accs = get_thresholded_direction_accuracies_dual(y_test_avg, y_test_center, y_pred)
        
        # Return results in the order: model_id, test_from, test_to, then all accuracy values
        thresholds = [f'{t:.1f}' for t in list(np.arange(0, 0.81, 0.1))]
        
        # First: PredictedAvg vs ActualAvg (up then down)
        avg_up_values = [f'{accs[f"test_up_acc_avg_thr_{t}"]:.6f}' for t in thresholds]
        avg_down_values = [f'{accs[f"test_down_acc_avg_thr_{t}"]:.6f}' for t in thresholds]
        
        # Second: PredictedAvg vs Actual (center label) (up then down)
        center_up_values = [f'{accs[f"test_up_acc_center_thr_{t}"]:.6f}' for t in thresholds]
        center_down_values = [f'{accs[f"test_down_acc_center_thr_{t}"]:.6f}' for t in thresholds]
        
        row = [model_id, test_from, test_to] + avg_up_values + avg_down_values + center_up_values + center_down_values
        return row
        
    except Exception as e:
        print(f"Error testing model {model_id}: {e}")
        # Return error row with empty accuracy values
        thresholds = [f'{t:.1f}' for t in list(np.arange(0, 0.81, 0.1))]
        num_accuracy_cols = len(thresholds) * 4  # 4 sets of threshold accuracies
        return [model_id, test_from, test_to] + [''] * num_accuracy_cols

def get_weekly_ranges(start_date, end_date):
    """
    Split a date range into weekly ranges (Sunday to Saturday).
    
    Args:
        start_date: Start date in YYYYMMDD format (string)
        end_date: End date in YYYYMMDD format (string)
    
    Returns:
        List of tuples [(week_start, week_end), ...]
    """
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    
    # Find the first Sunday (start of week)
    days_until_sunday = start_dt.weekday() + 1  # Monday=0, so Sunday=6, but we want Sunday=0
    if days_until_sunday == 7:  # Already Sunday
        first_sunday = start_dt
    else:
        first_sunday = start_dt - timedelta(days=days_until_sunday)
    
    # If first Sunday is before start_date, use start_date
    if first_sunday < start_dt:
        first_sunday = start_dt
    
    weekly_ranges = []
    current_sunday = first_sunday
    
    while current_sunday <= end_dt:
        # Calculate Saturday of this week
        saturday = current_sunday + timedelta(days=6)
        
        # Don't go beyond end_date
        week_end = min(saturday, end_dt)
        
        week_start_str = current_sunday.strftime('%Y%m%d')
        week_end_str = week_end.strftime('%Y%m%d')
        
        weekly_ranges.append((week_start_str, week_end_str))
        
        # Move to next Sunday
        current_sunday = current_sunday + timedelta(days=7)
        if current_sunday > end_dt:
            break
    
    return weekly_ranges

# Command line arguments for start and end model IDs
if len(sys.argv) < 3 or len(sys.argv) > 6:
    print("Usage: python batch_test_models_avg.py <start_model_id> <end_model_id> [test_type] [date_from] [date_to]")
    print("Examples:")
    print("  python batch_test_models_avg.py 1 5                    # Test on test periods (default)")
    print("  python batch_test_models_avg.py 1 5 test               # Test on test periods (columns 6,7)")
    print("  python batch_test_models_avg.py 1 5 val                # Test on validation periods (columns 4,5)")
    print("  python batch_test_models_avg.py 1 5 custom 20250101 20250131  # Test on custom date range")
    sys.exit(1)

# Convert input to 5-digit zero-padded format
start_model_id = f"{int(sys.argv[1]):05d}"
end_model_id = f"{int(sys.argv[2]):05d}"

# Determine test type
test_type = sys.argv[3] if len(sys.argv) >= 4 else "test"
custom_from = sys.argv[4] if len(sys.argv) >= 5 else None
custom_to = sys.argv[5] if len(sys.argv) >= 6 else None

if test_type not in ["test", "val", "custom"]:
    print("Error: test_type must be 'test', 'val', or 'custom'")
    sys.exit(1)

if test_type == "custom" and (not custom_from or not custom_to):
    print("Error: custom test type requires date_from and date_to parameters")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
model_log_path = os.path.join(project_root, 'models', 'model_log_avg.csv')  # Use avg log
model_test_path = os.path.join(project_root, 'models', 'model_test_avg.csv')  # Output to avg test file

with open(model_log_path, newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Create header with dual accuracy types
thresholds = [f'{t:.1f}' for t in list(np.arange(0, 0.81, 0.1))]
header = ['model_id', 'test_from', 'test_to']

# First set: PredictedAvg vs ActualAvg (like original)
for t in thresholds:
    header.append(f'test_up_acc_avg_thr_{t}')
for t in thresholds:
    header.append(f'test_down_acc_avg_thr_{t}')

# Second set: PredictedAvg vs Actual (center label)
for t in thresholds:
    header.append(f'test_up_acc_center_thr_{t}')
for t in thresholds:
    header.append(f'test_down_acc_center_thr_{t}')

with open(model_test_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    
    # Get all unique model_ids and filter by start and end range
    all_model_ids = sorted(set(row['model_id'] for row in rows if row.get('model_id')))
    
    # Filter model_ids to only include those in the specified range
    model_ids = []
    for model_id in all_model_ids:
        if start_model_id <= model_id <= end_model_id:
            model_ids.append(model_id)
    
    if not model_ids:
        print(f"No models found in range {start_model_id} to {end_model_id}")
        sys.exit(1)
    
    print(f"Processing averaging models from {start_model_id} to {end_model_id}: {model_ids}")
    
    # Get test_ranges based on test_type
    test_ranges = []
    
    if test_type == "custom":
        # Generate weekly ranges from custom date range
        test_ranges = get_weekly_ranges(custom_from, custom_to)
        print(f"Custom weekly test ranges from {custom_from} to {custom_to}: {test_ranges}")
    else:
        # Get test_ranges from the selected models
        for row in rows:
            if row.get('model_id') and start_model_id <= row['model_id'] <= end_model_id:
                if test_type == "test" and row.get('test_from') and row.get('test_to'):
                    test_range = (row['test_from'], row['test_to'])
                elif test_type == "val" and row.get('val_from') and row.get('val_to'):
                    test_range = (row['val_from'], row['val_to'])
                else:
                    continue
                
                if test_range not in test_ranges:  # Avoid duplicates
                    test_ranges.append(test_range)
        
        period_name = "test" if test_type == "test" else "validation"
        print(f"{period_name.capitalize()} ranges from selected models: {test_ranges}")
    
    # Check if we have any test ranges
    if not test_ranges:
        print(f"No {test_type} ranges found for the specified models. Exiting.")
        sys.exit(1)
    
    # Create a mapping of model_id to its training periods
    model_train_periods = {}
    for row in rows:
        if row.get('model_id') and row.get('train_from') and row.get('train_to'):
            model_train_periods[row['model_id']] = (row['train_from'], row['train_to'])

    for idx, model_id in enumerate(model_ids):
        for test_from, test_to in test_ranges:
            # Skip if test period overlaps with training period for this model
            if model_id in model_train_periods:
                train_from, train_to = model_train_periods[model_id]
                # Check if test period overlaps with training period
                if (int(test_from) >= int(train_from) and int(test_from) <= int(train_to)) or \
                   (int(test_to) >= int(train_from) and int(test_to) <= int(train_to)) or \
                   (int(test_from) <= int(train_from) and int(test_to) >= int(train_to)):
                    print(f"Skipping: model_id={model_id}, test_from={test_from}, test_to={test_to} (overlaps with training period {train_from}-{train_to})")
                    continue
            
            print(f"Processing: model_id={model_id}, test_from={test_from}, test_to={test_to}")
            
            # Call test_model_avg_dual function
            csv_row = test_model_avg_dual(model_id, test_from, test_to)
            
            # Check if we got a valid result
            if len(csv_row) != len(header):
                print(f"Error or invalid output for model_id={model_id}, test_from={test_from}, test_to={test_to}")
                csv_row = [model_id, test_from, test_to] + [''] * (len(header) - 3)
            
            writer.writerow(csv_row)
        
        # Insert a blank row between model blocks (except after the last model)
        if idx < len(model_ids) - 1:
            writer.writerow([])

print(f"Dual accuracy testing completed! Results saved to {model_test_path}")
print("\nColumn explanations:")
print("- test_up_acc_avg_thr_X.X / test_down_acc_avg_thr_X.X: PredictedAvg vs ActualAvg (like original)")
print("- test_up_acc_center_thr_X.X / test_down_acc_center_thr_X.X: PredictedAvg vs Actual (center label)")
