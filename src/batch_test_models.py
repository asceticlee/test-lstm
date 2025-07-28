import csv
import subprocess
import os
import sys
import numpy as np
from datetime import datetime, timedelta
from test_model import test_model

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
    print("Usage: python batch_test_models.py <start_model_id> <end_model_id> [test_type] [date_from] [date_to]")
    print("Examples:")
    print("  python batch_test_models.py 1 5                    # Test on test periods (default)")
    print("  python batch_test_models.py 1 5 test               # Test on test periods (columns 6,7)")
    print("  python batch_test_models.py 1 5 val                # Test on validation periods (columns 4,5)")
    print("  python batch_test_models.py 1 5 custom 20250101 20250131  # Test on custom date range")
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
model_log_path = os.path.join(project_root, 'models', 'model_log.csv')
model_test_path = os.path.join(project_root, 'models', 'model_test.csv')

with open(model_log_path, newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)


thresholds = [f'{t:.1f}' for t in list(np.arange(0, 0.81, 0.1))]
header = ['model_id', 'test_from', 'test_to']
# First all up thresholds, then all down thresholds
for t in thresholds:
    header.append(f'test_up_acc_thr_{t}')
for t in thresholds:
    header.append(f'test_down_acc_thr_{t}')

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
    
    print(f"Processing models from {start_model_id} to {end_model_id}: {model_ids}")
    
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
            
            # Call test_model function directly instead of subprocess
            csv_row = test_model(model_id, test_from, test_to, accuracy_only=True)
            
            # Check if we got a valid result
            if len(csv_row) != len(header):
                print(f"Error or invalid output for model_id={model_id}, test_from={test_from}, test_to={test_to}")
                csv_row = [model_id, test_from, test_to] + [''] * (len(header) - 3)
            
            writer.writerow(csv_row)
        # Insert a blank row between model blocks (except after the last model)
        if idx < len(model_ids) - 1:
            writer.writerow([])