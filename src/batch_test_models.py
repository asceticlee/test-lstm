import csv
import subprocess
import os
import numpy as np
from test_model import test_model

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
    # Get all unique model_ids and all test_from/test_to pairs
    model_ids = sorted(set(row['model_id'] for row in rows if row.get('model_id')))
    test_ranges = [(row['test_from'], row['test_to']) for row in rows if row.get('test_from') and row.get('test_to')]
    
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