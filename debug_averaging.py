#!/usr/bin/env python3
"""Debug script to check if the averaging logic is working correctly"""
import pandas as pd
import numpy as np
import sys
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir  # Since we're running from test-lstm directory
data_dir = os.path.join(project_root, 'data/')
data_file = os.path.join(data_dir, 'trainingData.csv')

# Test parameters
labelNumber = 5  # Should average labels 3,4,5,6,7
half_window = labelNumber // 2
start_label = labelNumber - half_window
end_label = labelNumber + half_window
label_range = list(range(start_label, end_label + 1))

print(f"Testing labelNumber {labelNumber}")
print(f"Should average labels: {label_range}")

# Load a small sample of data
full_df = pd.read_csv(data_file)
sample_df = full_df.head(1000)  # Just first 1000 rows for testing

# Check if required columns exist
required_label_cols = [f'Label_{i}' for i in label_range]
print(f"Required columns: {required_label_cols}")

for col in required_label_cols:
    if col not in sample_df.columns:
        print(f"ERROR: Column {col} not found!")
        sys.exit(1)

# Test the averaging calculation
print("\nTesting averaging calculation on first 10 rows:")
for i in range(10):
    individual_values = sample_df.iloc[i][required_label_cols].values
    calculated_avg = np.mean(individual_values)
    print(f"Row {i}: {individual_values} -> avg = {calculated_avg:.6f}")

# Also test what the original single label would be
single_label_col = f'Label_{labelNumber}'
print(f"\nComparison with single Label_{labelNumber}:")
for i in range(10):
    individual_values = sample_df.iloc[i][required_label_cols].values
    calculated_avg = np.mean(individual_values)
    single_value = sample_df.iloc[i][single_label_col]
    print(f"Row {i}: avg = {calculated_avg:.6f}, single = {single_value:.6f}, diff = {abs(calculated_avg - single_value):.6f}")

print("\nDebugging completed successfully!")
