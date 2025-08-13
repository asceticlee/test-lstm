#!/usr/bin/env python3
"""
Model Prediction Status Checker

This script checks the status of model prediction files in the model_predictions directory.
It shows which models have been processed and the number of records in each file.

Usage:
    python check_model_prediction_status.py [start_model] [end_model]
    
Examples:
    python check_model_prediction_status.py          # Check all files
    python check_model_prediction_status.py 1 10     # Check models 1-10
"""

import os
import sys
import csv
from pathlib import Path

def count_csv_lines(file_path):
    """Count lines in CSV file (excluding header)"""
    try:
        with open(file_path, 'r') as f:
            return sum(1 for line in f) - 1  # Subtract 1 for header
    except Exception as e:
        return f"Error: {e}"

def check_model_prediction_status(start_model=None, end_model=None):
    """Check status of model prediction files"""
    
    # Paths
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    prediction_dir = project_root / 'model_predictions'
    
    if not prediction_dir.exists():
        print(f"Model prediction directory not found: {prediction_dir}")
        return
    
    # Get all prediction files
    prediction_files = list(prediction_dir.glob('model_*_prediction.csv'))
    
    if not prediction_files:
        print("No model prediction files found.")
        return
    
    # Extract model numbers and sort
    model_data = []
    for file_path in prediction_files:
        try:
            # Extract model number from filename like "model_00377_prediction.csv"
            model_num = int(file_path.stem.split('_')[1])
            model_data.append((model_num, file_path))
        except (ValueError, IndexError):
            print(f"Warning: Could not parse model number from {file_path.name}")
    
    model_data.sort(key=lambda x: x[0])
    
    # Filter by range if specified
    if start_model is not None and end_model is not None:
        model_data = [(num, path) for num, path in model_data 
                     if start_model <= num <= end_model]
    
    if not model_data:
        print(f"No models found in range {start_model}-{end_model}")
        return
    
    # Display status
    print("Model Prediction File Status")
    print("=" * 50)
    print(f"{'Model':<8} {'Records':<10} {'File Size':<12} {'Status'}")
    print("-" * 50)
    
    total_records = 0
    total_files = 0
    
    for model_num, file_path in model_data:
        model_id = f"{model_num:05d}"
        record_count = count_csv_lines(file_path)
        
        try:
            file_size = file_path.stat().st_size
            if file_size > 1024*1024:
                size_str = f"{file_size / (1024*1024):.1f} MB"
            elif file_size > 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size} B"
        except Exception:
            size_str = "Unknown"
        
        if isinstance(record_count, int):
            status = "OK"
            total_records += record_count
            total_files += 1
        else:
            status = "Error"
        
        print(f"{model_id:<8} {str(record_count):<10} {size_str:<12} {status}")
    
    print("-" * 50)
    print(f"Total: {total_files} files, {total_records:,} records")
    
    # Show missing models in range if specified
    if start_model is not None and end_model is not None:
        existing_models = {num for num, _ in model_data}
        missing_models = []
        for num in range(start_model, end_model + 1):
            if num not in existing_models:
                missing_models.append(f"{num:05d}")
        
        if missing_models:
            print(f"\nMissing models: {', '.join(missing_models)}")

def main():
    start_model = None
    end_model = None
    
    if len(sys.argv) == 3:
        try:
            start_model = int(sys.argv[1])
            end_model = int(sys.argv[2])
        except ValueError:
            print("Error: Model numbers must be integers")
            sys.exit(1)
    elif len(sys.argv) != 1:
        print("Usage: python check_model_prediction_status.py [start_model] [end_model]")
        sys.exit(1)
    
    check_model_prediction_status(start_model, end_model)

if __name__ == "__main__":
    main()
