#!/usr/bin/env python3
"""
Model Cleanup Script

This script removes all files and records associated with a specific model ID.
It cleans up:
1. Model record from model_log.csv
2. Keras model file (lstm_stock_model_{model_id}.keras)
3. Scaler parameters file (scaler_params_{model_id}.json)
4. All prediction CSV files (train, validation, test)

Usage:
    python cleanup_model.py <model_id> [--dry-run]
    
Example:
    python cleanup_model.py 326
    python cleanup_model.py 00326
    python cleanup_model.py 326 --dry-run    # Show what would be deleted without actually deleting

The script accepts model_id with or without leading zeros.
Use --dry-run to see what files would be deleted without actually removing them.
"""

import os
import sys
import csv
import glob
from pathlib import Path


def format_model_id(model_id_input):
    """Convert model_id to 5-digit zero-padded format"""
    try:
        # Remove any leading zeros and convert to int, then back to 5-digit string
        model_id_int = int(str(model_id_input).lstrip('0') or '0')
        return f"{model_id_int:05d}"
    except ValueError:
        raise ValueError(f"Invalid model_id: {model_id_input}")


def get_project_paths():
    """Get project directory paths"""
    script_dir = Path(__file__).parent.absolute()  # This is src/
    project_root = script_dir.parent.absolute()    # Go up one level to project root
    models_dir = project_root / 'models'
    return project_root, models_dir


def remove_model_from_log(model_log_path, model_id, dry_run=False):
    """Remove model record from model_log.csv"""
    if not model_log_path.exists():
        print(f"Model log file does not exist: {model_log_path}")
        return False
    
    # Read all rows except the target model
    rows_to_keep = []
    model_found = False
    
    with open(model_log_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            rows_to_keep.append(header)
        
        for row in reader:
            if row and row[0] == model_id:
                model_found = True
                if dry_run:
                    print(f"[DRY RUN] Would remove model {model_id} from model_log.csv")
                else:
                    print(f"✓ Found model {model_id} in model_log.csv")
            else:
                rows_to_keep.append(row)
    
    if model_found and not dry_run:
        # Write back the filtered rows
        with open(model_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_keep)
        print(f"✓ Removed model {model_id} from model_log.csv")
        return True
    elif model_found and dry_run:
        return True
    else:
        print(f"⚠ Model {model_id} not found in model_log.csv")
        return False


def remove_file_if_exists(file_path, description, dry_run=False):
    """Remove a file if it exists and report the result"""
    if file_path.exists():
        if dry_run:
            print(f"[DRY RUN] Would remove {description}: {file_path.name}")
            return True
        else:
            try:
                file_path.unlink()
                print(f"✓ Removed {description}: {file_path.name}")
                return True
            except Exception as e:
                print(f"✗ Failed to remove {description}: {e}")
                return False
    else:
        print(f"⚠ {description} not found: {file_path.name}")
        return False


def find_prediction_files(project_root, model_id):
    """Find all prediction CSV files for the given model_id"""
    patterns = [
        f"train_predictions_regression_{model_id}_*.csv",
        f"validation_predictions_regression_{model_id}_*.csv", 
        f"test_predictions_regression_{model_id}_*.csv"
    ]
    
    found_files = []
    for pattern in patterns:
        files = list(project_root.glob(pattern))
        found_files.extend(files)
    
    return found_files


def cleanup_model(model_id_input, dry_run=False):
    """Main cleanup function"""
    try:
        model_id = format_model_id(model_id_input)
    except ValueError as e:
        print(f"Error: {e}")
        return False
    
    mode_text = "[DRY RUN] " if dry_run else ""
    print(f"{mode_text}Starting cleanup for model {model_id}")
    print("=" * 50)
    
    project_root, models_dir = get_project_paths()
    
    # Track what was actually removed
    removed_count = 0
    total_expected = 0
    
    # 1. Remove from model_log.csv
    model_log_path = models_dir / 'model_log.csv'
    if remove_model_from_log(model_log_path, model_id, dry_run):
        removed_count += 1
    total_expected += 1
    
    # 2. Remove model file
    model_file_path = models_dir / f'lstm_stock_model_{model_id}.keras'
    if remove_file_if_exists(model_file_path, "Model file", dry_run):
        removed_count += 1
    total_expected += 1
    
    # 3. Remove scaler parameters file
    scaler_file_path = models_dir / f'scaler_params_{model_id}.json'
    if remove_file_if_exists(scaler_file_path, "Scaler parameters file", dry_run):
        removed_count += 1
    total_expected += 1
    
    # 4. Remove prediction files
    prediction_files = find_prediction_files(project_root, model_id)
    total_expected += len(prediction_files)
    
    if prediction_files:
        print(f"\nFound {len(prediction_files)} prediction files:")
        for pred_file in prediction_files:
            if remove_file_if_exists(pred_file, "Prediction file", dry_run):
                removed_count += 1
    else:
        print("⚠ No prediction files found")
    
    # Summary
    print("\n" + "=" * 50)
    if dry_run:
        print(f"Dry Run Summary for Model {model_id}:")
        print(f"Files that would be removed: {removed_count}")
        print(f"Total files found: {total_expected}")
        if removed_count == 0:
            print("⚠ No files found - model might not exist")
        else:
            print("✓ Run without --dry-run to actually delete these files")
    else:
        print(f"Cleanup Summary for Model {model_id}:")
        print(f"Files removed: {removed_count}")
        print(f"Expected files: {total_expected}")
        
        if removed_count == 0:
            print("⚠ No files were removed - model might not exist or already cleaned")
            return False
        elif removed_count == total_expected:
            print("✓ All model files successfully removed!")
            return True
        else:
            print("⚠ Partial cleanup completed - some files may not have existed")
            return True
    
    return True


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python cleanup_model.py <model_id> [--dry-run]")
        print("Example: python cleanup_model.py 326")
        print("Example: python cleanup_model.py 00326")
        print("Example: python cleanup_model.py 326 --dry-run")
        sys.exit(1)
    
    model_id_input = sys.argv[1]
    dry_run = len(sys.argv) == 3 and sys.argv[2] == '--dry-run'
    
    # For dry run, just show what would be deleted
    if dry_run:
        success = cleanup_model(model_id_input, dry_run=True)
        sys.exit(0 if success else 1)
    
    # Confirm deletion for actual cleanup
    try:
        formatted_id = format_model_id(model_id_input)
        response = input(f"Are you sure you want to delete all files for model {formatted_id}? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            sys.exit(0)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    success = cleanup_model(model_id_input, dry_run=False)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
