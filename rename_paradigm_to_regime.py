#!/usr/bin/env python3
"""
Script to rename all "paradigm" references to "regime" throughout the project.
This includes:
1. Directory names
2. File names  
3. Content within files
4. Column names in CSV files
"""

import os
import shutil
import pandas as pd
from pathlib import Path
import re

def rename_directories(base_path):
    """Rename directories containing 'paradigm' to 'regime'"""
    print("Step 1: Renaming directories...")
    
    paradigm_analysis_dir = base_path / 'paradigm_analysis'
    regime_analysis_dir = base_path / 'regime_analysis'
    
    if paradigm_analysis_dir.exists():
        print(f"Renaming {paradigm_analysis_dir} to {regime_analysis_dir}")
        shutil.move(str(paradigm_analysis_dir), str(regime_analysis_dir))
    
def rename_files(base_path):
    """Rename files containing 'paradigm' to 'regime'"""
    print("Step 2: Renaming files...")
    
    # Get all files that need renaming
    paradigm_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if 'paradigm' in file.lower():
                paradigm_files.append(Path(root) / file)
    
    # Rename files
    for old_path in paradigm_files:
        new_name = old_path.name.replace('paradigm', 'regime').replace('Paradigm', 'Regime')
        new_path = old_path.parent / new_name
        print(f"Renaming {old_path} to {new_path}")
        shutil.move(str(old_path), str(new_path))

def update_file_contents(base_path):
    """Update content within all relevant files"""
    print("Step 3: Updating file contents...")
    
    # File extensions to process
    extensions = ['.py', '.csv', '.md', '.txt', '.json']
    
    # Files to process
    files_to_update = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                files_to_update.append(Path(root) / file)
    
    # Update each file
    for file_path in files_to_update:
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count replacements for reporting
            original_content = content
            
            # Replace paradigm/Paradigm with regime/Regime
            content = re.sub(r'\bparadigm\b', 'regime', content)
            content = re.sub(r'\bParadigm\b', 'Regime', content)
            content = re.sub(r'\bPARADIGM\b', 'REGIME', content)
            
            # Special replacements for specific patterns
            content = content.replace('paradigm_analysis', 'regime_analysis')
            content = content.replace('paradigm_assignments', 'regime_assignments')
            content = content.replace('paradigm_summary', 'regime_summary')
            content = content.replace('paradigm_model_log', 'regime_model_log')
            content = content.replace('lstm_paradigm_model', 'lstm_regime_model')
            content = content.replace('test_all_models_paradigms', 'test_all_models_regimes')
            content = content.replace('stockprice_lstm_paradigm_regression', 'stockprice_lstm_regime_regression')
            content = content.replace('market_paradigm_classifier', 'market_regime_classifier')
            content = content.replace('test_paradigm_model', 'test_regime_model')
            content = content.replace('weekly_paradigms', 'weekly_regimes')
            content = content.replace('model_paradigm_test_results', 'model_regime_test_results')
            content = content.replace('best_paradigm_summary', 'best_regime_summary')
            
            # Write back if changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Updated {file_path}")
                
        except Exception as e:
            print(f"Error updating {file_path}: {e}")

def update_csv_column_names(base_path):
    """Update column names in CSV files"""
    print("Step 4: Updating CSV column names...")
    
    csv_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(Path(root) / file)
    
    for csv_file in csv_files:
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Check if any column names need updating
            original_columns = df.columns.tolist()
            new_columns = []
            
            for col in original_columns:
                new_col = col.replace('paradigm', 'regime').replace('Paradigm', 'Regime')
                new_columns.append(new_col)
            
            # Update if changed
            if new_columns != original_columns:
                df.columns = new_columns
                df.to_csv(csv_file, index=False)
                print(f"Updated column names in {csv_file}")
                print(f"  Old: {original_columns}")
                print(f"  New: {new_columns}")
                
        except Exception as e:
            print(f"Error updating CSV {csv_file}: {e}")

def main():
    base_path = Path('/home/stephen/projects/Testing/TestPy/test-lstm')
    
    print("Starting paradigm -> regime renaming process...")
    print(f"Base path: {base_path}")
    
    # Step 1: Rename directories
    rename_directories(base_path)
    
    # Step 2: Rename files
    rename_files(base_path)
    
    # Step 3: Update file contents
    update_file_contents(base_path)
    
    # Step 4: Update CSV column names
    update_csv_column_names(base_path)
    
    print("\nRenaming process completed!")
    print("All 'paradigm' references have been changed to 'regime'")

if __name__ == "__main__":
    main()
