#!/usr/bin/env python3
"""
Fix remaining references in stockprice_lstm_regime_regression.py
"""

import re

def fix_regime_regression_file():
    file_path = '/home/stephen/projects/Testing/TestPy/test-lstm/src/stockprice_lstm_regime_regression.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix function names and variable names
    replacements = [
        ('load_paradigm_data', 'load_regime_data'),
        ('split_paradigm_data', 'split_regime_data'),
        ('paradigm_number', 'regime_number'),
        ('paradigm_weeks', 'regime_weeks'),
        ('train_paradigm_model', 'train_regime_model'),
        ('f"P{regime_number:02d}_{model_id:05d}"', 'f"R{regime_number:02d}_{model_id:05d}"'),
        ('scaler_params_{model_id_str}.json', 'scaler_params_{model_id_str}.json'),
        ('lstm_paradigm_model_{model_id_str}.keras', 'lstm_regime_model_{model_id_str}.keras'),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    # Special replacement for model ID string
    content = re.sub(r'f"P\{', 'f"R{', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed stockprice_lstm_regime_regression.py")

if __name__ == "__main__":
    fix_regime_regression_file()
