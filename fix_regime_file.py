#!/usr/bin/env python3
"""
Carefully update stockprice_lstm_regime_regression.py
"""

def fix_regime_file():
    file_path = '/home/stephen/projects/Testing/TestPy/test-lstm/src/stockprice_lstm_regime_regression.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Make replacements carefully
    content = content.replace('Paradigm-Specific LSTM Training', 'Regime-Specific LSTM Training')
    content = content.replace('market paradigm', 'market regime')
    content = content.replace('paradigm classifications', 'regime classifications')
    content = content.replace('specified paradigm', 'specified regime')
    content = content.replace('<paradigm_number>', '<regime_number>')
    content = content.replace('stockprice_lstm_paradigm_regression.py', 'stockprice_lstm_regime_regression.py')
    
    # Fix directory references
    content = content.replace("paradigm_dir = project_root / 'paradigm_analysis'", "regime_dir = project_root / 'regime_analysis'")
    content = content.replace('paradigm_dir', 'regime_dir')
    content = content.replace('paradigm_file', 'regime_file')
    
    # Fix function names
    content = content.replace('def load_paradigm_data(paradigm_number):', 'def load_regime_data(regime_number):')
    content = content.replace('def split_paradigm_data(df, paradigm_weeks, validation_split=0.2, random_seed=42):', 'def split_regime_data(df, regime_weeks, validation_split=0.2, random_seed=42):')
    content = content.replace('def train_paradigm_model(paradigm_number, label_number, validation_split=0.2,', 'def train_regime_model(regime_number, label_number, validation_split=0.2,')
    
    # Fix variable names
    content = content.replace('paradigm_number', 'regime_number')
    content = content.replace('paradigm_weeks', 'regime_weeks')
    content = content.replace('paradigm_data', 'regime_data')
    content = content.replace('paradigm_trading_days', 'regime_trading_days')
    content = content.replace('paradigm_df', 'regime_df')
    content = content.replace('paradigm_file', 'regime_file')
    
    # Fix file references
    content = content.replace("'paradigm_assignments.csv'", "'regime_assignments.csv'")
    content = content.replace("'paradigm_model_log.csv'", "'regime_model_log.csv'")
    content = content.replace('lstm_paradigm_model_', 'lstm_regime_model_')
    content = content.replace('Paradigm', 'Regime')
    
    # Fix model ID format
    content = content.replace('f"P{regime_number:02d}_{model_id:05d}"', 'f"R{regime_number:02d}_{model_id:05d}"')
    
    # Fix function calls
    content = content.replace('train_paradigm_model(', 'train_regime_model(')
    content = content.replace('load_paradigm_data(', 'load_regime_data(')
    content = content.replace('split_paradigm_data(', 'split_regime_data(')
    
    # Fix comments and print statements
    content = content.replace('Loading data for paradigm', 'Loading data for regime')
    content = content.replace('paradigm {', 'regime {')
    content = content.replace('Training LSTM model for Paradigm', 'Training LSTM model for Regime')
    content = content.replace('for Paradigm', 'for Regime')
    content = content.replace('paradigm analysis', 'regime analysis')
    content = content.replace('Validate paradigm number', 'Validate regime number')
    content = content.replace('Regime number must be', 'Regime number must be')
    content = content.replace('Available paradigms:', 'Available regimes:')
    content = content.replace('check against actual paradigm analysis', 'check against actual regime analysis')
    content = content.replace('paradigm assignments file', 'regime assignments file')
    content = content.replace('Regime analysis directory', 'Regime analysis directory')
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Successfully updated stockprice_lstm_regime_regression.py")

if __name__ == "__main__":
    fix_regime_file()
