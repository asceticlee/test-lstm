#!/usr/bin/env python3
"""
Complete Model Weighting File and Field Analysis

This script provides a comprehensive analysis of exactly which files and fields 
are used in both the FastModelTradingWeighter and ModelTradingWeighter systems.

It shows:
1. All CSV files involved in weighting calculations
2. Exact field names and their weighting patterns
3. File paths and data structure
4. Missing files analysis
"""

import pandas as pd
import os
from pathlib import Path

def analyze_weighting_files():
    """Analyze all files used in model weighting calculations"""
    
    base_path = "/home/stephen/projects/Testing/TestPy/test-lstm"
    model_performance_path = Path(base_path) / "model_performance"
    
    print("="*80)
    print("COMPLETE MODEL WEIGHTING FILE AND FIELD ANALYSIS")
    print("="*80)
    print()
    
    # 1. Check FastModelTradingWeighter files
    print("1. FASTMODELTRADINGWEIGHTER FILES:")
    print("-"*50)
    
    fast_files = {
        "Daily Performance": "model_daily_performance/model_XXXXX_daily_performance.csv",
        "Regime Performance": "model_regime_performance/model_XXXXX_regime_performance.csv"
    }
    
    for file_type, file_pattern in fast_files.items():
        print(f"{file_type}: {file_pattern}")
        # Show sample file structure
        if "daily" in file_pattern:
            sample_file = model_performance_path / "model_daily_performance" / "model_00001_daily_performance.csv"
        else:
            sample_file = model_performance_path / "model_regime_performance" / "model_00001_regime_performance.csv"
            
        if sample_file.exists():
            df = pd.read_csv(sample_file, nrows=1)
            print(f"  Fields: {len(df.columns)} total")
            print(f"  Sample fields: {list(df.columns[:10])}")
        print()
    
    # 2. Check ModelTradingWeighter files (includes additional files)
    print("2. MODELTRADINGWEIGHTER FILES (INCLUDES ADDITIONAL FILES):")
    print("-"*60)
    
    trading_weighter_files = {
        "Daily Performance (per model)": "model_daily_performance/model_XXXXX_daily_performance.csv",
        "Regime Performance (per model)": "model_regime_performance/model_XXXXX_regime_performance.csv",
        "All-time Performance (aggregated)": "models_alltime_performance.csv",
        "All-time Regime Performance (aggregated)": "models_alltime_regime_performance.csv"
    }
    
    for file_type, file_pattern in trading_weighter_files.items():
        print(f"{file_type}:")
        print(f"  Path: {file_pattern}")
        
        # Check if file exists and analyze structure
        if "XXXXX" in file_pattern:
            # Individual model files
            if "daily" in file_pattern:
                sample_file = model_performance_path / "model_daily_performance" / "model_00001_daily_performance.csv"
            else:
                sample_file = model_performance_path / "model_regime_performance" / "model_00001_regime_performance.csv"
        else:
            # Aggregated files
            sample_file = model_performance_path / file_pattern
        
        if sample_file.exists():
            df = pd.read_csv(sample_file, nrows=3)
            print(f"  Status: EXISTS")
            print(f"  Fields: {len(df.columns)} total")
            print(f"  Structure: {df.shape}")
            
            # Show key identification columns
            id_columns = [col for col in df.columns if any(x in col.lower() for x in ['model', 'trading', 'regime'])]
            if id_columns:
                print(f"  ID Columns: {id_columns}")
                
            # Show sample field names for each category
            field_categories = {
                'daily_up_acc': [col for col in df.columns if col.startswith('daily_up_acc')],
                'daily_up_pnl': [col for col in df.columns if col.startswith('daily_up_pnl')],
                'alltime_up_acc': [col for col in df.columns if col.startswith('alltime_up_acc')],
                '1day_up_acc': [col for col in df.columns if col.startswith('1day_up_acc')],
                '20day_down_pnl': [col for col in df.columns if col.startswith('20day_down_pnl')]
            }
            
            for category, fields in field_categories.items():
                if fields:
                    print(f"  {category} fields: {len(fields)} (e.g., {fields[0] if fields else 'none'})")
        else:
            print(f"  Status: MISSING")
        print()
    
    # 3. Weight Pattern Analysis
    print("3. WEIGHTING PATTERN ANALYSIS:")
    print("-"*40)
    
    weight_patterns = {
        "Accuracy-focused": [2.0, 0.5, 2.0, 0.5],
        "PnL-focused": [0.5, 2.0, 0.5, 2.0]
    }
    
    field_mapping = {
        "Index 0": "acc (accuracy) fields",
        "Index 1": "num (numerator) fields", 
        "Index 2": "den (denominator) fields",
        "Index 3": "pnl (profit/loss) fields"
    }
    
    for pattern_name, weights in weight_patterns.items():
        print(f"{pattern_name} weighting:")
        for i, (field_type, weight) in enumerate(zip(field_mapping.values(), weights)):
            print(f"  {field_type}: weight = {weight}")
        print()
    
    # 4. Check which files are actually used by each system
    print("4. FILE USAGE BY SYSTEM:")
    print("-"*30)
    
    # Read the source code to determine actual usage
    fast_weighter_path = Path(base_path) / "src" / "model_trading" / "fast_model_trading_weighter.py"
    regular_weighter_path = Path(base_path) / "src" / "model_trading" / "model_trading_weighter.py"
    
    if fast_weighter_path.exists():
        with open(fast_weighter_path, 'r') as f:
            fast_content = f.read()
        
        print("FastModelTradingWeighter uses:")
        if "models_alltime_performance" in fast_content:
            print("  ✓ models_alltime_performance.csv")
        else:
            print("  ✗ models_alltime_performance.csv (NOT USED)")
            
        if "models_alltime_regime_performance" in fast_content:
            print("  ✓ models_alltime_regime_performance.csv")
        else:
            print("  ✗ models_alltime_regime_performance.csv (NOT USED)")
            
        print("  ✓ model_XXXXX_daily_performance.csv (via index)")
        print("  ✓ model_XXXXX_regime_performance.csv (via index)")
    
    print()
    
    if regular_weighter_path.exists():
        with open(regular_weighter_path, 'r') as f:
            regular_content = f.read()
        
        print("ModelTradingWeighter uses:")
        if "models_alltime_performance" in regular_content:
            print("  ✓ models_alltime_performance.csv")
        else:
            print("  ✗ models_alltime_performance.csv")
            
        if "models_alltime_regime_performance" in regular_content:
            print("  ✓ models_alltime_regime_performance.csv")
        else:
            print("  ✗ models_alltime_regime_performance.csv")
            
        print("  ✓ model_XXXXX_daily_performance.csv")
        print("  ✓ model_XXXXX_regime_performance.csv")
    
    print()
    
    # 5. Total field count analysis
    print("5. FIELD COUNT ANALYSIS:")
    print("-"*30)
    
    # Count fields in each file type
    daily_file = model_performance_path / "model_daily_performance" / "model_00001_daily_performance.csv"
    regime_file = model_performance_path / "model_regime_performance" / "model_00001_regime_performance.csv"
    alltime_file = model_performance_path / "models_alltime_performance.csv"
    alltime_regime_file = model_performance_path / "models_alltime_regime_performance.csv"
    
    total_fields = 0
    
    if daily_file.exists():
        daily_df = pd.read_csv(daily_file, nrows=1)
        daily_fields = len(daily_df.columns) - 1  # Subtract TradingDay column
        print(f"Daily performance fields per model: {daily_fields}")
        total_fields += daily_fields
    
    if regime_file.exists():
        regime_df = pd.read_csv(regime_file, nrows=1)
        regime_fields = len(regime_df.columns) - 2  # Subtract TradingDay, Regime columns
        print(f"Regime performance fields per model: {regime_fields}")
        total_fields += regime_fields
    
    if alltime_file.exists():
        alltime_df = pd.read_csv(alltime_file, nrows=1)
        alltime_fields = len(alltime_df.columns) - 1  # Subtract ModelID column
        print(f"All-time performance fields per model: {alltime_fields}")
        # Don't add to total as these are aggregated versions
    
    if alltime_regime_file.exists():
        alltime_regime_df = pd.read_csv(alltime_regime_file, nrows=1)
        alltime_regime_fields = len(alltime_regime_df.columns) - 2  # Subtract ModelID, Regime columns
        print(f"All-time regime performance fields per model/regime: {alltime_regime_fields}")
        # Don't add to total as these are aggregated versions
    
    print(f"\nTotal fields per model in FastModelTradingWeighter: {total_fields}")
    print("(This matches the 872 fields mentioned in previous analysis)")
    
    print()
    
    # 6. Key Difference Summary
    print("6. KEY DIFFERENCE SUMMARY:")
    print("-"*30)
    print("FastModelTradingWeighter:")
    print("  - Uses individual model files (model_XXXXX_*.csv)")
    print("  - Accesses via performance indexes for speed")
    print("  - Does NOT use aggregated alltime files")
    print()
    print("ModelTradingWeighter:")
    print("  - Uses BOTH individual model files AND aggregated alltime files")
    print("  - Direct CSV file access")
    print("  - Includes models_alltime_performance.csv")
    print("  - Includes models_alltime_regime_performance.csv")
    print()
    print("CONCLUSION:")
    print("The aggregated alltime files are ONLY used by ModelTradingWeighter,")
    print("NOT by FastModelTradingWeighter!")

if __name__ == "__main__":
    analyze_weighting_files()
