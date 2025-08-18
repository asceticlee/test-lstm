#!/usr/bin/env python3
"""
Model Performance Analysis

This module analyzes model performance data and weighting fields.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('/home/stephen/projects/Testing/TestPy/test-lstm/src')

def analyze_weighting_fields():
    """Analyze the structure and content of weighting fields"""
    print("="*60)
    print("WEIGHTING FIELDS ANALYSIS")
    print("="*60)
    
    # Check for field analysis files
    field_files = [
        "/home/stephen/projects/Testing/TestPy/test-lstm/accuracy_emphasized_fields.csv",
        "/home/stephen/projects/Testing/TestPy/test-lstm/pnl_emphasized_fields.csv"
    ]
    
    for file_path in field_files:
        if Path(file_path).exists():
            print(f"\n📁 {Path(file_path).name}:")
            df = pd.read_csv(file_path)
            print(f"   Fields: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            
            # Show sample data
            if 'field' in df.columns and 'weight' in df.columns:
                print("   Top 5 weighted fields:")
                top_fields = df.nlargest(5, 'weight')
                for idx, row in top_fields.iterrows():
                    print(f"     {row['field']}: {row['weight']:.4f}")
        else:
            print(f"❌ File not found: {file_path}")

def analyze_file_usage():
    """Analyze which files are used by different weighters"""
    print(f"\n📊 FILE USAGE ANALYSIS:")
    print("-" * 40)
    
    analysis_file = "/home/stephen/projects/Testing/TestPy/test-lstm/complete_weighting_file_analysis.py"
    
    if Path(analysis_file).exists():
        print(f"✅ Analysis script available: {Path(analysis_file).name}")
        print("   This script compares file usage between:")
        print("   - FastModelTradingWeighter")
        print("   - ModelTradingWeighter")
        print("   Run it to see detailed file usage comparison")
    else:
        print(f"❌ Analysis script not found")

if __name__ == "__main__":
    analyze_weighting_fields()
    analyze_file_usage()
