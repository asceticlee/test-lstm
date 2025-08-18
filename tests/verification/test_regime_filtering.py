#!/usr/bin/env python3
"""
Regime Filtering Verification

This module provides detailed verification that regime-specific data filtering
works correctly in FastModelTradingWeighter.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('/home/stephen/projects/Testing/TestPy/test-lstm/src')

from model_trading.fast_model_trading_weighter import FastModelTradingWeighter

def verify_regime_data_differences():
    """Verify that different regimes produce different performance values"""
    print("="*70)
    print("REGIME FILTERING VERIFICATION")
    print("="*70)
    
    weighter = FastModelTradingWeighter()
    
    # Test parameters
    test_model = "00001"
    test_day = "20200102"
    test_regimes = [1, 2, 3]
    
    print(f"\nTesting Model: {test_model}")
    print(f"Trading Day: {test_day}")
    print(f"Regimes: {test_regimes}")
    
    regime_values = {}
    
    for regime in test_regimes:
        performance_data = weighter._get_performance_data_fast([test_model], test_day, regime)
        
        if not performance_data.empty:
            # Get alltime regime field values
            alltime_regime_cols = [col for col in performance_data.columns if col.startswith('alltime_regime_')]
            
            if alltime_regime_cols:
                sample_field = alltime_regime_cols[0]
                value = performance_data.iloc[0][sample_field]
                regime_values[regime] = value
                print(f"   Regime {regime}: {sample_field} = {value:.6f}")
    
    # Verify differences
    print(f"\n📊 REGIME COMPARISON:")
    differences_found = False
    
    regimes = list(regime_values.keys())
    for i in range(len(regimes)):
        for j in range(i+1, len(regimes)):
            regime1, regime2 = regimes[i], regimes[j]
            val1, val2 = regime_values[regime1], regime_values[regime2]
            
            if val1 != val2:
                differences_found = True
                print(f"   ✅ Regime {regime1} vs {regime2}: {val1:.6f} ≠ {val2:.6f}")
            else:
                print(f"   ⚠️  Regime {regime1} vs {regime2}: {val1:.6f} = {val2:.6f}")
    
    if differences_found:
        print(f"\n✅ VERIFICATION PASSED: Regime filtering produces different values!")
    else:
        print(f"\n❌ VERIFICATION FAILED: All regimes produced same values")
    
    return differences_found

def examine_raw_regime_files():
    """Examine raw regime performance files to show data structure"""
    print(f"\n📋 RAW REGIME FILE ANALYSIS:")
    print("-" * 40)
    
    model_id = "00001"
    regime_file = Path(f"/home/stephen/projects/Testing/TestPy/test-lstm/model_performance/model_regime_performance/model_{model_id}_regime_performance.csv")
    
    if regime_file.exists():
        df = pd.read_csv(regime_file)
        print(f"   File: {regime_file.name}")
        print(f"   Total rows: {len(df):,}")
        print(f"   Available regimes: {sorted(df['Regime'].unique())}")
        
        # Show regime distribution
        for regime in sorted(df['Regime'].unique())[:5]:  # Show first 5 regimes
            regime_count = len(df[df['Regime'] == regime])
            print(f"   Regime {regime}: {regime_count:,} rows")
    else:
        print(f"   ❌ File not found: {regime_file}")

if __name__ == "__main__":
    success = verify_regime_data_differences()
    examine_raw_regime_files()
    
    print(f"\n{'='*70}")
    print("REGIME FILTERING VERIFICATION COMPLETE")
    print(f"Result: {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"{'='*70}")
