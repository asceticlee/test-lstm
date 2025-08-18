#!/usr/bin/env python3
"""
Test Regime-Specific Data Filtering in FastModelTradingWeighter

This script verifies that when different regimes are passed to the weighter,
only the specific rows for that regime are used from the model_xxxxx_regime_performance.csv files.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('/home/stephen/projects/Testing/TestPy/test-lstm/src')

from model_trading.fast_model_trading_weighter import FastModelTradingWeighter

def test_regime_specific_filtering():
    """Test that regime-specific data is correctly filtered"""
    print("="*70)
    print("TESTING REGIME-SPECIFIC DATA FILTERING")
    print("="*70)
    
    # Initialize the weighter
    weighter = FastModelTradingWeighter()
    
    # Test parameters
    trading_day = "20200102"
    test_models = ["00001", "00002", "00003"]  # Test with first 3 models
    test_regimes = [1, 2, 3]  # Test different regimes
    
    print(f"\nTesting with models: {test_models}")
    print(f"Trading day: {trading_day}")
    print(f"Test regimes: {test_regimes}")
    
    # Test each regime separately
    regime_data = {}
    
    for regime_id in test_regimes:
        print(f"\n{'='*50}")
        print(f"TESTING REGIME {regime_id}")
        print(f"{'='*50}")
        
        # Get performance data for this regime
        performance_data = weighter._get_performance_data_fast(test_models, trading_day, regime_id)
        
        if not performance_data.empty:
            print(f"✅ Performance data retrieved for regime {regime_id}")
            print(f"   Models: {len(performance_data)} records")
            print(f"   Total fields: {len(performance_data.columns)}")
            
            # Store regime data for comparison
            regime_data[regime_id] = performance_data
            
            # Check regime-specific fields
            regime_fields = [col for col in performance_data.columns if col.startswith('regime_')]
            alltime_regime_fields = [col for col in performance_data.columns if col.startswith('alltime_regime_')]
            
            print(f"   Regime-specific fields: {len(regime_fields)}")
            print(f"   Alltime regime fields: {len(alltime_regime_fields)}")
            
            # Sample some regime-specific values for verification
            if regime_fields:
                sample_field = regime_fields[0]
                print(f"   Sample regime field '{sample_field}':")
                for idx, row in performance_data.head(3).iterrows():
                    model_id = row['ModelID']
                    value = row[sample_field]
                    print(f"     Model {model_id}: {value}")
                    
            if alltime_regime_fields:
                sample_alltime_field = alltime_regime_fields[0]
                print(f"   Sample alltime regime field '{sample_alltime_field}':")
                for idx, row in performance_data.head(3).iterrows():
                    model_id = row['ModelID']
                    value = row[sample_alltime_field]
                    print(f"     Model {model_id}: {value}")
        else:
            print(f"❌ No performance data retrieved for regime {regime_id}")
    
    # Compare data between regimes to verify they're different
    print(f"\n{'='*60}")
    print("COMPARING REGIME-SPECIFIC DATA DIFFERENCES")
    print(f"{'='*60}")
    
    if len(regime_data) >= 2:
        regime_ids = list(regime_data.keys())[:2]  # Compare first two regimes
        
        regime1_data = regime_data[regime_ids[0]]
        regime2_data = regime_data[regime_ids[1]]
        
        print(f"\nComparing Regime {regime_ids[0]} vs Regime {regime_ids[1]}:")
        
        # Compare regime-specific fields
        regime_fields = [col for col in regime1_data.columns if col.startswith('regime_')]
        
        if regime_fields:
            sample_field = regime_fields[0]
            print(f"\nComparing field '{sample_field}':")
            
            differences_found = False
            for model_id in test_models[:3]:
                if model_id in regime1_data['ModelID'].values and model_id in regime2_data['ModelID'].values:
                    val1 = regime1_data[regime1_data['ModelID'] == model_id][sample_field].iloc[0]
                    val2 = regime2_data[regime2_data['ModelID'] == model_id][sample_field].iloc[0]
                    
                    print(f"  Model {model_id}: Regime {regime_ids[0]}={val1}, Regime {regime_ids[1]}={val2}")
                    
                    if val1 != val2:
                        differences_found = True
            
            if differences_found:
                print("✅ Regime-specific differences detected - filtering is working correctly!")
            else:
                print("⚠️  No differences found - this might indicate a filtering issue")
        
        # Compare alltime regime fields
        alltime_regime_fields = [col for col in regime1_data.columns if col.startswith('alltime_regime_')]
        
        if alltime_regime_fields:
            sample_alltime_field = alltime_regime_fields[0]
            print(f"\nComparing alltime regime field '{sample_alltime_field}':")
            
            differences_found = False
            for model_id in test_models[:3]:
                if model_id in regime1_data['ModelID'].values and model_id in regime2_data['ModelID'].values:
                    val1 = regime1_data[regime1_data['ModelID'] == model_id][sample_alltime_field].iloc[0]
                    val2 = regime2_data[regime2_data['ModelID'] == model_id][sample_alltime_field].iloc[0]
                    
                    print(f"  Model {model_id}: Regime {regime_ids[0]}={val1}, Regime {regime_ids[1]}={val2}")
                    
                    if val1 != val2:
                        differences_found = True
            
            if differences_found:
                print("✅ Alltime regime-specific differences detected - filtering is working correctly!")
            else:
                print("⚠️  No alltime regime differences found - this might indicate a filtering issue")

def verify_raw_regime_files():
    """Verify the raw regime performance files to understand the data structure"""
    print(f"\n{'='*60}")
    print("VERIFYING RAW REGIME PERFORMANCE FILES")
    print(f"{'='*60}")
    
    model_performance_dir = Path("/home/stephen/projects/Testing/TestPy/test-lstm/model_performance")
    regime_performance_dir = model_performance_dir / "model_regime_performance"
    
    if not regime_performance_dir.exists():
        print(f"❌ Regime performance directory not found: {regime_performance_dir}")
        return
    
    # Check a few model regime files
    test_models = ["00001", "00002", "00003"]
    
    for model_id in test_models:
        regime_file = regime_performance_dir / f"model_{model_id}_regime_performance.csv"
        
        if regime_file.exists():
            print(f"\n📁 Checking {regime_file.name}:")
            
            try:
                df = pd.read_csv(regime_file)
                print(f"   Total rows: {len(df)}")
                print(f"   Columns: {len(df.columns)}")
                
                # Check if there's a regime column or regime identifier
                regime_cols = [col for col in df.columns if 'regime' in col.lower()]
                print(f"   Regime-related columns: {regime_cols}")
                
                # Check unique regimes in the file
                if 'Regime' in df.columns:
                    unique_regimes = df['Regime'].unique()
                    print(f"   Unique regimes in file: {sorted(unique_regimes)}")
                    
                    # Show sample data for each regime
                    for regime in sorted(unique_regimes)[:3]:  # Show first 3 regimes
                        regime_rows = df[df['Regime'] == regime]
                        print(f"   Regime {regime}: {len(regime_rows)} rows")
                        
                        # Show first few columns for this regime
                        if len(regime_rows) > 0:
                            sample_cols = df.columns[:5]  # First 5 columns
                            sample_row = regime_rows.iloc[0]
                            print(f"     Sample data: {dict(zip(sample_cols, sample_row[sample_cols]))}")
                else:
                    print("   No 'Regime' column found")
                    print(f"   First few rows:")
                    print(df.head(3))
                    
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
        else:
            print(f"❌ File not found: {regime_file}")

def test_regime_index_lookup():
    """Test the regime index lookup mechanism"""
    print(f"\n{'='*60}")
    print("TESTING REGIME INDEX LOOKUP MECHANISM")
    print(f"{'='*60}")
    
    weighter = FastModelTradingWeighter()
    
    # Test regime index functionality
    if hasattr(weighter, 'regime_performance_index') and weighter.regime_performance_index is not None:
        print("✅ Regime performance index is loaded")
        
        # Test lookup for different regimes
        test_models = ["00001", "00002"]
        test_regimes = [1, 2, 3]
        
        for model_id in test_models:
            print(f"\n📊 Testing regime lookups for Model {model_id}:")
            
            for regime_id in test_regimes:
                try:
                    # Test the regime lookup mechanism
                    if hasattr(weighter, '_get_regime_performance_indices'):
                        indices = weighter._get_regime_performance_indices(model_id, regime_id)
                        print(f"   Regime {regime_id}: {len(indices) if indices is not None else 0} records found")
                    else:
                        print(f"   Regime {regime_id}: Direct lookup method not found")
                        
                except Exception as e:
                    print(f"   Regime {regime_id}: Error - {e}")
    else:
        print("❌ Regime performance index not loaded or not available")

if __name__ == "__main__":
    # Run all tests
    test_regime_specific_filtering()
    verify_raw_regime_files()
    test_regime_index_lookup()
    
    print(f"\n{'='*70}")
    print("REGIME FILTERING VERIFICATION COMPLETE")
    print(f"{'='*70}")
