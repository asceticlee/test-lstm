#!/usr/bin/env python3
"""
Test FastModelTradingWeighter with Alltime Files

This script tests that the FastModelTradingWeighter is now properly loading
and using the alltime performance files.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('/home/stephen/projects/Testing/TestPy/test-lstm/src')

from model_trading.fast_model_trading_weighter import FastModelTradingWeighter

def test_alltime_loading():
    """Test that alltime files are being loaded"""
    print("="*60)
    print("TESTING FASTMODELTRADINGWEIGHTER WITH ALLTIME FILES")
    print("="*60)
    
    # Initialize the weighter
    weighter = FastModelTradingWeighter()
    
    # Check if alltime data was loaded
    print("\n1. ALLTIME DATA LOADING STATUS:")
    print("-"*40)
    
    if hasattr(weighter, 'alltime_performance_data') and weighter.alltime_performance_data is not None:
        print(f"✅ Alltime Performance Data: {len(weighter.alltime_performance_data)} models")
        print(f"   Fields: {len(weighter.alltime_performance_data.columns)} total")
        print(f"   Sample fields: {list(weighter.alltime_performance_data.columns[:5])}")
    else:
        print("❌ Alltime Performance Data: NOT LOADED")
    
    if hasattr(weighter, 'alltime_regime_performance_data') and weighter.alltime_regime_performance_data is not None:
        print(f"✅ Alltime Regime Performance Data: {len(weighter.alltime_regime_performance_data)} records")
        print(f"   Fields: {len(weighter.alltime_regime_performance_data.columns)} total")
        print(f"   Sample fields: {list(weighter.alltime_regime_performance_data.columns[:5])}")
    else:
        print("❌ Alltime Regime Performance Data: NOT LOADED")
    
    # Test data retrieval with alltime fields
    print("\n2. TESTING DATA RETRIEVAL WITH ALLTIME FIELDS:")
    print("-"*50)
    
    try:
        trading_day = "20200102"
        regime_id = 1
        
        # Get some test models
        available_models = weighter._get_available_models_for_trading_day(trading_day)
        test_models = available_models[:3] if available_models else []
        
        if test_models:
            print(f"Testing with models: {test_models}")
            print(f"Trading day: {trading_day}, Regime: {regime_id}")
            
            # Get performance data (should now include alltime fields)
            performance_data = weighter._get_performance_data_fast(test_models, trading_day, regime_id)
            
            if not performance_data.empty:
                print(f"✅ Performance data retrieved: {len(performance_data)} models")
                print(f"   Total fields: {len(performance_data.columns)}")
                
                # Check for different field types
                field_types = {
                    'daily_': [col for col in performance_data.columns if col.startswith('daily_')],
                    'regime_': [col for col in performance_data.columns if col.startswith('regime_')],
                    'alltime_': [col for col in performance_data.columns if col.startswith('alltime_')],
                    'alltime_regime_': [col for col in performance_data.columns if col.startswith('alltime_regime_')]
                }
                
                for field_type, fields in field_types.items():
                    if fields:
                        print(f"   {field_type} fields: {len(fields)} (e.g., {fields[0]})")
                    else:
                        print(f"   {field_type} fields: 0")
                
                # Test actual weighting
                print(f"\n3. TESTING WEIGHTING WITH EXPANDED FIELD SET:")
                print("-"*45)
                
                # Use a simple weighting array
                weighting_array = [2.0, 0.5, 2.0, 0.5]  # Accuracy-focused
                
                try:
                    model_id, direction, threshold = weighter.weight_and_select_model_fast(
                        trading_day, regime_id, weighting_array
                    )
                    print(f"✅ Weighting successful!")
                    print(f"   Selected model: {model_id}")
                    print(f"   Direction: {direction}")
                    print(f"   Threshold: {threshold}")
                    
                except Exception as e:
                    print(f"❌ Weighting failed: {e}")
                
            else:
                print("❌ No performance data retrieved")
        else:
            print("❌ No test models available")
            
    except Exception as e:
        print(f"❌ Error in data retrieval test: {e}")
    
    print("\n4. FIELD COUNT COMPARISON:")
    print("-"*30)
    
    # Compare with previous analysis
    if not performance_data.empty:
        current_fields = len(performance_data.columns)
        print(f"Current total fields: {current_fields}")
        print("Previous analysis showed:")
        print("  - Daily fields: 792")
        print("  - Regime fields: 576") 
        print("  - Alltime fields: 72")
        print("  - Alltime regime fields: 576")
        print(f"  - Expected total: ~2016 (plus prefixes and metadata)")
        
        if current_fields > 1500:
            print("✅ Field count suggests alltime data is included")
        else:
            print("⚠️  Field count seems low - alltime data may not be fully included")

if __name__ == "__main__":
    test_alltime_loading()
