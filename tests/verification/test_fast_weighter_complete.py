#!/usr/bin/env python3
"""
FastModelTradingWeighter Verification Test Suite

This module provides comprehensive verification of the FastModelTradingWeighter
functionality, including alltime files integration and regime filtering.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('/home/stephen/projects/Testing/TestPy/test-lstm/src')

from model_trading.fast_model_trading_weighter import FastModelTradingWeighter

def test_alltime_integration():
    """Test that alltime files are properly integrated"""
    print("="*60)
    print("TESTING ALLTIME FILES INTEGRATION")
    print("="*60)
    
    # Initialize the weighter
    weighter = FastModelTradingWeighter()
    
    # Check if alltime data was loaded
    print("\n1. ALLTIME DATA LOADING STATUS:")
    print("-"*40)
    
    alltime_loaded = (hasattr(weighter, 'alltime_performance_data') and 
                     weighter.alltime_performance_data is not None)
    alltime_regime_loaded = (hasattr(weighter, 'alltime_regime_performance_data') and 
                            weighter.alltime_regime_performance_data is not None)
    
    if alltime_loaded:
        print(f"✅ Alltime Performance Data: {len(weighter.alltime_performance_data)} models")
    else:
        print("❌ Alltime Performance Data: NOT LOADED")
    
    if alltime_regime_loaded:
        print(f"✅ Alltime Regime Performance Data: {len(weighter.alltime_regime_performance_data)} records")
    else:
        print("❌ Alltime Regime Performance Data: NOT LOADED")
    
    return alltime_loaded and alltime_regime_loaded

def test_regime_filtering():
    """Test that regime-specific filtering works correctly"""
    print("\n2. TESTING REGIME-SPECIFIC FILTERING:")
    print("-"*40)
    
    weighter = FastModelTradingWeighter()
    
    # Test parameters
    trading_day = "20200102"
    test_models = ["00001", "00002"]
    test_regimes = [1, 2, 3]
    
    regime_results = {}
    
    for regime_id in test_regimes:
        performance_data = weighter._get_performance_data_fast(test_models, trading_day, regime_id)
        
        if not performance_data.empty:
            # Check alltime regime fields for regime-specific values
            alltime_regime_fields = [col for col in performance_data.columns if col.startswith('alltime_regime_')]
            
            if alltime_regime_fields:
                sample_field = alltime_regime_fields[0]
                model_values = {}
                for idx, row in performance_data.iterrows():
                    model_id = row['ModelID']
                    value = row[sample_field]
                    model_values[model_id] = value
                
                regime_results[regime_id] = model_values
                print(f"   Regime {regime_id}: Retrieved {len(performance_data)} models")
    
    # Verify different regimes produce different values
    regime_differences = False
    if len(regime_results) >= 2:
        regimes = list(regime_results.keys())[:2]
        model_id = test_models[0]
        
        if (model_id in regime_results[regimes[0]] and 
            model_id in regime_results[regimes[1]]):
            val1 = regime_results[regimes[0]][model_id]
            val2 = regime_results[regimes[1]][model_id]
            
            if val1 != val2:
                regime_differences = True
                print(f"   ✅ Model {model_id}: Regime {regimes[0]}={val1:.4f}, Regime {regimes[1]}={val2:.4f} (Different)")
            else:
                print(f"   ⚠️  Model {model_id}: Same values across regimes")
    
    return regime_differences

def test_model_selection():
    """Test complete model selection workflow"""
    print("\n3. TESTING MODEL SELECTION WORKFLOW:")
    print("-"*40)
    
    weighter = FastModelTradingWeighter()
    
    try:
        # Test model selection
        trading_day = "20200102"
        regime_id = 1
        weighting_array = [2.0, 0.5, 2.0, 0.5]  # Accuracy-focused
        
        model_id, direction, threshold = weighter.weight_and_select_model_fast(
            trading_day, regime_id, weighting_array
        )
        
        print(f"   ✅ Model selection successful:")
        print(f"      Selected model: {model_id}")
        print(f"      Direction: {direction}")
        print(f"      Threshold: {threshold}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model selection failed: {e}")
        return False

def run_verification_suite():
    """Run the complete verification suite"""
    print("="*80)
    print("FASTMODELTRADINGWEIGHTER VERIFICATION SUITE")
    print("="*80)
    
    results = {}
    
    # Test 1: Alltime integration
    results['alltime_integration'] = test_alltime_integration()
    
    # Test 2: Regime filtering
    results['regime_filtering'] = test_regime_filtering()
    
    # Test 3: Model selection
    results['model_selection'] = test_model_selection()
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - FastModelTradingWeighter is working correctly!")
    else:
        print("⚠️  SOME TESTS FAILED - Please review the implementation")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    run_verification_suite()
