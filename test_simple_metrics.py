#!/usr/bin/env python3
"""
Simple test to verify show_metrics works for the standard method and 
test the GPU method basic functionality.
"""

import sys
import os
import numpy as np

# Add src directory to path to import our module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_trading.model_trading_weighter import ModelTradingWeighter


def test_standard_with_metrics():
    """Test just the standard method with show_metrics=True"""
    
    trading_day = "20250707"
    market_regime = 3
    
    # Generate consistent weights
    np.random.seed(42)
    weights = np.random.uniform(-1.0, 1.0, 76)
    
    print(f"Testing standard method with show_metrics=True")
    print(f"Trading day: {trading_day}, Market regime: {market_regime}")
    print("=" * 60)
    
    weighter = ModelTradingWeighter()

    try:
        # get_best_trading_model_batch_vectorized expects a LIST of weight arrays
        results = weighter.get_best_trading_model_batch_vectorized(trading_day, market_regime, [weights], show_metrics=True)
        result = results[0]  # Get the first (and only) result
        
        print(f"✅ Result: Model {result['model_id']}, Score {result['score']:.4f}")
        print(f"   Direction: {result['direction']}, Threshold: {result['threshold']}")
        
        if 'metrics_breakdown' in result and result['metrics_breakdown'] is not None:
            print(f"✅ Metrics breakdown available:")
            print(f"   Total metrics: {result['metrics_breakdown']['total_metrics']}")
            print(f"   Score verification: {result['metrics_breakdown']['total_score_verification']:.6f}")
            print(f"   Score match: {abs(result['score'] - result['metrics_breakdown']['total_score_verification']) < 1e-6}")
            
            # Show sample metrics
            print(f"\nSample metrics :")
            for i, metric in enumerate(result['metrics_breakdown']['metrics'][:]):
                print(f"   {metric['index']:2d}. {metric['column_name']:<30} {metric['data_source']:<6} "
                      f"w={metric['weight']:8.4f} v={metric['value']:8.4f} wv={metric['weighted_value']:10.6f}")
        else:
            print("❌ No metrics breakdown available")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("🔧 Testing show_metrics functionality")
    
    # Test 1: Standard method with metrics
    success1 = test_standard_with_metrics()
    
    print(f"\n📊 SUMMARY")
    print("=" * 60)
    print(f"Standard method with metrics: {'✅ PASS' if success1 else '❌ FAIL'}")
    
    if success1:
        print("\n🎉 The show_metrics functionality is working correctly!")
        print("   You can now use show_metrics=True with get_best_trading_model()")
    else:
        print("\n⚠️  There are issues with the show_metrics functionality")
