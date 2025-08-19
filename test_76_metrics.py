#!/usr/bin/env python3
"""
Quick Test: Model Trading Weighter

Tests basic functionality of the 76-metric approach.
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from model_trading.model_trading_weighter import ModelTradingWeighter, get_best_trading_model

def quick_test():
    """Quick functionality test."""
    print("🎯 Quick Test: Model Trading Weighter")
    print("=" * 50)
    
    # Test parameters
    trading_day = "20250707"
    market_regime = 1
    weights = np.random.randn(76)
    
    print(f"Testing: {trading_day}, regime {market_regime}")
    
    try:
        result = get_best_trading_model(trading_day, market_regime, weights, mode='standard')
        print(f"✓ SUCCESS!")
        print(f"  Best model: {result['model_id']}")
        print(f"  Score: {result['score']:.2f}")
        print(f"  Direction: {result['direction']}")
        print(f"  Threshold: {result['threshold']}")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ Model trading weighter is working correctly!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
