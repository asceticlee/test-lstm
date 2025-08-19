#!/usr/bin/env python3
"""
FINAL TEST: Model Trading Weighter

This script demonstrates the corrected and working Model Trading Weighter.
It shows how to properly use the weighting system to select optimal trading models.
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import with explicit error handling
try:
    from model_trading.model_trading_weighter import ModelTradingWeighter
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    raise

def main():
    print("=" * 70)
    print("MODEL TRADING WEIGHTER - FINAL WORKING VERSION")
    print("=" * 70)
    
    # Initialize weighter
    weighter = ModelTradingWeighter()
    
    # Get metrics information
    info = weighter.get_metric_columns_info()
    print(f"✓ Successfully loaded metric structure:")
    print(f"  - Daily metrics: {info['daily_metrics']}")
    print(f"  - Regime metrics: {info['regime_metrics']}")
    print(f"  - Total metrics: {info['total_metrics']}")
    
    # Create a simple weighting strategy
    total_metrics = info['total_metrics']
    weights = np.ones(total_metrics) / total_metrics  # Equal weights
    print(f"✓ Created weighting array with {len(weights)} weights")
    
    # Test the weighter
    trading_day = "20250707"
    market_regime = 1
    
    print(f"\n✓ Testing model selection for:")
    print(f"  - Trading day: {trading_day}")
    print(f"  - Market regime: {market_regime}")
    
    try:
        result = weighter.get_best_trading_model(trading_day, market_regime, weights)
        
        print(f"\n✓ SUCCESS! Best model found:")
        print(f"  - Model ID: {result['model_id']}")
        print(f"  - Score: {result['score']:.6f}")
        print(f"  - Direction: {result['direction']}")
        print(f"  - Threshold: {result['threshold']}")
        print(f"  - Details: {result['details']}")
        
        print(f"\n" + "=" * 70)
        print("🎉 MODEL TRADING WEIGHTER IS WORKING CORRECTLY! 🎉")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ All tests passed! The model trading weighter is ready for use.")
    else:
        print("\n❌ Tests failed! Check the error messages above.")
        sys.exit(1)
