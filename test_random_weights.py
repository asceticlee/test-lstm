#!/usr/bin/env python3
"""
Test script for model_trading_weighter.py with random weights.

Usage:
    python test_random_weights.py <trading_day> <market_regime>

Example:
    python test_random_weights.py 20250707 3
"""

import sys
import os
import numpy as np
from typing import Dict

# Add src directory to path to import our module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_trading.model_trading_weighter import ModelTradingWeighter


def generate_random_weights(num_weights: int = 76, 
                          weight_range: tuple = (-1.0, 1.0),
                          seed: int = None) -> np.ndarray:
    """
    Generate random weights for the model evaluation.
    
    Args:
        num_weights: Number of weights to generate (default: 76)
        weight_range: Tuple of (min_weight, max_weight) (default: (-1.0, 1.0))
        seed: Random seed for reproducibility (default: None for random)
        
    Returns:
        numpy array of random weights
    """
    if seed is not None:
        np.random.seed(seed)
    
    min_weight, max_weight = weight_range
    weights = np.random.uniform(min_weight, max_weight, num_weights)
    
    return weights


def test_model_weighter(trading_day: str, market_regime: int, 
                       mode: str = 'standard', 
                       weight_range: tuple = (-1.0, 1.0),
                       seed: int = None) -> Dict:
    """
    Test the model weighter with random weights.
    
    Args:
        trading_day: Trading day in format 'YYYYMMDD'
        market_regime: Market regime (0-4)
        mode: Evaluation mode ('standard', 'fast', or 'gpu')
        weight_range: Range for random weight generation
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with test results
    """
    print(f"🎯 Testing Model Trading Weighter")
    print(f"   Trading Day: {trading_day}")
    print(f"   Market Regime: {market_regime}")
    print(f"   Mode: {mode}")
    print(f"   Weight Range: {weight_range}")
    if seed is not None:
        print(f"   Random Seed: {seed}")
    print("-" * 50)
    
    # Generate random weights
    print("🎲 Generating 76 random weights...")
    weights = generate_random_weights(76, weight_range, seed)
    
    print(f"   Sample weights (first 10): {weights[:10].round(3)}")
    print(f"   Weight statistics: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    
    # Initialize weighter
    print("\n🔧 Initializing ModelTradingWeighter...")
    weighter = ModelTradingWeighter()
    
    # Test the model selection
    print(f"\n🚀 Finding best trading model using {mode} mode...")
    try:
        if mode == 'gpu':
            result = weighter.get_best_trading_model_gpu(trading_day, market_regime, weights)
        elif mode == 'fast':
            result = weighter.get_best_trading_model_fast(trading_day, market_regime, weights)
        else:
            result = weighter.get_best_trading_model(trading_day, market_regime, weights)
        
        return result
        
    except Exception as e:
        print(f"❌ Error during model evaluation: {e}")
        return None


def print_results(result: Dict):
    """Print the results in a formatted way."""
    if result is None:
        print("❌ No results to display.")
        return
    
    print("\n" + "=" * 60)
    print("🏆 BEST TRADING MODEL RESULTS")
    print("=" * 60)
    print(f"📊 Model ID:      {result['model_id']}")
    print(f"📈 Score:         {result['score']:.4f}")
    print(f"🎯 Direction:     {result['direction']}")
    print(f"⚡ Threshold:     {result['threshold']}")
    
    if 'trading_day' in result:
        print(f"📅 Trading Day:   {result['trading_day']}")
    if 'market_regime' in result:
        print(f"🌊 Market Regime: {result['market_regime']}")
    if 'details' in result:
        print(f"ℹ️  Details:       {result['details']}")
    
    print("=" * 60)
    
    # Additional analysis
    print(f"\n📋 ANALYSIS:")
    print(f"   → Recommended action: Trade {result['direction']}side")
    print(f"   → Use model {result['model_id']} with threshold {result['threshold']}")
    print(f"   → Expected performance score: {result['score']:.4f}")


def main():
    """Main function to handle command line arguments and run the test."""
    if len(sys.argv) != 3:
        print("Usage: python test_random_weights.py <trading_day> <market_regime>")
        print("Example: python test_random_weights.py 20250707 3")
        sys.exit(1)
    
    try:
        trading_day = sys.argv[1]
        market_regime = int(sys.argv[2])
        
        # Validate inputs
        if len(trading_day) != 8 or not trading_day.isdigit():
            raise ValueError("Trading day must be in YYYYMMDD format")
        
        if market_regime < 0 or market_regime > 4:
            raise ValueError("Market regime must be between 0 and 4")
        
    except ValueError as e:
        print(f"❌ Invalid arguments: {e}")
        sys.exit(1)
    
    # Run the test with different configurations
    test_configs = [
        {'mode': 'standard', 'weight_range': (-1.0, 1.0), 'seed': 42},
        # {'mode': 'fast', 'weight_range': (-1.0, 1.0), 'seed': 42},
        # {'mode': 'gpu', 'weight_range': (-1.0, 1.0), 'seed': 42},
    ]
    
    for i, config in enumerate(test_configs):
        if i > 0:
            print("\n" + "🔄" * 30)
        
        result = test_model_weighter(trading_day, market_regime, **config)
        print_results(result)
        
        if result is not None:
            # Test with different random seed for comparison
            print(f"\n🔄 Running again with different random weights...")
            result2 = test_model_weighter(trading_day, market_regime, 
                                        mode=config['mode'], 
                                        weight_range=config['weight_range'], 
                                        seed=None)
            
            if result2 is not None:
                print(f"\n🔍 COMPARISON:")
                print(f"   First run:  Model {result['model_id']}, Score {result['score']:.4f}")
                print(f"   Second run: Model {result2['model_id']}, Score {result2['score']:.4f}")
                
                if result['model_id'] == result2['model_id']:
                    print("   → ✅ Same model selected with different weights")
                else:
                    print("   → 🔄 Different model selected - weight sensitivity detected")
    
    print(f"\n✅ Test completed successfully!")


if __name__ == "__main__":
    main()
