#!/usr/bin/env python3
"""
Simple test script for model_trading_weighter.py with random weights.

Usage:
    python simple_test_weights.py <trading_day> <market_regime>

Example:
    python simple_test_weights.py 20250707 3
"""

import sys
import os
import numpy as np

# Add src directory to path to import our module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_trading.model_trading_weighter import ModelTradingWeighter


def main():
    """Main function to handle command line arguments and run the test."""
    if len(sys.argv) != 3:
        print("Usage: python simple_test_weights.py <trading_day> <market_regime>")
        print("Example: python simple_test_weights.py 20250707 3")
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
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Testing with trading_day={trading_day}, market_regime={market_regime}")
    
    # Generate 76 random weights between -1 and 1
    print("Generating 76 random weights...")
    weights = np.random.uniform(-1.0, 1.0, 76)
    print(f"Sample weights: {weights[:5].round(3)} ... (showing first 5 of 76)")
    
    # Initialize weighter and get best model
    print("Finding best trading model...")
    weighter = ModelTradingWeighter()
    
    try:
        result = weighter.get_best_trading_model(trading_day, market_regime, weights, show_metrics=True)
        
        print("\n" + "="*50)
        print("RESULT:")
        print(f"Best Model ID: {result['model_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Direction: {result['direction']}")
        print(f"Threshold: {result['threshold']}")
        print("="*50)
        
        # Print detailed metrics breakdown if available
        if 'metrics_breakdown' in result and result['metrics_breakdown'] is not None:
            print("\n" + "="*80)
            print("DETAILED METRICS BREAKDOWN (76 metrics)")
            print("="*80)
            print(f"{'#':<3} {'Column Name':<40} {'Source':<8} {'Weight':<12} {'Value':<12} {'Weighted':<12}")
            print("-" * 80)
            
            for metric in result['metrics_breakdown']['metrics']:
                print(f"{metric['index']:<3} {metric['column_name']:<40} {metric['data_source']:<8} "
                      f"{metric['weight']:<12.6f} {metric['value']:<12.6f} {metric['weighted_value']:<12.6f}")
            
            print("-" * 80)
            print(f"Total Score (verification): {result['metrics_breakdown']['total_score_verification']:.6f}")
            print(f"Reported Score:            {result['score']:.6f}")
            print(f"Match: {'✓' if abs(result['metrics_breakdown']['total_score_verification'] - result['score']) < 1e-6 else '✗'}")
            print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
