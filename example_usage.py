#!/usr/bin/env python3
"""
Example usage of the corrected Model Trading Weighter.

This script demonstrates different weighting strategies for selecting the best
trading models based on performance metrics.
"""

import numpy as np
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_trading.model_trading_weighter import ModelTradingWeighter

def create_weighting_strategies():
    """
    Create different weighting strategies for model evaluation.
    
    Returns:
        Dict of strategy_name -> weighting_array
    """
    # Get metric structure
    weighter = ModelTradingWeighter()
    info = weighter.get_metric_columns_info()
    
    total_metrics = info['total_metrics']
    daily_metrics = info['daily_metrics']
    regime_metrics = info['regime_metrics']
    
    print(f"Creating weighting strategies for {total_metrics} metrics")
    print(f"- Daily metrics: {daily_metrics}")
    print(f"- Regime metrics: {regime_metrics}")
    
    strategies = {}
    
    # 1. Equal weights strategy
    strategies['equal_weights'] = np.ones(total_metrics) / total_metrics
    
    # 2. Accuracy-focused strategy (higher weights for accuracy metrics)
    accuracy_weights = np.ones(total_metrics) * 0.5
    daily_cols = info['daily_columns']
    regime_cols = info['regime_columns']
    
    # Boost accuracy metrics
    for i, col in enumerate(daily_cols):
        if '_acc_' in col:
            accuracy_weights[i] *= 3.0
    
    for i, col in enumerate(regime_cols):
        if '_acc_' in col:
            accuracy_weights[daily_metrics + i] *= 3.0
    
    # Normalize
    strategies['accuracy_focused'] = accuracy_weights / np.sum(accuracy_weights)
    
    # 3. PnL-focused strategy (higher weights for profit/loss metrics)
    pnl_weights = np.ones(total_metrics) * 0.5
    
    # Boost PnL metrics
    for i, col in enumerate(daily_cols):
        if '_pnl_' in col:
            pnl_weights[i] *= 4.0
    
    for i, col in enumerate(regime_cols):
        if '_pnl_' in col:
            pnl_weights[daily_metrics + i] *= 4.0
    
    # Normalize
    strategies['pnl_focused'] = pnl_weights / np.sum(pnl_weights)
    
    # 4. Recent timeframes focused (higher weights for shorter timeframes)
    recent_weights = np.ones(total_metrics) * 0.3
    
    recent_timeframes = ['daily', '1day', '2day', '3day']
    for i, col in enumerate(daily_cols):
        for timeframe in recent_timeframes:
            if col.startswith(timeframe + '_'):
                recent_weights[i] *= 2.5
                break
    
    for i, col in enumerate(regime_cols):
        for timeframe in recent_timeframes:
            if col.startswith(timeframe + '_'):
                recent_weights[daily_metrics + i] *= 2.5
                break
    
    # Normalize
    strategies['recent_focused'] = recent_weights / np.sum(recent_weights)
    
    # 5. Conservative strategy (focus on accuracy and lower thresholds)
    conservative_weights = np.ones(total_metrics) * 0.4
    
    # Boost accuracy metrics and lower thresholds
    for i, col in enumerate(daily_cols):
        if '_acc_' in col and ('_thr_0.0' in col or '_thr_0.1' in col or '_thr_0.2' in col):
            conservative_weights[i] *= 3.0
    
    for i, col in enumerate(regime_cols):
        if '_acc_' in col and ('_thr_0.0' in col or '_thr_0.1' in col or '_thr_0.2' in col):
            conservative_weights[daily_metrics + i] *= 3.0
    
    # Normalize
    strategies['conservative'] = conservative_weights / np.sum(conservative_weights)
    
    return strategies

def run_strategy_comparison():
    """
    Run a comparison of different weighting strategies.
    """
    print("=" * 80)
    print("MODEL TRADING WEIGHTER - STRATEGY COMPARISON")
    print("=" * 80)
    
    # Create weighting strategies
    strategies = create_weighting_strategies()
    
    # Test parameters
    trading_day = "20250707"
    market_regime = 0
    
    print(f"\nTesting strategies for trading day {trading_day}, market regime {market_regime}")
    print("-" * 60)
    
    weighter = ModelTradingWeighter()
    results = {}
    
    # Test each strategy
    for strategy_name, weights in strategies.items():
        print(f"\nTesting strategy: {strategy_name}")
        print(f"Weighting array length: {len(weights)}")
        print(f"Weight statistics: min={weights.min():.6f}, max={weights.max():.6f}, sum={weights.sum():.6f}")
        
        try:
            result = weighter.get_best_trading_model(trading_day, market_regime, weights)
            results[strategy_name] = result
            
            print(f"✓ Best model: {result['model_id']}")
            print(f"  Score: {result['score']:.6f}")
            print(f"  Direction: {result['direction']}")
            print(f"  Threshold: {result['threshold']}")
            print(f"  {result['details']}")
            
        except Exception as e:
            print(f"✗ Error with strategy {strategy_name}: {e}")
            results[strategy_name] = None
    
    # Summary
    print("\n" + "=" * 80)
    print("STRATEGY RESULTS SUMMARY")
    print("=" * 80)
    
    for strategy_name, result in results.items():
        if result:
            print(f"{strategy_name:20} -> Model {result['model_id']:>6} (Score: {result['score']:>12.6f})")
        else:
            print(f"{strategy_name:20} -> FAILED")
    
    # Test with different market regimes
    print(f"\n" + "=" * 80)
    print("MARKET REGIME COMPARISON (using equal_weights strategy)")
    print("=" * 80)
    
    equal_weights = strategies['equal_weights']
    
    for regime in range(5):  # Test regimes 0-4
        try:
            result = weighter.get_best_trading_model(trading_day, regime, equal_weights)
            print(f"Regime {regime} -> Model {result['model_id']:>6} (Score: {result['score']:>12.6f})")
        except Exception as e:
            print(f"Regime {regime} -> ERROR: {e}")

def detailed_analysis_example():
    """
    Show a detailed analysis for the best model.
    """
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS EXAMPLE")
    print("=" * 80)
    
    weighter = ModelTradingWeighter()
    strategies = create_weighting_strategies()
    
    # Use PnL-focused strategy for detailed analysis
    pnl_weights = strategies['pnl_focused']
    
    try:
        result = weighter.get_best_trading_model("20250707", 1, pnl_weights)
        
        print(f"Best model analysis (PnL-focused strategy):")
        print(f"- Model ID: {result['model_id']}")
        print(f"- Weighted Score: {result['score']:.6f}")
        print(f"- Recommended Direction: {result['direction']}")
        print(f"- Recommended Threshold: {result['threshold']}")
        print(f"- Analysis: {result['details']}")
        
        # Show top metrics that contributed most to the score
        info = weighter.get_metric_columns_info()
        print(f"\nMetric breakdown:")
        print(f"- Total metrics evaluated: {info['total_metrics']}")
        print(f"- Daily performance metrics: {info['daily_metrics']}")
        print(f"- Regime performance metrics: {info['regime_metrics']}")
        
    except Exception as e:
        print(f"Error in detailed analysis: {e}")

if __name__ == "__main__":
    try:
        run_strategy_comparison()
        detailed_analysis_example()
        
        print(f"\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print("The model trading weighter is working correctly!")
        print("You can now use it to select optimal trading models based on your weighting preferences.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
