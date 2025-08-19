#!/usr/bin/env python3
"""
Enhanced test script for model_trading_weighter.py with detailed metrics breakdown.

Usage:
    python enhanced_test_weights.py <trading_day> <market_regime>

Example:
    python enhanced_test_weights.py 20250707 3
"""

import sys
import os
import numpy as np

# Add src directory to path to import our module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_trading.model_trading_weighter import ModelTradingWeighter


def print_metrics_summary(metrics_breakdown):
    """Print a summary of the metrics breakdown."""
    if not metrics_breakdown or 'metrics' not in metrics_breakdown:
        print("No metrics breakdown available.")
        return
    
    metrics = metrics_breakdown['metrics']
    
    # Group metrics by type and source
    daily_acc = [m for m in metrics if m['data_source'] == 'daily' and '_acc_' in m['column_name'] and '_ws_acc_' not in m['column_name']]
    daily_ws_acc = [m for m in metrics if m['data_source'] == 'daily' and '_ws_acc_' in m['column_name']]
    daily_pnl = [m for m in metrics if m['data_source'] == 'daily' and '_pnl_' in m['column_name'] and '_ppt_' not in m['column_name']]
    daily_ppt = [m for m in metrics if m['data_source'] == 'daily' and '_ppt_' in m['column_name']]
    
    regime_acc = [m for m in metrics if m['data_source'] == 'regime' and '_acc_' in m['column_name'] and '_ws_acc_' not in m['column_name']]
    regime_ws_acc = [m for m in metrics if m['data_source'] == 'regime' and '_ws_acc_' in m['column_name']]
    regime_pnl = [m for m in metrics if m['data_source'] == 'regime' and '_pnl_' in m['column_name'] and '_ppt_' not in m['column_name']]
    regime_ppt = [m for m in metrics if m['data_source'] == 'regime' and '_ppt_' in m['column_name']]
    
    print("\n" + "="*60)
    print("METRICS SUMMARY BY TYPE")
    print("="*60)
    
    categories = [
        ("Daily Accuracy", daily_acc),
        ("Daily Wilson Score Accuracy", daily_ws_acc),
        ("Daily PnL", daily_pnl),
        ("Daily PnL Per Trade", daily_ppt),
        ("Regime Accuracy", regime_acc),
        ("Regime Wilson Score Accuracy", regime_ws_acc),
        ("Regime PnL", regime_pnl),
        ("Regime PnL Per Trade", regime_ppt)
    ]
    
    for category_name, category_metrics in categories:
        if category_metrics:
            total_contribution = sum(m['weighted_value'] for m in category_metrics)
            print(f"\n{category_name} ({len(category_metrics)} metrics):")
            print(f"   Total Contribution: {total_contribution:12.6f}")
            print(f"   Avg Weight:         {np.mean([m['weight'] for m in category_metrics]):12.6f}")
            print(f"   Avg Value:          {np.mean([m['value'] for m in category_metrics]):12.6f}")
    
    print("\n" + "="*60)
    print(f"TOTAL SCORE: {metrics_breakdown['total_score_verification']:.6f}")
    print("="*60)


def main():
    """Main function to handle command line arguments and run the test."""
    if len(sys.argv) != 3:
        print("Usage: python enhanced_test_weights.py <trading_day> <market_regime>")
        print("Example: python enhanced_test_weights.py 20250707 3")
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
    print(f"Weight stats: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    
    # Initialize weighter and get best model
    print("\nFinding best trading model with detailed metrics...")
    weighter = ModelTradingWeighter()
    
    try:
        result = weighter.get_best_trading_model(trading_day, market_regime, weights, show_metrics=True)
        
        print("\n" + "="*50)
        print("OPTIMAL TRADING STRATEGY")
        print("="*50)
        print(f"📊 Model ID:    {result['model_id']}")
        print(f"📈 Score:       {result['score']:.4f}")
        print(f"🎯 Direction:   {result['direction']}")
        print(f"⚡ Threshold:   {result['threshold']}")
        print("="*50)
        
        # Print metrics summary
        if 'metrics_breakdown' in result and result['metrics_breakdown'] is not None:
            print_metrics_summary(result['metrics_breakdown'])
            
            # Ask if user wants to see full breakdown
            print("\nWould you like to see the full 76-line metrics breakdown? (y/n): ", end="")
            try:
                choice = input().lower().strip()
                if choice in ['y', 'yes']:
                    print("\n" + "="*90)
                    print("FULL METRICS BREAKDOWN (76 metrics)")
                    print("="*90)
                    print(f"{'#':<3} {'Column Name':<40} {'Src':<6} {'Weight':<10} {'Value':<10} {'Weighted':<12}")
                    print("-" * 90)
                    
                    for metric in result['metrics_breakdown']['metrics']:
                        print(f"{metric['index']:<3} {metric['column_name']:<40} {metric['data_source']:<6} "
                              f"{metric['weight']:<10.4f} {metric['value']:<10.4f} {metric['weighted_value']:<12.6f}")
                    
                    print("-" * 90)
                    print(f"Total: {result['metrics_breakdown']['total_score_verification']:.6f}")
                    print("="*90)
            except (EOFError, KeyboardInterrupt):
                print("\nSkipping full breakdown.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
