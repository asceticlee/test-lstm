#!/usr/bin/env python3
"""
Model Trading Calculation Demo

This module demonstrates how model weighting calculations work with examples.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('/home/stephen/projects/Testing/TestPy/test-lstm/src')

def demonstrate_calculation_logic():
    """Show how weighting calculations work with examples"""
    print("="*60)
    print("MODEL TRADING CALCULATION DEMO")
    print("="*60)
    
    print("\n🧮 WEIGHTING CALCULATION FORMULA:")
    print("-" * 40)
    print("For each model:")
    print("1. Load performance data (daily + regime + alltime + alltime_regime)")
    print("2. Apply weighting array to specific fields")
    print("3. Calculate weighted sum for each model")
    print("4. Select model with highest weighted score")
    
    print("\n📊 EXAMPLE CALCULATION:")
    print("-" * 40)
    
    # Example data
    example_models = {
        "Model_001": {"accuracy": 0.85, "pnl": 150.0, "sharpe": 1.2, "trades": 100},
        "Model_002": {"accuracy": 0.82, "pnl": 180.0, "sharpe": 1.4, "trades": 120},
        "Model_003": {"accuracy": 0.88, "pnl": 140.0, "sharpe": 1.1, "trades": 90}
    }
    
    # Example weighting arrays
    weighting_scenarios = {
        "Accuracy Focus": [2.0, 0.5, 1.0, 0.3],  # Emphasize accuracy
        "PnL Focus": [0.5, 2.0, 1.0, 0.3],       # Emphasize profit
        "Balanced": [1.0, 1.0, 1.0, 1.0]         # Equal weights
    }
    
    fields = ["accuracy", "pnl", "sharpe", "trades"]
    
    for scenario_name, weights in weighting_scenarios.items():
        print(f"\n🎯 {scenario_name} (weights: {weights}):")
        
        scores = {}
        for model_id, data in example_models.items():
            weighted_sum = sum(data[field] * weight for field, weight in zip(fields, weights))
            scores[model_id] = weighted_sum
            print(f"   {model_id}: {weighted_sum:.2f}")
        
        best_model = max(scores, key=scores.get)
        print(f"   → Selected: {best_model} (score: {scores[best_model]:.2f})")

def show_field_structure():
    """Show the structure of performance fields"""
    print(f"\n📋 PERFORMANCE DATA STRUCTURE:")
    print("-" * 40)
    
    field_categories = {
        "Daily Fields": ["daily_up_acc_thr_X", "daily_up_pnl_X", "daily_down_acc_thr_X"],
        "Regime Fields": ["regime_Xday_up_acc_thr_Y", "regime_Xday_up_pnl_Y"],
        "Alltime Fields": ["alltime_up_acc_X", "alltime_total_pnl", "alltime_sharpe_ratio"],
        "Alltime Regime": ["alltime_regime_Xday_up_acc_thr_Y", "alltime_regime_Xday_trades_Y"]
    }
    
    for category, examples in field_categories.items():
        print(f"\n   {category}:")
        for example in examples[:3]:  # Show first 3 examples
            print(f"     - {example}")
        if len(examples) > 3:
            print(f"     ... and more")

if __name__ == "__main__":
    demonstrate_calculation_logic()
    show_field_structure()
