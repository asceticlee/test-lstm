#!/usr/bin/env python3
"""
FastModelTradingWeighter Complete Calculation Walkthrough

This demonstrates the EXACT calculation process that FastModelTradingWeighter
uses internally, including index-based data access, regime adjustments,
and vectorized weighting operations.

Example scenario:
- Trading Day: 20200108
- Models: 00001, 00002, 00003
- Regimes: 0 (Bull), 1 (Bear)  
- Weighting Arrays: [2.0, 0.5, 2.0, 0.5, ...] (Accuracy-focused)
                   [0.5, 2.0, 0.5, 2.0, ...] (PnL-focused)
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append('src/model_trading')
sys.path.append('src')

def demonstrate_fast_weighter_calculation():
    """Demonstrate the complete FastModelTradingWeighter calculation process"""
    
    print("=" * 80)
    print("FastModelTradingWeighter EXACT Calculation Process")
    print("=" * 80)
    
    # Initialize the actual FastModelTradingWeighter
    try:
        from fast_model_trading_weighter import FastModelTradingWeighter
        print("✓ FastModelTradingWeighter imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import FastModelTradingWeighter: {e}")
        return
    
    # Configuration
    trading_day = "20200108"
    test_regimes = [0, 1]
    
    # Define two different weighting strategies
    weighting_strategies = {
        "Accuracy-Focused": {
            "description": "Emphasizes accuracy metrics (2.0) over PnL (0.5)",
            "weights": [2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5] * 10  # 100 weights
        },
        "PnL-Focused": {
            "description": "Emphasizes PnL metrics (2.0) over accuracy (0.5)", 
            "weights": [0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0] * 10  # 100 weights
        }
    }
    
    print(f"\n📅 ANALYSIS CONFIGURATION")
    print(f"   Trading Day: {trading_day}")
    print(f"   Test Regimes: {test_regimes}")
    print(f"   Weighting Strategies: {list(weighting_strategies.keys())}")
    
    # Step 1: Initialize FastModelTradingWeighter
    print(f"\n1️⃣ INITIALIZING FastModelTradingWeighter")
    print("-" * 50)
    
    try:
        weighter = FastModelTradingWeighter()
        print("✓ FastModelTradingWeighter initialized")
        print("✓ Performance indexes loaded into memory")
        print("✓ Model metadata loaded")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Step 2: Get available models for trading day
    print(f"\n2️⃣ GETTING AVAILABLE MODELS")
    print("-" * 50)
    
    try:
        available_models = weighter._get_available_models_for_trading_day(trading_day)
        print(f"✓ Found {len(available_models)} models available for {trading_day}")
        
        # Focus on first 3 models for detailed analysis
        focus_models = available_models[:3] if len(available_models) >= 3 else available_models
        print(f"📊 Focusing on models: {focus_models}")
        
    except Exception as e:
        print(f"❌ Error getting available models: {e}")
        return
    
    # Step 3: Demonstrate data access for each regime and strategy
    print(f"\n3️⃣ CALCULATION RESULTS")
    print("-" * 50)
    
    results_summary = {}
    
    for regime_id in test_regimes:
        regime_name = ["Bull Market", "Bear Market"][regime_id]
        print(f"\n🏛️ REGIME {regime_id}: {regime_name}")
        
        results_summary[regime_id] = {}
        
        for strategy_name, strategy_info in weighting_strategies.items():
            print(f"\n  📊 {strategy_name} Strategy:")
            print(f"     {strategy_info['description']}")
            
            try:
                # This is the actual FastModelTradingWeighter call
                start_time = pd.Timestamp.now()
                
                selected_model, direction, threshold = weighter.weight_and_select_model_fast(
                    trading_day=trading_day,
                    regime_id=regime_id, 
                    weighting_array=strategy_info['weights']
                )
                
                end_time = pd.Timestamp.now()
                calculation_time = (end_time - start_time).total_seconds()
                
                print(f"     🏆 Selected Model: {selected_model}")
                print(f"     🎯 Strategy: {direction} / threshold {threshold}")
                print(f"     ⚡ Calculation Time: {calculation_time:.4f} seconds")
                
                # Store results
                results_summary[regime_id][strategy_name] = {
                    'model': selected_model,
                    'direction': direction,
                    'threshold': threshold,
                    'time': calculation_time
                }
                
            except Exception as e:
                print(f"     ❌ Error: {e}")
                results_summary[regime_id][strategy_name] = {'error': str(e)}
    
    # Step 4: Show internal calculation details (simulated)
    print(f"\n4️⃣ INTERNAL CALCULATION DETAILS")
    print("-" * 50)
    print("Here's what happens inside weight_and_select_model_fast():")
    
    regime_id = 0  # Use regime 0 for detailed breakdown
    strategy_name = "Accuracy-Focused"
    weighting_array = weighting_strategies[strategy_name]['weights']
    
    print(f"\nDetailed breakdown for Regime {regime_id}, {strategy_name}:")
    
    try:
        # Simulate the internal steps
        print(f"\n  Step A: Get available models")
        available_models = weighter._get_available_models_for_trading_day(trading_day)
        print(f"           → {len(available_models)} models found")
        
        print(f"\n  Step B: Load performance data using indexes")
        performance_data = weighter._get_performance_data_fast(
            available_models[:10], trading_day, regime_id  # Limit to 10 for speed
        )
        print(f"           → Performance data loaded: {len(performance_data)} models")
        
        if not performance_data.empty:
            print(f"           → Data columns: {performance_data.shape[1]} metrics per model")
            print(f"           → Sample columns: {list(performance_data.columns)[:5]}...")
        
        print(f"\n  Step C: Apply weighting array")
        weighted_scores = weighter._apply_weighting_optimized(performance_data, weighting_array)
        print(f"           → Weighted scores calculated for {len(weighted_scores)} models")
        
        if not weighted_scores.empty and 'total_weighted_score' in weighted_scores.columns:
            print(f"           → Score range: {weighted_scores['total_weighted_score'].min():.3f} to {weighted_scores['total_weighted_score'].max():.3f}")
            
            # Show top 3 models
            top_models = weighted_scores.nlargest(3, 'total_weighted_score')
            print(f"           → Top 3 models:")
            for idx, row in top_models.iterrows():
                model_id = row['ModelID']
                score = row['total_weighted_score']
                print(f"             {model_id}: {score:.4f}")
        
        print(f"\n  Step D: Select optimal model")
        model_id, direction, threshold, score = weighter._select_optimal_model_simple(weighted_scores)
        print(f"           → Best model: {model_id}")
        print(f"           → Score: {score:.4f}")
        print(f"           → Strategy: {direction}, threshold {threshold}")
        
    except Exception as e:
        print(f"  ❌ Error in detailed breakdown: {e}")
    
    # Step 5: Results summary table
    print(f"\n5️⃣ RESULTS SUMMARY TABLE")
    print("-" * 50)
    
    print(f"{'Regime':<8} {'Strategy':<20} {'Model':<8} {'Direction':<10} {'Threshold':<10} {'Time(s)':<10}")
    print(f"{'-'*8} {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    
    for regime_id in test_regimes:
        regime_name = f"R{regime_id}"
        for strategy_name in weighting_strategies.keys():
            result = results_summary[regime_id].get(strategy_name, {})
            
            if 'error' not in result:
                model = result.get('model', 'N/A')
                direction = result.get('direction', 'N/A')
                threshold = result.get('threshold', 'N/A')
                time_val = f"{result.get('time', 0):.4f}"
                
                print(f"{regime_name:<8} {strategy_name:<20} {model:<8} {direction:<10} {threshold:<10} {time_val:<10}")
            else:
                print(f"{regime_name:<8} {strategy_name:<20} {'ERROR':<8} {'':<10} {'':<10} {'':<10}")
    
    # Step 6: Mathematical explanation
    print(f"\n6️⃣ MATHEMATICAL EXPLANATION")
    print("-" * 50)
    print("""
The FastModelTradingWeighter calculates model scores using:

1. Index Lookup: O(1) access to performance data using pre-generated indexes
   
2. Regime Adjustment: Performance metrics are adjusted based on market regime
   - Regime 0 (Bull): Favors upward predictions, boosts PnL metrics
   - Regime 1 (Bear): Favors downward predictions, different PnL weighting
   
3. Vectorized Weighting: For each model i and metric j:
   
   weighted_score_i = Σ(metric_i,j × weight_j) for all j
   
   Where:
   - metric_i,j = performance value for model i, metric j
   - weight_j = weighting array value for metric j
   
4. Model Selection: argmax(weighted_score_i) across all valid models
   
5. Strategy Determination: Extract direction/threshold from highest-contributing metric

Key advantages:
- Sub-second execution through index-based data access
- Vectorized numpy operations for mathematical calculations  
- Parallel data loading for multiple models
- Memory-efficient selective data loading
""")
    
    print(f"\n✅ CALCULATION DEMONSTRATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_fast_weighter_calculation()
