#!/usr/bin/env python3
"""
FastModelTradingWeighter Calculation Example

This script demonstrates EXACTLY how the FastModelTradingWeighter calculates
model selection results with a concrete example using:
- 1 trading day: 20200108
- 3 models: 00001, 00002, 00003  
- 2 regimes: 0, 1
- 2 weighting arrays: Accuracy-focused vs PnL-focused

Shows step-by-step calculation process for transparency.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the source path
sys.path.append('src/model_trading')
sys.path.append('src')

def load_sample_performance_data():
    """Load actual performance data for our 3 sample models on 20200108"""
    
    base_path = Path("/home/stephen/projects/Testing/TestPy/test-lstm")
    daily_perf_dir = base_path / "model_performance" / "model_daily_performance"
    
    trading_day = "20200108"
    sample_models = ["00001", "00002", "00003"]
    
    model_data = {}
    
    for model_id in sample_models:
        file_path = daily_perf_dir / f"model_{model_id}_daily_performance.csv"
        
        if file_path.exists():
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Filter for our trading day
            day_data = df[df['TradingDay'] == int(trading_day)]
            
            if not day_data.empty:
                model_data[model_id] = day_data.iloc[0].to_dict()
                print(f"✓ Loaded data for Model {model_id}")
            else:
                print(f"✗ No data for Model {model_id} on {trading_day}")
        else:
            print(f"✗ File not found for Model {model_id}")
    
    return model_data, trading_day

def simulate_regime_performance():
    """Simulate regime-specific performance adjustments"""
    
    # Regime adjustments (these would typically come from regime performance files)
    regime_adjustments = {
        0: {  # Regime 0: Bull market - favor upward predictions
            'up_multiplier': 1.2,
            'down_multiplier': 0.8,
            'pnl_boost': 0.1
        },
        1: {  # Regime 1: Bear market - favor downward predictions  
            'up_multiplier': 0.8,
            'down_multiplier': 1.2,
            'pnl_boost': 0.05
        }
    }
    
    return regime_adjustments

def extract_key_metrics(model_data):
    """Extract key metrics that would typically be weighted"""
    
    key_metrics = [
        'daily_up_acc_thr_0.0', 'daily_up_pnl_thr_0.0',
        'daily_down_acc_thr_0.0', 'daily_down_pnl_thr_0.0',
        'daily_up_acc_thr_0.2', 'daily_up_pnl_thr_0.2',
        'daily_down_acc_thr_0.2', 'daily_down_pnl_thr_0.2',
        '1week_up_acc_thr_0.0', '1week_up_pnl_thr_0.0',
        '1week_down_acc_thr_0.0', '1week_down_pnl_thr_0.0'
    ]
    
    extracted = {}
    for model_id, data in model_data.items():
        model_metrics = {}
        for metric in key_metrics:
            if metric in data:
                model_metrics[metric] = data[metric]
            else:
                model_metrics[metric] = 0.0  # Default value
        extracted[model_id] = model_metrics
    
    return extracted, key_metrics

def apply_regime_adjustments(model_metrics, regime_id, regime_adjustments):
    """Apply regime-specific adjustments to metrics"""
    
    if regime_id not in regime_adjustments:
        return model_metrics  # No adjustments
    
    adjustments = regime_adjustments[regime_id]
    adjusted_metrics = {}
    
    for model_id, metrics in model_metrics.items():
        adjusted = {}
        
        for metric_name, value in metrics.items():
            adjusted_value = value
            
            # Apply regime multipliers
            if '_up_' in metric_name:
                adjusted_value *= adjustments['up_multiplier']
            elif '_down_' in metric_name:
                adjusted_value *= adjustments['down_multiplier']
                
            # Apply PnL boost
            if '_pnl_' in metric_name:
                adjusted_value += adjustments['pnl_boost']
            
            adjusted[metric_name] = adjusted_value
        
        adjusted_metrics[model_id] = adjusted
    
    return adjusted_metrics

def calculate_weighted_scores(model_metrics, weighting_array, metric_names):
    """Calculate weighted scores for each model"""
    
    # Ensure weighting array matches number of metrics
    if len(weighting_array) != len(metric_names):
        if len(weighting_array) < len(metric_names):
            weighting_array = weighting_array + [1.0] * (len(metric_names) - len(weighting_array))
        else:
            weighting_array = weighting_array[:len(metric_names)]
    
    weighted_scores = {}
    
    for model_id, metrics in model_metrics.items():
        total_score = 0.0
        individual_scores = {}
        
        for i, metric_name in enumerate(metric_names):
            metric_value = metrics.get(metric_name, 0.0)
            weight = weighting_array[i]
            weighted_value = metric_value * weight
            
            individual_scores[metric_name] = weighted_value
            total_score += weighted_value
        
        weighted_scores[model_id] = {
            'total_score': total_score,
            'individual_scores': individual_scores
        }
    
    return weighted_scores

def select_best_model(weighted_scores):
    """Select the model with the highest total weighted score"""
    
    best_model = None
    best_score = float('-inf')
    
    for model_id, scores in weighted_scores.items():
        if scores['total_score'] > best_score:
            best_score = scores['total_score']
            best_model = model_id
    
    return best_model, best_score

def determine_strategy(best_model, weighted_scores, metric_names):
    """Determine the best strategy based on the highest-contributing metric"""
    
    individual_scores = weighted_scores[best_model]['individual_scores']
    
    # Find metric with highest contribution
    best_metric = max(individual_scores.keys(), key=lambda k: individual_scores[k])
    best_contribution = individual_scores[best_metric]
    
    # Extract direction and threshold
    if '_up_' in best_metric:
        direction = 'up'
    elif '_down_' in best_metric:
        direction = 'down'
    else:
        direction = 'neutral'
    
    # Extract threshold
    if '_thr_' in best_metric:
        threshold_part = best_metric.split('_thr_')[-1]
        threshold = threshold_part.replace('_', '.')
    else:
        threshold = "0.0"
    
    return best_metric, direction, threshold, best_contribution

def run_calculation_example():
    """Run the complete calculation example"""
    
    print("=" * 80)
    print("FastModelTradingWeighter Calculation Example")
    print("=" * 80)
    
    # Step 1: Load sample performance data
    print("\n1. LOADING PERFORMANCE DATA")
    print("-" * 40)
    
    model_data, trading_day = load_sample_performance_data()
    
    if not model_data:
        print("❌ No performance data found. Please check the data files.")
        return
    
    print(f"Trading Day: {trading_day}")
    print(f"Models loaded: {list(model_data.keys())}")
    
    # Step 2: Extract key metrics
    print("\n2. EXTRACTING KEY METRICS")
    print("-" * 40)
    
    model_metrics, metric_names = extract_key_metrics(model_data)
    
    print(f"Key metrics extracted: {len(metric_names)}")
    print("Sample metrics:", metric_names[:4], "...")
    
    # Display raw metric values
    print("\nRaw metric values:")
    for model_id in sorted(model_metrics.keys()):
        print(f"  Model {model_id}:")
        for i, metric in enumerate(metric_names[:4]):  # Show first 4 metrics
            value = model_metrics[model_id][metric]
            print(f"    {metric}: {value:.4f}")
        print("    ...")
    
    # Step 3: Define weighting arrays
    print("\n3. DEFINING WEIGHTING ARRAYS")
    print("-" * 40)
    
    weighting_scenarios = {
        "Accuracy-Focused": [
            2.0 if '_acc_' in metric else 0.5 for metric in metric_names
        ],
        "PnL-Focused": [
            2.0 if '_pnl_' in metric else 0.5 for metric in metric_names
        ]
    }
    
    for scenario_name, weights in weighting_scenarios.items():
        print(f"{scenario_name}:")
        acc_weights = [w for i, w in enumerate(weights) if '_acc_' in metric_names[i]]
        pnl_weights = [w for i, w in enumerate(weights) if '_pnl_' in metric_names[i]]
        print(f"  Accuracy weights: {acc_weights[:3]}... (avg: {np.mean(acc_weights):.1f})")
        print(f"  PnL weights: {pnl_weights[:3]}... (avg: {np.mean(pnl_weights):.1f})")
    
    # Step 4: Calculate for different regimes and weighting arrays
    print("\n4. CALCULATION RESULTS")
    print("-" * 40)
    
    regime_adjustments = simulate_regime_performance()
    
    for regime_id in [0, 1]:
        print(f"\n📊 REGIME {regime_id} RESULTS:")
        print(f"   {['Bull Market (up-favored)', 'Bear Market (down-favored)'][regime_id]}")
        
        # Apply regime adjustments
        adjusted_metrics = apply_regime_adjustments(model_metrics, regime_id, regime_adjustments)
        
        for scenario_name, weighting_array in weighting_scenarios.items():
            print(f"\n  🎯 {scenario_name} Weighting:")
            
            # Calculate weighted scores
            weighted_scores = calculate_weighted_scores(adjusted_metrics, weighting_array, metric_names)
            
            # Select best model
            best_model, best_score = select_best_model(weighted_scores)
            
            # Determine strategy
            best_metric, direction, threshold, contribution = determine_strategy(
                best_model, weighted_scores, metric_names
            )
            
            print(f"     🏆 Best Model: {best_model}")
            print(f"     📈 Total Score: {best_score:.4f}")
            print(f"     🎲 Strategy: {direction} / threshold {threshold}")
            print(f"     🔍 Key Metric: {best_metric} (contrib: {contribution:.4f})")
            
            # Show all model scores for comparison
            print(f"     📊 All Models:")
            for model_id in sorted(weighted_scores.keys()):
                score = weighted_scores[model_id]['total_score']
                rank = "👑" if model_id == best_model else "  "
                print(f"        {rank} Model {model_id}: {score:.4f}")
    
    # Step 5: Show detailed calculation for one scenario
    print("\n5. DETAILED CALCULATION BREAKDOWN")
    print("-" * 40)
    print("Showing detailed calculation for Regime 0, Accuracy-Focused weighting:")
    
    regime_id = 0
    scenario_name = "Accuracy-Focused"
    weighting_array = weighting_scenarios[scenario_name]
    
    adjusted_metrics = apply_regime_adjustments(model_metrics, regime_id, regime_adjustments)
    
    print("\nStep-by-step calculation for each model:")
    for model_id in sorted(adjusted_metrics.keys()):
        print(f"\n  Model {model_id}:")
        
        total = 0.0
        metrics = adjusted_metrics[model_id]
        
        for i, metric_name in enumerate(metric_names[:6]):  # Show first 6 for brevity
            value = metrics[metric_name]
            weight = weighting_array[i]
            contribution = value * weight
            total += contribution
            
            print(f"    {metric_name}: {value:.3f} × {weight:.1f} = {contribution:.3f}")
        
        print(f"    ... (and {len(metric_names)-6} more metrics)")
        
        # Calculate full total
        full_total = sum(metrics[metric] * weighting_array[i] for i, metric in enumerate(metric_names))
        print(f"    🎯 TOTAL SCORE: {full_total:.4f}")

if __name__ == "__main__":
    run_calculation_example()
