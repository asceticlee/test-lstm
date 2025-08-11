#!/usr/bin/env python3
"""
Advanced Regime Prediction Analysis

This script provides advanced analysis including:
1. Prediction confidence analysis
2. Regime stability over time
3. False positive/negative analysis
4. Optimal trading window identification

Usage:
    python advanced_regime_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent
sys.path.insert(0, str(src_dir))

def analyze_prediction_confidence():
    """Analyze prediction confidence and stability"""
    
    results_file = script_dir / "../../market_regime/prediction_accuracy/regime_prediction_accuracy_results.csv"
    
    if not results_file.exists():
        print("Results file not found. Please run regime_prediction_accuracy_test.py first.")
        return
    
    results = pd.read_csv(results_file)
    
    print("="*80)
    print("ADVANCED MARKET REGIME PREDICTION ANALYSIS")
    print("="*80)
    
    # 1. CONFIDENCE THRESHOLD ANALYSIS
    print("\n🎯 CONFIDENCE THRESHOLD ANALYSIS:")
    print("-" * 60)
    
    confidence_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("Threshold | Time Required | Remaining Time | Efficiency | Assessment")
    print("-" * 60)
    
    for threshold in confidence_thresholds:
        threshold_rows = results[results['overall_accuracy'] >= threshold]
        
        if len(threshold_rows) > 0:
            first_time = threshold_rows.iloc[0]
            time_str = first_time['end_time_str']
            duration = first_time['duration_minutes']
            remaining = 85 - duration
            efficiency = remaining / 85 * 100
            
            # Assessment
            if efficiency >= 50:
                assessment = "🟢 EXCELLENT"
            elif efficiency >= 30:
                assessment = "🟡 GOOD"
            elif efficiency >= 15:
                assessment = "🟠 LIMITED"
            else:
                assessment = "🔴 POOR"
            
            print(f"  {threshold:4.0%}   |   {time_str:8}   |   {remaining:5.1f} min   |   {efficiency:5.1f}%   | {assessment}")
        else:
            print(f"  {threshold:4.0%}   |   Never     |     0.0 min   |    0.0%   | 🔴 NEVER")
    
    # 2. ACCURACY IMPROVEMENT RATE
    print("\n📈 ACCURACY IMPROVEMENT ANALYSIS:")
    print("-" * 60)
    
    # Calculate rate of accuracy improvement
    if len(results) > 1:
        results['accuracy_change'] = results['overall_accuracy'].diff()
        results['time_change'] = results['duration_minutes'].diff()
        results['improvement_rate'] = results['accuracy_change'] / results['time_change']
        
        # Find periods of fastest improvement
        fastest_improvement = results.nlargest(3, 'improvement_rate')
        
        print("Periods of fastest accuracy improvement:")
        for i, row in fastest_improvement.iterrows():
            if not pd.isna(row['improvement_rate']):
                time_str = row['end_time_str']
                rate = row['improvement_rate'] * 100  # Convert to percentage per minute
                accuracy = row['overall_accuracy']
                print(f"• {time_str}: +{rate:.2f}% per minute (reached {accuracy:.1%})")
    
    # 3. CRITICAL DECISION POINTS
    print("\n⚡ CRITICAL DECISION POINTS:")
    print("-" * 60)
    
    # Define decision points based on accuracy levels
    decision_points = [
        (0.3, "Initial signal", "Consider preliminary analysis"),
        (0.5, "Moderate confidence", "Begin position sizing calculations"),
        (0.7, "High confidence", "Execute primary strategy"),
        (0.8, "Very high confidence", "Consider maximum position size"),
        (0.9, "Exceptional confidence", "Full conviction trades")
    ]
    
    for threshold, label, action in decision_points:
        threshold_rows = results[results['overall_accuracy'] >= threshold]
        
        if len(threshold_rows) > 0:
            first_time = threshold_rows.iloc[0]
            time_str = first_time['end_time_str']
            duration = first_time['duration_minutes']
            remaining = 85 - duration
            
            print(f"• {label:20} ({threshold:3.0%}): {time_str} | {remaining:4.1f}min left | {action}")
        else:
            print(f"• {label:20} ({threshold:3.0%}): Never reached")
    
    # 4. REGIME-SPECIFIC PERFORMANCE
    print("\n🎭 REGIME-SPECIFIC PERFORMANCE:")
    print("-" * 60)
    
    # Analyze how different regimes perform over time
    regime_cols = [col for col in results.columns if col.startswith('regime_') and col.endswith('_accuracy')]
    
    if regime_cols:
        print("Regime accuracy progression (early → mid → late trading):")
        
        # Get early, mid, and late results
        n_results = len(results)
        early_idx = min(2, n_results - 1)  # ~30% through
        mid_idx = n_results // 2           # ~50% through  
        late_idx = max(n_results - 2, 0)   # ~80% through
        
        periods = [
            (early_idx, "Early", results.iloc[early_idx]['end_time_str']),
            (mid_idx, "Mid", results.iloc[mid_idx]['end_time_str']),
            (late_idx, "Late", results.iloc[late_idx]['end_time_str'])
        ]
        
        for idx, period, time_str in periods:
            if idx < len(results):
                row = results.iloc[idx]
                print(f"\n{period} trading ({time_str}):")
                
                for col in regime_cols:
                    if col in row and not pd.isna(row[col]):
                        regime_num = col.split('_')[1]
                        accuracy = row[col]
                        
                        # Color coding
                        if accuracy >= 0.8:
                            status = "🟢"
                        elif accuracy >= 0.6:
                            status = "🟡"
                        elif accuracy >= 0.4:
                            status = "🟠"
                        else:
                            status = "🔴"
                        
                        print(f"  Regime {regime_num}: {accuracy:5.1%} {status}")
    
    # 5. TRADING STRATEGY RECOMMENDATIONS
    print("\n🚀 TRADING STRATEGY RECOMMENDATIONS:")
    print("-" * 60)
    
    # Find the sweet spot for trading
    good_accuracy_rows = results[results['overall_accuracy'] >= 0.6]
    
    if len(good_accuracy_rows) > 0:
        best_balance_idx = good_accuracy_rows['duration_minutes'].idxmin()
        best_balance = results.loc[best_balance_idx]
        
        time_str = best_balance['end_time_str']
        accuracy = best_balance['overall_accuracy']
        remaining = 85 - best_balance['duration_minutes']
        
        print(f"🎯 OPTIMAL TRADING WINDOW:")
        print(f"   Start regime-based trading at: {time_str}")
        print(f"   Expected accuracy: {accuracy:.1%}")
        print(f"   Execution window: {remaining:.1f} minutes")
        print(f"   Risk level: {'Low' if accuracy >= 0.7 else 'Medium'}")
        
        # Strategy suggestions
        if remaining >= 30:
            print(f"   Strategy: Full regime-based positioning")
        elif remaining >= 15:
            print(f"   Strategy: Quick execution with reduced size")
        else:
            print(f"   Strategy: High-conviction, small positions only")
        
        print(f"\n📋 IMPLEMENTATION CHECKLIST:")
        print(f"   □ Monitor regime signals starting at 10:35 AM")
        print(f"   □ Prepare positions but wait until {time_str}")
        print(f"   □ Execute regime-based strategy with {remaining:.0f}-minute window")
        print(f"   □ Consider partial positions if early signals are strong")
        print(f"   □ Have exit strategy ready for day-end")
        
    # 6. RISK ASSESSMENT
    print(f"\n⚠️  RISK ASSESSMENT:")
    print("-" * 60)
    
    early_accuracy = results['overall_accuracy'].iloc[0] if len(results) > 0 else 0
    mid_accuracy = results['overall_accuracy'].iloc[len(results)//2] if len(results) > 1 else 0
    late_accuracy = results['overall_accuracy'].iloc[-1] if len(results) > 0 else 0
    
    print(f"Early prediction risk: {(1-early_accuracy)*100:.0f}% chance of wrong regime")
    print(f"Mid-day prediction risk: {(1-mid_accuracy)*100:.0f}% chance of wrong regime")
    print(f"Late prediction risk: {(1-late_accuracy)*100:.0f}% chance of wrong regime")
    
    if early_accuracy < 0.3:
        print("🚨 High risk of early false signals - use caution")
    if mid_accuracy < 0.6:
        print("🚨 Moderate risk even at mid-day - consider position sizing")
    if late_accuracy < 0.8:
        print("🚨 Even late predictions have significant risk")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_prediction_confidence()
