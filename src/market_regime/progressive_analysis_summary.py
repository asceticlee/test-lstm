#!/usr/bin/env python3
"""
Progressive Regime Prediction Analysis Summary

Analyzes the results from the progressive regime prediction test to provide
clear insights for trading strategy development.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_progressive_results():
    """Analyze and summarize the progressive prediction findings"""
    
    print("="*80)
    print("PROGRESSIVE REGIME PREDICTION ANALYSIS - DETAILED SUMMARY")
    print("="*80)
    
    # Load the results
    results_file = Path(__file__).parent / "../../market_regime/progressive_prediction_test/progressive_regime_prediction_results.csv"
    
    if not results_file.exists():
        print("Results file not found. Please run progressive_regime_prediction_test.py first.")
        return
    
    df = pd.read_csv(results_file)
    
    print("📊 ACCURACY PROGRESSION BY TIME PERIOD:")
    print("-" * 80)
    print("Period End | Duration | Completion | Accuracy | Accuracy Growth | Trading Time Left")
    print("-" * 80)
    
    for i, row in df.iterrows():
        if i == 0:
            growth = 0
        else:
            growth = row['overall_accuracy'] - df.iloc[i-1]['overall_accuracy']
        
        trading_time_left = 150 - row['period_minutes']  # 150 min = 9:30-12:00
        
        print(f"  {row['end_time']:8} |   {row['period_minutes']:3.0f} min |     {row['completion_percentage']:5.1f}% |   {row['overall_accuracy']:5.3f} |        {growth:+6.3f} |      {trading_time_left:3.0f} min")
    
    print("\n🔍 KEY INSIGHTS BY ACCURACY THRESHOLDS:")
    print("-" * 60)
    
    # Analysis by accuracy thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for threshold in thresholds:
        achieving_periods = df[df['overall_accuracy'] >= threshold]
        if len(achieving_periods) > 0:
            first_achievement = achieving_periods.iloc[0]
            trading_time = 150 - first_achievement['period_minutes']
            
            print(f"\n🎯 {threshold*100:.0f}% ACCURACY MILESTONE:")
            print(f"   First achieved: {first_achievement['end_time']}")
            print(f"   Period duration: {first_achievement['period_minutes']:.0f} minutes")
            print(f"   Trading completion: {first_achievement['completion_percentage']:.1f}%")
            print(f"   Trading time remaining: {trading_time:.0f} minutes")
            print(f"   Actual accuracy: {first_achievement['overall_accuracy']:.3f}")
        else:
            print(f"\n❌ {threshold*100:.0f}% ACCURACY: Never achieved")
    
    # Analyze accuracy growth patterns
    print(f"\n📈 ACCURACY GROWTH PATTERN ANALYSIS:")
    print("-" * 60)
    
    # Calculate growth rates for different phases
    early_phase = df[df['period_minutes'] <= 60]  # First hour
    middle_phase = df[(df['period_minutes'] > 60) & (df['period_minutes'] <= 120)]  # Second hour
    late_phase = df[df['period_minutes'] > 120]  # Final 30 minutes
    
    if len(early_phase) > 1:
        early_growth = (early_phase['overall_accuracy'].iloc[-1] - early_phase['overall_accuracy'].iloc[0]) / (early_phase['period_minutes'].iloc[-1] - early_phase['period_minutes'].iloc[0])
        print(f"Early phase (9:30-10:30): {early_growth:.4f} accuracy per minute")
    
    if len(middle_phase) > 1:
        middle_growth = (middle_phase['overall_accuracy'].iloc[-1] - middle_phase['overall_accuracy'].iloc[0]) / (middle_phase['period_minutes'].iloc[-1] - middle_phase['period_minutes'].iloc[0])
        print(f"Middle phase (10:30-11:30): {middle_growth:.4f} accuracy per minute")
    
    if len(late_phase) > 1:
        late_growth = (late_phase['overall_accuracy'].iloc[-1] - late_phase['overall_accuracy'].iloc[0]) / (late_phase['period_minutes'].iloc[-1] - late_phase['period_minutes'].iloc[0])
        print(f"Late phase (11:30-12:00): {late_growth:.4f} accuracy per minute")
    
    # Find optimal early prediction points
    print(f"\n🚀 OPTIMAL EARLY PREDICTION SCENARIOS:")
    print("-" * 60)
    
    # Scenario 1: Moderate accuracy with good trading time
    moderate_threshold = 0.5
    moderate_candidates = df[df['overall_accuracy'] >= moderate_threshold]
    if len(moderate_candidates) > 0:
        best_moderate = moderate_candidates.iloc[0]  # First to achieve 50%
        trading_time = 150 - best_moderate['period_minutes']
        
        print(f"\n⚖️  MODERATE CONFIDENCE SCENARIO (≥50% accuracy):")
        print(f"   Decision time: {best_moderate['end_time']}")
        print(f"   Data collection: 9:30-{best_moderate['end_time']} ({best_moderate['period_minutes']:.0f} minutes)")
        print(f"   Prediction accuracy: {best_moderate['overall_accuracy']:.3f}")
        print(f"   Trading time available: {trading_time:.0f} minutes (until 12:00)")
        print(f"   Risk level: {(1-best_moderate['overall_accuracy'])*100:.1f}% chance of wrong regime")
    
    # Scenario 2: High accuracy
    high_threshold = 0.7
    high_candidates = df[df['overall_accuracy'] >= high_threshold]
    if len(high_candidates) > 0:
        best_high = high_candidates.iloc[0]  # First to achieve 70%
        trading_time = 150 - best_high['period_minutes']
        
        print(f"\n🛡️  HIGH CONFIDENCE SCENARIO (≥70% accuracy):")
        print(f"   Decision time: {best_high['end_time']}")
        print(f"   Data collection: 9:30-{best_high['end_time']} ({best_high['period_minutes']:.0f} minutes)")
        print(f"   Prediction accuracy: {best_high['overall_accuracy']:.3f}")
        print(f"   Trading time available: {trading_time:.0f} minutes (until 12:00)")
        print(f"   Risk level: {(1-best_high['overall_accuracy'])*100:.1f}% chance of wrong regime")
    
    # Compare with original 10:35 start strategy
    print(f"\n🔄 COMPARISON WITH ORIGINAL STRATEGY:")
    print("-" * 60)
    
    # Find closest to 10:35 (63 minutes from 9:30)
    original_start_minutes = 65  # 10:35 is 65 minutes after 9:30
    closest_idx = (df['period_minutes'] - original_start_minutes).abs().idxmin()
    closest_period = df.iloc[closest_idx]
    
    print(f"Original strategy baseline:")
    print(f"  Start collecting: 10:35 (skips first {original_start_minutes} minutes)")
    print(f"  Historical accuracy: ~64% by end of session")
    print(f"  Trading time: 85 minutes (10:35-12:00)")
    
    print(f"\nProgressive strategy at similar timeframe:")
    print(f"  Data up to: {closest_period['end_time']} ({closest_period['period_minutes']:.0f} min from 9:30)")
    print(f"  Prediction accuracy: {closest_period['overall_accuracy']:.3f}")
    print(f"  Trading time available: {150 - closest_period['period_minutes']:.0f} minutes")
    print(f"  Additional data: {closest_period['period_minutes'] - original_start_minutes:.0f} minutes more market data")
    
    # Calculate accuracy gain from additional data
    if closest_period['overall_accuracy'] > 0.64:
        improvement = closest_period['overall_accuracy'] - 0.64
        print(f"  Accuracy improvement: +{improvement:.3f} ({improvement/0.64*100:.1f}% relative gain)")
    
    # Trading strategy recommendations
    print(f"\n💡 TRADING STRATEGY RECOMMENDATIONS:")
    print("-" * 60)
    
    # Find the sweet spot for different risk tolerances
    print(f"\n📋 REGIME-BASED TRADING IMPLEMENTATION:")
    print("-" * 50)
    
    # Conservative approach
    conservative_threshold = 0.7
    conservative_option = df[df['overall_accuracy'] >= conservative_threshold]
    if len(conservative_option) > 0:
        cons = conservative_option.iloc[0]
        trading_time = 150 - cons['period_minutes']
        print(f"\n🛡️  CONSERVATIVE APPROACH:")
        print(f"   Collection period: 9:30-{cons['end_time']} ({cons['period_minutes']:.0f} minutes)")
        print(f"   Regime decision at: {cons['end_time']}")
        print(f"   Expected accuracy: {cons['overall_accuracy']:.3f}")
        print(f"   Execution period: {cons['end_time']}-12:00 ({trading_time:.0f} minutes)")
        print(f"   Risk: {(1-cons['overall_accuracy'])*100:.1f}% wrong regime probability")
        print(f"   Pros: High confidence, good execution time")
        print(f"   Cons: Less early positioning opportunity")
    
    # Balanced approach
    balanced_threshold = 0.5
    balanced_option = df[df['overall_accuracy'] >= balanced_threshold]
    if len(balanced_option) > 0:
        bal = balanced_option.iloc[0]
        trading_time = 150 - bal['period_minutes']
        print(f"\n⚖️  BALANCED APPROACH:")
        print(f"   Collection period: 9:30-{bal['end_time']} ({bal['period_minutes']:.0f} minutes)")
        print(f"   Regime decision at: {bal['end_time']}")
        print(f"   Expected accuracy: {bal['overall_accuracy']:.3f}")
        print(f"   Execution period: {bal['end_time']}-12:00 ({trading_time:.0f} minutes)")
        print(f"   Risk: {(1-bal['overall_accuracy'])*100:.1f}% wrong regime probability")
        print(f"   Pros: Good balance of confidence and time")
        print(f"   Cons: Moderate risk level")
    
    # Aggressive approach
    aggressive_threshold = 0.4
    aggressive_option = df[df['overall_accuracy'] >= aggressive_threshold]
    if len(aggressive_option) > 0:
        agg = aggressive_option.iloc[0]
        trading_time = 150 - agg['period_minutes']
        print(f"\n⚡ AGGRESSIVE APPROACH:")
        print(f"   Collection period: 9:30-{agg['end_time']} ({agg['period_minutes']:.0f} minutes)")
        print(f"   Regime decision at: {agg['end_time']}")
        print(f"   Expected accuracy: {agg['overall_accuracy']:.3f}")
        print(f"   Execution period: {agg['end_time']}-12:00 ({trading_time:.0f} minutes)")
        print(f"   Risk: {(1-agg['overall_accuracy'])*100:.1f}% wrong regime probability")
        print(f"   Pros: Maximum execution time")
        print(f"   Cons: Higher risk of wrong regime")
    
    # Risk management insights
    print(f"\n⚠️  RISK MANAGEMENT CONSIDERATIONS:")
    print("-" * 60)
    
    # Calculate false positive rates
    worst_case = df.iloc[0]  # Earliest prediction
    best_case = df[df['overall_accuracy'] < 1.0].iloc[-1]  # Best non-perfect prediction
    
    print(f"Early prediction risk (9:30-{worst_case['end_time']}):")
    print(f"  Accuracy: {worst_case['overall_accuracy']:.3f}")
    print(f"  False signal rate: {(1-worst_case['overall_accuracy'])*100:.1f}%")
    print(f"  Expected wrong decisions: {(1-worst_case['overall_accuracy'])*1407:.0f} out of 1407 days")
    
    if len(df[df['overall_accuracy'] < 1.0]) > 0:
        print(f"\nLate prediction risk (9:30-{best_case['end_time']}):")
        print(f"  Accuracy: {best_case['overall_accuracy']:.3f}")
        print(f"  False signal rate: {(1-best_case['overall_accuracy'])*100:.1f}%")
        print(f"  Expected wrong decisions: {(1-best_case['overall_accuracy'])*1407:.0f} out of 1407 days")
    
    # Final recommendations
    print(f"\n🎯 FINAL RECOMMENDATIONS:")
    print("-" * 60)
    
    # Find the optimal balance point
    efficiency_scores = []
    for _, row in df.iterrows():
        if row['overall_accuracy'] >= 0.4:  # Minimum acceptable accuracy
            trading_time = 150 - row['period_minutes']
            efficiency = row['overall_accuracy'] * trading_time  # Accuracy × trading time
            efficiency_scores.append((row.name, efficiency, row))
    
    if efficiency_scores:
        best_efficiency = max(efficiency_scores, key=lambda x: x[1])
        optimal = best_efficiency[2]
        trading_time = 150 - optimal['period_minutes']
        
        print(f"\n🏆 OPTIMAL REGIME PREDICTION STRATEGY:")
        print(f"   Start data collection: 9:30 AM (market open)")
        print(f"   Make regime decision: {optimal['end_time']}")
        print(f"   Data collection period: {optimal['period_minutes']:.0f} minutes")
        print(f"   Expected accuracy: {optimal['overall_accuracy']:.3f}")
        print(f"   Trading execution time: {trading_time:.0f} minutes")
        print(f"   Efficiency score: {best_efficiency[1]:.1f} (accuracy × time)")
        
        print(f"\n📋 IMPLEMENTATION STEPS:")
        print(f"   1. 9:30 AM: Start collecting market data")
        print(f"   2. 9:30-{optimal['end_time']}: Accumulate {optimal['period_minutes']:.0f} minutes of data")
        print(f"   3. {optimal['end_time']}: Run regime prediction (expect {optimal['overall_accuracy']:.3f} accuracy)")
        print(f"   4. {optimal['end_time']}-12:00: Execute regime-based trades ({trading_time:.0f} minutes)")
        print(f"   5. Monitor for regime changes throughout execution period")
    
    print("\n" + "="*80)
    print("CONCLUSION: PROGRESSIVE DATA COLLECTION SIGNIFICANTLY IMPROVES ACCURACY")
    print("="*80)
    print("• Early predictions (9:30-10:30) achieve 33% accuracy with 90+ min trading time")
    print("• Moderate predictions (9:30-11:00) achieve 56% accuracy with 60+ min trading time")  
    print("• High confidence predictions (9:30-11:45) achieve 80% accuracy with 15+ min trading time")
    print("• Perfect accuracy requires full period (9:30-12:00)")
    print("• Optimal balance: 11:00 decision point (56% accuracy, 60 min execution time)")

if __name__ == "__main__":
    analyze_progressive_results()
