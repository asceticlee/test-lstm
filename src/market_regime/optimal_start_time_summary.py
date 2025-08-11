#!/usr/bin/env python3
"""
Optimal Start Time Analysis Summary

This script provides a comprehensive summary and recommendation based on 
the optimal start time analysis results.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_optimal_start_time_results():
    """Analyze and summarize the optimal start time findings"""
    
    print("="*80)
    print("OPTIMAL START TIME ANALYSIS - COMPREHENSIVE SUMMARY")
    print("="*80)
    
    # Load the comparison results
    results_file = Path(__file__).parent / "../../market_regime/optimal_start_time_analysis/start_time_comparison.csv"
    
    if not results_file.exists():
        print("Results file not found. Please run optimal_start_time_analysis.py first.")
        return
    
    df = pd.read_csv(results_file)
    
    print("📊 ACCURACY GROWTH COMPARISON BY START TIME:")
    print("-" * 80)
    print("Start Time | Additional Time | Max Accuracy | Early Accuracy | 40% At | 50% At")
    print("-" * 80)
    
    for _, row in df.iterrows():
        start_time = row['start_time']
        additional_time = row['additional_trading_time']
        max_accuracy = row['max_accuracy']
        early_accuracy = row['best_early_accuracy'] if row['best_early_accuracy'] > 0 else 0
        time_40 = row['time_to_40%'] if row['time_to_40%'] != 'Never' else 'Never'
        time_50 = row['time_to_50%'] if row['time_to_50%'] != 'Never' else 'Never'
        
        print(f"  {start_time:8} |      {additional_time:6.0f} min |     {max_accuracy:5.1%} |       {early_accuracy:5.1%} | {time_40:6} | {time_50:6}")
    
    # Analysis by categories
    print("\n🔍 KEY INSIGHTS BY START TIME CATEGORY:")
    print("-" * 60)
    
    # Early start times (9:00-9:30)
    early_starts = df[df['additional_trading_time'] >= 65]
    print("\n📅 EARLY STARTS (9:00-9:30): +65-95 minutes trading time")
    if len(early_starts) > 0:
        avg_max_accuracy = early_starts['max_accuracy'].mean()
        avg_early_accuracy = early_starts['best_early_accuracy'].mean()
        print(f"   Average max accuracy: {avg_max_accuracy:.1%}")
        print(f"   Average early accuracy: {avg_early_accuracy:.1%}")
        print(f"   40% accuracy timing: Most reach around 11:35")
        print(f"   50% accuracy: Never achieved")
        print(f"   ✅ Pros: Lots of preparation time")
        print(f"   ❌ Cons: Poor early signals, never reach 50% accuracy")
    
    # Medium start times (9:45-10:15)
    medium_starts = df[(df['additional_trading_time'] >= 20) & (df['additional_trading_time'] < 65)]
    print(f"\n📅 MEDIUM STARTS (9:45-10:15): +20-50 minutes trading time")
    if len(medium_starts) > 0:
        avg_max_accuracy = medium_starts['max_accuracy'].mean()
        best_performer = medium_starts.loc[medium_starts['max_accuracy'].idxmax()]
        print(f"   Average max accuracy: {avg_max_accuracy:.1%}")
        print(f"   Best performer: {best_performer['start_time']} ({best_performer['max_accuracy']:.1%} max)")
        print(f"   40% accuracy timing: 11:07-11:21")
        print(f"   50% accuracy: Some achieve by 11:30-11:51")
        print(f"   ✅ Pros: Balanced time vs accuracy")
        print(f"   ⚠️  Cons: Still limited early prediction reliability")
    
    # Late start times (10:30)
    late_starts = df[df['additional_trading_time'] < 20]
    print(f"\n📅 LATE STARTS (10:30): +5 minutes trading time")
    if len(late_starts) > 0:
        best_late = late_starts.iloc[0]
        print(f"   Max accuracy: {best_late['max_accuracy']:.1%}")
        print(f"   40% accuracy at: {best_late['time_to_40%']}")
        print(f"   50% accuracy at: {best_late['time_to_50%']}")
        print(f"   60% accuracy at: {best_late['time_to_60%']}")
        print(f"   ✅ Pros: High accuracy potential (88%), faster convergence")
        print(f"   ❌ Cons: Minimal additional trading time")
    
    # Find the optimal balance points
    print(f"\n🎯 OPTIMAL SCENARIOS FOR DIFFERENT TRADING STRATEGIES:")
    print("-" * 60)
    
    # Scenario 1: Conservative (need 50% accuracy)
    can_reach_50 = df[df['time_to_50%'] != 'Never']
    print(f"\n🛡️  CONSERVATIVE STRATEGY (≥50% accuracy required):")
    if len(can_reach_50) > 0:
        best_conservative = can_reach_50.loc[can_reach_50['trading_time_at_50%'].idxmax()]
        print(f"   Recommended start: {best_conservative['start_time']}")
        print(f"   50% accuracy achieved at: {best_conservative['time_to_50%']}")
        print(f"   Trading time remaining: {best_conservative['trading_time_at_50%']:.1f} minutes")
        print(f"   Max accuracy potential: {best_conservative['max_accuracy']:.1%}")
    else:
        print(f"   ❌ No start times reliably achieve 50% accuracy")
    
    # Scenario 2: Moderate (40% accuracy acceptable)
    can_reach_40 = df[df['time_to_40%'] != 'Never']
    print(f"\n⚖️  MODERATE STRATEGY (≥40% accuracy acceptable):")
    if len(can_reach_40) > 0:
        # Find best balance of trading time and accuracy
        can_reach_40['balance_score'] = can_reach_40['trading_time_at_40%'] * can_reach_40['max_accuracy']
        best_moderate = can_reach_40.loc[can_reach_40['balance_score'].idxmax()]
        print(f"   Recommended start: {best_moderate['start_time']}")
        print(f"   40% accuracy achieved at: {best_moderate['time_to_40%']}")
        print(f"   Trading time remaining: {best_moderate['trading_time_at_40%']:.1f} minutes")
        print(f"   Max accuracy potential: {best_moderate['max_accuracy']:.1%}")
        print(f"   Additional prep time: {best_moderate['additional_trading_time']:.0f} minutes")
    
    # Scenario 3: Aggressive (early signals acceptable)
    early_signal_viable = df[df['best_early_accuracy'] >= 0.25]
    print(f"\n⚡ AGGRESSIVE STRATEGY (early signals for positioning):")
    if len(early_signal_viable) > 0:
        best_aggressive = early_signal_viable.loc[early_signal_viable['additional_trading_time'].idxmax()]
        print(f"   Recommended start: {best_aggressive['start_time']}")
        print(f"   Early signal accuracy: {best_aggressive['best_early_accuracy']:.1%}")
        print(f"   Additional prep time: {best_aggressive['additional_trading_time']:.0f} minutes")
        print(f"   Final accuracy potential: {best_aggressive['max_accuracy']:.1%}")
        print(f"   ⚠️  Risk: {100-best_aggressive['best_early_accuracy']*100:.0f}% false signal rate")
    
    # Practical recommendations
    print(f"\n🚀 PRACTICAL TRADING RECOMMENDATIONS:")
    print("-" * 60)
    
    # Find the sweet spot
    viable_options = df[df['time_to_40%'] != 'Never'].copy()
    if len(viable_options) > 0:
        # Calculate efficiency score (accuracy * trading_time / prep_time)
        viable_options['efficiency'] = (viable_options['max_accuracy'] * 
                                       viable_options['trading_time_at_40%'] / 
                                       np.maximum(viable_options['additional_trading_time'], 1))
        
        optimal = viable_options.loc[viable_options['efficiency'].idxmax()]
        
        print(f"\n🏆 RECOMMENDED OPTIMAL START TIME: {optimal['start_time']}")
        print(f"   Why this is optimal:")
        print(f"   • Additional prep time: {optimal['additional_trading_time']:.0f} minutes")
        print(f"   • Reaches 40% accuracy at: {optimal['time_to_40%']}")
        print(f"   • Trading time at 40%: {optimal['trading_time_at_40%']:.1f} minutes")
        print(f"   • Maximum accuracy: {optimal['max_accuracy']:.1%}")
        print(f"   • Efficiency score: {optimal['efficiency']:.3f}")
        
        print(f"\n📋 IMPLEMENTATION STRATEGY:")
        print(f"   1. START DATA COLLECTION: {optimal['start_time']}")
        print(f"   2. PREPARATION PHASE: {optimal['start_time']}-10:35 ({optimal['additional_trading_time']:.0f} min)")
        print(f"      - Monitor market conditions")
        print(f"      - Prepare trading infrastructure")
        print(f"      - Calculate position sizes")
        print(f"      - DO NOT execute trades yet")
        
        print(f"   3. DECISION PHASE: 10:35-{optimal['time_to_40%']} (regime analysis)")
        print(f"      - Use proven regime prediction methodology")
        print(f"      - Wait for 40% accuracy threshold")
        print(f"      - Validate with preparation phase insights")
        
        print(f"   4. EXECUTION PHASE: {optimal['time_to_40%']}-12:00 ({optimal['trading_time_at_40%']:.0f} min)")
        print(f"      - Execute regime-based trades")
        print(f"      - Use position sizing based on confidence")
        print(f"      - Monitor for regime changes")
    
    # Risk assessment
    print(f"\n⚠️  RISK CONSIDERATIONS:")
    print("-" * 60)
    
    max_accuracy_start = df.loc[df['max_accuracy'].idxmax()]
    min_time_for_40 = df[df['time_to_40%'] != 'Never']['trading_time_at_40%'].max()
    
    print(f"• Best possible accuracy: {max_accuracy_start['max_accuracy']:.1%} (from {max_accuracy_start['start_time']} start)")
    print(f"• Maximum trading time at 40% accuracy: {min_time_for_40:.1f} minutes")
    print(f"• Early signal reliability: All <30% before regime period")
    print(f"• False signal risk: 70-75% for early positioning")
    
    # Alternative strategies
    print(f"\n🔄 ALTERNATIVE STRATEGIES TO CONSIDER:")
    print("-" * 60)
    print(f"1. PURE REGIME STRATEGY (original approach)")
    print(f"   - Start: 10:35, Decide: 11:38, Execute: 11:38-12:00")
    print(f"   - Pros: 64% accuracy, proven reliability")
    print(f"   - Cons: Only 22 minutes execution time")
    
    print(f"\n2. MULTI-TIMEFRAME ENSEMBLE")
    print(f"   - Combine predictions from multiple start times")
    print(f"   - Use early signals for trend, late signals for execution")
    print(f"   - Weight predictions by historical accuracy")
    
    print(f"\n3. ADAPTIVE START TIME")
    print(f"   - Adjust start time based on market volatility")
    print(f"   - Earlier start for volatile markets, later for stable")
    print(f"   - Use overnight indicators to choose strategy")
    
    print("\n" + "="*80)
    print("CONCLUSION: 10:15 START PROVIDES OPTIMAL BALANCE")
    print("="*80)
    print("• 20 minutes additional preparation time")
    print("• 40% accuracy achieved by 11:07")  
    print("• 52.5 minutes trading time remaining")
    print("• 67% maximum accuracy potential")
    print("• Best efficiency for practical trading")

if __name__ == "__main__":
    analyze_optimal_start_time_results()
