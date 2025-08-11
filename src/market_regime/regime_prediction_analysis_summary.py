#!/usr/bin/env python3
"""
Regime Prediction Analysis Summary

This script provides a practical summary of regime prediction accuracy results
to help understand when market regime predictions become reliable enough for trading.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_regime_prediction_results(results_file):
    """Analyze the regime prediction accuracy results and provide practical insights"""
    
    # Load results
    results = pd.read_csv(results_file)
    
    print("="*70)
    print("MARKET REGIME PREDICTION ACCURACY ANALYSIS")
    print("="*70)
    
    print(f"Analysis period: 10:35 AM to 12:00 PM (85 minutes)")
    print(f"Test points: {len(results)}")
    print(f"Days tested: {results['n_test_days'].iloc[0]:,}")
    
    print("\n📈 ACCURACY PROGRESSION:")
    print("-" * 50)
    
    # Key milestones
    milestones = [
        (10, "Early morning (10% complete)"),
        (25, "Quarter way through (25% complete)"),
        (50, "Halfway point (50% complete)"),
        (75, "Three-quarters (75% complete)"),
        (90, "Near end (90% complete)")
    ]
    
    for pct, desc in milestones:
        # Find closest match
        closest_idx = np.argmin(np.abs(results['completion_percentage'] - pct))
        closest_row = results.iloc[closest_idx]
        
        if np.abs(closest_row['completion_percentage'] - pct) < 15:  # Within 15%
            accuracy = closest_row['overall_accuracy']
            time_str = closest_row['end_time_str']
            duration = closest_row['duration_minutes']
            
            # Color coding for accuracy levels
            if accuracy >= 0.8:
                status = "🟢 EXCELLENT"
            elif accuracy >= 0.6:
                status = "🟡 GOOD"
            elif accuracy >= 0.4:
                status = "🟠 MODERATE"
            else:
                status = "🔴 POOR"
            
            print(f"{desc:25} | {time_str} ({duration:4.1f}min) | {accuracy:5.1%} | {status}")
    
    print("\n🎯 TRADING DECISION THRESHOLDS:")
    print("-" * 50)
    
    # Find when accuracy crosses key thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        threshold_rows = results[results['overall_accuracy'] >= threshold]
        if len(threshold_rows) > 0:
            first_time = threshold_rows.iloc[0]
            time_str = first_time['end_time_str']
            duration = first_time['duration_minutes']
            completion = first_time['completion_percentage']
            
            print(f"{threshold:4.0%} accuracy reached at: {time_str} ({duration:4.1f}min, {completion:4.1f}% complete)")
    
    print("\n📊 PRACTICAL IMPLICATIONS:")
    print("-" * 50)
    
    # Calculate time to reliable prediction
    reliable_threshold = 0.7  # 70% accuracy threshold
    reliable_rows = results[results['overall_accuracy'] >= reliable_threshold]
    
    if len(reliable_rows) > 0:
        reliable_time = reliable_rows.iloc[0]
        minutes_needed = reliable_time['duration_minutes']
        time_str = reliable_time['end_time_str']
        remaining_minutes = 85 - minutes_needed
        
        print(f"• Reliable predictions (≥70%) available from: {time_str}")
        print(f"• Time needed for reliable prediction: {minutes_needed:.1f} minutes")
        print(f"• Remaining trading time after reliable prediction: {remaining_minutes:.1f} minutes")
        print(f"• Trading window efficiency: {remaining_minutes/85*100:.1f}% of total period")
    
    # Early prediction quality
    early_threshold = 30  # First 30% of trading day
    early_rows = results[results['completion_percentage'] <= early_threshold]
    if len(early_rows) > 0:
        early_accuracy = early_rows['overall_accuracy'].mean()
        print(f"• Early prediction accuracy (first 30%): {early_accuracy:.1%}")
        
        if early_accuracy >= 0.4:
            print("  → Early predictions may be useful for initial positioning")
        else:
            print("  → Early predictions not reliable - wait for more data")
    
    print("\n⚡ REGIME-SPECIFIC INSIGHTS:")
    print("-" * 50)
    
    # Analyze regime-specific accuracy in the final results
    final_results = results.iloc[-1]
    regime_cols = [col for col in results.columns if col.startswith('regime_') and col.endswith('_accuracy')]
    
    print("Final regime prediction accuracy by regime:")
    for col in regime_cols:
        if col in final_results:
            regime_num = col.split('_')[1]
            accuracy = final_results[col]
            print(f"• Regime {regime_num}: {accuracy:.1%}")
    
    print("\n🚀 RECOMMENDATIONS:")
    print("-" * 50)
    
    if len(reliable_rows) > 0 and reliable_time['duration_minutes'] < 60:
        print("✅ Market regime prediction is VIABLE for trading:")
        print(f"   - Wait until {time_str} for reliable regime identification")
        print(f"   - Execute trades with {remaining_minutes:.0f} minutes remaining")
        print("   - Consider early indicators for preliminary positioning")
    elif len(reliable_rows) > 0:
        print("⚠️  Market regime prediction has LIMITED trading utility:")
        print(f"   - Reliable prediction only available at {time_str}")
        print(f"   - Only {remaining_minutes:.0f} minutes left for execution")
        print("   - Better suited for next-day strategy preparation")
    else:
        print("❌ Market regime prediction NOT suitable for same-day trading:")
        print("   - Accuracy remains low throughout the trading period")
        print("   - Consider alternative regime detection methods")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    results_file = Path(__file__).parent / "../../market_regime/prediction_accuracy/regime_prediction_accuracy_results.csv"
    
    if results_file.exists():
        analyze_regime_prediction_results(results_file)
    else:
        print("Results file not found. Please run regime_prediction_accuracy_test.py first.")
