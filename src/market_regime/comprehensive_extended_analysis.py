#!/usr/bin/env python3
"""
Comprehensive Extended Period Analysis

This script compares different extended start times to find the optimal
balance between prediction accuracy and additional trading time.

Tests multiple scenarios:
- 9:00 AM start (95 min additional)
- 9:15 AM start (80 min additional) 
- 9:30 AM start (65 min additional)
- 9:45 AM start (50 min additional)
- 10:00 AM start (35 min additional)
- 10:15 AM start (20 min additional)

Usage:
    python comprehensive_extended_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import subprocess
import json

# Add the src directory to the path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent
sys.path.insert(0, str(src_dir))

def time_to_ms(time_str):
    """Convert time string (HH:MM) to milliseconds of day"""
    hours, minutes = map(int, time_str.split(':'))
    return hours * 3600000 + minutes * 60000

def ms_to_time(ms):
    """Convert milliseconds of day to readable time format"""
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    return f"{hours:02d}:{minutes:02d}"

def run_extended_test(extended_start_time, output_suffix=""):
    """Run extended period test for a specific start time"""
    print(f"\n🔄 Running extended test for {extended_start_time} start...")
    
    # Get paths
    python_exe = "/home/stephen/projects/Testing/TestPy/test-lstm/venv-test-lstm/bin/python"
    script_path = script_dir / "extended_period_regime_prediction_test.py"
    
    # Custom output directory for this test
    output_dir = f"../../market_regime/extended_prediction_accuracy_{extended_start_time.replace(':', '')}{output_suffix}"
    
    # Run the test
    cmd = [
        python_exe, str(script_path),
        "--extended_start", extended_start_time,
        "--output_dir", output_dir,
        "--test_intervals", "15",
        "--min_duration", "10"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
        
        if result.returncode == 0:
            print(f"✅ Test completed for {extended_start_time}")
            
            # Load the results
            results_file = script_dir / output_dir.replace("../..", ".") / "extended_regime_prediction_accuracy_results.csv"
            summary_file = script_dir / output_dir.replace("../..", ".") / "extended_accuracy_test_summary.json"
            
            if results_file.exists() and summary_file.exists():
                results_df = pd.read_csv(results_file)
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                return results_df, summary
            else:
                print(f"⚠️  Results files not found for {extended_start_time}")
                return None, None
        else:
            print(f"❌ Test failed for {extended_start_time}: {result.stderr}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error running test for {extended_start_time}: {e}")
        return None, None

def analyze_comprehensive_results():
    """Run comprehensive analysis across multiple extended start times"""
    
    print("="*80)
    print("COMPREHENSIVE EXTENDED PERIOD REGIME PREDICTION ANALYSIS")
    print("="*80)
    
    # Define test scenarios
    test_scenarios = [
        "09:00",  # 95 min additional
        "09:15",  # 80 min additional
        "09:30",  # 65 min additional
        "09:45",  # 50 min additional
        "10:00",  # 35 min additional
        "10:15",  # 20 min additional
    ]
    
    regime_start_ms = time_to_ms("10:35")
    regime_end_ms = time_to_ms("12:00")
    
    all_results = []
    all_summaries = []
    
    # Run tests for each scenario
    for start_time in test_scenarios:
        extended_start_ms = time_to_ms(start_time)
        additional_time = (regime_start_ms - extended_start_ms) / 60000
        
        print(f"\n📊 Testing scenario: {start_time} start (+{additional_time:.0f} min trading time)")
        
        # Run the test
        results_df, summary = run_extended_test(start_time)
        
        if results_df is not None and summary is not None:
            # Add scenario info
            summary['extended_start_time'] = start_time
            summary['additional_trading_minutes'] = additional_time
            
            all_results.append(results_df)
            all_summaries.append(summary)
        else:
            print(f"⚠️  Skipping {start_time} due to test failure")
    
    if len(all_summaries) == 0:
        print("❌ No successful tests - cannot perform analysis")
        return
    
    # Create comprehensive comparison
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS RESULTS")
    print("="*80)
    
    # Create comparison table
    comparison_data = []
    for summary in all_summaries:
        start_time = summary['extended_start_time']
        additional_time = summary['additional_trading_minutes']
        
        # Find best early prediction (before regime start)
        best_early_accuracy = summary.get('best_early_accuracy', 0)
        accuracy_at_regime_start = summary.get('accuracy_at_regime_start', 0)
        final_accuracy = summary.get('final_accuracy', 0)
        
        comparison_data.append({
            'start_time': start_time,
            'additional_trading_time': additional_time,
            'best_early_accuracy': best_early_accuracy,
            'accuracy_at_regime_start': accuracy_at_regime_start,
            'final_accuracy': final_accuracy,
            'viable_for_trading': summary.get('viable_for_trading', False)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    print("\n📋 SCENARIO COMPARISON:")
    print("-" * 80)
    print("Start Time | Additional Time | Best Early | At Regime Start | Final | Viable")
    print("-" * 80)
    
    for _, row in comparison_df.iterrows():
        start_time = row['start_time']
        additional_time = row['additional_trading_time']
        best_early = row['best_early_accuracy']
        regime_start = row['accuracy_at_regime_start']
        final = row['final_accuracy']
        viable = "✅" if row['viable_for_trading'] else "❌"
        
        print(f"  {start_time:8} |      {additional_time:6.0f} min |    {best_early:5.1%} |         {regime_start:5.1%} | {final:5.1%} |   {viable}")
    
    # Find optimal scenarios
    print("\n🎯 OPTIMAL SCENARIOS:")
    print("-" * 50)
    
    # Scenario 1: Best early accuracy
    best_early_idx = comparison_df['best_early_accuracy'].idxmax()
    best_early_scenario = comparison_df.iloc[best_early_idx]
    
    print(f"Best early accuracy: {best_early_scenario['start_time']} start")
    print(f"  - Early accuracy: {best_early_scenario['best_early_accuracy']:.1%}")
    print(f"  - Additional time: {best_early_scenario['additional_trading_time']:.0f} minutes")
    
    # Scenario 2: Best balance (accuracy > 30% with most additional time)
    viable_scenarios = comparison_df[comparison_df['best_early_accuracy'] >= 0.30]
    if len(viable_scenarios) > 0:
        best_balance_idx = viable_scenarios['additional_trading_time'].idxmax()
        best_balance_scenario = comparison_df.iloc[best_balance_idx]
        
        print(f"\nBest balance (≥30% accuracy): {best_balance_scenario['start_time']} start")
        print(f"  - Early accuracy: {best_balance_scenario['best_early_accuracy']:.1%}")
        print(f"  - Additional time: {best_balance_scenario['additional_trading_time']:.0f} minutes")
    else:
        print(f"\nNo scenarios achieve ≥30% early accuracy")
    
    # Analysis conclusions
    print(f"\n🔍 KEY INSIGHTS:")
    print("-" * 50)
    
    max_early_accuracy = comparison_df['best_early_accuracy'].max()
    min_additional_time_for_max = comparison_df[comparison_df['best_early_accuracy'] == max_early_accuracy]['additional_trading_time'].min()
    
    print(f"• Maximum early accuracy achieved: {max_early_accuracy:.1%}")
    print(f"• Minimum additional time for max accuracy: {min_additional_time_for_max:.0f} minutes")
    
    if max_early_accuracy < 0.5:
        print(f"• ⚠️  All extended scenarios show poor early accuracy (<50%)")
        print(f"• 💡 Recommendation: Stick with original timing, use extended data for confirmation only")
    else:
        print(f"• ✅ Extended period strategy may be viable")
        print(f"• 💡 Recommendation: Use extended data for early positioning")
    
    # Trading strategy recommendations
    print(f"\n🚀 TRADING STRATEGY RECOMMENDATIONS:")
    print("-" * 50)
    
    if max_early_accuracy >= 0.4:
        best_scenario = comparison_df.iloc[comparison_df['best_early_accuracy'].idxmax()]
        print(f"✅ VIABLE EXTENDED STRATEGY:")
        print(f"   - Start data collection: {best_scenario['start_time']}")
        print(f"   - Begin preliminary positioning based on early signals")
        print(f"   - Expected early accuracy: {best_scenario['best_early_accuracy']:.1%}")
        print(f"   - Additional execution time: {best_scenario['additional_trading_time']:.0f} minutes")
        print(f"   - Use position sizing appropriate for accuracy level")
    elif max_early_accuracy >= 0.3:
        print(f"⚠️  LIMITED EXTENDED STRATEGY:")
        print(f"   - Extended data provides weak early signals")
        print(f"   - Use for market preparation, not primary trading signals")
        print(f"   - Wait for regime period data for reliable predictions")
    else:
        print(f"❌ EXTENDED STRATEGY NOT RECOMMENDED:")
        print(f"   - Early predictions too unreliable (<30% accuracy)")
        print(f"   - Focus on optimizing execution speed within regime period")
        print(f"   - Consider alternative regime detection methods")
    
    # Save comprehensive results
    output_dir = script_dir / "../../market_regime/comprehensive_extended_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_dir / "extended_scenarios_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    
    summary_file = output_dir / "comprehensive_analysis_summary.json"
    comprehensive_summary = {
        'test_scenarios': test_scenarios,
        'max_early_accuracy': float(max_early_accuracy),
        'best_early_scenario': best_early_scenario['start_time'],
        'recommendation': 'viable' if max_early_accuracy >= 0.4 else 'limited' if max_early_accuracy >= 0.3 else 'not_recommended',
        'scenarios': comparison_data
    }
    
    with open(summary_file, 'w') as f:
        json.dump(comprehensive_summary, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"   - Scenario comparison: {comparison_file}")
    print(f"   - Summary: {summary_file}")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    analyze_comprehensive_results()
