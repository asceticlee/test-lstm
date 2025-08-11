#!/usr/bin/env python3
"""
Test prediction accuracy of simplified overnight gap features
"""

import sys
import os
import subprocess
from pathlib import Path

def run_prediction_test(scenario_name, regime_path, description):
    """Run a single prediction accuracy test"""
    print(f"\n{description}")
    print("-" * 50)
    
    # Run the progressive prediction test
    cmd = [
        "./venv-test-lstm/bin/python", 
        "src/market_regime/progressive_regime_prediction_test.py",
        "--regime_assignments", regime_path,
        "--test_intervals", "15"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/stephen/projects/Testing/TestPy/test-lstm")
        
        # Parse the output to find 50% accuracy time
        lines = result.stdout.split('\n')
        accuracy_50_time = None
        
        for line in lines:
            if "50% accuracy achieved at:" in line:
                time_part = line.split("50% accuracy achieved at:")[-1].strip()
                accuracy_50_time = time_part
                break
        
        print(f"{scenario_name} 50% accuracy achieved at: {accuracy_50_time if accuracy_50_time else 'Never'}")
        return accuracy_50_time
        
    except Exception as e:
        print(f"Error running test for {scenario_name}: {e}")
        return None

def main():
    """Test simplified overnight features against standard approach"""
    
    print("="*80)
    print("TESTING SIMPLIFIED OVERNIGHT GAP FEATURES")
    print("="*80)
    
    # Test standard approach (for comparison)
    standard_50_time = run_prediction_test(
        "Standard", 
        "../../market_regime/overnight_analysis/standard/daily_regime_assignments.csv",
        "1. STANDARD APPROACH (No Overnight Features)"
    )
    
    # Test simplified overnight approach
    simplified_50_time = run_prediction_test(
        "Simplified Overnight",
        "../../market_regime/simplified_overnight/daily_regime_assignments.csv", 
        "2. SIMPLIFIED OVERNIGHT APPROACH"
    )
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"Standard approach:          50% at {standard_50_time if standard_50_time else 'Never'}")
    print(f"Simplified overnight:       50% at {simplified_50_time if simplified_50_time else 'Never'}")
    
    if standard_50_time and simplified_50_time:
        # Convert times to minutes for comparison
        try:
            std_hours, std_mins = map(int, standard_50_time.split(':'))
            simp_hours, simp_mins = map(int, simplified_50_time.split(':'))
            
            std_total_mins = std_hours * 60 + std_mins
            simp_total_mins = simp_hours * 60 + simp_mins
            
            if simp_total_mins < std_total_mins:
                print(f"✅ Simplified overnight is BETTER by {std_total_mins - simp_total_mins} minutes")
            elif simp_total_mins == std_total_mins:
                print("🤝 Both approaches achieve 50% at the same time")
            else:
                print(f"❌ Standard approach is better by {simp_total_mins - std_total_mins} minutes")
        except:
            print("Could not compare times")
    elif standard_50_time and not simplified_50_time:
        print("❌ Standard approach achieves 50%, simplified overnight does not")
    elif simplified_50_time and not standard_50_time:
        print("✅ Simplified overnight achieves 50%, standard does not")
    else:
        print("❌ Neither approach achieves 50% accuracy")

if __name__ == "__main__":
    main()
