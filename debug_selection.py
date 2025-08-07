#!/usr/bin/env python3
"""
Test daily winner selection logic
"""
import sys
sys.path.append('src')

from test_all_models_regimes import ModelTester

# Create tester instance
tester = ModelTester()

# Check if performance data loaded
print(f"Daily performance is None: {tester.daily_performance is None}")
if tester.daily_performance is not None:
    print(f"Daily performance shape: {tester.daily_performance.shape}")

# Test the daily winner selection
candidates = ['00001', '00002', '00003']
test_date = 20200102
test_regime = 1
test_direction = 'upside'

print(f"\nTesting daily winner selection:")
print(f"Candidates: {candidates}")
print(f"Date: {test_date}")
print(f"Regime: {test_regime}")
print(f"Direction: {test_direction}")

# Load regime base data
import pandas as pd
import os

regime_base_file = os.path.join('test_results', 'best_regime_summary_1_425.csv')
if os.path.exists(regime_base_file):
    regime_base_df = pd.read_csv(regime_base_file)
    print(f"Loaded regime base data: {len(regime_base_df)} records")
    
    result = tester._select_daily_winner(candidates, test_date, test_regime, test_direction, regime_base_df, None)
    print(f"Result: {result}")
else:
    print(f"Regime base file not found: {regime_base_file}")
