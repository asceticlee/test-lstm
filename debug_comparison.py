#!/usr/bin/env python3
"""
Debug why daily selection is using static fallback while weekly works
"""
import sys
sys.path.append('src')

from test_all_models_regimes import ModelTester
import pandas as pd

# Create tester instance
tester = ModelTester()

# Test specific scenario
candidates = ['00001', '00002', '00003']
test_date = 20200102
regime_type = 1
direction = 'upside'

# Load regime base data
regime_base_file = 'test_results/best_regime_summary_1_425.csv'
regime_base_df = pd.read_csv(regime_base_file)

print(f"Testing with candidates: {candidates}")
print(f"Daily performance data shape: {tester.daily_performance.shape if tester.daily_performance is not None else 'None'}")

# Add some debug prints inside the method by calling directly
print(f"\nCalling _select_daily_winner...")
result = tester._select_daily_winner(candidates, test_date, regime_type, direction, regime_base_df, None)
print(f"Daily result: {result}")

# Now test weekly for comparison
test_dates = [20200102, 20200103]
print(f"\nCalling _select_weekly_winner...")
weekly_result = tester._select_weekly_winner(candidates, test_dates, regime_type, direction, regime_base_df, None)
print(f"Weekly result: {weekly_result}")
