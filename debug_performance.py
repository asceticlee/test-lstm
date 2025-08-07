#!/usr/bin/env python3
"""
Debug performance file usage
"""
import pandas as pd
import os

# Load the performance file
daily_perf_path = 'src/model_daily_performance_1_425.csv'
daily_performance = pd.read_csv(daily_perf_path)
daily_performance['date'] = pd.to_datetime(daily_performance['date'])

print(f"Daily performance shape: {daily_performance.shape}")
print(f"Date range: {daily_performance['date'].min()} to {daily_performance['date'].max()}")
print(f"Model IDs: {sorted(daily_performance['model_id'].unique())[:10]}...")
print(f"Sample columns: {list(daily_performance.columns)}")

# Test specific date lookup
test_date = pd.to_datetime('2020-01-02')
date_performance = daily_performance[daily_performance['date'] == test_date]
print(f"\nPerformance records for {test_date.date()}: {len(date_performance)}")

if len(date_performance) > 0:
    # Test candidate filtering
    candidates = ['00001', '00002', '00003']
    candidate_ints = [int(candidate) for candidate in candidates]
    candidate_performance = date_performance[date_performance['model_id'].isin(candidate_ints)]
    print(f"Candidate performance records: {len(candidate_performance)}")
    
    if len(candidate_performance) > 0:
        print(f"Available models for {test_date.date()}: {sorted(candidate_performance['model_id'].unique())}")
        print(f"Sample upside_0.0 scores: {candidate_performance[['model_id', 'upside_0.0']].head()}")
    else:
        print("No candidate performance found!")
else:
    print("No performance data for test date!")
