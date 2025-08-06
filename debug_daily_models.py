#!/usr/bin/env python3

import pandas as pd
import numpy as np

# Load the daily best models data
daily_best_models = pd.read_csv('test_results/daily_best_models_1_425.csv', dtype={'best_upside_model_id': str, 'best_downside_model_id': str})

print("Daily best models data loaded:")
print(f"Total rows: {len(daily_best_models)}")
print(f"Columns: {list(daily_best_models.columns)}")
print(f"Data types: {daily_best_models.dtypes}")
print()

# Test specific case: looking for regime 3 on 20250701, should find data from 20250630
prediction_date = 20250701
regime = 3

print(f"Testing prediction for date {prediction_date}, regime {regime}")
print()

# Search backwards up to 10 days
for days_back in range(1, 11):
    search_date = prediction_date - days_back
    print(f"  Searching {days_back} days back: {search_date}")
    
    # Filter data
    daily_data = daily_best_models[
        (daily_best_models['trading_day'] == search_date) &
        (daily_best_models['test_regime'] == regime)
    ]
    
    print(f"    Found {len(daily_data)} rows")
    if len(daily_data) > 0:
        row = daily_data.iloc[0]
        print(f"    Found data: upside_model={row['best_upside_model_id']}, downside_model={row['best_downside_model_id']}")
        break
    else:
        # Check if the trading day exists at all
        day_exists = len(daily_best_models[daily_best_models['trading_day'] == search_date]) > 0
        print(f"    Trading day {search_date} exists in data: {day_exists}")
        
        # Check if regime 3 exists for any day
        regime_exists = len(daily_best_models[daily_best_models['test_regime'] == regime]) > 0
        print(f"    Regime {regime} exists in data: {regime_exists}")

print()
print("Checking data around 20250630:")
nearby_data = daily_best_models[
    (daily_best_models['trading_day'] >= 20250628) & 
    (daily_best_models['trading_day'] <= 20250702) &
    (daily_best_models['test_regime'] == regime)
]
print(nearby_data[['trading_day', 'test_regime', 'best_upside_model_id', 'best_downside_model_id']])
