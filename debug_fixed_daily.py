#!/usr/bin/env python3

import pandas as pd
import numpy as np

# Load the daily best models data
daily_best_models = pd.read_csv('test_results/daily_best_models_1_425.csv', dtype={'best_upside_model_id': str, 'best_downside_model_id': str})

def get_best_models_daily_fixed(regime, prediction_date):
    """Fixed version of get_best_models_daily"""
    prediction_date_int = int(prediction_date)
    
    # Get all available trading days for this regime, sorted in descending order
    available_days = daily_best_models[
        daily_best_models['test_regime'] == regime
    ]['trading_day'].unique()
    available_days = sorted(available_days, reverse=True)
    
    print(f"Available trading days for regime {regime}: {available_days[:10]}...")  # Show first 10
    
    # Find the most recent trading day before the prediction date
    for trading_day in available_days:
        if trading_day < prediction_date_int:
            daily_data = daily_best_models[
                (daily_best_models['trading_day'] == trading_day) &
                (daily_best_models['test_regime'] == regime)
            ]
            
            if len(daily_data) > 0:
                row = daily_data.iloc[0]
                
                print(f"Found daily best models from {trading_day} for regime {regime} on {prediction_date}")
                print(f"  Upside model: {row['best_upside_model_id']}")
                print(f"  Downside model: {row['best_downside_model_id']}")
                return True
    
    print(f"No daily best models found for regime {regime}")
    return False

# Test the fixed version
print("Testing fixed daily model selection:")
print("=" * 50)

test_cases = [
    (20250701, 3),
    (20250702, 3),
    (20200107, 3),
]

for prediction_date, regime in test_cases:
    print(f"\nTesting prediction_date={prediction_date}, regime={regime}")
    get_best_models_daily_fixed(regime, prediction_date)
