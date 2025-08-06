#!/usr/bin/env python3
"""
Debug weekly selection to find why Model 53 is being selected 
for the week 20250120-20250126 despite training overlap
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
results_dir = Path('test_results')
regime_base_df = pd.read_csv(results_dir / 'best_regime_summary_1_425.csv')
df = pd.read_csv(results_dir / 'model_regime_test_results_1_425.csv')

def _model_trained_during_week(model_id, week_trading_days, regime_base_df):
    model_row = regime_base_df[regime_base_df['model_id'] == int(model_id)]
    if len(model_row) == 0:
        return True  # Conservative: exclude if model not found
        
    training_from = int(model_row.iloc[0]['training_from'])
    training_to = int(model_row.iloc[0]['training_to'])
    
    # Check if any day in the week falls within training period
    return any(
        training_from <= trading_day <= training_to 
        for trading_day in week_trading_days
    )

# Focus on the problematic week
week_trading_days = [20250120, 20250121, 20250122, 20250123, 20250124, 20250125, 20250126]
test_regime = 0
start_model = 40
end_model = 70

print(f"Debugging weekly selection for week {week_trading_days[0]}-{week_trading_days[-1]}, regime {test_regime}")
print("=" * 80)

# Build regime models (same logic as weekly method)
valid_models = set(f"{i:05d}" for i in range(start_model, end_model + 1))
upside_regime_models = {}
downside_regime_models = {}

for _, row in regime_base_df.iterrows():
    model_id = f"{int(row['model_id']):05d}"  # Convert back to 00001 format
    
    if model_id not in valid_models:
        continue
    
    # Group by upside regime base
    upside_regime = row['model_regime_upside']
    if upside_regime not in upside_regime_models:
        upside_regime_models[upside_regime] = []
    upside_regime_models[upside_regime].append(model_id)
    
    # Group by downside regime base
    downside_regime = row['model_regime_downside'] 
    if downside_regime not in downside_regime_models:
        downside_regime_models[downside_regime] = []
    downside_regime_models[downside_regime].append(model_id)

# Get candidates for this specific regime
upside_candidates = upside_regime_models.get(test_regime, [])
downside_candidates = downside_regime_models.get(test_regime, [])

print(f"Initial upside candidates for regime {test_regime}: {len(upside_candidates)}")
print(f"Candidates: {upside_candidates}")
print(f"Model 00053 in initial candidates: {'00053' in upside_candidates}")
print()

# Apply filtering
print("Applying training period filter...")
filtered_upside_candidates = [
    model_id for model_id in upside_candidates
    if not _model_trained_during_week(model_id, week_trading_days, regime_base_df)
]

filtered_downside_candidates = [
    model_id for model_id in downside_candidates
    if not _model_trained_during_week(model_id, week_trading_days, regime_base_df)
]

print(f"After filtering - upside candidates: {len(filtered_upside_candidates)}")
print(f"Filtered upside candidates: {filtered_upside_candidates}")
print(f"Model 00053 in filtered candidates: {'00053' in filtered_upside_candidates}")

print(f"After filtering - downside candidates: {len(filtered_downside_candidates)}")
print(f"Filtered downside candidates: {filtered_downside_candidates}")
print(f"Model 00053 in filtered downside candidates: {'00053' in filtered_downside_candidates}")
print()

# If model 53 is being filtered out correctly, where is it coming from?
print("Checking if model 00053 was filtered correctly:")
should_be_excluded = _model_trained_during_week('00053', week_trading_days, regime_base_df)
print(f"Model 00053 should be excluded: {should_be_excluded}")

# Check training period
model_info = regime_base_df[regime_base_df['model_id'] == 53].iloc[0]
training_from = int(model_info['training_from'])
training_to = int(model_info['training_to'])
print(f"Model 00053 training period: {training_from} to {training_to}")

# Check overlap with week days
overlapping_days = [day for day in week_trading_days if training_from <= day <= training_to]
print(f"Overlapping days: {overlapping_days}")
print()

# Simulate the competition (what should happen)
print("Simulating upside competition...")
if filtered_upside_candidates:
    upside_results = []
    for model_id in filtered_upside_candidates:
        print(f"  Checking model {model_id}...")
        model_data = df[(df['model_id'] == int(model_id)) & (df['regime'] == test_regime)]
        print(f"    Found {len(model_data)} performance records")
        if len(model_data) > 0:
            row = model_data.iloc[0]
            upside_accs = [row[f'upside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
            avg_upside_score = np.mean(upside_accs)
            print(f"    Average upside score: {avg_upside_score:.4f}")
            upside_results.append((model_id, avg_upside_score))
    
    print(f"  Total upside results collected: {len(upside_results)}")
    if upside_results:
        upside_results.sort(key=lambda x: x[1], reverse=True)
        best_upside_model, best_upside_score = upside_results[0]
        print(f"Best upside model should be: {best_upside_model} with score {best_upside_score:.4f}")
        print(f"Top 3 upside results: {upside_results[:3]}")
    else:
        print("No upside results found")
        best_upside_model = None
else:
    print("No upside candidates after filtering")
    best_upside_model = None

print()
print("Simulating downside competition...")
if filtered_downside_candidates:
    downside_results = []
    for model_id in filtered_downside_candidates:
        model_data = df[(df['model_id'] == model_id) & (df['regime'] == test_regime)]
        if len(model_data) > 0:
            row = model_data.iloc[0]
            downside_accs = [row[f'downside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
            avg_downside_score = np.mean(downside_accs)
            downside_results.append((model_id, avg_downside_score))
    
    if downside_results:
        downside_results.sort(key=lambda x: x[1], reverse=True)
        best_downside_model, best_downside_score = downside_results[0]
        print(f"Best downside model should be: {best_downside_model} with score {best_downside_score:.4f}")
        print(f"Top 3 downside results: {downside_results[:3]}")
    else:
        print("No downside results found")
        best_downside_model = None
else:
    print("No downside candidates after filtering")
    best_downside_model = None

print()
print("=" * 80)
print("EXPECTED RESULTS:")
print(f"Best upside model: {best_upside_model}")
print(f"Best downside model: {best_downside_model}")

# Compare with actual results
weekly_df = pd.read_csv('test_results/weekly_best_models_40_70.csv')
actual_results = weekly_df[
    (weekly_df['week_start'] == 20250120) & 
    (weekly_df['test_regime'] == test_regime)
]

if len(actual_results) > 0:
    actual = actual_results.iloc[0]
    print("\nACTUAL RESULTS FROM FILE:")
    print(f"Best upside model: {actual['best_upside_model_id']}")
    print(f"Best downside model: {actual['best_downside_model_id']}")
    
    if str(actual['best_upside_model_id']) == '53.0':
        print("\n🚨 CONFIRMED BUG: Model 53 was selected as upside model!")
    if str(actual['best_downside_model_id']) == '53.0':
        print("\n🚨 CONFIRMED BUG: Model 53 was selected as downside model!")
else:
    print("\nNo actual results found in file")
