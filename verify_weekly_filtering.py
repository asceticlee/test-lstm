#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

def verify_weekly_training_filtering():
    """Verify that weekly models are properly excluded when their training period overlaps the week"""
    
    # Read the files
    regime_df = pd.read_csv('test_results/best_regime_summary_1_425.csv')
    weekly_df = pd.read_csv('test_results/weekly_best_models_40_70.csv')
    
    print("Verifying Weekly Training Period Filtering")
    print("=" * 50)
    print(f"Loaded {len(regime_df)} models from regime base file")
    print(f"Loaded {len(weekly_df)} weekly selections")
    
    violations = 0
    total_checks = 0
    
    # Sample some weeks for checking
    sample_weeks = weekly_df.sample(min(50, len(weekly_df)))
    
    for _, row in sample_weeks.iterrows():
        week_start = int(row['week_start'])
        week_end = int(row['week_end'])
        
        # Check upside model
        if pd.notna(row['best_upside_model_id']):
            model_id = int(row['best_upside_model_id'])
            model_info = regime_df[regime_df['model_id'] == model_id]
            
            if len(model_info) > 0:
                training_from = int(model_info.iloc[0]['training_from'])
                training_to = int(model_info.iloc[0]['training_to'])
                
                # Check if training period overlaps with the week
                # This is more complex than daily - need to check if ANY day in week overlaps training
                week_overlaps = not (training_to < week_start or training_from > week_end)
                
                if week_overlaps:
                    violations += 1
                    if violations <= 5:  # Show first few violations
                        print(f'VIOLATION: Week {week_start}-{week_end}, Upside Model {model_id:05d} trained {training_from}-{training_to}')
            total_checks += 1
        
        # Check downside model  
        if pd.notna(row['best_downside_model_id']):
            model_id = int(row['best_downside_model_id'])
            model_info = regime_df[regime_df['model_id'] == model_id]
            
            if len(model_info) > 0:
                training_from = int(model_info.iloc[0]['training_from'])
                training_to = int(model_info.iloc[0]['training_to'])
                
                # Check if training period overlaps with the week
                week_overlaps = not (training_to < week_start or training_from > week_end)
                
                if week_overlaps:
                    violations += 1
                    if violations <= 5:  # Show first few violations
                        print(f'VIOLATION: Week {week_start}-{week_end}, Downside Model {model_id:05d} trained {training_from}-{training_to}')
            total_checks += 1
    
    print(f'Total weekly model selections checked: {total_checks:,}')
    print(f'Training period violations: {violations}')
    
    if violations == 0:
        print('✅ PERFECT: All weekly models properly filtered - no training period violations!')
    else:
        print(f'❌ VIOLATIONS DETECTED: {violations} cases where models were selected during their training period')
    
    # Show some examples of proper filtering
    print(f'\\nSample of properly filtered weekly selections:')
    for _, row in sample_weeks.head(5).iterrows():
        week_start = int(row['week_start'])
        week_end = int(row['week_end'])
        
        if pd.notna(row['best_upside_model_id']):
            model_id = int(row['best_upside_model_id'])
            model_info = regime_df[regime_df['model_id'] == model_id]
            if len(model_info) > 0:
                training_from = int(model_info.iloc[0]['training_from'])
                training_to = int(model_info.iloc[0]['training_to'])
                week_overlaps = not (training_to < week_start or training_from > week_end)
                print(f'  Week {week_start}-{week_end}, Regime {row["test_regime"]} Upside: Model {model_id:05d} '
                      f'(trained {training_from}-{training_to}) - {"❌ VIOLATION" if week_overlaps else "✅ Valid"}')

if __name__ == "__main__":
    verify_weekly_training_filtering()
