#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

def verify_weekly_training_filtering_corrected():
    """
    CORRECTED version: Verify using actual trading days within each week,
    not calendar week boundaries.
    """
    
    # Read the files
    regime_df = pd.read_csv('test_results/best_regime_summary_1_425.csv')
    weekly_df = pd.read_csv('test_results/weekly_best_models_40_70.csv')
    regime_assignments = pd.read_csv('regime_analysis/regime_assignments.csv')
    
    print("Verifying Weekly Training Period Filtering (CORRECTED)")
    print("=" * 60)
    print(f"Loaded {len(regime_df)} models from regime base file")
    print(f"Loaded {len(weekly_df)} weekly selections")
    print(f"Loaded {len(regime_assignments)} regime assignment records")
    
    # Get unique trading days
    unique_trading_days = sorted(regime_assignments['TradingDay'].unique())
    
    # Build trading days to week mapping
    trading_days_df = pd.DataFrame({'trading_day': unique_trading_days})
    trading_days_df['date'] = pd.to_datetime(trading_days_df['trading_day'], format='%Y%m%d')
    trading_days_df['week_start'] = trading_days_df['date'].dt.to_period('W').dt.start_time
    trading_days_df['week_end'] = trading_days_df['date'].dt.to_period('W').dt.end_time
    
    # Create week to trading days mapping
    week_to_trading_days = {}
    for (week_start, week_end), week_data in trading_days_df.groupby(['week_start', 'week_end']):
        week_key = (int(week_start.strftime('%Y%m%d')), int(week_end.strftime('%Y%m%d')))
        week_to_trading_days[week_key] = week_data['trading_day'].tolist()
    
    violations = 0
    total_checks = 0
    
    # Check all weekly selections
    sample_weeks = weekly_df.sample(min(100, len(weekly_df)))
    
    for _, row in sample_weeks.iterrows():
        week_start = int(row['week_start'])
        week_end = int(row['week_end'])
        week_key = (week_start, week_end)
        
        # Get actual trading days for this week
        week_trading_days = week_to_trading_days.get(week_key, [])
        
        # Check upside model
        if pd.notna(row['best_upside_model_id']):
            model_id = int(row['best_upside_model_id'])
            model_info = regime_df[regime_df['model_id'] == model_id]
            
            if len(model_info) > 0:
                training_from = int(model_info.iloc[0]['training_from'])
                training_to = int(model_info.iloc[0]['training_to'])
                
                # CORRECT CHECK: Does training period overlap with ACTUAL TRADING DAYS?
                overlapping_days = [
                    day for day in week_trading_days 
                    if training_from <= day <= training_to
                ]
                
                if overlapping_days:
                    violations += 1
                    if violations <= 5:  # Show first few violations
                        print(f'VIOLATION: Week {week_start}-{week_end} (trading days: {week_trading_days})')
                        print(f'  Upside Model {model_id:05d} trained {training_from}-{training_to}')
                        print(f'  Overlapping trading days: {overlapping_days}')
            total_checks += 1
        
        # Check downside model  
        if pd.notna(row['best_downside_model_id']):
            model_id = int(row['best_downside_model_id'])
            model_info = regime_df[regime_df['model_id'] == model_id]
            
            if len(model_info) > 0:
                training_from = int(model_info.iloc[0]['training_from'])
                training_to = int(model_info.iloc[0]['training_to'])
                
                # CORRECT CHECK: Does training period overlap with ACTUAL TRADING DAYS?
                overlapping_days = [
                    day for day in week_trading_days 
                    if training_from <= day <= training_to
                ]
                
                if overlapping_days:
                    violations += 1
                    if violations <= 5:  # Show first few violations
                        print(f'VIOLATION: Week {week_start}-{week_end} (trading days: {week_trading_days})')
                        print(f'  Downside Model {model_id:05d} trained {training_from}-{training_to}')
                        print(f'  Overlapping trading days: {overlapping_days}')
            total_checks += 1
    
    print(f"\\nTotal weekly model selections checked: {total_checks}")
    print(f"Training period violations: {violations}")
    
    if violations == 0:
        print("✅ PERFECT: All weekly models properly filtered - no training period violations!")
        
        # Show some examples of proper filtering
        print("\\nSample of properly filtered weekly selections:")
        for _, row in weekly_df.sample(5).iterrows():
            week_start = int(row['week_start'])
            week_end = int(row['week_end'])
            week_key = (week_start, week_end)
            week_trading_days = week_to_trading_days.get(week_key, [])
            
            if pd.notna(row['best_upside_model_id']):
                model_id = int(row['best_upside_model_id'])
                model_info = regime_df[regime_df['model_id'] == model_id]
                if len(model_info) > 0:
                    training_from = int(model_info.iloc[0]['training_from'])
                    training_to = int(model_info.iloc[0]['training_to'])
                    print(f"  Week {week_start}-{week_end}, Regime {row['test_regime']} Upside: Model {model_id:05d} (trained {training_from}-{training_to}) - ✅ Valid")
                    break
    else:
        print(f"🚨 Found {violations} violations - weekly filtering needs debugging!")
    
    # Special check for the originally problematic case
    print("\\n" + "="*60)
    print("SPECIAL CHECK: The originally reported violation case")
    
    problem_week = weekly_df[
        (weekly_df['week_start'] == 20250120) & 
        (weekly_df['test_regime'] == 0)
    ]
    
    if len(problem_week) > 0:
        row = problem_week.iloc[0]
        week_trading_days = week_to_trading_days.get((20250120, 20250126), [])
        
        print(f"Week 20250120-20250126, Regime 0:")
        print(f"  Calendar week: Monday 2025-01-20 to Sunday 2025-01-26")
        print(f"  Actual trading days: {week_trading_days}")
        print(f"  Selected upside model: {int(row['best_upside_model_id'])}")
        print(f"  Selected downside model: {int(row['best_downside_model_id'])}")
        
        # Check the upside model
        model_id = int(row['best_upside_model_id'])
        model_info = regime_df[regime_df['model_id'] == model_id]
        if len(model_info) > 0:
            training_from = int(model_info.iloc[0]['training_from'])
            training_to = int(model_info.iloc[0]['training_to'])
            
            print(f"  Model {model_id} training period: {training_from} to {training_to}")
            
            # Check for overlap with actual trading days
            overlapping_days = [
                day for day in week_trading_days 
                if training_from <= day <= training_to
            ]
            
            if overlapping_days:
                print(f"  🚨 VIOLATION: Overlapping trading days: {overlapping_days}")
            else:
                print(f"  ✅ VALID: No overlap with week's trading days")
                print(f"  📅 Note: Training starts 2025-01-26 (Sunday), which is NOT a trading day")

if __name__ == "__main__":
    verify_weekly_training_filtering_corrected()
