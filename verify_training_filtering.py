#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

def verify_training_period_filtering():
    """Verify that models are properly excluded when their training period covers the trading day"""
    
    # Read the enhanced regime base file
    regime_file = Path('test_results/best_regime_summary_1_425.csv')
    regime_df = pd.read_csv(regime_file)
    
    # Read the daily results file  
    daily_file = Path('test_results/daily_best_models_40_70.csv')
    daily_df = pd.read_csv(daily_file)
    
    print("Verifying training period filtering...")
    print(f"Loaded {len(regime_df)} models from regime base file")
    print(f"Loaded {len(daily_df)} daily selections")
    
    violations = []
    checked_combinations = 0
    
    # Sample some trading days and check if selected models violate training period rule
    sample_days = sorted(daily_df['trading_day'].unique())[::100]  # Every 100th day
    
    for trading_day in sample_days:
        day_selections = daily_df[daily_df['trading_day'] == trading_day]
        
        for _, row in day_selections.iterrows():
            checked_combinations += 1
            
            # Check upside model
            if pd.notna(row['best_upside_model_id']):
                model_id = int(row['best_upside_model_id'])
                model_info = regime_df[regime_df['model_id'] == model_id]
                
                if len(model_info) > 0:
                    training_from = int(model_info.iloc[0]['training_from'])
                    training_to = int(model_info.iloc[0]['training_to'])
                    
                    if training_from <= trading_day <= training_to:
                        violations.append({
                            'trading_day': trading_day,
                            'regime': row['test_regime'],
                            'direction': 'upside',
                            'model_id': f"{model_id:05d}",
                            'training_from': training_from,
                            'training_to': training_to,
                            'violation': 'Model trained on target day'
                        })
            
            # Check downside model
            if pd.notna(row['best_downside_model_id']):
                model_id = int(row['best_downside_model_id'])
                model_info = regime_df[regime_df['model_id'] == model_id]
                
                if len(model_info) > 0:
                    training_from = int(model_info.iloc[0]['training_from'])
                    training_to = int(model_info.iloc[0]['training_to'])
                    
                    if training_from <= trading_day <= training_to:
                        violations.append({
                            'trading_day': trading_day,
                            'regime': row['test_regime'],
                            'direction': 'downside', 
                            'model_id': f"{model_id:05d}",
                            'training_from': training_from,
                            'training_to': training_to,
                            'violation': 'Model trained on target day'
                        })
    
    print(f"\nVerification Results:")
    print(f"  Checked {checked_combinations} combinations across {len(sample_days)} sample days")
    print(f"  Training period violations found: {len(violations)}")
    
    if violations:
        print(f"\nVIOLATIONS DETECTED:")
        for v in violations[:10]:  # Show first 10 violations
            print(f"  Day {v['trading_day']}, Regime {v['regime']}, {v['direction']}: "
                  f"Model {v['model_id']} trained {v['training_from']}-{v['training_to']}")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more violations")
    else:
        print(f"\n✅ VERIFICATION PASSED: No training period violations detected!")
        
    # Show some examples of proper filtering
    print(f"\nSample of properly filtered selections:")
    for trading_day in sample_days[:3]:
        day_selections = daily_df[daily_df['trading_day'] == trading_day].head(2)
        print(f"\nTrading Day {trading_day}:")
        
        for _, row in day_selections.iterrows():
            if pd.notna(row['best_upside_model_id']):
                model_id = int(row['best_upside_model_id'])
                model_info = regime_df[regime_df['model_id'] == model_id]
                if len(model_info) > 0:
                    training_from = int(model_info.iloc[0]['training_from'])
                    training_to = int(model_info.iloc[0]['training_to'])
                    print(f"  Regime {row['test_regime']} Upside: Model {model_id:05d} "
                          f"(trained {training_from}-{training_to}) - {'✅ Valid' if not (training_from <= trading_day <= training_to) else '❌ INVALID'}")

if __name__ == "__main__":
    verify_training_period_filtering()
