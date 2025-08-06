#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

def add_regime_base_columns():
    """Add model_regime_upside and model_regime_downside columns to best_regime_summary"""
    
    # Read the existing file
    input_file = Path('test_results/best_regime_summary_1_425.csv')
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} models from {input_file}")
    
    # Add regime base columns
    model_regime_upside = []
    model_regime_downside = []
    
    for _, row in df.iterrows():
        # Extract upside scores for all regimes
        upside_scores = {
            0: row['regime_0_up'],
            1: row['regime_1_up'], 
            2: row['regime_2_up'],
            3: row['regime_3_up'],
            4: row['regime_4_up']
        }
        
        # Extract downside scores for all regimes  
        downside_scores = {
            0: row['regime_0_down'],
            1: row['regime_1_down'],
            2: row['regime_2_down'], 
            3: row['regime_3_down'],
            4: row['regime_4_down']
        }
        
        # Find regime with lowest (best) upside score
        best_upside_regime = min(upside_scores.keys(), key=lambda k: upside_scores[k])
        
        # Find regime with lowest (best) downside score
        best_downside_regime = min(downside_scores.keys(), key=lambda k: downside_scores[k])
        
        model_regime_upside.append(best_upside_regime)
        model_regime_downside.append(best_downside_regime)
        
        # Print first few examples for verification
        if len(model_regime_upside) <= 5:
            print(f"Model {row['model_id']}: upside regime {best_upside_regime} (score {upside_scores[best_upside_regime]}), "
                  f"downside regime {best_downside_regime} (score {downside_scores[best_downside_regime]})")
    
    # Add new columns
    df['model_regime_upside'] = model_regime_upside
    df['model_regime_downside'] = model_regime_downside
    
    # Save enhanced file
    output_file = input_file  # Overwrite the original
    df.to_csv(output_file, index=False)
    print(f"\nSaved enhanced file with regime base columns to {output_file}")
    
    # Show distribution of regime bases
    print("\nUpside regime base distribution:")
    upside_dist = df['model_regime_upside'].value_counts().sort_index()
    for regime, count in upside_dist.items():
        print(f"  Regime {regime}: {count} models")
        
    print("\nDownside regime base distribution:")
    downside_dist = df['model_regime_downside'].value_counts().sort_index()  
    for regime, count in downside_dist.items():
        print(f"  Regime {regime}: {count} models")
    
    return df

if __name__ == "__main__":
    enhanced_df = add_regime_base_columns()
