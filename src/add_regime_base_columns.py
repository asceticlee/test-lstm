#!/usr/bin/env python3
"""
Add regime base columns to the best regime summary file.

This script adds model_regime_upside and model_regime_downside columns
that identify each model's best regime for upside and downside performance.
"""

import pandas as pd
import numpy as np
import os

def add_regime_base_columns():
    """Add regime base columns to the best regime summary"""
    
    # Load the existing summary
    summary_path = '../test_results/best_regime_summary_1_425.csv'
    if not os.path.exists(summary_path):
        print(f"ERROR: Summary file not found: {summary_path}")
        return False
    
    df = pd.read_csv(summary_path)
    print(f"Loaded summary with {len(df)} models")
    print(f"Current columns: {list(df.columns)}")
    
    # Get unique regimes (extract from column names)
    upside_cols = [col for col in df.columns if col.endswith('_up')]
    downside_cols = [col for col in df.columns if col.endswith('_down')]
    
    regimes = []
    for col in upside_cols:
        regime = col.split('_')[1]  # Extract regime number from 'regime_X_up'
        regimes.append(int(regime))
    
    regimes = sorted(regimes)
    print(f"Found regimes: {regimes}")
    
    # For each model, find the best regime for upside and downside
    best_upside_regimes = []
    best_downside_regimes = []
    
    for _, row in df.iterrows():
        # Find best upside regime (lowest rank = best)
        upside_ranks = []
        for regime in regimes:
            rank = row[f'regime_{regime}_up']
            upside_ranks.append((regime, rank))
        
        # Sort by rank (lowest = best)
        upside_ranks.sort(key=lambda x: x[1])
        best_upside_regime = upside_ranks[0][0]
        best_upside_regimes.append(best_upside_regime)
        
        # Find best downside regime (lowest rank = best)
        downside_ranks = []
        for regime in regimes:
            rank = row[f'regime_{regime}_down']
            downside_ranks.append((regime, rank))
        
        # Sort by rank (lowest = best)
        downside_ranks.sort(key=lambda x: x[1])
        best_downside_regime = downside_ranks[0][0]
        best_downside_regimes.append(best_downside_regime)
    
    # Add the new columns
    df['model_regime_upside'] = best_upside_regimes
    df['model_regime_downside'] = best_downside_regimes
    
    # Calculate average performance for each model's best regimes
    # This will be used as fallback scores in static selection
    best_upside_scores = []
    best_downside_scores = []
    
    for i, row in df.iterrows():
        best_upside_regime = row['model_regime_upside']
        best_downside_regime = row['model_regime_downside']
        
        # Get the rank for the best regime (lower rank = better performance)
        # Convert rank to a score (higher score = better)
        upside_rank = row[f'regime_{best_upside_regime}_up']
        downside_rank = row[f'regime_{best_downside_regime}_down']
        
        # Convert ranks to scores: rank 1 = high score, higher ranks = lower scores
        # Use a simple inversion: score = 1 / rank
        upside_score = 1.0 / upside_rank if upside_rank > 0 else 0.0
        downside_score = 1.0 / downside_rank if downside_rank > 0 else 0.0
        
        best_upside_scores.append(upside_score)
        best_downside_scores.append(downside_score)
    
    df['model_regime_upside_score'] = best_upside_scores
    df['model_regime_downside_score'] = best_downside_scores
    
    # Save the updated file
    df.to_csv(summary_path, index=False)
    print(f"Updated summary file saved with new columns:")
    print(f"  - model_regime_upside: best regime for upside performance")
    print(f"  - model_regime_downside: best regime for downside performance") 
    print(f"  - model_regime_upside_score: fallback score for upside selection")
    print(f"  - model_regime_downside_score: fallback score for downside selection")
    
    # Show distribution of regime assignments
    print(f"\nUpside regime distribution:")
    upside_dist = pd.Series(best_upside_regimes).value_counts().sort_index()
    for regime, count in upside_dist.items():
        print(f"  Regime {regime}: {count} models")
    
    print(f"\nDownside regime distribution:")
    downside_dist = pd.Series(best_downside_regimes).value_counts().sort_index()
    for regime, count in downside_dist.items():
        print(f"  Regime {regime}: {count} models")
    
    # Show some examples
    print(f"\nSample regime assignments:")
    sample_df = df.sample(min(10, len(df)))
    for _, row in sample_df.iterrows():
        print(f"  Model {row['model_id']}: upside regime {row['model_regime_upside']} "
              f"(score {row['model_regime_upside_score']:.4f}), "
              f"downside regime {row['model_regime_downside']} "
              f"(score {row['model_regime_downside_score']:.4f})")
    
    return True

if __name__ == "__main__":
    success = add_regime_base_columns()
    if success:
        print("\nRegime base columns added successfully!")
    else:
        print("\nFailed to add regime base columns.")
    exit(0 if success else 1)
