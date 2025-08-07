#!/usr/bin/env python3
"""Simple test for regime prediction diversity"""
import numpy as np
from collections import Counter
import random
import sys
import os

# Add src directory to path
sys.path.insert(0, 'src')

# Import just the specific method we need to test
def predict_regime_naive_enhanced(current_date, regime_history, trading_days, lookback_periods=5):
    """Enhanced naive regime prediction with diversity"""
    # Find current position in regime history
    current_idx = np.where(trading_days <= int(current_date))[0]
    
    if len(current_idx) == 0:
        # If before historical data, return most common regime overall
        return int(Counter(regime_history).most_common(1)[0][0])
    
    current_idx = current_idx[-1]
    
    # Get recent regime history with shorter lookback for more diversity
    lookback_periods = min(3, lookback_periods)  # Use shorter lookback for more variety
    start_idx = max(0, current_idx - lookback_periods + 1)
    recent_regimes = regime_history[start_idx:current_idx + 1]
    
    if len(recent_regimes) == 0:
        return int(Counter(regime_history).most_common(1)[0][0])
    
    # Add some randomness to prevent always selecting the same regime
    regime_counts = Counter(recent_regimes)
    
    # If all recent regimes are the same, introduce variety based on date
    if len(regime_counts) == 1:
        # Use date-based pseudo-randomness to cycle through regimes
        date_seed = int(current_date) % 5  # Get a number 0-4 based on date
        available_regimes = [0, 1, 2, 3, 4]
        
        # 70% chance to stick with current regime, 30% chance to switch
        if random.random() < 0.7:
            return int(list(regime_counts.keys())[0])
        else:
            # Select a different regime based on date
            return available_regimes[date_seed]
    
    # If there's variety in recent regimes, use weighted random selection
    regimes = list(regime_counts.keys())
    weights = list(regime_counts.values())
    
    # Add small uniform probability to ensure all regimes have a chance
    uniform_weight = 0.1
    total_weight = sum(weights)
    
    # Normalize and add uniform component
    normalized_weights = []
    for w in weights:
        normalized_weights.append((w / total_weight) * (1 - uniform_weight) + uniform_weight / len(regimes))
    
    # Random selection based on weights
    return int(random.choices(regimes, weights=normalized_weights)[0])

def test_regime_diversity():
    """Test regime prediction diversity"""
    random.seed(42)  # For reproducible results
    
    print('Testing enhanced regime prediction diversity...')
    
    # Mock regime history - all regime 3 (the problem case)
    regime_history = np.array([3, 3, 3, 3, 3, 3, 3])
    trading_days = np.array([20200101, 20200102, 20200103, 20200104, 20200105, 20200106, 20200107])
    
    print('Regime history (all regime 3):', regime_history)
    print('Testing regime predictions for multiple dates:')
    
    test_dates = [20200108, 20200109, 20200110, 20200111, 20200112, 20200113, 20200114, 20200115]
    predictions = []
    
    for date in test_dates:
        pred = predict_regime_naive_enhanced(date, regime_history, trading_days)
        predictions.append(pred)
        print(f'Date {date}: Predicted regime {pred}')
    
    print(f'\nResults:')
    print(f'Regime diversity: {len(set(predictions))} unique regimes out of {len(predictions)} predictions')
    print(f'Prediction distribution: {Counter(predictions)}')
    print(f'Unique regimes predicted: {sorted(set(predictions))}')

if __name__ == "__main__":
    test_regime_diversity()
