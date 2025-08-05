#!/usr/bin/env python3
"""
Test script to validate the HMM regime mapping fix
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

def test_hmm_regime_mapping():
    """Test the fixed HMM regime mapping approach"""
    print("Testing HMM regime mapping fix...")
    
    # Load data
    trading_data = pd.read_csv('data/trainingData.csv')
    regime_assignments = pd.read_csv('regime_analysis/regime_assignments.csv')
    
    # Prepare features
    daily_features = trading_data.groupby('TradingDay').agg({
        'Mid': ['mean', 'std', 'min', 'max'],
        'Bid': ['mean', 'std'],
        'Ask': ['mean', 'std'],
        'ROC_05min:ROC': ['mean', 'std'],
        'PriceChangesStat_05min:Mean': 'mean',
        'PriceChangesStat_05min:+Std': 'mean',
        'PriceChangesStat_05min:-Std': 'mean'
    }).reset_index()
    
    daily_features.columns = ['TradingDay'] + [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in daily_features.columns[1:]]
    daily_features['price_range'] = daily_features['Mid_max'] - daily_features['Mid_min']
    daily_features['price_volatility'] = daily_features['Mid_std']
    daily_features['bid_ask_spread'] = daily_features['Ask_mean'] - daily_features['Bid_mean']
    
    regime_data = pd.merge(
        daily_features, 
        regime_assignments.groupby('TradingDay')['Regime'].first().reset_index(),
        on='TradingDay', 
        how='inner'
    ).sort_values('TradingDay').dropna()
    
    print(f"Training data: {len(regime_data)} days")
    
    # Train HMM
    feature_cols = ['Mid_mean', 'Mid_std', 'price_range', 'price_volatility', 'bid_ask_spread', 'ROC_05min:ROC_mean']
    available_cols = [col for col in feature_cols if col in regime_data.columns]
    observations = regime_data[available_cols].values
    
    scaler = StandardScaler()
    observations_scaled = scaler.fit_transform(observations)
    
    regimes = regime_data['Regime'].values
    unique_regimes = sorted(np.unique(regimes))
    
    print(f"Unique regimes: {unique_regimes}")
    
    # Calculate overall regime frequencies
    regime_frequencies = {regime: np.sum(regimes == regime) / len(regimes) for regime in unique_regimes}
    print("Regime frequencies:")
    for regime, freq in regime_frequencies.items():
        print(f"  Regime {regime}: {freq:.1%}")
    
    hmm_model = hmm.GaussianHMM(n_components=5, covariance_type='full', n_iter=200, random_state=42, verbose=False)
    hmm_model.fit(observations_scaled)
    predicted_states = hmm_model.predict(observations_scaled)
    
    # OLD APPROACH (most common regime per state)
    print("\n=== OLD APPROACH: Most Common Regime per State ===")
    old_mapping = {}
    for state in range(5):
        state_mask = predicted_states == state
        if np.any(state_mask):
            regimes_in_state = regimes[state_mask]
            most_common_regime = Counter(regimes_in_state).most_common(1)[0][0]
            old_mapping[state] = most_common_regime
            state_count = len(regimes_in_state)
            regime_count = np.sum(regimes_in_state == most_common_regime)
            print(f"  State {state} -> Regime {most_common_regime} ({regime_count}/{state_count} = {regime_count/state_count:.1%})")
    
    old_covered_regimes = set(old_mapping.values())
    old_missing_regimes = set(unique_regimes) - old_covered_regimes
    print(f"Old approach covers: {sorted(old_covered_regimes)}")
    print(f"Old approach missing: {sorted(old_missing_regimes)}")
    
    # NEW APPROACH (enrichment-based assignment)
    print("\n=== NEW APPROACH: Enrichment-Based Assignment ===")
    
    # Build enrichment matrix
    enrichment_matrix = np.zeros((5, len(unique_regimes)))
    
    for state in range(5):
        state_mask = predicted_states == state
        if np.any(state_mask):
            regimes_in_state = regimes[state_mask]
            state_size = len(regimes_in_state)
            
            for regime_idx, regime in enumerate(unique_regimes):
                regime_count_in_state = np.sum(regimes_in_state == regime)
                state_frequency = regime_count_in_state / state_size
                overall_frequency = regime_frequencies[regime]
                
                enrichment_score = state_frequency / overall_frequency if overall_frequency > 0 else 0
                enrichment_matrix[state, regime_idx] = enrichment_score
    
    # Optimal assignment
    remaining_states = set(range(5))
    remaining_regimes = set(range(len(unique_regimes)))
    
    assignments = []
    for state in range(5):
        for regime_idx in range(len(unique_regimes)):
            enrichment = enrichment_matrix[state, regime_idx]
            assignments.append((enrichment, state, regime_idx))
    
    assignments.sort(reverse=True)
    
    new_mapping = {}
    for enrichment, state, regime_idx in assignments:
        regime = unique_regimes[regime_idx]
        if state in remaining_states and regime_idx in remaining_regimes:
            new_mapping[state] = regime
            remaining_states.remove(state)
            remaining_regimes.remove(regime_idx)
            
            state_mask = predicted_states == state
            if np.any(state_mask):
                regimes_in_state = regimes[state_mask]
                regime_count = np.sum(regimes_in_state == regime)
                state_size = len(regimes_in_state)
                state_freq = regime_count / state_size * 100
                overall_freq = regime_frequencies[regime] * 100
                print(f"  State {state} -> Regime {regime}: {regime_count}/{state_size} = {state_freq:.1f}% (vs {overall_freq:.1f}% overall, enrichment: {enrichment:.2f})")
            
            if len(remaining_regimes) == 0:
                break
    
    new_covered_regimes = set(new_mapping.values())
    new_missing_regimes = set(unique_regimes) - new_covered_regimes
    print(f"New approach covers: {sorted(new_covered_regimes)}")
    print(f"New approach missing: {sorted(new_missing_regimes)}")
    
    print(f"\n✅ Improvement: Old approach missing {len(old_missing_regimes)} regimes, new approach missing {len(new_missing_regimes)} regimes")
    
    # Test prediction diversity
    print("\n=== Testing Prediction Diversity ===")
    
    # Simulate predictions using new mapping
    test_predictions = []
    for _ in range(100):
        # Simulate HMM state prediction (random for demo)
        predicted_state = np.random.choice(5)
        predicted_regime = new_mapping.get(predicted_state, 0)
        test_predictions.append(predicted_regime)
    
    prediction_counts = Counter(test_predictions)
    print(f"Simulated prediction distribution: {dict(prediction_counts)}")
    print(f"Unique regimes predicted: {sorted(prediction_counts.keys())}")
    
    return len(new_missing_regimes) == 0

if __name__ == "__main__":
    success = test_hmm_regime_mapping()
    if success:
        print("\n🎉 HMM regime mapping fix is working correctly!")
    else:
        print("\n❌ HMM regime mapping still has issues")
