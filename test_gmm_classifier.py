#!/usr/bin/env python3
"""
Test script to demonstrate the new GMM-based regime classification
"""

import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent.absolute()
src_dir = script_dir / 'src'
sys.path.insert(0, str(src_dir))

from market_regime_classifier import MarketRegimeClassifier
import pandas as pd
import numpy as np

def test_gmm_classifier():
    """Test the new GMM-based market regime classifier"""
    print("Testing GMM-based Market Regime Classifier")
    print("=" * 50)
    
    # Check if data file exists
    data_path = Path('data/trainingData.csv')
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please ensure the trading data file exists.")
        return False
    
    try:
        # Initialize classifier
        classifier = MarketRegimeClassifier(
            data_path=str(data_path),
            output_dir='regime_analysis_test'
        )
        
        # Load data
        classifier.load_data()
        print(f"✓ Data loaded successfully: {len(classifier.raw_data):,} rows")
        
        # Engineer features
        classifier.engineer_weekly_features()
        print(f"✓ Features engineered: {len(classifier.weekly_features)} weeks")
        
        # Test optimal cluster determination
        n_regimes, X_pca = classifier.determine_optimal_clusters(max_clusters=6)
        print(f"✓ Optimal number of regimes determined: {n_regimes}")
        
        # Classify regimes using GMM
        classifier.classify_regimes(n_regimes=n_regimes)
        print(f"✓ Regimes classified using GMM")
        
        # Check regime assignments
        unique_regimes = classifier.weekly_features['Regime'].unique()
        print(f"✓ Found {len(unique_regimes)} unique regimes: {sorted(unique_regimes)}")
        
        # Check regime probabilities
        avg_prob = classifier.weekly_features['Regime_Probability'].mean()
        print(f"✓ Average assignment probability: {avg_prob:.3f}")
        
        # Test prediction on new data
        sample_features = classifier.weekly_features.iloc[:1].copy()
        predicted_regime, probabilities = classifier.predict_regime(sample_features)
        print(f"✓ Prediction test successful: Regime {predicted_regime}, Max prob: {np.max(probabilities):.3f}")
        
        print("\n🎉 All tests passed! GMM-based regime classification is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gmm_classifier()
    if success:
        print("\nThe new GMM-based market regime classifier is ready to use!")
    else:
        print("\nPlease check the error messages above.")
