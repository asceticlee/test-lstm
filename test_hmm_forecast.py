#!/usr/bin/env python3
"""
Test script for the enhanced HMM forecaster
"""

import sys
import os
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir / 'src'
sys.path.insert(0, str(src_dir))

def test_daily_mode():
    """Test daily forecasting mode"""
    print("="*60)
    print("TESTING DAILY MODE")
    print("="*60)
    
    # Import after path setup
    from market_regime_forecast.market_regime_hmm_forecast import HMMRegimeForecaster
    
    # Initialize forecaster in daily mode
    forecaster = HMMRegimeForecaster(
        mode='daily',
        n_components=3,  # Use fewer components for testing
        n_features=10    # Use fewer features for testing
    )
    
    # Test data paths
    data_file = script_dir / 'data' / 'history_spot_quote.csv'
    regime_file = script_dir / 'regime_analysis' / 'regime_assignments.csv'
    
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        return False
    
    if not regime_file.exists():
        print(f"ERROR: Regime file not found: {regime_file}")
        return False
    
    try:
        # Load data
        print("Loading data...")
        market_data, regime_data = forecaster.load_and_prepare_data(
            str(data_file), str(regime_file)
        )
        
        # Extract features
        print("Extracting features...")
        daily_features = forecaster.extract_daily_features(market_data)
        
        print(f"✓ Daily mode test successful!")
        print(f"  Market data shape: {market_data.shape}")
        print(f"  Regime data shape: {regime_data.shape}")
        print(f"  Daily features shape: {daily_features.shape}")
        
        # Show sample features including overnight gap
        feature_cols = [col for col in daily_features.columns if col not in ['trading_day', 'reference_time_ms', 'reference_price', 'num_observations']]
        print(f"  Total features extracted: {len(feature_cols)}")
        
        # Check for overnight gap features
        gap_features = [col for col in feature_cols if 'overnight_gap' in col]
        print(f"  Overnight gap features: {gap_features}")
        
        if len(gap_features) > 0:
            print(f"  ✓ Overnight gap features successfully added!")
        
        return True
        
    except Exception as e:
        print(f"✗ Daily mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_intraday_mode():
    """Test intraday forecasting mode"""
    print("="*60)
    print("TESTING INTRADAY MODE")
    print("="*60)
    
    # Import after path setup
    from market_regime_forecast.market_regime_hmm_forecast import HMMRegimeForecaster
    
    # Initialize forecaster in intraday mode
    forecaster = HMMRegimeForecaster(
        mode='intraday',
        n_components=3,  # Use fewer components for testing
        n_features=10    # Use fewer features for testing
    )
    
    # Test data paths
    data_file = script_dir / 'data' / 'history_spot_quote.csv'
    regime_file = script_dir / 'regime_analysis' / 'regime_assignments.csv'
    
    try:
        # Load data
        print("Loading data...")
        market_data, regime_data = forecaster.load_and_prepare_data(
            str(data_file), str(regime_file)
        )
        
        # Extract features
        print("Extracting features...")
        daily_features = forecaster.extract_daily_features(market_data)
        
        print(f"✓ Intraday mode test successful!")
        print(f"  Market data shape: {market_data.shape}")
        print(f"  Regime data shape: {regime_data.shape}")
        print(f"  Daily features shape: {daily_features.shape}")
        
        # Show sample features
        feature_cols = [col for col in daily_features.columns if col not in ['trading_day', 'reference_time_ms', 'reference_price', 'num_observations']]
        print(f"  Total features extracted: {len(feature_cols)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Intraday mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_statistical_features():
    """Test the enhanced statistical features module"""
    print("="*60)
    print("TESTING STATISTICAL FEATURES")
    print("="*60)
    
    try:
        from market_data_stat.statistical_features import StatisticalFeatureExtractor
        import pandas as pd
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        n_days = 3
        n_points_per_day = 100
        
        sample_data = []
        for day in range(20200102, 20200102 + n_days):
            for i in range(n_points_per_day):
                sample_data.append({
                    'trading_day': day,
                    'ms_of_day': 34200000 + i * 60000,  # Start at 9:30 AM, increment by 1 minute
                    'mid': 100 + np.random.randn() * 0.1 + (day - 20200102) * 0.5  # Slight upward trend
                })
        
        sample_df = pd.DataFrame(sample_data)
        
        # Initialize feature extractor
        extractor = StatisticalFeatureExtractor()
        
        # Extract features
        features = extractor.extract_daily_features(
            daily_data=sample_df,
            price_column='mid',
            volume_column=None,
            reference_time_ms=38100000,  # 10:35 AM
            trading_day_column='trading_day',
            time_column='ms_of_day',
            use_relative=True,
            include_overnight_gap=True
        )
        
        print(f"✓ Statistical features test successful!")
        print(f"  Sample data shape: {sample_df.shape}")
        print(f"  Features shape: {features.shape}")
        
        # Check for overnight gap features
        gap_features = [col for col in features.columns if 'overnight_gap' in col]
        print(f"  Overnight gap features: {gap_features}")
        
        if len(gap_features) > 0:
            print(f"  ✓ Overnight gap calculation working!")
            # Show gap values for verification
            for col in gap_features:
                print(f"    {col}: {features[col].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Statistical features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("HMM FORECASTER ENHANCED TESTING")
    print("="*60)
    
    tests = [
        ("Statistical Features", test_statistical_features),
        ("Daily Mode", test_daily_mode),
        ("Intraday Mode", test_intraday_mode)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
        print()
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:20} {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
