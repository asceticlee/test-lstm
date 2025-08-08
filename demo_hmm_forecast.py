#!/usr/bin/env python3
"""
HMM Market Regime Forecasting Demo

This script demonstrates the two forecasting modes of the enhanced HMM forecaster:
1. Daily Mode: Use complete daily data to predict next day's regime
2. Intraday Mode: Use data before 10:35 AM to predict 10:36-12:00 regime

Usage:
    python demo_hmm_forecast.py [--mode daily|intraday] [--full_training]
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir / 'src'
sys.path.insert(0, str(src_dir))

from market_regime_forecast.market_regime_hmm_forecast import HMMRegimeForecaster

def demonstrate_mode(mode, full_training=False):
    """
    Demonstrate a specific forecasting mode
    
    Args:
        mode: 'daily' or 'intraday'
        full_training: If True, run full training. If False, just demonstrate data loading and feature extraction.
    """
    print("="*80)
    print(f"HMM REGIME FORECASTING DEMONSTRATION - {mode.upper()} MODE")
    print("="*80)
    
    # Initialize forecaster
    forecaster = HMMRegimeForecaster(
        mode=mode,
        n_components=4,  # Use reasonable number of components
        n_features=15,   # Use reasonable number of features
        random_state=42
    )
    
    # Data files
    data_file = script_dir / 'data' / 'history_spot_quote.csv'
    regime_file = script_dir / 'regime_analysis' / 'regime_assignments.csv'
    
    print(f"Data source: {data_file}")
    print(f"Regime assignments: {regime_file}")
    print()
    
    # Load and prepare data
    print("Step 1: Loading market data and regime assignments...")
    market_data, regime_data = forecaster.load_and_prepare_data(str(data_file), str(regime_file))
    print()
    
    # Extract features
    print("Step 2: Extracting statistical features...")
    daily_features = forecaster.extract_daily_features(market_data)
    
    # Show feature summary
    feature_cols = [col for col in daily_features.columns if col not in ['trading_day', 'reference_time_ms', 'reference_price', 'num_observations']]
    print(f"Features extracted per day: {len(feature_cols)}")
    
    # Show overnight gap features specifically
    gap_features = [col for col in feature_cols if 'overnight_gap' in col]
    print(f"Overnight gap features: {gap_features}")
    
    # Show some sample feature values
    if len(daily_features) > 0:
        print(f"\\nSample data (first 3 days):")
        sample_data = daily_features.head(3)
        for _, row in sample_data.iterrows():
            print(f"  Day {int(row['trading_day'])}: {row['num_observations']} observations")
            for gap_feat in gap_features:
                if gap_feat in row:
                    print(f"    {gap_feat}: {row[gap_feat]:.4f}")
    print()
    
    if full_training:
        print("Step 3: Preparing training data...")
        
        # Use first 80% for training, last 20% for testing
        all_days = sorted(daily_features['trading_day'].unique())
        split_idx = int(len(all_days) * 0.8)
        train_end = all_days[split_idx]
        test_start = all_days[split_idx + 1] if split_idx + 1 < len(all_days) else all_days[-1]
        
        print(f"Training period: {all_days[0]} to {train_end}")
        print(f"Testing period: {test_start} to {all_days[-1]}")
        
        # Prepare training data
        X_train, y_train, feature_names, train_days = forecaster.prepare_training_data(
            daily_features, regime_data,
            train_end=train_end
        )
        
        print("Step 4: Training HMM model...")
        training_results = forecaster.train_hmm_model(X_train, y_train, train_days)
        
        print("Step 5: Evaluating on test data...")
        X_test, y_test, _, test_days = forecaster.prepare_training_data(
            daily_features, regime_data,
            train_start=test_start
        )
        
        test_results = forecaster.evaluate_model(X_test, y_test, test_days)
        
        print(f"\\n{mode.upper()} MODE RESULTS:")
        print(f"Training accuracy: {training_results.get('train_accuracy', 'N/A'):.3f}")
        print(f"Test accuracy: {test_results.get('accuracy', 'N/A'):.3f}")
        print(f"Selected features: {training_results.get('selected_features', [])[5]}")
        
        # Show regime transition matrix
        if hasattr(forecaster.hmm_model, 'transmat_'):
            print(f"\\nTransition Matrix:")
            trans_mat = forecaster.hmm_model.transmat_
            for i in range(len(trans_mat)):
                row_str = " ".join([f"{trans_mat[i][j]:.3f}" for j in range(len(trans_mat[i]))])
                print(f"  Regime {i}: [{row_str}]")
    
    else:
        print("Step 3: Skipping training (use --full_training to run complete demo)")
        print(f"\\n{mode.upper()} MODE SUMMARY:")
        print(f"  ✓ Successfully loaded {len(market_data):,} market data points")
        print(f"  ✓ Successfully loaded {len(regime_data):,} regime assignments") 
        print(f"  ✓ Successfully extracted features for {len(daily_features)} trading days")
        print(f"  ✓ Generated {len(feature_cols)} features per day including overnight gaps")
    
    print("="*80)

def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description='HMM Market Regime Forecasting Demo')
    parser.add_argument('--mode', type=str, choices=['daily', 'intraday', 'both'], 
                       default='both', help='Demonstration mode')
    parser.add_argument('--full_training', action='store_true',
                       help='Run full training and evaluation (takes longer)')
    
    args = parser.parse_args()
    
    print("HMM MARKET REGIME FORECASTING DEMONSTRATION")
    print("="*80)
    print("This demo showcases the enhanced HMM forecasting system with two modes:")
    print()
    print("1. DAILY MODE:")
    print("   - Uses complete daily history_spot_quote.csv data")
    print("   - Calculates comprehensive statistical features") 
    print("   - Includes overnight gap (prior day last → current day first price)")
    print("   - Predicts next trading day's market regime")
    print()
    print("2. INTRADAY MODE:")
    print("   - Uses data before 10:35 AM only")
    print("   - Calculates same statistical features on limited timeframe")
    print("   - Includes overnight gap feature")
    print("   - Predicts market regime for 10:36 AM - 12:00 PM window")
    print()
    
    if args.mode in ['daily', 'both']:
        demonstrate_mode('daily', args.full_training)
        
    if args.mode in ['intraday', 'both']:
        demonstrate_mode('intraday', args.full_training)
    
    print()
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("Key Features Demonstrated:")
    print("✓ Two distinct forecasting modes (daily vs intraday)")
    print("✓ Enhanced statistical features with overnight gap calculation")
    print("✓ history_spot_quote.csv data format support")
    print("✓ Mode-specific data filtering and feature extraction")
    print("✓ HMM-based regime transition modeling")
    if args.full_training:
        print("✓ Complete training and evaluation pipeline")
    print()
    print("To run with full training: python demo_hmm_forecast.py --full_training")
    print("To test specific mode: python demo_hmm_forecast.py --mode daily --full_training")

if __name__ == "__main__":
    main()
