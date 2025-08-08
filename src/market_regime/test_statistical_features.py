#!/usr/bin/env python3
"""
Example usage of the Statistical Features Module

This script demonstrates how to use the statistical_features module
for extracting comprehensive market features from price data.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))

from market_data_stat.statistical_features import StatisticalFeatureExtractor, calculate_technical_indicators

def example_single_series():
    """Example: Extract features from a single price series"""
    print("="*60)
    print("EXAMPLE 1: Single Price Series Feature Extraction")
    print("="*60)
    
    # Generate sample price data (100 time points)
    np.random.seed(42)
    base_price = 450.0
    returns = np.random.normal(0, 0.001, 100)
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    reference_price = prices[0]  # Use first price as reference
    
    # Generate sample volume data
    volumes = np.random.lognormal(10, 0.3, len(prices))
    
    print(f"Sample data: {len(prices)} price points")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"Reference price: ${reference_price:.2f}")
    
    # Method 1: Use convenience function
    features_conv = calculate_technical_indicators(prices, volumes, reference_price, use_relative=True)
    
    # Method 2: Use feature extractor class
    extractor = StatisticalFeatureExtractor()
    features_class = extractor.calculate_all_features(prices, volumes, reference_price, use_relative=True)
    
    print(f"\nFeatures extracted: {len(features_conv)}")
    print("\nSample features (relative to reference price):")
    
    # Display some key features
    key_features = [
        'rel_price_mean', 'rel_price_std', 'rel_price_range', 
        'return_mean', 'return_std', 'realized_volatility',
        'rel_trend_slope', 'rel_momentum_final', 'num_peaks', 'autocorr_lag1'
    ]
    
    for feature in key_features:
        if feature in features_conv:
            print(f"  {feature:20}: {features_conv[feature]:8.4f}")

def example_daily_data():
    """Example: Extract features from daily trading data"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Daily Trading Data Feature Extraction")
    print("="*60)
    
    # Create sample daily trading data
    np.random.seed(123)
    
    trading_days = ['20250101', '20250102', '20250103']
    all_data = []
    
    for day in trading_days:
        # Generate intraday data for each day (10:35 AM to 12:00 PM)
        start_ms = 38100000  # 10:35 AM
        end_ms = 43200000    # 12:00 PM
        
        # Create time points every minute
        time_points = list(range(start_ms, end_ms + 1, 60000))  # Every minute
        
        # Generate price walk for the day
        base_price = 445 + np.random.normal(0, 5)  # Random daily opening
        returns = np.random.normal(0, 0.0005, len(time_points))
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate volumes
        volumes = np.random.lognormal(9, 0.4, len(time_points))
        
        # Create DataFrame for this day
        day_data = pd.DataFrame({
            'TradingDay': day,
            'TradingMsOfDay': time_points,
            'Mid': prices,
            'Volume': volumes
        })
        
        all_data.append(day_data)
    
    # Combine all days
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"Sample data: {len(combined_data)} total observations")
    print(f"Trading days: {len(trading_days)}")
    print(f"Time range: 10:35 AM to 12:00 PM")
    
    # Extract daily features using the module
    extractor = StatisticalFeatureExtractor()
    daily_features = extractor.extract_daily_features(
        daily_data=combined_data,
        price_column='Mid',
        volume_column='Volume',
        reference_time_ms=38100000,  # 10:35 AM
        use_relative=True
    )
    
    print(f"\nDaily features extracted:")
    print(f"  Days processed: {len(daily_features)}")
    print(f"  Features per day: {len([col for col in daily_features.columns if col not in ['TradingDay', 'reference_time_ms', 'reference_price', 'num_observations']])}")
    
    # Show summary statistics for each day
    print("\nDaily feature summary:")
    for _, day in daily_features.iterrows():
        trading_day = day['TradingDay']
        ref_price = day['reference_price']
        rel_range = day['rel_price_range']
        trend_slope = day['rel_trend_slope']
        volatility = day['realized_volatility']
        
        print(f"  {trading_day}: Ref=${ref_price:.2f}, Range={rel_range:.3f}%, Trend={trend_slope:.4f}, Vol={volatility:.4f}")

def example_feature_comparison():
    """Example: Compare relative vs absolute features"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Relative vs Absolute Feature Comparison")
    print("="*60)
    
    # Create two price series at different levels but similar patterns
    np.random.seed(456)
    
    # Series 1: Lower price level
    base_price_1 = 200.0
    pattern = np.sin(np.linspace(0, 4*np.pi, 50)) * 0.02 + np.random.normal(0, 0.005, 50)
    prices_1 = base_price_1 * (1 + np.cumsum(pattern))
    
    # Series 2: Higher price level, same pattern
    base_price_2 = 800.0
    prices_2 = base_price_2 * (1 + np.cumsum(pattern))
    
    extractor = StatisticalFeatureExtractor()
    
    # Extract absolute features (no reference price)
    features_1_abs = extractor.calculate_all_features(prices_1, use_relative=False)
    features_2_abs = extractor.calculate_all_features(prices_2, use_relative=False)
    
    # Extract relative features (using first price as reference)
    features_1_rel = extractor.calculate_all_features(prices_1, reference_price=prices_1[0], use_relative=True)
    features_2_rel = extractor.calculate_all_features(prices_2, reference_price=prices_2[0], use_relative=True)
    
    print(f"Series 1 price range: ${prices_1.min():.2f} - ${prices_1.max():.2f}")
    print(f"Series 2 price range: ${prices_2.min():.2f} - ${prices_2.max():.2f}")
    
    print("\nAbsolute features (should be very different):")
    abs_features = ['price_mean', 'price_std', 'price_range']
    for feature in abs_features:
        if feature in features_1_abs and feature in features_2_abs:
            print(f"  {feature:15}: Series1={features_1_abs[feature]:8.2f}, Series2={features_2_abs[feature]:8.2f}")
    
    print("\nRelative features (should be similar):")
    rel_features = ['rel_price_mean', 'rel_price_std', 'rel_price_range']
    for feature in rel_features:
        if feature in features_1_rel and feature in features_2_rel:
            print(f"  {feature:15}: Series1={features_1_rel[feature]:8.4f}, Series2={features_2_rel[feature]:8.4f}")
    
    print("\nReturn-based features (should be similar):")
    return_features = ['return_mean', 'return_std', 'realized_volatility']
    for feature in return_features:
        if feature in features_1_rel and feature in features_2_rel:
            print(f"  {feature:15}: Series1={features_1_rel[feature]:8.6f}, Series2={features_2_rel[feature]:8.6f}")

def main():
    """Run all examples"""
    print("Statistical Features Module - Usage Examples")
    print("This demonstrates how other scripts can use the statistical_features module")
    
    # Run examples
    example_single_series()
    example_daily_data()
    example_feature_comparison()
    
    print("\n" + "="*60)
    print("USAGE SUMMARY")
    print("="*60)
    print("1. Import: from market_data_stat.statistical_features import StatisticalFeatureExtractor")
    print("2. Create: extractor = StatisticalFeatureExtractor()")
    print("3. Extract: features = extractor.calculate_all_features(prices, volumes, ref_price)")
    print("4. Or use: features = calculate_technical_indicators(prices, volumes, ref_price)")
    print("5. For daily data: extractor.extract_daily_features(df, price_col, volume_col)")
    print("\nKey benefits:")
    print("- Consistent feature extraction across all scripts")
    print("- Relative price calculations remove absolute price bias")
    print("- Comprehensive statistical and technical indicators")
    print("- Easy integration with pandas DataFrames")

if __name__ == "__main__":
    main()
