#!/usr/bin/env python3
"""
Statistical Features Module

This module provides comprehensive statistical feature extraction functions
for financial time series analysis, particularly for intraday market data.

Features include:
- Price-based statistics (relative and absolute)
- Volatility measures
- Momentum indicators  
- Trend analysis
- Peak/trough detection
- Autocorrelation analysis
- Jump detection
- Volume analysis

Usage:
    from statistical_features import StatisticalFeatureExtractor
    
    extractor = StatisticalFeatureExtractor()
    features = extractor.calculate_all_features(prices, volumes, reference_price)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class StatisticalFeatureExtractor:
    """
    Comprehensive statistical feature extraction for financial time series
    """
    
    def __init__(self, rolling_window=5, momentum_short_window=5, momentum_long_window=10):
        """
        Initialize the feature extractor
        
        Args:
            rolling_window: Window size for rolling statistics
            momentum_short_window: Short period for momentum calculation
            momentum_long_window: Long period for momentum calculation
        """
        self.rolling_window = rolling_window
        self.momentum_short_window = momentum_short_window
        self.momentum_long_window = momentum_long_window
    
    def calculate_price_features(self, prices, reference_price=None, use_relative=True):
        """
        Calculate basic price-level features
        
        Args:
            prices: Array of price values
            reference_price: Reference price for relative calculations
            use_relative: Whether to use relative prices (percentage from reference)
            
        Returns:
            dict: Price feature dictionary
        """
        prices = np.array(prices)
        n = len(prices)
        
        if n < 1:
            return {}
        
        # Convert to relative prices if requested and reference provided
        if use_relative and reference_price is not None and reference_price > 0:
            working_prices = (prices / reference_price - 1) * 100  # Percentage change
            prefix = 'rel_'
        else:
            working_prices = prices
            prefix = ''
        
        features = {
            f'{prefix}price_mean': np.mean(working_prices),
            f'{prefix}price_std': np.std(working_prices),
            f'{prefix}price_min': np.min(working_prices),
            f'{prefix}price_max': np.max(working_prices),
            f'{prefix}price_range': np.max(working_prices) - np.min(working_prices),
        }
        
        # Add range percentage only for relative prices (already in percentage)
        if use_relative and reference_price is not None:
            features[f'{prefix}price_range_pct'] = features[f'{prefix}price_range']
        elif not use_relative:
            features[f'{prefix}price_range_pct'] = (features[f'{prefix}price_range'] / np.mean(working_prices)) * 100 if np.mean(working_prices) != 0 else 0
        
        return features
    
    def calculate_return_features(self, prices):
        """
        Calculate return-based features (always use actual prices for returns)
        
        Args:
            prices: Array of price values
            
        Returns:
            dict: Return feature dictionary
        """
        prices = np.array(prices)
        n = len(prices)
        
        if n < 2:
            return {
                'return_mean': 0,
                'return_std': 0,
                'return_skewness': 0,
                'return_kurtosis': 0,
            }
        
        # Calculate returns using actual prices
        returns = np.diff(prices) / prices[:-1]
        
        features = {
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'return_skewness': stats.skew(returns) if len(returns) > 2 else 0,
            'return_kurtosis': stats.kurtosis(returns) if len(returns) > 3 else 0,
        }
        
        return features
    
    def calculate_volatility_features(self, prices, reference_price=None, use_relative=True):
        """
        Calculate volatility-based features
        
        Args:
            prices: Array of price values
            reference_price: Reference price for relative calculations
            use_relative: Whether to use relative prices
            
        Returns:
            dict: Volatility feature dictionary
        """
        prices = np.array(prices)
        n = len(prices)
        
        if n < 2:
            return {}
        
        # Get returns for realized volatility (always use actual returns)
        returns = np.diff(prices) / prices[:-1]
        
        # Get working prices for other volatility measures
        if use_relative and reference_price is not None and reference_price > 0:
            working_prices = (prices / reference_price - 1) * 100
            prefix = 'rel_'
        else:
            working_prices = prices
            prefix = ''
        
        price_changes = np.diff(working_prices)
        
        features = {
            'realized_volatility': np.sqrt(np.sum(returns**2)) if len(returns) > 0 else 0,
            f'{prefix}price_change_std': np.std(price_changes) if len(price_changes) > 0 else 0,
            f'max_{prefix}price_change': np.max(np.abs(price_changes)) if len(price_changes) > 0 else 0,
        }
        
        return features
    
    def calculate_rolling_features(self, prices, reference_price=None, use_relative=True):
        """
        Calculate rolling window statistics
        
        Args:
            prices: Array of price values
            reference_price: Reference price for relative calculations
            use_relative: Whether to use relative prices
            
        Returns:
            dict: Rolling feature dictionary
        """
        prices = np.array(prices)
        n = len(prices)
        
        if n < self.rolling_window:
            return {}
        
        # Get working prices
        if use_relative and reference_price is not None and reference_price > 0:
            working_prices = (prices / reference_price - 1) * 100
            prefix = 'rel_'
        else:
            working_prices = prices
            prefix = ''
        
        rolling_std = pd.Series(working_prices).rolling(
            window=self.rolling_window, min_periods=1
        ).std()
        
        features = {
            f'{prefix}rolling_vol_mean': np.mean(rolling_std),
            f'{prefix}rolling_vol_std': np.std(rolling_std),
            f'{prefix}rolling_vol_max': np.max(rolling_std),
        }
        
        return features
    
    def calculate_momentum_features(self, prices, reference_price=None, use_relative=True):
        """
        Calculate momentum indicators
        
        Args:
            prices: Array of price values
            reference_price: Reference price for relative calculations
            use_relative: Whether to use relative prices
            
        Returns:
            dict: Momentum feature dictionary
        """
        prices = np.array(prices)
        n = len(prices)
        
        if n < self.momentum_long_window:
            return {}
        
        # Get working prices
        if use_relative and reference_price is not None and reference_price > 0:
            working_prices = (prices / reference_price - 1) * 100
            prefix = 'rel_'
        else:
            working_prices = prices
            prefix = ''
        
        short_ma = pd.Series(working_prices).rolling(window=self.momentum_short_window).mean()
        long_ma = pd.Series(working_prices).rolling(window=self.momentum_long_window).mean()
        momentum = short_ma - long_ma
        
        features = {
            f'{prefix}momentum_mean': np.nanmean(momentum),
            f'{prefix}momentum_std': np.nanstd(momentum),
            f'{prefix}momentum_final': momentum.iloc[-1] if not pd.isna(momentum.iloc[-1]) else 0,
        }
        
        return features
    
    def calculate_trend_features(self, prices, reference_price=None, use_relative=True):
        """
        Calculate trend analysis features
        
        Args:
            prices: Array of price values
            reference_price: Reference price for relative calculations
            use_relative: Whether to use relative prices
            
        Returns:
            dict: Trend feature dictionary
        """
        prices = np.array(prices)
        n = len(prices)
        
        if n < 3:
            return {}
        
        # Get working prices
        if use_relative and reference_price is not None and reference_price > 0:
            working_prices = (prices / reference_price - 1) * 100
            prefix = 'rel_'
        else:
            working_prices = prices
            prefix = ''
        
        # Linear trend analysis
        x = np.arange(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, working_prices)
        
        # Direction consistency
        price_changes = np.diff(working_prices)
        price_directions = np.sign(price_changes)
        direction_changes = np.sum(np.diff(price_directions) != 0)
        
        features = {
            f'{prefix}trend_slope': slope,
            f'{prefix}trend_r_squared': r_value**2,
            f'{prefix}trend_p_value': p_value,
            f'{prefix}trend_strength': abs(slope) * (r_value**2),
            'direction_changes': direction_changes,
            'direction_consistency': 1 - (direction_changes / max(1, len(price_changes)-1)),
        }
        
        return features
    
    def calculate_peak_trough_features(self, prices, reference_price=None, use_relative=True, min_distance=2):
        """
        Calculate peak and trough detection features
        
        Args:
            prices: Array of price values
            reference_price: Reference price for relative calculations
            use_relative: Whether to use relative prices
            min_distance: Minimum distance between peaks/troughs
            
        Returns:
            dict: Peak/trough feature dictionary
        """
        prices = np.array(prices)
        n = len(prices)
        
        if n < 5:
            return {
                'num_peaks': 0,
                'num_troughs': 0,
                'peak_trough_ratio': 1,
                'reversal_frequency': 0,
            }
        
        # Get working prices
        if use_relative and reference_price is not None and reference_price > 0:
            working_prices = (prices / reference_price - 1) * 100
        else:
            working_prices = prices
        
        try:
            peaks, _ = find_peaks(working_prices, distance=min_distance)
            troughs, _ = find_peaks(-working_prices, distance=min_distance)
            
            features = {
                'num_peaks': len(peaks),
                'num_troughs': len(troughs),
                'peak_trough_ratio': len(peaks) / max(1, len(troughs)),
                'reversal_frequency': (len(peaks) + len(troughs)) / n * 100,
            }
        except:
            features = {
                'num_peaks': 0,
                'num_troughs': 0,
                'peak_trough_ratio': 1,
                'reversal_frequency': 0,
            }
        
        return features
    
    def calculate_autocorrelation_features(self, prices, max_lag=1):
        """
        Calculate autocorrelation features (always use actual returns)
        
        Args:
            prices: Array of price values
            max_lag: Maximum lag for autocorrelation
            
        Returns:
            dict: Autocorrelation feature dictionary
        """
        prices = np.array(prices)
        n = len(prices)
        
        if n < 10:
            return {'autocorr_lag1': 0}
        
        # Calculate returns using actual prices
        returns = np.diff(prices) / prices[:-1]
        
        try:
            if len(returns) > max_lag:
                autocorr_1 = np.corrcoef(returns[:-max_lag], returns[max_lag:])[0, 1]
                autocorr_1 = autocorr_1 if not np.isnan(autocorr_1) else 0
            else:
                autocorr_1 = 0
        except:
            autocorr_1 = 0
        
        return {'autocorr_lag1': autocorr_1}
    
    def calculate_jump_features(self, prices, threshold_multiplier=2):
        """
        Calculate jump detection features (always use actual returns)
        
        Args:
            prices: Array of price values
            threshold_multiplier: Multiplier for standard deviation threshold
            
        Returns:
            dict: Jump feature dictionary
        """
        prices = np.array(prices)
        n = len(prices)
        
        if n < 2:
            return {
                'num_jumps': 0,
                'jump_frequency': 0,
                'max_jump_size': 0,
            }
        
        # Calculate returns using actual prices
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) == 0:
            return {
                'num_jumps': 0,
                'jump_frequency': 0,
                'max_jump_size': 0,
            }
        
        return_threshold = threshold_multiplier * np.std(returns) if np.std(returns) > 0 else 0.001
        jumps = np.abs(returns) > return_threshold
        
        features = {
            'num_jumps': np.sum(jumps),
            'jump_frequency': np.sum(jumps) / len(returns) * 100,
            'max_jump_size': np.max(np.abs(returns)),
        }
        
        return features
    
    def calculate_volume_features(self, volumes, prices=None, reference_price=None, use_relative=True):
        """
        Calculate volume-based features
        
        Args:
            volumes: Array of volume values
            prices: Array of price values (for correlation)
            reference_price: Reference price for relative calculations
            use_relative: Whether to use relative prices for correlation
            
        Returns:
            dict: Volume feature dictionary
        """
        if volumes is None or len(volumes) == 0:
            return {}
        
        volumes = np.array(volumes)
        
        features = {
            'volume_mean': np.mean(volumes),
            'volume_std': np.std(volumes),
            'volume_skewness': stats.skew(volumes) if len(volumes) > 2 else 0,
        }
        
        # Price-volume correlation if prices provided
        if prices is not None and len(prices) == len(volumes) and len(prices) > 1:
            if use_relative and reference_price is not None and reference_price > 0:
                working_prices = (np.array(prices) / reference_price - 1) * 100
                corr_name = 'rel_price_volume_corr'
            else:
                working_prices = np.array(prices)
                corr_name = 'price_volume_corr'
            
            try:
                correlation = np.corrcoef(working_prices, volumes)[0, 1]
                features[corr_name] = correlation if not np.isnan(correlation) else 0
            except:
                features[corr_name] = 0
        
        return features
    
    def calculate_all_features(self, prices, volumes=None, reference_price=None, 
                             use_relative=True, include_volume=True):
        """
        Calculate all statistical features for a price series
        
        Args:
            prices: Array of price values
            volumes: Array of volume values (optional)
            reference_price: Reference price for relative calculations
            use_relative: Whether to use relative prices for most features
            include_volume: Whether to include volume features
            
        Returns:
            dict: Complete feature dictionary
        """
        prices = np.array(prices)
        n = len(prices)
        
        if n < 2:
            return {}
        
        # Combine all feature sets
        all_features = {}
        
        # Price features
        all_features.update(self.calculate_price_features(prices, reference_price, use_relative))
        
        # Return features (always use actual prices)
        all_features.update(self.calculate_return_features(prices))
        
        # Volatility features
        all_features.update(self.calculate_volatility_features(prices, reference_price, use_relative))
        
        # Rolling features
        if n >= self.rolling_window:
            all_features.update(self.calculate_rolling_features(prices, reference_price, use_relative))
        
        # Momentum features
        if n >= self.momentum_long_window:
            all_features.update(self.calculate_momentum_features(prices, reference_price, use_relative))
        
        # Trend features
        if n >= 3:
            all_features.update(self.calculate_trend_features(prices, reference_price, use_relative))
        
        # Peak/trough features
        if n >= 5:
            all_features.update(self.calculate_peak_trough_features(prices, reference_price, use_relative))
        
        # Autocorrelation features
        if n >= 10:
            all_features.update(self.calculate_autocorrelation_features(prices))
        
        # Jump features
        all_features.update(self.calculate_jump_features(prices))
        
        # Volume features
        if include_volume and volumes is not None:
            all_features.update(self.calculate_volume_features(volumes, prices, reference_price, use_relative))
        
        return all_features
    
    def extract_daily_features(self, daily_data, price_column, volume_column=None, 
                             reference_time_ms=38100000, trading_day_column='TradingDay',
                             time_column='TradingMsOfDay', use_relative=True,
                             include_overnight_gap=True):
        """
        Extract features for multiple trading days
        
        Args:
            daily_data: DataFrame with trading data
            price_column: Name of price column
            volume_column: Name of volume column (optional)
            reference_time_ms: Reference time in milliseconds for relative prices
            trading_day_column: Name of trading day column
            time_column: Name of time column
            use_relative: Whether to use relative prices
            include_overnight_gap: Whether to include overnight gap features
            
        Returns:
            pd.DataFrame: Features for each trading day
        """
        daily_features_list = []
        previous_day_last_price = None
        
        for trading_day in sorted(daily_data[trading_day_column].unique()):
            day_data = daily_data[daily_data[trading_day_column] == trading_day].copy()
            
            if len(day_data) < 5:  # Skip days with insufficient data
                continue
            
            prices = day_data[price_column].values
            volumes = day_data[volume_column].values if volume_column and volume_column in day_data.columns else None
            
            # Find reference price if using relative prices
            reference_price = None
            if use_relative:
                # Look for exact time match first
                reference_rows = day_data[day_data[time_column] == reference_time_ms]
                if len(reference_rows) > 0:
                    reference_price = reference_rows[price_column].iloc[0]
                else:
                    # Find closest time before reference
                    before_ref = day_data[day_data[time_column] <= reference_time_ms]
                    if len(before_ref) > 0:
                        closest_idx = before_ref[time_column].idxmax()
                        reference_price = before_ref.loc[closest_idx, price_column]
                    else:
                        reference_price = prices[0]
            
            # Calculate features
            features = self.calculate_all_features(prices, volumes, reference_price, use_relative)
            
            # Add overnight gap features if previous day data is available
            if include_overnight_gap and previous_day_last_price is not None:
                current_day_first_price = prices[0]
                gap_absolute = current_day_first_price - previous_day_last_price
                
                features.update({
                    'overnight_gap_absolute': gap_absolute,
                })
            else:
                # Set default values for first day or when feature is disabled
                features.update({
                    'overnight_gap_absolute': 0,
                })
            
            # Store last price of current day for next iteration
            if include_overnight_gap:
                previous_day_last_price = prices[-1]
            
            # Add metadata
            features.update({
                trading_day_column: trading_day,
                'reference_time_ms': reference_time_ms,
                'reference_price': reference_price,
                'num_observations': len(day_data),
            })
            
            daily_features_list.append(features)
        
        return pd.DataFrame(daily_features_list)

# Convenience function for quick usage
def calculate_technical_indicators(prices, volumes=None, reference_price=None, use_relative=True):
    """
    Convenience function to calculate all technical indicators
    
    Args:
        prices: Array of price values
        volumes: Array of volume values (optional)
        reference_price: Reference price for relative calculations
        use_relative: Whether to use relative prices
        
    Returns:
        dict: Complete feature dictionary
    """
    extractor = StatisticalFeatureExtractor()
    return extractor.calculate_all_features(prices, volumes, reference_price, use_relative)

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
    volumes = np.random.lognormal(10, 0.5, 100)
    reference_price = prices[0]
    
    # Extract features
    extractor = StatisticalFeatureExtractor()
    features = extractor.calculate_all_features(prices, volumes, reference_price, use_relative=True)
    
    print("Sample feature extraction:")
    for name, value in list(features.items())[:10]:  # Show first 10 features
        print(f"  {name}: {value:.6f}")
    print(f"Total features extracted: {len(features)}")
