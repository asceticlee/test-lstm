#!/usr/bin/env python3
"""
Enhanced HMM Market Regime Forecaster with Proper Temporal Logic

A robust implementation for next-day market regime forecasting using Hidden Markov Models.
CORRECTLY uses day T's market data (10:35-12:00) to predict day T+1's market regime.

Key Features:
- Proper temporal alignment: Day T features → Day T+1 regime prediction
- Enhanced technical analysis features with volatility, momentum, and pattern detection
- Mutual information-based feature selection for better regime discrimination
- Robust preprocessing with outlier handling and enhanced scaling
- Multi-configuration HMM training with validation
- Uses 10:35-12:00 time window for consistency with regime labels

Enhanced Accuracy Features:
- 60+ technical indicators including volatility regimes and momentum patterns
- RobustScaler preprocessing for better outlier handling
- Dynamic component selection based on data size
- Confidence-weighted state-to-regime mapping
- Multiple random restarts for robust model training

Usage:
    python market_regime_hmm_forecaster.py --train_end 20211231 --test_start 20220101 --n_components 7 --n_features 25
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json

# Machine Learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from hmmlearn import hmm
from scipy import stats
from scipy.signal import find_peaks

# Suppress warnings
warnings.filterwarnings('ignore')

class TechnicalAnalysisFeatures:
    """
    Custom technical analysis feature extractor optimized for regime detection
    """
    
    @staticmethod
    def calculate_returns(prices):
        """Calculate various return metrics"""
        if len(prices) < 2:
            return {}
        
        returns = np.diff(prices) / prices[:-1]
        
        return {
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'return_skewness': stats.skew(returns) if len(returns) > 2 else 0,
            'return_kurtosis': stats.kurtosis(returns) if len(returns) > 3 else 0,
            'return_max': np.max(returns),
            'return_min': np.min(returns),
            'positive_return_ratio': np.sum(returns > 0) / len(returns),
        }
    
    @staticmethod
    def calculate_volatility_features(prices):
        """Calculate volatility-based features"""
        if len(prices) < 2:
            return {}
        
        returns = np.diff(prices) / prices[:-1]
        price_changes = np.diff(prices)
        
        # Realized volatility
        realized_vol = np.sqrt(np.sum(returns**2))
        
        # Price range features
        price_range = np.max(prices) - np.min(prices)
        price_range_pct = price_range / np.mean(prices) * 100
        
        # Volatility clustering
        abs_returns = np.abs(returns)
        vol_clustering = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1] if len(abs_returns) > 2 else 0
        
        # Rolling volatility features
        if len(returns) >= 5:
            rolling_vol_short = np.std(returns[-3:])
            rolling_vol_medium = np.std(returns[-5:])
            vol_of_vol = np.std([np.std(returns[i:i+3]) for i in range(len(returns)-2)])
        else:
            rolling_vol_short = rolling_vol_medium = vol_of_vol = np.std(returns)
        
        # Enhanced volatility features for better regime detection
        if len(returns) > 1:
            # Volatility persistence and jumps
            vol_persistence = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1] if len(abs_returns) > 2 else 0
            vol_threshold = 2 * np.std(abs_returns)
            vol_jumps = np.sum(abs_returns > vol_threshold)
            
            # Volatility regimes within the day
            vol_percentiles = [np.percentile(abs_returns, p) for p in [25, 50, 75, 90]]
            vol_p25, vol_p50, vol_p75, vol_p90 = vol_percentiles
            
            # High volatility periods
            high_vol_ratio = np.sum(abs_returns > vol_p75) / len(abs_returns)
            extreme_vol_ratio = np.sum(abs_returns > vol_p90) / len(abs_returns)
            
            # Volatility skewness and concentration
            vol_skew = stats.skew(abs_returns)
            vol_concentration = vol_p90 / max(vol_p50, 1e-6)  # Avoid division by zero
            
        else:
            vol_persistence = vol_jumps = 0
            vol_p25 = vol_p50 = vol_p75 = vol_p90 = np.std(returns)
            high_vol_ratio = extreme_vol_ratio = 0
            vol_skew = 0
            vol_concentration = 1
        
        # Price level volatility
        price_level_vol = np.std(prices) / np.mean(prices) if np.mean(prices) != 0 else 0
        
        # Intraday volatility decay (early vs late period)
        if len(returns) >= 4:
            early_returns = returns[:len(returns)//2]
            late_returns = returns[len(returns)//2:]
            early_vol = np.std(early_returns)
            late_vol = np.std(late_returns)
            vol_decay = (early_vol - late_vol) / max(early_vol, 1e-6)
        else:
            vol_decay = 0
        
        return {
            'realized_volatility': realized_vol,
            'price_range': price_range,
            'price_range_pct': price_range_pct,
            'price_change_std': np.std(price_changes),
            'max_price_change': np.max(np.abs(price_changes)),
            'volatility_clustering': vol_clustering if not np.isnan(vol_clustering) else 0,
            'rolling_vol_short': rolling_vol_short,
            'rolling_vol_medium': rolling_vol_medium,
            'volatility_of_volatility': vol_of_vol if not np.isnan(vol_of_vol) else 0,
            'volatility_persistence': vol_persistence if not np.isnan(vol_persistence) else 0,
            'volatility_jumps': vol_jumps,
            'price_level_volatility': price_level_vol,
            # Enhanced volatility features
            'vol_p25': vol_p25,
            'vol_p50': vol_p50,
            'vol_p75': vol_p75,
            'vol_p90': vol_p90,
            'high_vol_ratio': high_vol_ratio,
            'extreme_vol_ratio': extreme_vol_ratio,
            'vol_skewness': vol_skew if not np.isnan(vol_skew) else 0,
            'vol_concentration': vol_concentration,
            'vol_decay': vol_decay,
        }
    
    @staticmethod
    def calculate_momentum_features(prices):
        """Calculate momentum and trend features"""
        if len(prices) < 5:
            return {}
        
        n = len(prices)
        
        # Linear trend
        x = np.arange(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        # Price position relative to period
        first_price = prices[0]
        last_price = prices[-1]
        max_price = np.max(prices)
        min_price = np.min(prices)
        
        # Where does the last price sit relative to the range?
        price_position = (last_price - min_price) / (max_price - min_price) if max_price != min_price else 0.5
        
        # Moving averages for momentum
        short_ma = np.mean(prices[-3:]) if len(prices) >= 3 else last_price
        medium_ma = np.mean(prices[-5:]) if len(prices) >= 5 else last_price
        long_ma = np.mean(prices)
        momentum = (short_ma - long_ma) / long_ma * 100
        momentum_medium = (medium_ma - long_ma) / long_ma * 100
        
        # Direction consistency
        price_directions = np.sign(np.diff(prices))
        direction_changes = np.sum(np.diff(price_directions) != 0)
        direction_consistency = 1 - (direction_changes / max(1, len(price_directions)-1))
        
        # Rate of change features
        if len(prices) >= 3:
            roc_short = (prices[-1] - prices[-3]) / prices[-3] * 100
        else:
            roc_short = 0
            
        if len(prices) >= 5:
            roc_medium = (prices[-1] - prices[-5]) / prices[-5] * 100
        else:
            roc_medium = 0
        
        # Acceleration (second derivative of price)
        if len(prices) >= 3:
            first_diff = np.diff(prices)
            acceleration = np.mean(np.diff(first_diff))
        else:
            acceleration = 0
            
        # Velocity consistency
        returns = np.diff(prices) / prices[:-1]
        if len(returns) > 1:
            velocity_std = np.std(returns)
            velocity_mean = np.mean(returns)
        else:
            velocity_std = velocity_mean = 0
        
        # Enhanced momentum features
        # Price momentum strength and persistence
        if len(prices) >= 6:
            # Early vs late momentum
            early_prices = prices[:len(prices)//2]
            late_prices = prices[len(prices)//2:]
            early_momentum = (early_prices[-1] - early_prices[0]) / early_prices[0] * 100
            late_momentum = (late_prices[-1] - late_prices[0]) / late_prices[0] * 100
            momentum_acceleration = late_momentum - early_momentum
            
            # Momentum regime consistency
            returns_early = np.diff(early_prices) / early_prices[:-1]
            returns_late = np.diff(late_prices) / late_prices[:-1]
            momentum_regime_change = np.sign(np.mean(returns_early)) != np.sign(np.mean(returns_late))
        else:
            early_momentum = late_momentum = momentum_acceleration = 0
            momentum_regime_change = False
        
        # Price breakout patterns
        price_range = max_price - min_price
        breakout_threshold = price_range * 0.8
        near_high = (last_price - min_price) > breakout_threshold
        near_low = (max_price - last_price) > breakout_threshold
        
        # Momentum reversals
        if len(returns) >= 4:
            # Count momentum reversals
            momentum_signs = np.sign(returns)
            reversals = np.sum(np.diff(momentum_signs) != 0)
            reversal_frequency = reversals / len(momentum_signs)
            
            # Strength of final momentum
            final_momentum_strength = np.abs(np.mean(returns[-3:]))
        else:
            reversal_frequency = final_momentum_strength = 0
        
        # Trend strength categorization
        trend_strength_raw = abs(slope) * (r_value**2)
        if trend_strength_raw > 0.1:
            trend_category = 2  # Strong trend
        elif trend_strength_raw > 0.05:
            trend_category = 1  # Moderate trend
        else:
            trend_category = 0  # Weak/no trend
        
        return {
            'trend_slope': slope,
            'trend_r_squared': r_value**2,
            'trend_strength': trend_strength_raw,
            'price_momentum': momentum,
            'price_momentum_medium': momentum_medium,
            'price_position': price_position,
            'direction_consistency': direction_consistency,
            'total_price_change_pct': (last_price - first_price) / first_price * 100,
            'rate_of_change_short': roc_short,
            'rate_of_change_medium': roc_medium,
            'price_acceleration': acceleration,
            'velocity_consistency': 1 / (1 + velocity_std) if velocity_std > 0 else 1,
            'velocity_mean': velocity_mean,
            # Enhanced momentum features
            'early_momentum': early_momentum,
            'late_momentum': late_momentum,
            'momentum_acceleration': momentum_acceleration,
            'momentum_regime_change': float(momentum_regime_change),
            'near_high_breakout': float(near_high),
            'near_low_breakout': float(near_low),
            'reversal_frequency': reversal_frequency,
            'final_momentum_strength': final_momentum_strength,
            'trend_category': trend_category,
        }
    
    @staticmethod
    def calculate_pattern_features(prices):
        """Calculate pattern recognition features"""
        if len(prices) < 5:
            return {}
        
        # Peak and trough detection
        peaks, _ = find_peaks(prices, distance=2)
        troughs, _ = find_peaks(-prices, distance=2)
        
        # Autocorrelation
        if len(prices) > 10:
            autocorr_1 = np.corrcoef(prices[:-1], prices[1:])[0, 1]
            autocorr_2 = np.corrcoef(prices[:-2], prices[2:])[0, 1]
            autocorr_3 = np.corrcoef(prices[:-3], prices[3:])[0, 1]
        else:
            autocorr_1 = autocorr_2 = autocorr_3 = 0
        
        # Jump detection (large price movements)
        returns = np.diff(prices) / prices[:-1]
        threshold = 2 * np.std(returns) if len(returns) > 1 else 0
        jumps = np.abs(returns) > threshold
        
        # Price level features
        mean_price = np.mean(prices)
        price_std = np.std(prices)
        
        # Higher order moments of returns
        if len(returns) > 3:
            return_skew = stats.skew(returns)
            return_kurt = stats.kurtosis(returns)
        else:
            return_skew = return_kurt = 0
        
        # Hurst exponent approximation (mean reverting vs trending)
        if len(prices) > 10:
            def hurst_simple(ts):
                """Simple Hurst exponent calculation"""
                ts = np.array(ts)
                n = len(ts)
                if n < 10:
                    return 0.5
                
                # Calculate R/S statistic for different lags
                lags = [2, 3, 4, 5]
                rs_vals = []
                
                for lag in lags:
                    if lag >= n:
                        continue
                    
                    # Split into chunks
                    chunks = n // lag
                    rs_chunk = []
                    
                    for i in range(chunks):
                        chunk = ts[i*lag:(i+1)*lag]
                        if len(chunk) < 2:
                            continue
                        
                        mean_chunk = np.mean(chunk)
                        std_chunk = np.std(chunk)
                        
                        if std_chunk == 0:
                            continue
                        
                        # Cumulative deviations
                        cum_dev = np.cumsum(chunk - mean_chunk)
                        R = np.max(cum_dev) - np.min(cum_dev)
                        S = std_chunk
                        
                        if S > 0:
                            rs_chunk.append(R / S)
                    
                    if rs_chunk:
                        rs_vals.append(np.mean(rs_chunk))
                
                if len(rs_vals) >= 2:
                    # Simple linear regression to estimate Hurst
                    log_lags = np.log(lags[:len(rs_vals)])
                    log_rs = np.log(rs_vals)
                    try:
                        slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
                        return slope
                    except:
                        return 0.5
                return 0.5
            
            hurst_exp = hurst_simple(prices)
        else:
            hurst_exp = 0.5
        
        # Gap analysis (price jumps between consecutive observations)
        if len(prices) > 1:
            price_gaps = np.abs(np.diff(prices))
            avg_gap = np.mean(price_gaps)
            max_gap = np.max(price_gaps)
            gap_variance = np.var(price_gaps)
        else:
            avg_gap = max_gap = gap_variance = 0
        
        return {
            'num_peaks': len(peaks),
            'num_troughs': len(troughs),
            'peak_trough_ratio': len(peaks) / max(1, len(troughs)),
            'autocorr_lag1': autocorr_1 if not np.isnan(autocorr_1) else 0,
            'autocorr_lag2': autocorr_2 if not np.isnan(autocorr_2) else 0,
            'autocorr_lag3': autocorr_3 if not np.isnan(autocorr_3) else 0,
            'num_jumps': np.sum(jumps),
            'jump_frequency': np.sum(jumps) / len(returns) if len(returns) > 0 else 0,
            'price_coefficient_variation': price_std / mean_price if mean_price != 0 else 0,
            'return_skewness_pattern': return_skew if not np.isnan(return_skew) else 0,
            'return_kurtosis_pattern': return_kurt if not np.isnan(return_kurt) else 0,
            'hurst_exponent': hurst_exp if not np.isnan(hurst_exp) else 0.5,
            'avg_price_gap': avg_gap,
            'max_price_gap': max_gap,
            'price_gap_variance': gap_variance,
        }
    
    @staticmethod
    def calculate_relative_features(prices, reference_price):
        """Calculate features relative to reference price (10:35 AM price)"""
        if reference_price <= 0:
            return {}
        
        # Convert to relative prices (percentage from reference)
        rel_prices = (prices / reference_price - 1) * 100
        
        return {
            'rel_price_mean': np.mean(rel_prices),
            'rel_price_std': np.std(rel_prices),
            'rel_price_max': np.max(rel_prices),
            'rel_price_min': np.min(rel_prices),
            'rel_price_range': np.max(rel_prices) - np.min(rel_prices),
            'rel_price_final': rel_prices[-1],
        }
    
    @classmethod
    def extract_features(cls, prices, reference_price=None):
        """
        Extract all technical analysis features for a single day
        
        Args:
            prices: Array of prices for the day
            reference_price: Reference price (e.g., 10:35 AM price)
            
        Returns:
            dict: Complete feature dictionary
        """
        if len(prices) < 2:
            return {}
        
        # Use first price as reference if not provided
        if reference_price is None:
            reference_price = prices[0]
        
        features = {}
        
        # Add all feature categories
        features.update(cls.calculate_returns(prices))
        features.update(cls.calculate_volatility_features(prices))
        features.update(cls.calculate_momentum_features(prices))
        features.update(cls.calculate_pattern_features(prices))
        features.update(cls.calculate_relative_features(prices, reference_price))
        
        return features


class HMMRegimeForecaster:
    """
    Simplified HMM-based market regime forecaster
    """
    
    def __init__(self, n_components=7, n_features=25, random_state=42):
        """
        Initialize the forecaster
        
        Args:
            n_components: Number of HMM hidden states (should be >= number of regimes for better modeling)
            n_features: Number of top features to select
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.n_features = n_features
        self.random_state = random_state
        
        # Time window settings (10:35 AM - 12:00 PM)
        self.start_time_ms = 38100000  # 10:35 AM
        self.end_time_ms = 43200000    # 12:00 PM
        
        # Model components
        self.hmm_model = None
        self.scaler = StandardScaler()
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        self.selected_features = None
        self.feature_names = None
        
        print(f"Enhanced HMM Regime Forecaster initialized:")
        print(f"  HMM Components (hidden states): {n_components}")
        print(f"  Features: {n_features}")
        print(f"  Feature selection: Mutual Information")
        print(f"  Time window: 10:35 AM - 12:00 PM")
        print(f"  Prediction: Day T features -> Day T+1 regime")
    
    def load_market_data(self, data_file):
        """
        Load and filter market data
        
        Args:
            data_file: Path to history_spot_quote.csv
            
        Returns:
            pd.DataFrame: Filtered market data
        """
        print(f"Loading market data from: {data_file}")
        
        # Load data
        data = pd.read_csv(data_file)
        
        # Validate columns
        required_cols = ['trading_day', 'ms_of_day', 'mid']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter to our time window
        filtered_data = data[
            (data['ms_of_day'] >= self.start_time_ms) & 
            (data['ms_of_day'] <= self.end_time_ms)
        ].copy()
        
        print(f"Original data shape: {data.shape}")
        print(f"Filtered data shape: {filtered_data.shape}")
        print(f"Date range: {filtered_data['trading_day'].min()} to {filtered_data['trading_day'].max()}")
        print(f"Unique trading days: {filtered_data['trading_day'].nunique()}")
        
        return filtered_data
    
    def extract_daily_features(self, market_data):
        """
        Extract features for each trading day
        
        Args:
            market_data: Filtered market data DataFrame
            
        Returns:
            pd.DataFrame: Daily features
        """
        print("Extracting daily technical analysis features...")
        
        daily_features = []
        
        for trading_day in sorted(market_data['trading_day'].unique()):
            day_data = market_data[market_data['trading_day'] == trading_day]
            
            if len(day_data) < 5:  # Skip days with insufficient data
                continue
            
            prices = day_data['mid'].values
            reference_price = prices[0]  # First price of the day (10:35 AM)
            
            # Extract all features
            features = TechnicalAnalysisFeatures.extract_features(prices, reference_price)
            
            # Add metadata
            features['trading_day'] = trading_day
            features['num_observations'] = len(day_data)
            features['reference_price'] = reference_price
            
            daily_features.append(features)
        
        features_df = pd.DataFrame(daily_features)
        
        print(f"Extracted features for {len(features_df)} trading days")
        print(f"Feature columns: {len([c for c in features_df.columns if c not in ['trading_day', 'num_observations', 'reference_price']])}")
        
        return features_df
    
    def load_regime_labels(self, regime_file):
        """
        Load regime labels and filter to 5 regimes (0-4)
        
        Args:
            regime_file: Path to regime assignments CSV
            
        Returns:
            pd.DataFrame: Filtered regime data
        """
        print(f"Loading regime labels from: {regime_file}")
        
        regime_data = pd.read_csv(regime_file)
        
        # Determine trading day column
        if 'trading_day' in regime_data.columns:
            trading_day_col = 'trading_day'
        elif 'TradingDay' in regime_data.columns:
            trading_day_col = 'TradingDay'
        else:
            raise ValueError("Regime data must contain 'trading_day' or 'TradingDay' column")
        
        if 'Regime' not in regime_data.columns:
            raise ValueError("Regime data must contain 'Regime' column")
        
        # Filter to regimes 0-4 only
        regime_data = regime_data[regime_data['Regime'] <= 4].copy()
        
        # Rename column for consistency
        if trading_day_col != 'trading_day':
            regime_data = regime_data.rename(columns={trading_day_col: 'trading_day'})
        
        # Keep only necessary columns
        regime_data = regime_data[['trading_day', 'Regime']].drop_duplicates(subset=['trading_day'])
        
        print(f"Loaded {len(regime_data)} regime assignments")
        print(f"Available regimes: {sorted(regime_data['Regime'].unique())}")
        print(f"Regime distribution: {dict(regime_data['Regime'].value_counts().sort_index())}")
        
        return regime_data
    
    def prepare_training_data(self, features_df, regime_data, train_start=None, train_end=None):
        """
        Merge features with regime labels and prepare training data
        CRITICAL: Uses day T features to predict day T+1 regime
        
        Args:
            features_df: Daily features DataFrame
            regime_data: Regime labels DataFrame
            train_start: Training start date (YYYYMMDD)
            train_end: Training end date (YYYYMMDD)
            
        Returns:
            tuple: (X, y, trading_days)
        """
        print("Preparing training data with proper temporal alignment...")
        
        # Sort both dataframes by trading day
        features_df = features_df.sort_values('trading_day').reset_index(drop=True)
        regime_data = regime_data.sort_values('trading_day').reset_index(drop=True)
        
        # Create next-day regime mapping: day T features -> day T+1 regime
        # Shift regime data forward by 1 day (regime today will be predicted from yesterday's features)
        regime_shifted = regime_data.copy()
        regime_shifted['trading_day'] = regime_shifted['trading_day'] - 1  # Match with previous day's features
        regime_shifted = regime_shifted.rename(columns={'Regime': 'Next_Day_Regime'})
        
        # Merge features from day T with regime from day T+1
        merged_data = pd.merge(features_df, regime_shifted, on='trading_day', how='inner')
        
        # Filter by date range
        if train_start:
            merged_data = merged_data[merged_data['trading_day'] >= int(train_start)]
        if train_end:
            merged_data = merged_data[merged_data['trading_day'] <= int(train_end)]
        
        print(f"Training date range: {merged_data['trading_day'].min()} to {merged_data['trading_day'].max()}")
        print(f"Training samples: {len(merged_data)} (day T features -> day T+1 regime)")
        
        # Separate features and targets
        feature_columns = [col for col in merged_data.columns 
                          if col not in ['trading_day', 'Next_Day_Regime', 'num_observations', 'reference_price']]
        
        X = merged_data[feature_columns].values
        y = merged_data['Next_Day_Regime'].values
        trading_days = merged_data['trading_day'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store feature names
        self.feature_names = feature_columns
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Regime distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y, trading_days
    
    def train(self, X, y):
        """
        Train the HMM model with enhanced preprocessing and validation
        
        Args:
            X: Feature matrix
            y: Regime labels
            
        Returns:
            dict: Training results
        """
        print("Training Enhanced HMM model...")
        
        unique_regimes = sorted(np.unique(y))
        print(f"Training regimes: {unique_regimes}")
        print(f"Regime distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Enhanced preprocessing
        # 1. Handle infinite values and outliers
        X_clean = np.copy(X)
        
        # Replace infinite values
        X_clean = np.where(np.isinf(X_clean), 0, X_clean)
        X_clean = np.where(np.isnan(X_clean), 0, X_clean)
        
        # Outlier capping (3-sigma rule per feature)
        for i in range(X_clean.shape[1]):
            feature_data = X_clean[:, i]
            mean_val = np.mean(feature_data)
            std_val = np.std(feature_data)
            
            if std_val > 0:
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                X_clean[:, i] = np.clip(feature_data, lower_bound, upper_bound)
        
        print(f"Feature preprocessing: Handled {np.sum(np.isinf(X)) + np.sum(np.isnan(X))} invalid values")
        
        # 2. Scale features using RobustScaler for better outlier handling
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # 3. Enhanced feature selection with mutual information
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = [self.feature_names[i] for i in selected_indices]
        
        print(f"Selected {len(self.selected_features)} best features using mutual information:")
        feature_scores = [(self.feature_names[i], self.feature_selector.scores_[i]) 
                         for i in selected_indices]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, score) in enumerate(feature_scores[:15]):  # Show top 15
            print(f"  {i+1:2d}. {feature}: {score:.3f}")
        
        if len(feature_scores) > 15:
            print(f"  ... and {len(feature_scores) - 15} more features")
        
        # Store feature indices for later use
        self.feature_indices = selected_indices
        
        # 4. Enhanced HMM training with multiple trials and validation
        best_model = None
        best_score = -np.inf
        best_cov_type = 'diag'
        best_n_components = self.n_components
        
        # Try different numbers of components if training set is large enough
        min_samples_per_component = 10
        max_components = min(self.n_components + 2, len(X_selected) // min_samples_per_component)
        component_range = range(max(3, self.n_components - 1), max_components + 1)
        
        print(f"Testing {len(component_range)} different component counts: {list(component_range)}")
        
        # Try different covariance types and component counts
        covariance_types = ['diag', 'spherical', 'full']
        n_trials_per_config = 5  # Multiple random starts per configuration
        
        total_trials = 0
        successful_trials = 0
        
        for n_comp in component_range:
            print(f"  Testing {n_comp} components...")
            
            for cov_type in covariance_types:
                for trial in range(n_trials_per_config):
                    total_trials += 1
                    try:
                        model = hmm.GaussianHMM(
                            n_components=n_comp,
                            covariance_type=cov_type,
                            n_iter=300,  # More iterations for better convergence
                            random_state=self.random_state + trial,
                            tol=1e-6,
                            verbose=False,
                            init_params='stmc'  # Initialize all parameters
                        )
                        
                        model.fit(X_selected)
                        score = model.score(X_selected)
                        successful_trials += 1
                        
                        # Additional validation: check if model produces reasonable state transitions
                        predicted_states = model.predict(X_selected)
                        unique_states_used = len(np.unique(predicted_states))
                        
                        # Penalize models that don't use enough states
                        state_usage_penalty = (n_comp - unique_states_used) * 0.1
                        adjusted_score = score - state_usage_penalty
                        
                        if adjusted_score > best_score:
                            best_score = adjusted_score
                            best_model = model
                            best_cov_type = cov_type
                            best_n_components = n_comp
                            
                    except Exception as e:
                        continue
        
        print(f"Training completed: {successful_trials}/{total_trials} trials successful")
        
        if best_model is None:
            raise ValueError("All HMM training trials failed")
        
        print(f"  Best configuration: {best_n_components} components, {best_cov_type} covariance")
        print(f"  Best model log-likelihood: {best_score:.4f}")
        
        self.hmm_model = best_model
        self.n_components = best_n_components  # Update to best found
        
        # 5. Enhanced state-to-regime mapping with confidence weighting
        predicted_states = self.hmm_model.predict(X_selected)
        state_probabilities = self.hmm_model.predict_proba(X_selected)
        
        # Create state-to-regime mapping using confidence-weighted voting
        state_to_regime = {}
        state_confidence = {}
        
        for state in range(self.n_components):
            state_mask = predicted_states == state
            
            if np.sum(state_mask) > 0:
                actual_regimes = y[state_mask]
                state_probs = state_probabilities[state_mask, state]
                
                # Weight regime votes by prediction confidence
                weighted_votes = {}
                for regime, prob in zip(actual_regimes, state_probs):
                    if regime not in weighted_votes:
                        weighted_votes[regime] = 0
                    weighted_votes[regime] += prob
                
                # Select regime with highest weighted vote
                best_regime = max(weighted_votes.items(), key=lambda x: x[1])[0]
                state_to_regime[state] = best_regime
                state_confidence[state] = np.mean(state_probs)
            else:
                # Fallback for unused states
                state_to_regime[state] = state % len(unique_regimes)
                state_confidence[state] = 0.0
        
        self.state_to_regime = state_to_regime
        self.state_confidence = state_confidence
        
        # 6. Calculate enhanced training metrics
        regime_predictions = np.array([state_to_regime[state] for state in predicted_states])
        training_accuracy = accuracy_score(y, regime_predictions)
        
        # Calculate per-regime accuracy
        regime_accuracies = {}
        for regime in unique_regimes:
            regime_mask = (y == regime)
            if np.sum(regime_mask) > 0:
                regime_acc = accuracy_score(y[regime_mask], regime_predictions[regime_mask])
                regime_accuracies[int(regime)] = regime_acc  # Convert to int for JSON serialization
        
        # Overall confidence
        avg_confidence = np.mean([state_confidence[state] for state in predicted_states])
        
        print(f"Training Results:")
        print(f"  Overall accuracy: {training_accuracy:.4f}")
        print(f"  Average confidence: {avg_confidence:.4f}")
        print(f"  Per-regime accuracy:")
        for regime in sorted(regime_accuracies.keys()):
            print(f"    Regime {regime}: {regime_accuracies[regime]:.4f}")
        
        return {
            'log_likelihood': best_score,
            'training_accuracy': training_accuracy,
            'regime_accuracies': regime_accuracies,
            'average_confidence': avg_confidence,
            'state_to_regime': {int(k): int(v) for k, v in state_to_regime.items()},  # Convert for JSON
            'state_confidence': {int(k): float(v) for k, v in state_confidence.items()},  # Convert for JSON
            'selected_features': self.selected_features,
            'n_components': self.n_components,
            'n_features': len(self.selected_features),
            'covariance_type': best_cov_type,
            'successful_trials': successful_trials,
            'total_trials': total_trials
        }
    
    def predict(self, X):
        """
        Make regime predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            tuple: (predictions, probabilities, confidence_scores)
        """
        if self.hmm_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_selected = X_scaled[:, self.feature_indices]
        
        # Get predictions
        predicted_states = self.hmm_model.predict(X_selected)
        state_probabilities = self.hmm_model.predict_proba(X_selected)
        
        # Convert states to regimes
        regime_predictions = np.array([self.state_to_regime[state] for state in predicted_states])
        
        # Calculate confidence scores
        confidence_scores = np.max(state_probabilities, axis=1)
        
        return regime_predictions, state_probabilities, confidence_scores
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test feature matrix
            y_test: True regime labels
            
        Returns:
            dict: Evaluation metrics
        """
        predictions, probabilities, confidence_scores = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        class_report = classification_report(y_test, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Average confidence: {np.mean(confidence_scores):.4f}")
        
        return {
            'test_accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'average_confidence': np.mean(confidence_scores),
            'predictions': predictions.tolist(),
            'true_labels': y_test.tolist(),
            'confidence_scores': confidence_scores.tolist()
        }
    
    def save_model(self, output_dir, model_name='hmm_regime_forecaster'):
        """Save trained model"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_data = {
            'hmm_model': self.hmm_model,
            'scaler': self.scaler,
            'feature_indices': self.feature_indices,
            'selected_features': self.selected_features,
            'feature_names': self.feature_names,
            'state_to_regime': self.state_to_regime,
            'n_components': self.n_components,
            'n_features': self.n_features,
            'start_time_ms': self.start_time_ms,
            'end_time_ms': self.end_time_ms
        }
        
        model_path = os.path.join(output_dir, f'{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {model_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='HMM Market Regime Forecaster')
    
    # File paths
    parser.add_argument('--data_file', type=str, 
                       default='../../data/history_spot_quote.csv',
                       help='Path to market data CSV file')
    parser.add_argument('--regime_file', type=str,
                       default='../../market_regime/daily_regime_assignments.csv',
                       help='Path to regime assignments CSV file')
    parser.add_argument('--output_dir', type=str,
                       default='../../market_regime_forecast',
                       help='Output directory for results')
    
    # Date ranges
    parser.add_argument('--train_start', type=str, default=None,
                       help='Training start date (YYYYMMDD)')
    parser.add_argument('--train_end', type=str, default=None,
                       help='Training end date (YYYYMMDD)')
    parser.add_argument('--test_start', type=str, default=None,
                       help='Test start date (YYYYMMDD)')
    parser.add_argument('--test_end', type=str, default=None,
                       help='Test end date (YYYYMMDD)')
    
    # Model parameters
    parser.add_argument('--n_components', type=int, default=7,
                       help='Number of HMM hidden states (should be >= number of regimes)')
    parser.add_argument('--n_features', type=int, default=25,
                       help='Number of features to select (increased for better coverage)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Resolve file paths relative to script location
    script_dir = Path(__file__).parent
    data_file = script_dir / args.data_file
    regime_file = script_dir / args.regime_file
    output_dir = script_dir / args.output_dir
    
    # Verify files exist
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        sys.exit(1)
    
    if not regime_file.exists():
        print(f"ERROR: Regime file not found: {regime_file}")
        sys.exit(1)
    
    print("="*80)
    print("HMM MARKET REGIME FORECASTER")
    print("="*80)
    print(f"Data file: {data_file}")
    print(f"Regime file: {regime_file}")
    print(f"Output directory: {output_dir}")
    print(f"Components: {args.n_components}")
    print(f"Features: {args.n_features}")
    print()
    
    try:
        # Initialize forecaster
        forecaster = HMMRegimeForecaster(
            n_components=args.n_components,
            n_features=args.n_features,
            random_state=args.random_state
        )
        
        # Load and process data
        market_data = forecaster.load_market_data(data_file)
        regime_data = forecaster.load_regime_labels(regime_file)
        features_df = forecaster.extract_daily_features(market_data)
        
        # Prepare training data
        X_train, y_train, train_days = forecaster.prepare_training_data(
            features_df, regime_data, args.train_start, args.train_end
        )
        
        # Train model
        training_results = forecaster.train(X_train, y_train)
        
        # Prepare test data if specified
        test_results = None
        if args.test_start or args.test_end:
            # Generate test start if not provided
            if not args.test_start and args.train_end:
                train_end_date = datetime.strptime(args.train_end, '%Y%m%d')
                test_start_date = train_end_date + timedelta(days=1)
                test_start = test_start_date.strftime('%Y%m%d')
            else:
                test_start = args.test_start
            
            X_test, y_test, test_days = forecaster.prepare_training_data(
                features_df, regime_data, test_start, args.test_end
            )
            
            if len(X_test) > 0:
                print(f"\nEvaluating on {len(X_test)} test samples...")
                test_results = forecaster.evaluate(X_test, y_test)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training results
        with open(output_dir / 'training_results.json', 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        # Save test results if available
        if test_results:
            with open(output_dir / 'test_results.json', 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
        
        # Save model
        forecaster.save_model(output_dir)
        
        # Save features for inspection
        features_df.to_csv(output_dir / 'daily_features.csv', index=False)
        
        print("\n" + "="*80)
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {output_dir}")
        print(f"Training accuracy: {training_results['training_accuracy']:.4f}")
        if test_results:
            print(f"Test accuracy: {test_results['test_accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
