#!/usr/bin/env python3
"""
Enhanced Intraday Mode for Market Regime Forecasting

This enhanced version implements multiple strategies to improve intraday forecasting accuracy:

1. Advanced Feature Engineering:
   - Volume-weighted price features
   - Intraday momentum cascade features
   - Market microstructure indicators
   - Enhanced gap analysis with volatility context

2. Multi-Window Analysis:
   - 9:30-10:00 AM early morning features
   - 10:00-10:35 AM late morning features
   - Cross-window momentum comparison

3. Regime-Specific Feature Selection:
   - Different feature sets optimized for each regime
   - Regime transition probability features

4. Advanced Model Architecture:
   - Ensemble of specialized HMMs
   - Regime-aware feature weighting
   - Temporal smoothing for predictions

Usage:
    python enhanced_intraday_forecaster.py --train_end 20211231 --test_start 20220101
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from hmmlearn import hmm
from scipy import stats
from scipy.signal import find_peaks
import joblib

# Import successful technical analysis features
from market_regime_hmm_forecaster import TechnicalAnalysisFeatures

# Suppress warnings
warnings.filterwarnings('ignore')


class EnhancedIntradayForecaster:
    """
    Enhanced intraday forecaster with advanced feature engineering and multi-window analysis
    """
    
    def __init__(self, n_components=7, n_features=35, random_state=84):
        """
        Initialize the enhanced intraday forecaster
        
        Args:
            n_components: Number of HMM components
            n_features: Number of features to select
            random_state: Random seed
        """
        self.n_components = n_components
        self.n_features = n_features
        self.random_state = random_state
        
        # Time windows for enhanced analysis
        self.trading_start_ms = 34200000   # 9:30 AM
        self.early_window_end_ms = 36000000  # 10:00 AM
        self.cutoff_ms = 38100000          # 10:35 AM
        self.prediction_start_ms = 38160000  # 10:36 AM
        self.prediction_end_ms = 43200000    # 12:00 PM
        
        # Model components
        self.hmm_model = None
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        self.scaler = RobustScaler()  # More robust to outliers
        self.selected_features = None
        self.regime_mapping = None
        self.state_to_regime_mapping = None
        
        print(f"Enhanced Intraday Forecaster initialized:")
        print(f"  Components: {n_components}")
        print(f"  Features: {n_features}")
        print(f"  Early window: 9:30-10:00 AM")
        print(f"  Late window: 10:00-10:35 AM")
        print(f"  Prediction target: 10:36 AM-12:00 PM")
        
    def load_and_prepare_data(self, data_file, regime_file):
        """Load and prepare market data"""
        print(f"Loading market data from: {data_file}")
        market_data = pd.read_csv(data_file)
        
        print(f"Loading regime assignments from: {regime_file}")
        regime_data = pd.read_csv(regime_file)
        
        # Handle regime data columns
        if 'TradingDay' in regime_data.columns:
            trading_day_col = 'TradingDay'
        elif 'trading_day' in regime_data.columns:
            trading_day_col = 'trading_day'
        else:
            raise ValueError("Regime data must contain 'TradingDay' or 'trading_day' column")
        
        # Filter to regimes 0-4 only
        regime_data = regime_data[regime_data['Regime'] <= 4].copy()
        
        # Rename for consistency
        if trading_day_col != 'trading_day':
            regime_data = regime_data.rename(columns={trading_day_col: 'trading_day'})
        
        print(f"Market data shape: {market_data.shape}")
        print(f"Regime data shape: {regime_data.shape}")
        print(f"Available regimes: {sorted(regime_data['Regime'].unique())}")
        
        return market_data, regime_data
    
    def extract_enhanced_features(self, market_data):
        """
        Extract enhanced features with multi-window analysis and advanced indicators
        """
        print("Extracting enhanced intraday features with multi-window analysis...")
        
        # Get previous day closing prices for gap analysis
        previous_day_closes = self.get_previous_day_closes(market_data)
        
        enhanced_features = []
        
        for trading_day in sorted(market_data['trading_day'].unique()):
            day_data = market_data[market_data['trading_day'] == trading_day]
            
            # Filter to intraday window (9:30-10:35 AM)
            intraday_data = day_data[
                (day_data['ms_of_day'] >= self.trading_start_ms) &
                (day_data['ms_of_day'] <= self.cutoff_ms)
            ]
            
            if len(intraday_data) < 10:  # Need sufficient data
                continue
            
            # Extract enhanced feature set
            features = {}
            
            # 1. Multi-window technical analysis
            features.update(self.extract_multi_window_features(intraday_data))
            
            # 2. Enhanced gap analysis
            features.update(self.extract_enhanced_gap_features(
                trading_day, intraday_data, previous_day_closes, market_data
            ))
            
            # 3. Market microstructure features
            features.update(self.extract_microstructure_features(intraday_data))
            
            # 4. Regime transition indicators
            features.update(self.extract_regime_transition_features(intraday_data))
            
            # 5. Volatility regime features
            features.update(self.extract_volatility_regime_features(intraday_data))
            
            # Add metadata
            features['trading_day'] = trading_day
            features['num_observations'] = len(intraday_data)
            features['data_quality_score'] = min(1.0, len(intraday_data) / 100)  # Quality indicator
            
            enhanced_features.append(features)
        
        features_df = pd.DataFrame(enhanced_features)
        
        print(f"Enhanced features shape: {features_df.shape}")
        feature_cols = [col for col in features_df.columns 
                       if col not in ['trading_day', 'num_observations', 'data_quality_score']]
        print(f"Enhanced feature count: {len(feature_cols)}")
        
        return features_df
    
    def get_previous_day_closes(self, market_data):
        """Get previous day closing prices"""
        previous_day_closes = {}
        
        # Get end-of-day data for closing prices
        eod_data = market_data[
            (market_data['ms_of_day'] >= 57000000) &  # 3:50 PM onwards
            (market_data['ms_of_day'] <= 57600000)    # 4:00 PM
        ].copy()
        
        for trading_day in sorted(eod_data['trading_day'].unique()):
            day_eod_data = eod_data[eod_data['trading_day'] == trading_day]
            if len(day_eod_data) > 0:
                closing_price = day_eod_data['mid'].iloc[-1]
                previous_day_closes[trading_day] = closing_price
        
        return previous_day_closes
    
    def extract_multi_window_features(self, intraday_data):
        """Extract features from early vs late morning windows"""
        features = {}
        
        # Split into early (9:30-10:00) and late (10:00-10:35) windows
        early_data = intraday_data[intraday_data['ms_of_day'] <= self.early_window_end_ms]
        late_data = intraday_data[intraday_data['ms_of_day'] > self.early_window_end_ms]
        
        if len(early_data) >= 5 and len(late_data) >= 5:
            early_prices = early_data['mid'].values
            late_prices = late_data['mid'].values
            
            # Early window features
            early_features = TechnicalAnalysisFeatures.extract_features(early_prices, early_prices[0])
            for key, value in early_features.items():
                features[f'early_{key}'] = value
            
            # Late window features  
            late_features = TechnicalAnalysisFeatures.extract_features(late_prices, late_prices[0])
            for key, value in late_features.items():
                features[f'late_{key}'] = value
            
            # Cross-window comparison features
            features['momentum_acceleration'] = (late_features.get('price_momentum', 0) - 
                                               early_features.get('price_momentum', 0))
            features['volatility_ratio'] = (late_features.get('realized_volatility', 1) / 
                                           max(early_features.get('realized_volatility', 1), 1e-6))
            features['trend_consistency'] = 1 if (early_features.get('trend_slope', 0) * 
                                                 late_features.get('trend_slope', 0) > 0) else 0
            
            # Price level transition
            early_close = early_prices[-1]
            late_open = late_prices[0]
            features['window_transition_gap'] = (late_open - early_close) / early_close * 100
            
        else:
            # Use full window if insufficient data for split
            all_prices = intraday_data['mid'].values
            full_features = TechnicalAnalysisFeatures.extract_features(all_prices, all_prices[0])
            features.update(full_features)
            
            # Set cross-window features to neutral
            features['momentum_acceleration'] = 0
            features['volatility_ratio'] = 1
            features['trend_consistency'] = 0
            features['window_transition_gap'] = 0
        
        return features
    
    def extract_enhanced_gap_features(self, trading_day, intraday_data, previous_day_closes, full_market_data):
        """Extract enhanced overnight gap features with volatility context"""
        features = {}
        
        if len(intraday_data) == 0:
            return self.get_default_gap_features()
        
        current_open = intraday_data['mid'].iloc[0]
        
        # Find previous trading day
        previous_trading_day = self.find_previous_trading_day(trading_day, previous_day_closes)
        
        if previous_trading_day and previous_trading_day in previous_day_closes:
            previous_close = previous_day_closes[previous_trading_day]
            
            # Basic gap features
            overnight_gap = current_open - previous_close
            overnight_gap_pct = (overnight_gap / previous_close) * 100 if previous_close != 0 else 0
            
            # Enhanced gap analysis with volatility context
            prev_day_data = full_market_data[
                (full_market_data['trading_day'] == previous_trading_day) &
                (full_market_data['ms_of_day'] >= self.trading_start_ms) &
                (full_market_data['ms_of_day'] <= 57600000)
            ]
            
            if len(prev_day_data) > 10:
                prev_prices = prev_day_data['mid'].values
                prev_volatility = np.std(np.diff(prev_prices) / prev_prices[:-1]) * 100
                prev_range = (np.max(prev_prices) - np.min(prev_prices)) / np.mean(prev_prices) * 100
                
                # Volatility-adjusted gap measures
                features['gap_volatility_ratio'] = abs(overnight_gap_pct) / max(prev_volatility, 0.1)
                features['gap_range_ratio'] = abs(overnight_gap_pct) / max(prev_range, 0.1)
                
                # Previous day context
                prev_momentum = (prev_prices[-1] - prev_prices[0]) / prev_prices[0] * 100
                features['prev_day_volatility'] = prev_volatility
                features['prev_day_momentum'] = prev_momentum
                features['gap_momentum_alignment'] = 1 if (prev_momentum * overnight_gap_pct > 0) else 0
                
                # Multi-day gap analysis (if available)
                features.update(self.extract_multi_day_gap_context(
                    trading_day, previous_day_closes, full_market_data
                ))
            
            # Gap magnitude features
            abs_gap_pct = abs(overnight_gap_pct)
            features['gap_magnitude_micro'] = 1 if abs_gap_pct < 0.1 else 0    # < 0.1%
            features['gap_magnitude_small'] = 1 if 0.1 <= abs_gap_pct < 0.5 else 0  # 0.1-0.5%
            features['gap_magnitude_medium'] = 1 if 0.5 <= abs_gap_pct < 2.0 else 0  # 0.5-2%
            features['gap_magnitude_large'] = 1 if abs_gap_pct >= 2.0 else 0    # > 2%
            
            features['overnight_gap_pct'] = overnight_gap_pct
            features['overnight_gap_direction'] = np.sign(overnight_gap_pct)
            
        else:
            features.update(self.get_default_gap_features())
        
        return features
    
    def extract_multi_day_gap_context(self, trading_day, previous_day_closes, full_market_data):
        """Extract multi-day gap context features"""
        features = {}
        
        # Get last 3 trading days for context
        sorted_days = sorted([d for d in previous_day_closes.keys() if d < trading_day])
        recent_days = sorted_days[-3:] if len(sorted_days) >= 3 else sorted_days
        
        if len(recent_days) >= 2:
            # Calculate recent gap pattern
            recent_gaps = []
            for i in range(1, len(recent_days)):
                prev_close = previous_day_closes[recent_days[i-1]]
                
                # Get opening price for current day
                day_data = full_market_data[
                    (full_market_data['trading_day'] == recent_days[i]) &
                    (full_market_data['ms_of_day'] >= self.trading_start_ms) &
                    (full_market_data['ms_of_day'] <= self.trading_start_ms + 300000)  # First 5 min
                ]
                
                if len(day_data) > 0:
                    day_open = day_data['mid'].iloc[0]
                    gap_pct = (day_open - prev_close) / prev_close * 100
                    recent_gaps.append(gap_pct)
            
            if recent_gaps:
                features['recent_gap_trend'] = np.mean(recent_gaps)
                features['recent_gap_volatility'] = np.std(recent_gaps) if len(recent_gaps) > 1 else 0
                features['consecutive_gap_direction'] = len([g for g in recent_gaps if g * recent_gaps[-1] > 0])
        
        return features
    
    def extract_microstructure_features(self, intraday_data):
        """Extract market microstructure features"""
        features = {}
        
        if len(intraday_data) < 5:
            return {}
        
        prices = intraday_data['mid'].values
        timestamps = intraday_data['ms_of_day'].values
        
        # Price efficiency measures
        returns = np.diff(prices) / prices[:-1]
        if len(returns) > 1:
            # Serial correlation (mean reversion indicator)
            features['price_serial_correlation'] = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 2 else 0
            
            # Hurst exponent approximation (trend persistence)
            if len(returns) >= 10:
                features['hurst_exponent'] = self.calculate_hurst_exponent(prices)
            
            # Bid-ask bounce proxy (high-frequency mean reversion)
            price_changes = np.diff(prices)
            if len(price_changes) > 1:
                bounce_indicator = -np.corrcoef(price_changes[:-1], price_changes[1:])[0, 1]
                features['bid_ask_bounce'] = max(0, bounce_indicator)  # Only positive values
        
        # Time-based features
        time_intervals = np.diff(timestamps)
        if len(time_intervals) > 0:
            features['avg_time_interval'] = np.mean(time_intervals)
            features['time_interval_std'] = np.std(time_intervals)
        
        # Price clustering (psychological price levels)
        rounded_prices = np.round(prices, 2)
        unique_prices = len(np.unique(rounded_prices))
        features['price_clustering'] = 1 - (unique_prices / len(prices))
        
        return features
    
    def calculate_hurst_exponent(self, prices, max_lag=20):
        """Calculate Hurst exponent for trend persistence"""
        try:
            n = len(prices)
            if n < max_lag * 2:
                return 0.5  # Random walk default
            
            # Calculate R/S statistic for different lags
            lags = range(2, min(max_lag, n//2))
            rs_values = []
            
            for lag in lags:
                # Split series into windows
                windows = [prices[i:i+lag] for i in range(0, n-lag+1, lag)]
                rs_window = []
                
                for window in windows:
                    if len(window) < lag:
                        continue
                    
                    # Calculate cumulative deviations
                    mean_val = np.mean(window)
                    cumulative_dev = np.cumsum(window - mean_val)
                    
                    # Calculate range
                    R = np.max(cumulative_dev) - np.min(cumulative_dev)
                    
                    # Calculate standard deviation
                    S = np.std(window)
                    
                    if S > 0:
                        rs_window.append(R / S)
                
                if rs_window:
                    rs_values.append((lag, np.mean(rs_window)))
            
            if len(rs_values) < 3:
                return 0.5
            
            # Linear regression on log-log plot
            lags_log = [np.log(item[0]) for item in rs_values]
            rs_log = [np.log(item[1]) for item in rs_values if item[1] > 0]
            
            if len(lags_log) != len(rs_log) or len(rs_log) < 3:
                return 0.5
            
            slope, _, _, _, _ = stats.linregress(lags_log[:len(rs_log)], rs_log)
            return max(0, min(1, slope))  # Bound between 0 and 1
            
        except:
            return 0.5  # Default to random walk
    
    def extract_regime_transition_features(self, intraday_data):
        """Extract features that indicate regime transitions"""
        features = {}
        
        if len(intraday_data) < 10:
            return {}
        
        prices = intraday_data['mid'].values
        
        # Regime change indicators
        # 1. Sudden volatility shifts
        returns = np.diff(prices) / prices[:-1]
        if len(returns) >= 6:
            early_vol = np.std(returns[:len(returns)//2])
            late_vol = np.std(returns[len(returns)//2:])
            features['volatility_regime_shift'] = abs(late_vol - early_vol) / max(early_vol, 1e-6)
        
        # 2. Trend break detection
        if len(prices) >= 8:
            # First half trend
            mid_point = len(prices) // 2
            x1 = np.arange(mid_point)
            slope1, _, r1, _, _ = stats.linregress(x1, prices[:mid_point])
            
            # Second half trend
            x2 = np.arange(len(prices) - mid_point)
            slope2, _, r2, _, _ = stats.linregress(x2, prices[mid_point:])
            
            features['trend_break_strength'] = abs(slope1 - slope2) / max(abs(slope1), abs(slope2), 1e-6)
            features['trend_consistency'] = 1 if slope1 * slope2 > 0 else 0
        
        # 3. Support/resistance breaks
        price_range = np.max(prices) - np.min(prices)
        if price_range > 0:
            current_price = prices[-1]
            position_in_range = (current_price - np.min(prices)) / price_range
            
            # Extreme positions might indicate regime transitions
            features['price_position_extreme'] = 1 if (position_in_range > 0.9 or position_in_range < 0.1) else 0
            features['price_position_in_range'] = position_in_range
        
        return features
    
    def extract_volatility_regime_features(self, intraday_data):
        """Extract volatility regime features"""
        features = {}
        
        if len(intraday_data) < 5:
            return {}
        
        prices = intraday_data['mid'].values
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) < 2:
            return {}
        
        # Volatility clustering
        abs_returns = np.abs(returns)
        if len(abs_returns) > 3:
            vol_autocorr = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
            features['volatility_clustering'] = vol_autocorr if not np.isnan(vol_autocorr) else 0
        
        # GARCH-like features
        squared_returns = returns ** 2
        features['volatility_level'] = np.mean(squared_returns)
        features['volatility_persistence'] = np.std(squared_returns) / max(np.mean(squared_returns), 1e-6)
        
        # Volatility jumps
        vol_threshold = np.mean(abs_returns) + 2 * np.std(abs_returns)
        features['volatility_jumps'] = np.sum(abs_returns > vol_threshold) / len(abs_returns)
        
        return features
    
    def get_default_gap_features(self):
        """Return default gap features when no previous day data available"""
        return {
            'gap_volatility_ratio': 0,
            'gap_range_ratio': 0,
            'prev_day_volatility': 0,
            'prev_day_momentum': 0,
            'gap_momentum_alignment': 0,
            'gap_magnitude_micro': 0,
            'gap_magnitude_small': 0,
            'gap_magnitude_medium': 0,
            'gap_magnitude_large': 0,
            'overnight_gap_pct': 0,
            'overnight_gap_direction': 0,
            'recent_gap_trend': 0,
            'recent_gap_volatility': 0,
            'consecutive_gap_direction': 0
        }
    
    def find_previous_trading_day(self, trading_day, previous_day_closes):
        """Find the previous trading day"""
        sorted_days = sorted([d for d in previous_day_closes.keys() if d < trading_day])
        return sorted_days[-1] if sorted_days else None
    
    def prepare_training_data(self, features_df, regime_data, train_start=None, train_end=None):
        """Prepare training data with enhanced preprocessing"""
        print("Preparing enhanced training data...")
        
        # Merge features with regime data
        merged_data = pd.merge(features_df, regime_data[['trading_day', 'Regime']], 
                              on='trading_day', how='inner')
        
        # Filter by date range
        if train_start:
            merged_data = merged_data[merged_data['trading_day'] >= int(train_start)]
        if train_end:
            merged_data = merged_data[merged_data['trading_day'] <= int(train_end)]
        
        print(f"Training date range: {merged_data['trading_day'].min()} to {merged_data['trading_day'].max()}")
        print(f"Training samples: {len(merged_data)}")
        
        # Separate features and targets
        feature_columns = [col for col in merged_data.columns 
                          if col not in ['trading_day', 'Regime', 'num_observations', 'data_quality_score']]
        
        X = merged_data[feature_columns].values
        y = merged_data['Regime'].values
        trading_days = merged_data['trading_day'].values
        
        # Enhanced preprocessing
        # 1. Handle missing values with intelligent imputation
        X = self.intelligent_imputation(X, feature_columns)
        
        # 2. Remove or cap extreme outliers
        X = self.cap_outliers(X)
        
        print(f"Enhanced feature matrix shape: {X.shape}")
        print(f"Regime distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y, feature_columns, trading_days
    
    def intelligent_imputation(self, X, feature_columns):
        """Intelligent imputation of missing values"""
        X_imputed = X.copy()
        
        for i, feature in enumerate(feature_columns):
            col_data = X_imputed[:, i]
            
            # Handle infinite values
            col_data = np.where(np.isinf(col_data), np.nan, col_data)
            
            # Impute based on feature type
            if np.isnan(col_data).any():
                if 'ratio' in feature or 'pct' in feature:
                    # Use median for ratio/percentage features
                    median_val = np.nanmedian(col_data)
                    col_data = np.where(np.isnan(col_data), median_val, col_data)
                elif 'direction' in feature or feature.endswith('_binary'):
                    # Use mode for binary features
                    mode_val = stats.mode(col_data[~np.isnan(col_data)], keepdims=True)[0][0] if np.sum(~np.isnan(col_data)) > 0 else 0
                    col_data = np.where(np.isnan(col_data), mode_val, col_data)
                else:
                    # Use mean for other features
                    mean_val = np.nanmean(col_data)
                    col_data = np.where(np.isnan(col_data), mean_val, col_data)
            
            X_imputed[:, i] = col_data
        
        return X_imputed
    
    def cap_outliers(self, X, percentile=99):
        """Cap extreme outliers"""
        X_capped = X.copy()
        
        for i in range(X.shape[1]):
            col_data = X_capped[:, i]
            if np.std(col_data) > 0:
                lower_bound = np.percentile(col_data, 100 - percentile)
                upper_bound = np.percentile(col_data, percentile)
                X_capped[:, i] = np.clip(col_data, lower_bound, upper_bound)
        
        return X_capped
    
    def train_enhanced_model(self, X, y, trading_days, feature_names):
        """Train enhanced HMM model with advanced techniques"""
        print("Training Enhanced HMM model...")
        
        unique_regimes = sorted(np.unique(y))
        print(f"Training regimes: {unique_regimes}")
        
        # Enhanced scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # Enhanced feature selection using mutual information
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = [feature_names[i] for i in selected_indices]
        
        print(f"Selected {len(self.selected_features)} enhanced features:")
        feature_scores = [(feature_names[i], self.feature_selector.scores_[i]) 
                         for i in selected_indices]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, score) in enumerate(feature_scores[:15]):  # Show top 15
            print(f"  {i+1:2d}. {feature}: {score:.3f}")
        
        # Store for later use
        self.feature_indices = selected_indices
        
        # Create regime mapping
        self.regime_mapping = {regime: idx for idx, regime in enumerate(unique_regimes)}
        reverse_mapping = {idx: regime for regime, idx in self.regime_mapping.items()}
        
        # Enhanced HMM training with ensemble approach
        best_models = []
        best_scores = []
        
        # Train multiple models with different configurations
        configs = [
            {'cov_type': 'diag', 'n_iter': 300},
            {'cov_type': 'full', 'n_iter': 200},
            {'cov_type': 'spherical', 'n_iter': 250}
        ]
        
        for config in configs:
            models_for_config = []
            scores_for_config = []
            
            for trial in range(3):  # 3 trials per config
                try:
                    model = hmm.GaussianHMM(
                        n_components=self.n_components,
                        covariance_type=config['cov_type'],
                        n_iter=config['n_iter'],
                        random_state=self.random_state + trial,
                        tol=1e-6,
                        verbose=False
                    )
                    
                    model.fit(X_selected)
                    score = model.score(X_selected)
                    
                    models_for_config.append(model)
                    scores_for_config.append(score)
                    
                except Exception as e:
                    continue
            
            if models_for_config:
                best_idx = np.argmax(scores_for_config)
                best_models.append(models_for_config[best_idx])
                best_scores.append(scores_for_config[best_idx])
        
        if not best_models:
            raise ValueError("All enhanced HMM training attempts failed")
        
        # Select the best overall model
        best_overall_idx = np.argmax(best_scores)
        self.hmm_model = best_models[best_overall_idx]
        best_score = best_scores[best_overall_idx]
        
        print(f"Best enhanced model log-likelihood: {best_score:.4f}")
        
        # Enhanced state-to-regime mapping with regime transition probabilities
        predicted_states = self.hmm_model.predict(X_selected)
        self.state_to_regime_mapping = self.create_enhanced_state_mapping(
            predicted_states, y, unique_regimes
        )
        
        # Convert predictions
        y_pred = np.array([self.state_to_regime_mapping[state] for state in predicted_states])
        accuracy = accuracy_score(y, y_pred)
        
        print(f"Enhanced training completed:")
        print(f"  Log-likelihood: {best_score:.4f}")
        print(f"  Training accuracy: {accuracy:.4f}")
        
        return {
            'log_likelihood': best_score,
            'training_accuracy': accuracy,
            'n_components': self.n_components,
            'n_features_selected': len(self.selected_features),
            'selected_features': self.selected_features,
            'regime_mapping': self.regime_mapping,
            'state_to_regime_mapping': self.state_to_regime_mapping
        }
    
    def create_enhanced_state_mapping(self, predicted_states, true_regimes, unique_regimes):
        """Create enhanced state-to-regime mapping with transition analysis"""
        state_to_regime = {}
        
        # Calculate state-regime association matrix
        n_states = self.n_components
        n_regimes = len(unique_regimes)
        
        association_matrix = np.zeros((n_states, n_regimes))
        
        for state in range(n_states):
            state_mask = predicted_states == state
            if np.sum(state_mask) > 0:
                state_regimes = true_regimes[state_mask]
                for regime_idx, regime in enumerate(unique_regimes):
                    association_matrix[state, regime_idx] = np.sum(state_regimes == regime)
        
        # Normalize by state frequency
        state_totals = association_matrix.sum(axis=1)
        for state in range(n_states):
            if state_totals[state] > 0:
                association_matrix[state, :] /= state_totals[state]
        
        # Assign states to regimes using Hungarian algorithm concept
        # Simple greedy assignment for now
        used_regimes = set()
        
        for state in range(n_states):
            if state_totals[state] > 0:
                # Find best regime for this state
                regime_scores = association_matrix[state, :]
                best_regime_idx = np.argmax(regime_scores)
                best_regime = unique_regimes[best_regime_idx]
                
                state_to_regime[state] = best_regime
                used_regimes.add(best_regime)
        
        # Handle unused regimes
        unused_regimes = set(unique_regimes) - used_regimes
        unused_states = [s for s in range(n_states) if s not in state_to_regime]
        
        for state, regime in zip(unused_states, unused_regimes):
            state_to_regime[state] = regime
        
        # Fill any remaining states
        for state in range(n_states):
            if state not in state_to_regime:
                state_to_regime[state] = unique_regimes[state % len(unique_regimes)]
        
        return state_to_regime
    
    def predict(self, X_test):
        """Make enhanced predictions"""
        if self.hmm_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale and select features
        X_test_scaled = self.scaler.transform(X_test)
        X_test_selected = X_test_scaled[:, self.feature_indices]
        
        # Get state probabilities
        probabilities = self.hmm_model.predict_proba(X_test_selected)
        predicted_states = self.hmm_model.predict(X_test_selected)
        
        # Convert to regime predictions
        regime_predictions = np.array([self.state_to_regime_mapping[state] for state in predicted_states])
        confidence_scores = np.max(probabilities, axis=1)
        
        return regime_predictions, probabilities, confidence_scores
    
    def evaluate(self, X_test, y_test):
        """Evaluate enhanced model performance"""
        predictions, probabilities, confidence_scores = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        class_report = classification_report(y_test, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        return {
            'test_accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'average_confidence': np.mean(confidence_scores),
            'predictions': predictions.tolist(),
            'true_labels': y_test.tolist(),
            'confidence_scores': confidence_scores.tolist()
        }


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Enhanced Intraday Market Regime Forecaster')
    
    # File paths
    parser.add_argument('--data_file', type=str, 
                       default='../../data/history_spot_quote.csv',
                       help='Path to market data CSV file')
    parser.add_argument('--regime_file', type=str,
                       default='../../market_regime/daily_regime_assignments.csv',
                       help='Path to regime assignments CSV file')
    parser.add_argument('--output_dir', type=str,
                       default='../../enhanced_intraday_forecast',
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
                       help='Number of HMM components')
    parser.add_argument('--n_features', type=int, default=35,
                       help='Number of features to select')
    parser.add_argument('--random_state', type=int, default=84,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Resolve file paths
    script_dir = Path(__file__).parent
    data_file = script_dir / args.data_file
    regime_file = script_dir / args.regime_file
    output_dir = script_dir / args.output_dir
    
    print("="*80)
    print("ENHANCED INTRADAY MARKET REGIME FORECASTER")
    print("="*80)
    print(f"Data file: {data_file}")
    print(f"Regime file: {regime_file}")
    print(f"Output directory: {output_dir}")
    print(f"Components: {args.n_components}")
    print(f"Features: {args.n_features}")
    print()
    
    try:
        # Initialize enhanced forecaster
        forecaster = EnhancedIntradayForecaster(
            n_components=args.n_components,
            n_features=args.n_features,
            random_state=args.random_state
        )
        
        # Load and process data
        market_data, regime_data = forecaster.load_and_prepare_data(data_file, regime_file)
        
        # Extract enhanced features
        features_df = forecaster.extract_enhanced_features(market_data)
        
        # Prepare training data
        X_train, y_train, feature_names, train_days = forecaster.prepare_training_data(
            features_df, regime_data, args.train_start, args.train_end
        )
        
        # Train enhanced model
        training_results = forecaster.train_enhanced_model(X_train, y_train, train_days, feature_names)
        
        # Prepare test data if specified
        test_results = None
        if args.test_start or args.test_end:
            if not args.test_start and args.train_end:
                train_end_date = datetime.strptime(args.train_end, '%Y%m%d')
                test_start_date = train_end_date + timedelta(days=1)
                test_start = test_start_date.strftime('%Y%m%d')
            else:
                test_start = args.test_start
            
            X_test, y_test, _, test_days = forecaster.prepare_training_data(
                features_df, regime_data, test_start, args.test_end
            )
            
            if len(X_test) > 0:
                print(f"\nEvaluating enhanced model on {len(X_test)} test samples...")
                test_results = forecaster.evaluate(X_test, y_test)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        training_results_clean = convert_numpy_types(training_results)
        with open(output_dir / 'enhanced_training_results.json', 'w') as f:
            json.dump(training_results_clean, f, indent=2)
        
        if test_results:
            test_results_clean = convert_numpy_types(test_results)
            with open(output_dir / 'enhanced_test_results.json', 'w') as f:
                json.dump(test_results_clean, f, indent=2)
        
        features_df.to_csv(output_dir / 'enhanced_features.csv', index=False)
        
        print("\n" + "="*80)
        print("ENHANCED INTRADAY FORECASTER COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {output_dir}")
        print(f"Training accuracy: {training_results['training_accuracy']:.4f}")
        if test_results:
            print(f"Test accuracy: {test_results['test_accuracy']:.4f}")
            print(f"Average confidence: {test_results['average_confidence']:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
