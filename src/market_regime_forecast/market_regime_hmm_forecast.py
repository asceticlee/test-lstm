#!/usr/bin/env python3
"""
Hidden Markov Model Market Regime Forecasting

This script uses Hidden Markov Models (HMM) to predict market regimes based on 
statistical features extracted from history_spot_quote.csv data.

Two Forecasting Modes:
1. Daily Mode: Use today's complete trading day data (from market open) to predict next day's regime
2. Intraday Mode: Use today's data before 10:35 AM to predict regime for 10:36 AM - 12:00 PM period

Key Features:
- Extracts comprehensive statistical features from daily market data
- Includes overnight gap features (prior day's last price to current day's first price)
- Trains HMM models to capture regime transition dynamics
- Provides regime predictions with confidence scores
- Supports both daily and intraday forecasting modes

Usage:
    # Daily forecasting mode (uses default paths)
    python market_regime_hmm_forecast.py --mode daily
    
    # Intraday forecasting mode (uses default paths)
    python market_regime_hmm_forecast.py --mode intraday
    
    # Custom paths if needed
    python market_regime_hmm_forecast.py --mode daily --data_file ../../data/history_spot_quote.csv --regime_file ../../market_regime/daily_regime_assignments.csv
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

# Add the src directory to the path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent
sys.path.insert(0, str(src_dir))

# Machine Learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from hmmlearn import hmm
import joblib

# Import successful technical analysis features
from market_regime_hmm_forecaster import TechnicalAnalysisFeatures

# Statistical analysis
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Import our statistical features module (backup)
from market_data_stat.statistical_features import StatisticalFeatureExtractor

# Suppress warnings
warnings.filterwarnings('ignore')

class HMMRegimeForecaster:
    """
    Hidden Markov Model based market regime forecasting system
    Supports both daily and intraday forecasting modes
    """
    
    def __init__(self, mode='daily', n_components=7, n_features=25, covariance_type='full', 
                 n_iter=200, random_state=84, trading_start_time='09:30',
                 intraday_cutoff_time='10:35', intraday_start_time='10:36', 
                 intraday_end_time='12:00', auto_components=True):
        """
        Initialize the HMM regime forecaster
        
        Args:
            mode: Forecasting mode ('daily' or 'intraday')
            n_components: Number of hidden states (regimes) in HMM
            n_features: Number of top features to select for modeling
            covariance_type: Type of covariance parameters ('full', 'diag', 'tied', 'spherical')
            n_iter: Maximum number of iterations for HMM training
            random_state: Random seed for reproducibility
            trading_start_time: Daily trading start time (format: 'HH:MM')
            intraday_cutoff_time: Cutoff time for intraday mode feature calculation (format: 'HH:MM')
            intraday_start_time: Start of intraday prediction window (format: 'HH:MM')
            intraday_end_time: End of intraday prediction window (format: 'HH:MM')
            auto_components: Whether to automatically determine number of components from data
        """
        self.mode = mode
        self.n_components = n_components
        self.n_features = n_features
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.auto_components = auto_components
        
        # Time settings
        self.trading_start_time = trading_start_time
        self.intraday_cutoff_time = intraday_cutoff_time
        self.intraday_start_time = intraday_start_time
        self.intraday_end_time = intraday_end_time
        
        # Convert times to milliseconds
        self.trading_start_ms = self._time_to_ms(trading_start_time)
        self.intraday_cutoff_ms = self._time_to_ms(intraday_cutoff_time)  # 38100000ms (10:35)
        self.intraday_start_ms = self._time_to_ms(intraday_start_time)    # 38160000ms (10:36)
        self.intraday_end_ms = self._time_to_ms(intraday_end_time)        # 43200000ms (12:00)
        
        # Initialize components
        self.feature_extractor = StatisticalFeatureExtractor()
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        self.selected_features = None
        self.regime_mapping = None
        self.actual_n_components = n_components
        
        print(f"HMM Regime Forecaster initialized in {mode} mode")
        if mode == 'intraday':
            print(f"  Cutoff time: {intraday_cutoff_time} ({self.intraday_cutoff_ms}ms)")
            print(f"  Prediction window: {intraday_start_time} - {intraday_end_time}")
        print(f"  Auto-components: {auto_components}")
        if auto_components:
            print(f"  Components will be determined from training data (max: {n_components})")
    
    
    def _time_to_ms(self, time_str):
        """Convert time string (HH:MM) to milliseconds from midnight"""
        try:
            hour, minute = map(int, time_str.split(':'))
            # Convert to milliseconds from midnight
            total_milliseconds = (hour * 60 + minute) * 60 * 1000
            return total_milliseconds
        except:
            return 38100000  # Default to 10:35 AM
    
    def load_and_prepare_data(self, data_file, regime_file):
        """
        Load and prepare market data and regime assignments for history_spot_quote.csv format
        
        Args:
            data_file: Path to history_spot_quote.csv file
            regime_file: Path to regime assignments CSV file
            
        Returns:
            tuple: (market_data_df, regime_df)
        """
        print(f"Loading market data from: {data_file}")
        market_data = pd.read_csv(data_file)
        
        print(f"Loading regime assignments from: {regime_file}")
        regime_data = pd.read_csv(regime_file)
        
        # Validate required columns for history_spot_quote.csv format
        required_market_cols = ['trading_day', 'ms_of_day', 'mid']
        missing_cols = [col for col in required_market_cols if col not in market_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in market data: {missing_cols}")
        
        # Validate regime data columns - handle both old and new formats
        if 'TradingDay' in regime_data.columns:
            # Old format
            trading_day_col = 'TradingDay'
        elif 'trading_day' in regime_data.columns:
            # New format (from fresh GMM clustering)
            trading_day_col = 'trading_day'
        else:
            raise ValueError("Regime data must contain 'TradingDay' or 'trading_day' column")
            
        if 'Regime' not in regime_data.columns:
            raise ValueError("Regime data must contain 'Regime' column")
        
        # Filter to regimes 0-4 only (same as successful forecaster)
        print(f"Original regime data: {len(regime_data)} rows with regimes {sorted(regime_data['Regime'].unique())}")
        regime_data = regime_data[regime_data['Regime'] <= 4].copy()
        print(f"Filtered regime data: {len(regime_data)} rows with regimes {sorted(regime_data['Regime'].unique())}")
        
        # Convert trading_day to match regime data format if needed
        if market_data['trading_day'].dtype != regime_data[trading_day_col].dtype:
            market_data['trading_day'] = market_data['trading_day'].astype(int)
            regime_data[trading_day_col] = regime_data[trading_day_col].astype(int)
        
        print(f"Market data shape: {market_data.shape}")
        print(f"Regime data shape: {regime_data.shape}")
        print(f"Date range: {market_data['trading_day'].min()} to {market_data['trading_day'].max()}")
        print(f"Available regimes: {sorted(regime_data['Regime'].unique())}")
        
        return market_data, regime_data, trading_day_col
    
    def extract_daily_features(self, market_data):
        """
        Extract statistical features for each trading day using successful TechnicalAnalysisFeatures
        
        Args:
            market_data: DataFrame with market data (history_spot_quote.csv format)
            
        Returns:
            pd.DataFrame: Daily features
        """
        print(f"Extracting daily technical analysis features using successful approach...")
        
        # Prepare data based on mode (same logic as before)
        if self.mode == 'daily':
            # Use 10:35 AM - 12:00 PM time window for consistency with successful forecaster
            filtered_data = market_data[
                (market_data['ms_of_day'] >= 38100000) &  # 10:35 AM
                (market_data['ms_of_day'] <= 43200000)    # 12:00 PM
            ].copy()
            print(f"Daily mode: Using 10:35 AM - 12:00 PM time window for consistency with successful forecaster")
            
        elif self.mode == 'intraday':
            # Mode 2: Use data before 38100000ms to predict 38160000-43200000ms regime
            filtered_data = market_data[
                (market_data['ms_of_day'] >= self.trading_start_ms) & 
                (market_data['ms_of_day'] <= self.intraday_cutoff_ms)
            ].copy()
            print(f"Intraday mode: Using data before {self.intraday_cutoff_time} to predict {self.intraday_start_time}-{self.intraday_end_time}")
            print(f"Intraday mode: Will include overnight gap features from previous day's close")
            
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'daily' or 'intraday'")
        
        print(f"Data shape after filtering: {filtered_data.shape}")
        
        # For intraday mode, we need to calculate overnight gaps
        # Get previous day closing prices for gap calculation
        previous_day_closes = {}
        if self.mode == 'intraday':
            print("Calculating previous day closing prices for overnight gap features...")
            
            # Get end-of-day data (around 4:00 PM = 57600000ms) for gap calculation
            eod_data = market_data[
                (market_data['ms_of_day'] >= 57000000) &  # 3:50 PM onwards
                (market_data['ms_of_day'] <= 57600000)    # 4:00 PM
            ].copy()
            
            for trading_day in sorted(eod_data['trading_day'].unique()):
                day_eod_data = eod_data[eod_data['trading_day'] == trading_day]
                if len(day_eod_data) > 0:
                    # Use last available price as closing price
                    closing_price = day_eod_data['mid'].iloc[-1]
                    previous_day_closes[trading_day] = closing_price
            
            print(f"Found closing prices for {len(previous_day_closes)} days")
        
        # Extract features using the successful TechnicalAnalysisFeatures approach
        daily_features = []
        
        for trading_day in sorted(filtered_data['trading_day'].unique()):
            day_data = filtered_data[filtered_data['trading_day'] == trading_day]
            
            if len(day_data) < 5:  # Skip days with insufficient data
                continue
            
            prices = day_data['mid'].values
            reference_price = prices[0]  # First price of the day (9:30 AM for intraday, 10:35 AM for daily)
            
            # Extract all features using successful approach
            features = TechnicalAnalysisFeatures.extract_features(prices, reference_price)
            
            # Add overnight gap features for intraday mode
            if self.mode == 'intraday':
                gap_features = self.calculate_overnight_gap_features(
                    trading_day, reference_price, previous_day_closes, filtered_data
                )
                features.update(gap_features)
            
            # Add metadata
            features['trading_day'] = trading_day
            features['num_observations'] = len(day_data)
            features['reference_price'] = reference_price
            
            daily_features.append(features)
        
        features_df = pd.DataFrame(daily_features)
        
        print(f"Extracted features shape: {features_df.shape}")
        feature_cols = [col for col in features_df.columns if col not in ['trading_day', 'num_observations', 'reference_price']]
        print(f"Number of features per day: {len(feature_cols)}")
        print(f"Feature list preview: {feature_cols[:10]}...")  # Show first 10 features
        
        return features_df
    
    def calculate_overnight_gap_features(self, trading_day, current_open_price, previous_day_closes, full_market_data):
        """
        Calculate overnight gap features for intraday mode
        
        Args:
            trading_day: Current trading day
            current_open_price: Opening price of current day (9:30 AM price)
            previous_day_closes: Dict of trading_day -> closing_price
            full_market_data: Full market data for additional calculations
            
        Returns:
            dict: Overnight gap features
        """
        gap_features = {}
        
        # Find previous trading day
        previous_trading_day = None
        sorted_days = sorted(previous_day_closes.keys())
        
        try:
            current_day_idx = sorted_days.index(trading_day)
            if current_day_idx > 0:
                previous_trading_day = sorted_days[current_day_idx - 1]
        except ValueError:
            # Current day not in the list
            for i, day in enumerate(sorted_days):
                if day < trading_day:
                    previous_trading_day = day
                else:
                    break
        
        if previous_trading_day is not None and previous_trading_day in previous_day_closes:
            previous_close = previous_day_closes[previous_trading_day]
            
            # Basic overnight gap features
            overnight_gap_abs = current_open_price - previous_close
            overnight_gap_pct = (overnight_gap_abs / previous_close) * 100 if previous_close != 0 else 0
            overnight_gap_direction = 1 if overnight_gap_abs > 0 else (-1 if overnight_gap_abs < 0 else 0)
            
            # Gap magnitude categories
            gap_abs_pct = abs(overnight_gap_pct)
            gap_small = 1 if gap_abs_pct < 0.5 else 0  # Less than 0.5%
            gap_medium = 1 if 0.5 <= gap_abs_pct < 2.0 else 0  # 0.5% to 2%
            gap_large = 1 if gap_abs_pct >= 2.0 else 0  # Greater than 2%
            
            # Gap relative to previous day's trading range
            prev_day_data = full_market_data[
                (full_market_data['trading_day'] == previous_trading_day) &
                (full_market_data['ms_of_day'] >= self.trading_start_ms) &
                (full_market_data['ms_of_day'] <= 57600000)  # Full trading day
            ]
            
            if len(prev_day_data) > 0:
                prev_day_high = prev_day_data['mid'].max()
                prev_day_low = prev_day_data['mid'].min()
                prev_day_range = prev_day_high - prev_day_low
                
                # Gap relative to previous day's range
                gap_vs_range = abs(overnight_gap_abs) / prev_day_range if prev_day_range > 0 else 0
                
                # Whether gap breaks previous day's range
                gap_above_high = 1 if current_open_price > prev_day_high else 0
                gap_below_low = 1 if current_open_price < prev_day_low else 0
                
                # Previous day's momentum features
                prev_day_prices = prev_day_data['mid'].values
                if len(prev_day_prices) > 1:
                    prev_day_open = prev_day_prices[0]
                    prev_day_momentum = (previous_close - prev_day_open) / prev_day_open * 100 if prev_day_open != 0 else 0
                    
                    # Gap continuation vs reversal
                    gap_continues_momentum = 1 if (prev_day_momentum > 0 and overnight_gap_abs > 0) or (prev_day_momentum < 0 and overnight_gap_abs < 0) else 0
                    gap_reverses_momentum = 1 if (prev_day_momentum > 0 and overnight_gap_abs < 0) or (prev_day_momentum < 0 and overnight_gap_abs > 0) else 0
                else:
                    prev_day_momentum = 0
                    gap_continues_momentum = 0
                    gap_reverses_momentum = 0
            else:
                gap_vs_range = 0
                gap_above_high = 0
                gap_below_low = 0
                prev_day_momentum = 0
                gap_continues_momentum = 0
                gap_reverses_momentum = 0
            
            gap_features = {
                'overnight_gap_abs': overnight_gap_abs,
                'overnight_gap_pct': overnight_gap_pct,
                'overnight_gap_direction': overnight_gap_direction,
                'gap_magnitude_small': gap_small,
                'gap_magnitude_medium': gap_medium,
                'gap_magnitude_large': gap_large,
                'gap_vs_prev_range': gap_vs_range,
                'gap_above_prev_high': gap_above_high,
                'gap_below_prev_low': gap_below_low,
                'prev_day_momentum': prev_day_momentum,
                'gap_continues_momentum': gap_continues_momentum,
                'gap_reverses_momentum': gap_reverses_momentum,
                'has_gap_data': 1
            }
        else:
            # No previous day data available - set default values
            gap_features = {
                'overnight_gap_abs': 0,
                'overnight_gap_pct': 0,
                'overnight_gap_direction': 0,
                'gap_magnitude_small': 0,
                'gap_magnitude_medium': 0,
                'gap_magnitude_large': 0,
                'gap_vs_prev_range': 0,
                'gap_above_prev_high': 0,
                'gap_below_prev_low': 0,
                'prev_day_momentum': 0,
                'gap_continues_momentum': 0,
                'gap_reverses_momentum': 0,
                'has_gap_data': 0
            }
        
        return gap_features
    
    def prepare_training_data(self, daily_features, regime_data, trading_day_col, train_start=None, 
                            train_end=None):
        """
        Prepare training data by merging features with regime labels
        
        Args:
            daily_features: DataFrame with daily features
            regime_data: DataFrame with regime assignments
            trading_day_col: Name of the trading day column in regime_data
            train_start: Start date for training (YYYYMMDD format)
            train_end: End date for training (YYYYMMDD format)
            
        Returns:
            tuple: (X_train, y_train, feature_names, trading_days)
        """
        print("Preparing training data...")
        
        # Deduplicate regime data to have only one row per trading day
        # Take the first occurrence of each TradingDay (they should all have the same regime anyway)
        regime_data_dedup = regime_data.drop_duplicates(subset=[trading_day_col], keep='first')
        print(f"Deduplicated regime data: {len(regime_data)} -> {len(regime_data_dedup)} rows")
        
        # Merge features with regime data using the correct column mapping
        merged_data = pd.merge(
            daily_features, 
            regime_data_dedup[[trading_day_col, 'Regime']], 
            left_on='trading_day', 
            right_on=trading_day_col, 
            how='inner'
        )
        
        # Filter by date range if specified
        if train_start is not None:
            merged_data = merged_data[merged_data['trading_day'] >= int(train_start)]
        if train_end is not None:
            merged_data = merged_data[merged_data['trading_day'] <= int(train_end)]
        
        print(f"Training data date range: {merged_data['trading_day'].min()} to {merged_data['trading_day'].max()}")
        print(f"Training data shape: {merged_data.shape}")
        print(f"Unique trading days: {merged_data['trading_day'].nunique()}")
        
        # Separate features and targets
        feature_columns = [col for col in merged_data.columns 
                         if col not in ['trading_day', trading_day_col, 'Regime', 'reference_time_ms', 
                                      'reference_price', 'num_observations']]
        
        X = merged_data[feature_columns].values
        y = merged_data['Regime'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y, feature_columns, merged_data['trading_day'].values
    
    def train_hmm_model(self, X, y, trading_days):
        """
        Train the Hidden Markov Model with improved regime mapping
        
        Args:
            X: Feature matrix
            y: Regime labels
            trading_days: Array of trading days
            
        Returns:
            dict: Training results and metrics
        """
        print("Training HMM model...")
        
        # Determine actual number of components from data
        unique_regimes = sorted(np.unique(y))
        print(f"Found regimes in training data: {unique_regimes}")
        print(f"Regime distribution in training: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        if self.auto_components:
            # Use the requested number of components (allows capturing regime sub-variations)
            self.actual_n_components = self.n_components
            print(f"Auto-detected {len(unique_regimes)} unique regimes, using {self.actual_n_components} components as requested: {unique_regimes}")
        else:
            self.actual_n_components = min(self.n_components, len(unique_regimes))
            print(f"Using {self.actual_n_components} components for {len(unique_regimes)} unique regimes: {unique_regimes}")
        
        # Check for regime data issues
        if len(unique_regimes) != max(unique_regimes) + 1:
            print("WARNING: Non-consecutive regime labels detected!")
            print(f"Expected regimes 0-{max(unique_regimes)}, but found: {unique_regimes}")
            
        # Check for severe class imbalance
        regime_counts = dict(zip(*np.unique(y, return_counts=True)))
        total_samples = len(y)
        minority_regimes = []
        
        for regime, count in regime_counts.items():
            percentage = count / total_samples * 100
            if percentage < 5.0:  # Less than 5% of data
                minority_regimes.append((regime, count, percentage))
        
        if minority_regimes:
            print("WARNING: Severely underrepresented regimes detected:")
            for regime, count, pct in minority_regimes:
                print(f"  Regime {regime}: {count} samples ({pct:.1f}%) - May cause poor prediction accuracy")
            print("Applying SMOTE to balance minority classes...")
            
            # Apply SMOTE to balance the classes
            try:
                from imblearn.over_sampling import SMOTE
                
                # Use a more conservative approach - boost minorities to reasonable levels
                # without making everything perfectly balanced
                min_reasonable_samples = 30  # Minimum samples for reliable learning
                
                target_samples = {}
                needs_balancing = False
                
                for regime in unique_regimes:
                    current_count = regime_counts[regime]
                    if current_count < min_reasonable_samples:
                        # Boost severely underrepresented regimes
                        target_samples[regime] = min_reasonable_samples
                        needs_balancing = True
                    else:
                        # Keep well-represented regimes as-is
                        target_samples[regime] = current_count
                
                if needs_balancing:
                    print(f"Target sample distribution: {target_samples}")
                    
                    # Apply SMOTE
                    k_neighbors = min(5, min(regime_counts.values()) - 1)
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                    
                    print(f"Data resampled: {X.shape[0]} -> {X_resampled.shape[0]} samples")
                    print(f"New distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
                    
                    X, y = X_resampled, y_resampled
                else:
                    print("No SMOTE needed - all classes have sufficient samples")
                
            except ImportError:
                print("Warning: imbalanced-learn not available, proceeding without SMOTE")
            except Exception as e:
                print(f"Warning: SMOTE failed ({e}), proceeding with original data")
        else:
            print("Class distribution is reasonably balanced.")
        
        # Ensure minimum data per regime for stable training
        regime_counts = dict(zip(*np.unique(y, return_counts=True)))
        min_samples_per_regime = 5
        small_regimes = [regime for regime, count in regime_counts.items() if count < min_samples_per_regime]
        if small_regimes:
            print(f"WARNING: Some regimes have very few samples: {[(regime, regime_counts[regime]) for regime in small_regimes]}")
            print("This may cause training instability.")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Select best features
        self.feature_selector.k = min(self.n_features, X_scaled.shape[1])  # Ensure we don't select more features than available
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = [self.feature_columns[i] for i in selected_indices]
        
        print(f"Selected {len(self.selected_features)} best features out of {len(self.feature_columns)} available:")
        feature_scores = [(self.feature_columns[i], self.feature_selector.scores_[i]) 
                         for i in selected_indices]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, score) in enumerate(feature_scores[:10]):  # Show top 10
            print(f"  {i+1:2d}. {feature}: {score:.3f}")
        if len(feature_scores) > 10:
            print(f"  ... and {len(feature_scores) - 10} more features")
        
        # Create regime mapping (ensure regimes are 0-indexed for HMM)
        self.regime_mapping = {regime: idx for idx, regime in enumerate(unique_regimes)}
        reverse_mapping = {idx: regime for regime, idx in self.regime_mapping.items()}
        
        # Convert regimes to 0-indexed
        y_indexed = np.array([self.regime_mapping[regime] for regime in y])
        
        # Initialize and train HMM with improved parameters using multi-covariance strategy
        # This is the successful approach from market_regime_hmm_forecaster.py
        best_model = None
        best_score = -np.inf
        best_cov_type = 'diag'
        
        # Try different covariance types (strategy from successful forecaster)
        covariance_types = ['diag', 'full', 'spherical']
        n_trials_per_type = 7  # Total ~20 trials like the successful version
        
        print(f"Training with multiple covariance types (successful strategy)...")
        
        for cov_type in covariance_types:
            print(f"  Testing covariance type: {cov_type}")
            
            for trial in range(n_trials_per_type):
                try:
                    model = hmm.GaussianHMM(
                        n_components=self.actual_n_components,
                        covariance_type=cov_type,
                        n_iter=200,  # Increased iterations for better convergence
                        random_state=self.random_state + trial,
                        tol=1e-6,    # Tighter tolerance
                        verbose=False
                    )
                    
                    model.fit(X_selected)
                    score = model.score(X_selected)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_cov_type = cov_type
                        
                except Exception as e:
                    continue
        
        if best_model is None:
            raise ValueError("All HMM training trials failed")
        
        print(f"  Best covariance type: {best_cov_type}")
        self.hmm_model = best_model
        log_likelihood = best_score
        
        print(f"Best model log-likelihood: {log_likelihood:.4f}")
        
        # Get state sequence using Viterbi algorithm
        predicted_states = self.hmm_model.predict(X_selected)
        
        # Create better state-to-regime mapping using maximum likelihood
        state_to_regime = {}
        regime_to_states = {regime: [] for regime in unique_regimes}
        
        # First pass: assign each state to most frequent regime
        for state in range(self.actual_n_components):
            state_mask = predicted_states == state
            if np.sum(state_mask) > 0:
                # Find most frequent actual regime for this state
                actual_regimes_for_state = y_indexed[state_mask]
                if len(actual_regimes_for_state) > 0:
                    most_frequent_regime = stats.mode(actual_regimes_for_state, keepdims=True)[0][0]
                    original_regime = reverse_mapping[most_frequent_regime]
                    state_to_regime[state] = original_regime
                    regime_to_states[original_regime].append(state)
                else:
                    state_to_regime[state] = unique_regimes[state % len(unique_regimes)]
            else:
                state_to_regime[state] = unique_regimes[state % len(unique_regimes)]
        
        # Check for unmapped regimes and reassign states if necessary
        unmapped_regimes = [regime for regime, states in regime_to_states.items() if not states]
        overmapped_regimes = [(regime, states) for regime, states in regime_to_states.items() if len(states) > 1]
        
        if unmapped_regimes and overmapped_regimes:
            print(f"Reassigning states: {len(unmapped_regimes)} unmapped regimes, {len(overmapped_regimes)} overmapped regimes")
            
            # Reassign excess states to unmapped regimes
            for unmapped_regime in unmapped_regimes:
                if overmapped_regimes:
                    # Take one state from the regime with most states
                    overmapped_regimes.sort(key=lambda x: len(x[1]), reverse=True)
                    source_regime, source_states = overmapped_regimes[0]
                    
                    if len(source_states) > 1:
                        # Move the state with lowest purity to unmapped regime
                        state_purities = []
                        for state in source_states:
                            state_mask = predicted_states == state
                            actual_regimes = y_indexed[state_mask]
                            if len(actual_regimes) > 0:
                                regime_counts = np.bincount(actual_regimes)
                                purity = np.max(regime_counts) / len(actual_regimes)
                                state_purities.append((state, purity))
                        
                        if state_purities:
                            # Move state with lowest purity
                            state_purities.sort(key=lambda x: x[1])
                            state_to_move = state_purities[0][0]
                            
                            # Update mappings
                            state_to_regime[state_to_move] = unmapped_regime
                            regime_to_states[source_regime].remove(state_to_move)
                            regime_to_states[unmapped_regime].append(state_to_move)
                            
                            # Update overmapped_regimes list
                            overmapped_regimes[0] = (source_regime, regime_to_states[source_regime])
        
        self.state_to_regime_mapping = state_to_regime
        
        # Convert predictions back to original regime labels
        y_pred = np.array([state_to_regime[state] for state in predicted_states])
        
        # Calculate accuracy
        accuracy = accuracy_score(y, y_pred)
        
        # Show state mapping with purity information
        print("State to Regime Mapping:")
        for state, regime in state_to_regime.items():
            state_count = np.sum(predicted_states == state)
            if state_count > 0:
                state_mask = predicted_states == state
                actual_regimes = y_indexed[state_mask]
                if len(actual_regimes) > 0:
                    # Calculate purity (how often this state correctly predicts its assigned regime)
                    mapped_regime_idx = self.regime_mapping[regime]
                    correct_predictions = np.sum(actual_regimes == mapped_regime_idx)
                    purity = correct_predictions / len(actual_regimes)
                    print(f"  State {state} -> Regime {regime} ({state_count} observations, {purity:.1%} purity)")
                else:
                    print(f"  State {state} -> Regime {regime} ({state_count} observations, no data)")
            else:
                print(f"  State {state} -> Regime {regime} (0 observations, unused)")
        
        # Check regime coverage
        mapped_regimes = set(state_to_regime.values())
        unmapped_regimes = set(unique_regimes) - mapped_regimes
        if unmapped_regimes:
            print(f"WARNING: Regimes {unmapped_regimes} are not mapped to any state!")
        
        # Check for multiple states per regime
        regime_states = {}
        for state, regime in state_to_regime.items():
            if regime not in regime_states:
                regime_states[regime] = []
            regime_states[regime].append(state)
        
        multi_state_regimes = {regime: states for regime, states in regime_states.items() if len(states) > 1}
        if multi_state_regimes:
            print(f"Multiple states per regime: {multi_state_regimes}")
        
        training_results = {
            'log_likelihood': log_likelihood,
            'training_accuracy': accuracy,
            'n_components': self.actual_n_components,
            'n_features_selected': len(self.selected_features),
            'regime_mapping': self.regime_mapping,
            'state_to_regime_mapping': state_to_regime,
            'selected_features': self.selected_features,
            'transition_matrix': self.hmm_model.transmat_.tolist(),
            'start_probabilities': self.hmm_model.startprob_.tolist(),
            'unique_regimes': unique_regimes
        }
        
        print(f"Training completed:")
        print(f"  Log-likelihood: {log_likelihood:.4f}")
        print(f"  Training accuracy: {accuracy:.4f}")
        print(f"  Components: {self.actual_n_components}")
        print(f"  Selected features: {len(self.selected_features)}")
        
        return training_results
    
    def predict_next_regime(self, X_test, current_regimes=None):
        """
        Predict next day's regime for test data
        
        Args:
            X_test: Test feature matrix
            current_regimes: Current regime labels (optional, for sequence prediction)
            
        Returns:
            tuple: (predictions, probabilities, confidence_scores)
        """
        if self.hmm_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        print("Making regime predictions...")
        
        # Scale and select features
        X_test_scaled = self.scaler.transform(X_test)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Get state probabilities for each observation
        # Note: predict_proba already returns normalized probabilities, not log probabilities
        probabilities = self.hmm_model.predict_proba(X_test_selected)
        
        # Get most likely state sequence
        predicted_states = self.hmm_model.predict(X_test_selected)
        
        # Convert states back to regime labels
        state_to_regime = {}
        # Use the training mapping if available
        if hasattr(self, 'state_to_regime_mapping'):
            state_to_regime = self.state_to_regime_mapping
        else:
            # Fallback: simple mapping
            unique_regimes = sorted(list(self.regime_mapping.keys()))
            for i in range(self.n_components):
                state_to_regime[i] = unique_regimes[i % len(unique_regimes)]
        
        predictions = np.array([state_to_regime[state] for state in predicted_states])
        
        # Calculate confidence scores (max probability)
        confidence_scores = np.max(probabilities, axis=1)
        
        print(f"Prediction completed for {len(predictions)} samples")
        print(f"Average confidence: {np.mean(confidence_scores):.4f}")
        
        return predictions, probabilities, confidence_scores
    
    def evaluate_model(self, X_test, y_test, trading_days_test):
        """
        Evaluate model performance on test data
        
        Args:
            X_test: Test feature matrix
            y_test: True regime labels
            trading_days_test: Test trading days
            
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating model performance...")
        
        predictions, probabilities, confidence_scores = self.predict_next_regime(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        # Classification report
        class_report = classification_report(y_test, predictions, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, predictions)
        unique_regimes = sorted(np.unique(np.concatenate([y_test, predictions])))
        
        evaluation_results = {
            'test_accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'regime_labels': unique_regimes,
            'average_confidence': np.mean(confidence_scores),
            'predictions': predictions.tolist(),
            'true_labels': y_test.tolist(),
            'confidence_scores': confidence_scores.tolist(),
            'trading_days': trading_days_test.tolist()
        }
        
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Average confidence: {np.mean(confidence_scores):.4f}")
        
        # Analyze regime predictions vs actual
        from collections import Counter
        actual_counts = Counter(y_test)
        predicted_counts = Counter(predictions)
        
        print(f"\nActual regime distribution in test set:")
        for regime in sorted(actual_counts.keys()):
            print(f"  Regime {regime}: {actual_counts[regime]} samples ({actual_counts[regime]/len(y_test)*100:.1f}%)")
            
        print(f"\nPredicted regime distribution in test set:")
        for regime in sorted(predicted_counts.keys()):
            print(f"  Regime {regime}: {predicted_counts[regime]} samples ({predicted_counts[regime]/len(predictions)*100:.1f}%)")
        
        # Analyze transitions
        actual_transitions = []
        predicted_transitions = []
        for i in range(1, len(y_test)):
            actual_transitions.append((y_test[i-1], y_test[i]))
            predicted_transitions.append((predictions[i-1], predictions[i]))
        
        print(f"\nTop 5 most common actual transitions:")
        actual_trans_counts = Counter(actual_transitions)
        for trans, count in actual_trans_counts.most_common(5):
            print(f"  {trans[0]} → {trans[1]}: {count} times")
            
        print(f"\nTop 5 most common predicted transitions:")
        pred_trans_counts = Counter(predicted_transitions)
        for trans, count in pred_trans_counts.most_common(5):
            print(f"  {trans[0]} → {trans[1]}: {count} times")
        
        return evaluation_results
    
    def save_model(self, output_dir, model_name='hmm_regime_forecaster'):
        """
        Save the trained model and components
        
        Args:
            output_dir: Directory to save model files
            model_name: Base name for model files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model components
        model_path = os.path.join(output_dir, f'{model_name}.pkl')
        
        model_data = {
            'hmm_model': self.hmm_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'regime_mapping': self.regime_mapping,
            'state_to_regime_mapping': getattr(self, 'state_to_regime_mapping', {}),
            'n_components': self.n_components,
            'n_features': self.n_features,
            'covariance_type': self.covariance_type,
            'mode': self.mode,
            'trading_start_ms': self.trading_start_ms,
            'intraday_cutoff_ms': self.intraday_cutoff_ms,
            'intraday_start_ms': self.intraday_start_ms,
            'intraday_end_ms': self.intraday_end_ms
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path):
        """
        Load a previously trained model
        
        Args:
            model_path: Path to saved model file
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.hmm_model = model_data['hmm_model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.selected_features = model_data['selected_features']
        self.regime_mapping = model_data['regime_mapping']
        self.state_to_regime_mapping = model_data.get('state_to_regime_mapping', {})
        self.n_components = model_data['n_components']
        self.n_features = model_data['n_features']
        self.covariance_type = model_data['covariance_type']
        self.mode = model_data.get('mode', 'daily')
        self.trading_start_ms = model_data.get('trading_start_ms', 34200000)
        self.intraday_cutoff_ms = model_data.get('intraday_cutoff_ms', 38100000)
        self.intraday_start_ms = model_data.get('intraday_start_ms', 38160000)
        self.intraday_end_ms = model_data.get('intraday_end_ms', 43200000)
        
        print(f"Model loaded from: {model_path}")

def main():
    """Main execution function"""
    # Get script directory and set up default paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Default file paths
    default_data_file = project_root / 'data' / 'history_spot_quote.csv'
    default_regime_file = project_root / 'market_regime' / 'daily_regime_assignments.csv'
    default_output_dir = project_root / 'market_regime_forecast'
    
    parser = argparse.ArgumentParser(description='HMM Market Regime Forecasting')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='daily', choices=['daily', 'intraday'],
                      help='Forecasting mode: daily (predict next day) or intraday (predict same day window)')
    
    # Data paths with defaults
    parser.add_argument('--data_file', type=str, default=str(default_data_file),
                      help=f'Path to history_spot_quote.csv file (default: {default_data_file})')
    parser.add_argument('--regime_file', type=str, default=str(default_regime_file),
                      help=f'Path to regime assignments CSV file (default: {default_regime_file})')
    parser.add_argument('--output_dir', type=str, default=str(default_output_dir),
                      help=f'Output directory for results (default: {default_output_dir})')
    
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
                      help='Number of hidden states in HMM (max if auto_components=True)')
    parser.add_argument('--n_features', type=int, default=25,
                      help='Number of top features to select')
    parser.add_argument('--covariance_type', type=str, default='full',
                      choices=['full', 'diag', 'tied', 'spherical'],
                      help='Type of covariance parameters')
    parser.add_argument('--n_iter', type=int, default=200,
                      help='Maximum iterations for HMM training')
    parser.add_argument('--random_state', type=int, default=84,
                      help='Random seed for reproducibility')
    parser.add_argument('--auto_components', type=bool, default=True,
                      help='Automatically determine number of components from training data')
    
    # Trading parameters
    parser.add_argument('--trading_start', type=str, default='09:30',
                      help='Daily trading start time (HH:MM)')
    parser.add_argument('--intraday_cutoff', type=str, default='10:35',
                      help='Intraday mode cutoff time (HH:MM)')
    parser.add_argument('--intraday_start', type=str, default='10:36',
                      help='Intraday prediction window start (HH:MM)')
    parser.add_argument('--intraday_end', type=str, default='12:00',
                      help='Intraday prediction window end (HH:MM)')
    
    args = parser.parse_args()
    
    # Verify required files exist
    if not Path(args.data_file).exists():
        print(f"ERROR: Data file not found: {args.data_file}")
        print("Please ensure history_spot_quote.csv exists in the data/ directory")
        sys.exit(1)
    
    if not Path(args.regime_file).exists():
        print(f"ERROR: Regime file not found: {args.regime_file}")
        print("Please run the GMM regime clustering first to generate daily_regime_assignments.csv")
        sys.exit(1)
    
    print("="*80)
    print(f"HMM MARKET REGIME FORECASTING - {args.mode.upper()} MODE")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Data file: {args.data_file}")
    print(f"Regime file: {args.regime_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"HMM components: {args.n_components}")
    print(f"Selected features: {args.n_features}")
    if args.mode == 'daily':
        print(f"Daily mode: Using complete trading day data from {args.trading_start}")
        print(f"Predicting: Next trading day's regime")
    else:
        print(f"Intraday mode: Using data before {args.intraday_cutoff}")
        print(f"Predicting: {args.intraday_start} - {args.intraday_end} regime")
    print()
    
    # Initialize forecaster
    forecaster = HMMRegimeForecaster(
        mode=args.mode,
        n_components=args.n_components,
        n_features=args.n_features,
        covariance_type=args.covariance_type,
        n_iter=args.n_iter,
        random_state=args.random_state,
        auto_components=args.auto_components,
        trading_start_time=args.trading_start,
        intraday_cutoff_time=args.intraday_cutoff,
        intraday_start_time=args.intraday_start,
        intraday_end_time=args.intraday_end
    )
    
    try:
        # Load data (now uses history_spot_quote.csv format)
        market_data, regime_data, trading_day_col = forecaster.load_and_prepare_data(
            args.data_file, args.regime_file
        )
        
        # Extract daily features (mode-specific)
        daily_features = forecaster.extract_daily_features(market_data)
        
        # Prepare training data
        X_train, y_train, feature_names, train_days = forecaster.prepare_training_data(
            daily_features, regime_data, trading_day_col,
            train_start=args.train_start,
            train_end=args.train_end
        )
        
        # Store feature names for later use
        forecaster.feature_columns = feature_names
        
        # Train model
        training_results = forecaster.train_hmm_model(X_train, y_train, train_days)
        
        # Prepare test data
        test_data_available = False
        predictions_df = None
        evaluation_results = None
        
        if args.test_start is not None or args.test_end is not None:
            # Use explicitly provided test dates
            X_test, y_test, _, test_days = forecaster.prepare_training_data(
                daily_features, regime_data, trading_day_col,
                train_start=args.test_start,
                train_end=args.test_end
            )
            test_data_available = True
            
        else:
            # Auto-generate test data for period after training
            print("No test dates specified. Using all available data after training period for predictions...")
            
            # Determine test start date (day after train_end if specified, otherwise use all data)
            if args.train_end is not None:
                # Convert train_end to datetime and add 1 day
                from datetime import datetime, timedelta
                train_end_date = datetime.strptime(args.train_end, '%Y%m%d')
                test_start_date = train_end_date + timedelta(days=1)
                test_start_auto = test_start_date.strftime('%Y%m%d')
                print(f"Auto-generated test period starts from: {test_start_auto}")
                
                # Get all available dates after training
                available_test_data = daily_features.merge(
                    regime_data[[trading_day_col, 'Regime']], 
                    left_on='trading_day', 
                    right_on=trading_day_col, 
                    how='inner'
                )
                available_test_data = available_test_data[available_test_data['trading_day'] >= int(test_start_auto)]
                
                if len(available_test_data) > 0:
                    X_test, y_test, _, test_days = forecaster.prepare_training_data(
                        daily_features, regime_data, trading_day_col,
                        train_start=test_start_auto,
                        train_end=None
                    )
                    test_data_available = True
                    print(f"Auto-generated test data: {len(test_days)} days from {test_days.min()} to {test_days.max()}")
                else:
                    print("No data available after training period for testing.")
            else:
                print("No training end date specified. Skipping test predictions.")
        
        if test_data_available and len(test_days) > 0:
            # Evaluate model
            evaluation_results = forecaster.evaluate_model(X_test, y_test, test_days)
            
            # Create detailed predictions DataFrame
            predictions, probabilities, confidence_scores = forecaster.predict_next_regime(X_test)
            
            predictions_df = pd.DataFrame({
                'TradingDay': test_days,
                'True_Regime': y_test,
                'Predicted_Regime': predictions,
                'Confidence': confidence_scores
            })
            
            # Add probability columns for each regime
            unique_regimes = sorted(np.unique(np.concatenate([y_train, y_test])))
            for i, regime in enumerate(unique_regimes):
                if i < probabilities.shape[1]:
                    predictions_df[f'Prob_Regime_{regime}'] = probabilities[:, i]
        else:
            print("No valid test data available. Predictions will not be generated.")
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
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
        
        # Save training results
        training_results_clean = convert_numpy_types(training_results)
        with open(os.path.join(args.output_dir, 'hmm_training_results.json'), 'w') as f:
            json.dump(training_results_clean, f, indent=2)
        
        # Save evaluation results if available
        if evaluation_results is not None:
            evaluation_results_clean = convert_numpy_types(evaluation_results)
            with open(os.path.join(args.output_dir, 'hmm_evaluation_results.json'), 'w') as f:
                json.dump(evaluation_results_clean, f, indent=2)
        
        # Save predictions if available
        if predictions_df is not None and len(predictions_df) > 0:
            predictions_df.to_csv(
                os.path.join(args.output_dir, 'hmm_regime_predictions.csv'),
                index=False
            )
            print(f"Predictions saved for {len(predictions_df)} days")
        else:
            print("No predictions generated - no valid test data available")
        
        # Save model
        forecaster.save_model(args.output_dir, 'hmm_regime_forecaster')
        
        # Save daily features for future use
        daily_features.to_csv(
            os.path.join(args.output_dir, 'daily_features.csv'),
            index=False
        )
        
        print("\n" + "="*80)
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {args.output_dir}")
        print(f"Training accuracy: {training_results['training_accuracy']:.4f}")
        if evaluation_results is not None:
            print(f"Test accuracy: {evaluation_results['test_accuracy']:.4f}")
            print(f"Average confidence: {evaluation_results['average_confidence']:.4f}")
        else:
            print("No test evaluation performed - no valid test data available")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
