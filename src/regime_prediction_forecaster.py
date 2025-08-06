#!/usr/bin/env python3
"""
Regime Prediction and LSTM Model Selection Forecaster

This script:
1. Analyzes market data to predict future regimes using HMM or naive models
2. Selects best upside and downside models based on regime predictions
3. Makes minute-by-minute regression forecasts using selected models
4. Prevents using models trained on the prediction period

Usage:
    python regime_prediction_forecaster.py [--period daily|weekly] [--method hmm|naive] [--start_date YYYYMMDD] [--end_date YYYYMMDD]

Parameters:
    --period: Forecasting granularity
        - 'daily': Make regime predictions daily and forecast each day's minute data
        - 'weekly': Make regime predictions weekly and forecast entire week's minute data
    --method: Regime prediction approach
        - 'hmm': Use Hidden Markov Model with market features
        - 'naive': Use most common recent regime
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import argparse
from datetime import datetime, timedelta
import warnings
from hmmlearn import hmm
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

# Get project paths
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
data_dir = project_root / 'data'
models_dir = project_root / 'models'
regime_dir = project_root / 'regime_analysis'
results_dir = project_root / 'test_results'
output_dir = project_root / 'forecasts'

# Ensure directories exist
output_dir.mkdir(exist_ok=True)

class RegimePredictionForecaster:
    """
    Predicts market regimes and uses best models for forecasting
    """
    
    def __init__(self):
        # Data storage
        self.trading_data = None
        self.regime_assignments = None
        self.model_results = None
        self.model_log = None
        
        # Regime prediction models
        self.hmm_model = None
        self.regime_history = None
        
        print("Regime Prediction Forecaster initialized")
        
    def load_data(self):
        """Load all required data files"""
        print("Loading data files...")
        
        # Load trading data
        data_file = data_dir / 'trainingData.csv'
        if not data_file.exists():
            raise FileNotFoundError(f"Trading data file not found: {data_file}")
        self.trading_data = pd.read_csv(data_file)
        print(f"Loaded trading data: {len(self.trading_data):,} rows")
        
        # Load regime assignments
        regime_file = regime_dir / 'regime_assignments.csv'
        if not regime_file.exists():
            raise FileNotFoundError(f"Regime assignments file not found: {regime_file}")
        self.regime_assignments = pd.read_csv(regime_file)
        print(f"Loaded regime assignments: {len(self.regime_assignments):,} rows")
        
        # Load model test results (preserve model_id as string)
        # Try to find the most recent model test results file
        model_results_files = [
            results_dir / 'model_regime_test_results_1_5.csv',  # Test file
            results_dir / 'model_regime_test_results_1_425.csv'  # Full file
        ]
        
        model_results_file = None
        for file_path in model_results_files:
            if file_path.exists():
                model_results_file = file_path
                break
        
        if model_results_file is None:
            raise FileNotFoundError(f"No model test results file found. Tried: {[str(f) for f in model_results_files]}")
        
        self.model_results = pd.read_csv(model_results_file, dtype={'model_id': str})
        print(f"Loaded model test results from {model_results_file.name}: {len(self.model_results):,} rows")
        
        # Load model log
        model_log_file = models_dir / 'model_log.csv'
        if not model_log_file.exists():
            raise FileNotFoundError(f"Model log file not found: {model_log_file}")
        self.model_log = pd.read_csv(model_log_file, dtype={'model_id': str})
        print(f"Loaded model log: {len(self.model_log):,} rows")
        
    def load_best_models_data(self, model_selection):
        """Load daily or weekly best models data"""
        print(f"Loading {model_selection} best models data...")
        
        if model_selection == 'daily':
            # Try to find daily best models file
            daily_files = list(results_dir.glob('daily_best_models_*.csv'))
            if not daily_files:
                raise FileNotFoundError(f"No daily best models files found in {results_dir}")
            
            # Use the most recent daily file (by modification time)
            daily_file = max(daily_files, key=lambda f: f.stat().st_mtime)
            self.daily_best_models = pd.read_csv(daily_file, dtype={'best_upside_model_id': str, 'best_downside_model_id': str})
            print(f"Loaded daily best models from {daily_file.name}: {len(self.daily_best_models):,} rows")
            
        elif model_selection == 'weekly':
            # Try to find weekly best models file
            weekly_files = list(results_dir.glob('weekly_best_models_*.csv'))
            if not weekly_files:
                raise FileNotFoundError(f"No weekly best models files found in {results_dir}")
            
            # Use the most recent weekly file (by modification time)
            weekly_file = max(weekly_files, key=lambda f: f.stat().st_mtime)
            self.weekly_best_models = pd.read_csv(weekly_file, dtype={'best_upside_model_id': str, 'best_downside_model_id': str})
            print(f"Loaded weekly best models from {weekly_file.name}: {len(self.weekly_best_models):,} rows")
        
    def prepare_regime_history(self):
        """Prepare historical regime sequence for prediction models"""
        print("Preparing regime history...")
        
        # Get unique weeks with their regimes
        weekly_regimes = self.regime_assignments.groupby('TradingDay')['Regime'].first().reset_index()
        weekly_regimes = weekly_regimes.sort_values('TradingDay')
        
        # Create regime sequence
        self.regime_history = weekly_regimes['Regime'].values
        self.trading_days = weekly_regimes['TradingDay'].values
        
        print(f"Prepared regime history: {len(self.regime_history)} periods")
        print(f"Regime distribution: {Counter(self.regime_history)}")
        
    def train_hmm_model(self, n_components=None):
        """Train HMM model for regime prediction using market features as observations"""
        print("Training HMM model for regime prediction...")
        
        try:
            from hmmlearn import hmm
            
            # Prepare training data with market features as observations
            print("Preparing HMM training data with market features...")
            
            # Get daily aggregated market features from trading data
            daily_features = self.trading_data.groupby('TradingDay').agg({
                'Mid': ['mean', 'std', 'min', 'max'],  # Price statistics
                'Bid': ['mean', 'std'],  # Bid statistics
                'Ask': ['mean', 'std'],  # Ask statistics
                'ROC_05min:ROC': ['mean', 'std'],  # 5-min ROC statistics
                'PriceChangesStat_05min:Mean': 'mean',  # Price change statistics
                'PriceChangesStat_05min:+Std': 'mean',
                'PriceChangesStat_05min:-Std': 'mean'
            }).reset_index()
            
            # Flatten column names
            daily_features.columns = ['TradingDay'] + [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in daily_features.columns[1:]]
            
            # Calculate additional features
            daily_features['price_range'] = daily_features['Mid_max'] - daily_features['Mid_min']
            daily_features['price_volatility'] = daily_features['Mid_std']
            daily_features['bid_ask_spread'] = daily_features['Ask_mean'] - daily_features['Bid_mean']
            
            # Merge with regime assignments
            regime_data = pd.merge(
                daily_features, 
                self.regime_assignments.groupby('TradingDay')['Regime'].first().reset_index(),
                on='TradingDay', 
                how='inner'
            ).sort_values('TradingDay').dropna()
            
            if len(regime_data) < 100:
                print(f"Insufficient data for HMM training: {len(regime_data)} days")
                self.hmm_model = None
                return
            
            # Get unique regimes from the actual data
            regimes = regime_data['Regime'].values
            unique_regimes = sorted(np.unique(regimes))
            
            # Set number of components explicitly to number of unique regimes
            if n_components is None:
                n_components = len(unique_regimes)
            
            print(f"Training HMM with {n_components} components for regimes: {unique_regimes}")
            
            # Select meaningful features for HMM
            feature_cols = [
                'Mid_mean', 'Mid_std', 'price_range', 'price_volatility', 
                'bid_ask_spread', 'ROC_05min:ROC_mean', 'PriceChangesStat_05min:Mean_'
            ]
            
            # Filter available columns
            available_cols = [col for col in feature_cols if col in regime_data.columns]
            
            if len(available_cols) == 0:
                print("No valid features available for HMM training")
                self.hmm_model = None
                return
            
            observations = regime_data[available_cols].values
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            self.hmm_scaler = StandardScaler()
            observations_scaled = self.hmm_scaler.fit_transform(observations)
            
            # Train Gaussian HMM with market features as observations
            self.hmm_model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type="full",
                n_iter=200,
                random_state=42,
                verbose=False
            )
            
            # Fit the model with observed market features
            self.hmm_model.fit(observations_scaled)
            
            # Get predicted states for training data to create proper state-to-regime mapping
            predicted_states = self.hmm_model.predict(observations_scaled)
            
            # Create state-to-regime mapping using optimal assignment
            state_to_regime_map = {}
            
            # Calculate overall regime frequencies for comparison
            regime_frequencies = {regime: np.sum(regimes == regime) / len(regimes) 
                                for regime in unique_regimes}
            
            # Build enrichment matrix
            enrichment_matrix = np.zeros((n_components, len(unique_regimes)))
            
            for state in range(n_components):
                state_mask = predicted_states == state
                if np.any(state_mask):
                    regimes_in_state = regimes[state_mask]
                    state_size = len(regimes_in_state)
                    
                    for regime_idx, regime in enumerate(unique_regimes):
                        regime_count_in_state = np.sum(regimes_in_state == regime)
                        state_frequency = regime_count_in_state / state_size
                        overall_frequency = regime_frequencies[regime]
                        
                        # Score based on relative over-representation
                        if overall_frequency > 0:
                            enrichment_score = state_frequency / overall_frequency
                        else:
                            enrichment_score = 0
                        enrichment_matrix[state, regime_idx] = enrichment_score
            
            # Use simple greedy assignment ensuring each regime gets a state
            remaining_states = set(range(n_components))
            remaining_regimes = set(range(len(unique_regimes)))
            
            # Create all possible assignments sorted by enrichment score
            assignments = []
            for state in range(n_components):
                for regime_idx in range(len(unique_regimes)):
                    enrichment = enrichment_matrix[state, regime_idx]
                    assignments.append((enrichment, state, regime_idx))
            
            assignments.sort(reverse=True)  # Highest enrichment first
            
            # Assign each regime to best available state, ensuring coverage
            for enrichment, state, regime_idx in assignments:
                regime = unique_regimes[regime_idx]
                if state in remaining_states and regime_idx in remaining_regimes:
                    state_to_regime_map[state] = regime
                    remaining_states.remove(state)
                    remaining_regimes.remove(regime_idx)
                    
                    # Print assignment details
                    state_mask = predicted_states == state
                    if np.any(state_mask):
                        regimes_in_state = regimes[state_mask]
                        regime_count = np.sum(regimes_in_state == regime)
                        state_size = len(regimes_in_state)
                        state_freq = regime_count / state_size * 100
                        overall_freq = regime_frequencies[regime] * 100
                        print(f"  State {state} -> Regime {regime}: {regime_count}/{state_size} = {state_freq:.1f}% (vs {overall_freq:.1f}% overall, enrichment: {enrichment:.2f})")
                        regimes_in_state = regimes[state_mask]
                        regime_count = np.sum(regimes_in_state == regime)
                        state_size = len(regimes_in_state)
                        state_freq = regime_count / state_size * 100
                        overall_freq = regime_frequencies[regime] * 100
                        print(f"  State {state} -> Regime {regime}: {regime_count}/{state_size} = {state_freq:.1f}% (vs {overall_freq:.1f}% overall, enrichment: {enrichment:.2f})")
                    
                    if len(remaining_regimes) == 0:
                        break
            
            # Handle any remaining unmapped states
            for state in remaining_states:
                # Find the best regime for this state from what's available  
                best_regime = None
                best_score = -1
                for regime_idx, regime in enumerate(unique_regimes):
                    if regime not in state_to_regime_map.values():
                        score = enrichment_matrix[state, regime_idx] if state < len(enrichment_matrix) else 0
                        if score > best_score:
                            best_score = score
                            best_regime = regime
                            best_score = score
                            best_regime = regime
                
                if best_regime is not None:
                    state_to_regime_map[state] = best_regime
                    print(f"  State {state} -> Regime {best_regime}: (secondary assignment, enrichment: {best_score:.2f})")
                else:
                    # Fallback to first available regime
                    available_regimes = set(unique_regimes) - set(state_to_regime_map.values())
                    if available_regimes:
                        fallback_regime = sorted(available_regimes)[0]
                        state_to_regime_map[state] = fallback_regime
                        print(f"  State {state} -> Regime {fallback_regime}: (fallback assignment)")
            
            # Verify all regimes are covered
            mapped_regimes = set(state_to_regime_map.values())
            missing_regimes = set(unique_regimes) - mapped_regimes
            if missing_regimes:
                print(f"  WARNING: Missing regimes {missing_regimes} in state mapping!")
                # Force assignment of missing regimes
                for missing_regime in missing_regimes:
                    # Find state with lowest current assignment score and reassign
                    worst_state = None
                    worst_score = float('inf')
                    for state, assigned_regime in state_to_regime_map.items():
                        regime_idx = list(unique_regimes).index(assigned_regime)
                        score = enrichment_matrix[state, regime_idx]
                        if score < worst_score:
                            worst_score = score
                            worst_state = state
                            worst_score = score
                            worst_state = state
                    
                    if worst_state is not None:
                        old_regime = state_to_regime_map[worst_state]
                        state_to_regime_map[worst_state] = missing_regime
                        print(f"  FORCED: State {worst_state} -> Regime {missing_regime} (was Regime {old_regime})")
                        
            print(f"Final mapping covers regimes: {sorted(state_to_regime_map.values())}")
            
            # Update mapping to use proper state-to-regime correspondence
            self.reverse_mapping = state_to_regime_map
            self.regime_mapping = {v: k for k, v in state_to_regime_map.items()}
            
            # Store training data for reference
            self.hmm_training_regimes = regimes
            self.hmm_training_dates = regime_data['TradingDay'].values
            self.hmm_feature_cols = available_cols
            
            # Evaluate model fit and state mapping
            log_likelihood = self.hmm_model.score(observations_scaled)
            print(f"HMM model trained with log-likelihood: {log_likelihood:.3f}")
            print(f"Using {len(available_cols)} market features: {available_cols}")
            print(f"Training data: {len(regime_data)} days")
            
            # Print state-to-regime mapping for verification
            print("State-to-regime mapping (enrichment-based):")
            for state, regime in state_to_regime_map.items():
                state_regime_count = np.sum((predicted_states == state) & (regimes == regime))
                state_total = np.sum(predicted_states == state)
                regime_idx = list(unique_regimes).index(regime)
                enrichment = enrichment_matrix[state, regime_idx] if regime_idx < enrichment_matrix.shape[1] else 0
                print(f"  State {state} -> Regime {regime} (enrichment: {enrichment:.2f}x, coverage: {state_regime_count}/{state_total})")
            
            print(f"Feature columns saved: {hasattr(self, 'hmm_feature_cols') and self.hmm_feature_cols is not None}")
            print(f"Reverse mapping saved: {hasattr(self, 'reverse_mapping') and self.reverse_mapping is not None}")
            print("HMM training completed successfully")
            
        except Exception as e:
            print(f"HMM training failed: {str(e)}")
            print("Falling back to naive prediction method")
            self.hmm_model = None
        
    def predict_regime_hmm(self, current_date, lookback_periods=10):
        """Predict next regime using HMM with market features"""
        if self.hmm_model is None:
            return self.predict_regime_naive(current_date)
        
        try:
            current_date_int = int(current_date)
            
            # Get recent market data for the prediction date and lookback period
            end_date = current_date_int
            start_date = end_date - lookback_periods
            
            recent_data = self.trading_data[
                (self.trading_data['TradingDay'] >= start_date) & 
                (self.trading_data['TradingDay'] < end_date)
            ]
            
            if len(recent_data) == 0:
                print(f"No recent market data for HMM prediction, falling back to naive")
                return self.predict_regime_naive(current_date)
            
            # Aggregate daily features (same as in training)
            daily_features = recent_data.groupby('TradingDay').agg({
                'Mid': ['mean', 'std', 'min', 'max'],
                'Bid': ['mean', 'std'],
                'Ask': ['mean', 'std'],
                'ROC_05min:ROC': ['mean', 'std'],
                'PriceChangesStat_05min:Mean': 'mean',
                'PriceChangesStat_05min:+Std': 'mean',
                'PriceChangesStat_05min:-Std': 'mean'
            }).reset_index()
            
            # Flatten column names
            daily_features.columns = ['TradingDay'] + [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in daily_features.columns[1:]]
            
            # Calculate additional features
            daily_features['price_range'] = daily_features['Mid_max'] - daily_features['Mid_min']
            daily_features['price_volatility'] = daily_features['Mid_std']
            daily_features['bid_ask_spread'] = daily_features['Ask_mean'] - daily_features['Bid_mean']
            
            # Use same feature columns as training
            if not hasattr(self, 'hmm_feature_cols'):
                print("HMM feature columns not available, falling back to naive")
                return self.predict_regime_naive(current_date)
            
            available_cols = [col for col in self.hmm_feature_cols if col in daily_features.columns]
            
            if len(available_cols) == 0:
                print("No valid features for HMM prediction, falling back to naive")
                return self.predict_regime_naive(current_date)
            
            # Get recent observations and remove NaN
            observations = daily_features[available_cols].dropna().values
            
            if len(observations) == 0:
                print("No valid observations for HMM prediction, falling back to naive")
                return self.predict_regime_naive(current_date)
            
            # Scale using the same scaler from training
            if hasattr(self, 'hmm_scaler'):
                observations_scaled = self.hmm_scaler.transform(observations)
            else:
                print("HMM scaler not available, falling back to naive")
                return self.predict_regime_naive(current_date)
            
            # Predict hidden states for recent observations
            hidden_states = self.hmm_model.predict(observations_scaled)
            
            # Use the most recent predicted state as current state
            current_state = hidden_states[-1]
            
            # For better regime diversity, combine transition probabilities with uniform sampling
            # This helps ensure all regimes have a chance to be predicted
            transition_probs = self.hmm_model.transmat_[current_state]
            
            # Add small uniform noise to prevent getting stuck in deterministic states
            uniform_noise = 0.1  # 10% uniform contribution
            num_states = len(transition_probs)
            uniform_probs = np.ones(num_states) / num_states
            
            # Combine transition probabilities with uniform distribution
            combined_probs = (1 - uniform_noise) * transition_probs + uniform_noise * uniform_probs
            
            # Sample from the combined probability distribution
            predicted_state = np.random.choice(len(combined_probs), p=combined_probs)
            
            # Map back to regime
            if predicted_state in self.reverse_mapping:
                predicted_regime = self.reverse_mapping[predicted_state]
                return int(predicted_regime)
            else:
                print(f"Invalid state {predicted_state}, falling back to naive")
                return self.predict_regime_naive(current_date)
                
        except Exception as e:
            print(f"HMM prediction failed: {str(e)}, falling back to naive method")
            return self.predict_regime_naive(current_date)
            
    def predict_regime_naive(self, current_date, lookback_periods=5):
        """Predict next regime using naive approach (most common recent regime)"""
        # Find current position in regime history
        current_idx = np.where(self.trading_days <= int(current_date))[0]
        
        if len(current_idx) == 0:
            # If before historical data, return most common regime overall
            return int(Counter(self.regime_history).most_common(1)[0][0])
        
        current_idx = current_idx[-1]
        
        # Get recent regime history
        start_idx = max(0, current_idx - lookback_periods + 1)
        recent_regimes = self.regime_history[start_idx:current_idx + 1]
        
        if len(recent_regimes) == 0:
            return int(Counter(self.regime_history).most_common(1)[0][0])
        
        # Return most common recent regime
        return int(Counter(recent_regimes).most_common(1)[0][0])
    
    def get_best_models_for_regime(self, regime, prediction_date):
        """Get best upside and downside models for a given regime using pre-calculated rankings"""
        prediction_date_int = int(prediction_date)
        
        # Filter models that don't overlap with prediction date
        valid_models = []
        for _, model_row in self.model_log.iterrows():
            train_from = int(model_row['train_from'])
            train_to = int(model_row['train_to'])
            
            # Exclude models if prediction date falls within training period
            if not (train_from <= prediction_date_int <= train_to):
                # Model IDs in model_log are already 5-digit strings like "00001"
                model_id_str = str(model_row['model_id']).strip()
                valid_models.append(model_id_str)
        
        if len(valid_models) == 0:
            print(f"Warning: No valid models found for regime {regime} on date {prediction_date}")
            return None, None
        
        print(f"Found {len(valid_models)} valid models for regime {regime} on date {prediction_date}")
        print(f"Valid models: {valid_models[:10]}...")  # Show first 10 for debugging
        
        # Filter model_results for the specific regime and valid models
        regime_results = self.model_results[
            (self.model_results['regime'] == regime) &
            (self.model_results['model_id'].isin(valid_models))
        ]
        
        if len(regime_results) == 0:
            print(f"Warning: No models found in model_results for regime {regime}")
            print(f"Available regimes: {self.model_results['regime'].unique()}")
            return None, None
        
        # Check if the ranking columns exist
        if 'model_regime_upside_rank' not in regime_results.columns or 'model_regime_downside_rank' not in regime_results.columns:
            print(f"Warning: Ranking columns not found in model_results")
            print(f"Available columns: {list(regime_results.columns)}")
            return None, None
        
        # Find best upside model (lowest rank number = best)
        best_upside_row = regime_results.loc[regime_results['model_regime_upside_rank'].idxmin()]
        best_upside_rank = int(best_upside_row['model_regime_upside_rank'])
        
        # Find best downside model (lowest rank number = best)
        best_downside_row = regime_results.loc[regime_results['model_regime_downside_rank'].idxmin()]
        best_downside_rank = int(best_downside_row['model_regime_downside_rank'])
        
        # Create result objects with model info and rankings
        best_upside = {
            'model_id': int(best_upside_row['model_id']),
            'model_regime_rank': best_upside_rank,
            'regime': regime
        }
        
        best_downside = {
            'model_id': int(best_downside_row['model_id']),
            'model_regime_rank': best_downside_rank,
            'regime': regime
        }
        
        return best_upside, best_downside
    
    def get_best_models_daily(self, regime, prediction_date):
        """Get best models based on yesterday's performance for the given regime"""
        prediction_date_int = int(prediction_date)
        
        # Get all available trading days for this regime, sorted in descending order
        available_days = self.daily_best_models[
            self.daily_best_models['test_regime'] == regime
        ]['trading_day'].unique()
        available_days = sorted(available_days, reverse=True)
        
        # Find the most recent trading day before the prediction date
        for trading_day in available_days:
            if trading_day < prediction_date_int:
                daily_data = self.daily_best_models[
                    (self.daily_best_models['trading_day'] == trading_day) &
                    (self.daily_best_models['test_regime'] == regime)
                ]
                
                if len(daily_data) > 0:
                    row = daily_data.iloc[0]
                    
                    best_upside = {
                        'model_id': int(row['best_upside_model_id']) if pd.notna(row['best_upside_model_id']) else None,
                        'model_regime_rank': int(row['best_upside_rank']) if pd.notna(row['best_upside_rank']) else None,
                        'regime': regime,
                        'selection_method': 'daily',
                        'reference_date': trading_day
                    }
                    
                    best_downside = {
                        'model_id': int(row['best_downside_model_id']) if pd.notna(row['best_downside_model_id']) else None,
                        'model_regime_rank': int(row['best_downside_rank']) if pd.notna(row['best_downside_rank']) else None,
                        'regime': regime,
                        'selection_method': 'daily',
                        'reference_date': trading_day
                    }
                    
                    print(f"Using daily best models from {trading_day} for regime {regime} on {prediction_date}")
                    return best_upside, best_downside
        
        print(f"No daily best models found for regime {regime}, falling back to historical")
        return self.get_best_models_for_regime(regime, prediction_date)
    
    def get_best_models_weekly(self, regime, prediction_date):
        """Get best models based on last week's performance for the given regime"""
        prediction_date_int = int(prediction_date)
        
        # Convert prediction date to datetime for week calculations
        pred_datetime = datetime.strptime(str(prediction_date_int), '%Y%m%d')
        
        # Find last week's start and end dates
        days_since_monday = pred_datetime.weekday()
        current_week_start = pred_datetime - timedelta(days=days_since_monday)
        last_week_start = current_week_start - timedelta(days=7)
        last_week_end = current_week_start - timedelta(days=1)
        
        # Format dates back to YYYYMMDD
        last_week_start_str = last_week_start.strftime('%Y%m%d')
        last_week_end_str = last_week_end.strftime('%Y%m%d')
        
        # Search for last week's data, go back up to 4 weeks if needed
        for weeks_back in range(1, 5):
            search_week_start = last_week_start - timedelta(days=7 * (weeks_back - 1))
            search_week_start_str = search_week_start.strftime('%Y%m%d')
            
            weekly_data = self.weekly_best_models[
                (self.weekly_best_models['week_start'] == search_week_start_str) &
                (self.weekly_best_models['test_regime'] == regime)
            ]
            
            if len(weekly_data) > 0:
                row = weekly_data.iloc[0]
                
                best_upside = {
                    'model_id': int(row['best_upside_model_id']) if pd.notna(row['best_upside_model_id']) else None,
                    'model_regime_rank': int(row['best_upside_rank']) if pd.notna(row['best_upside_rank']) else None,
                    'regime': regime,
                    'selection_method': 'weekly',
                    'reference_week': search_week_start_str
                }
                
                best_downside = {
                    'model_id': int(row['best_downside_model_id']) if pd.notna(row['best_downside_model_id']) else None,
                    'model_regime_rank': int(row['best_downside_rank']) if pd.notna(row['best_downside_rank']) else None,
                    'regime': regime,
                    'selection_method': 'weekly',
                    'reference_week': search_week_start_str
                }
                
                print(f"Using weekly best models from week {search_week_start_str} for regime {regime} on {prediction_date}")
                return best_upside, best_downside
        
        print(f"No weekly best models found for regime {regime}, falling back to historical")
        return self.get_best_models_for_regime(regime, prediction_date)
    
    def get_best_models(self, regime, prediction_date, model_selection='historical'):
        """Unified method to get best models based on selection strategy"""
        if model_selection == 'daily':
            return self.get_best_models_daily(regime, prediction_date)
        elif model_selection == 'weekly':
            return self.get_best_models_weekly(regime, prediction_date)
        else:  # historical
            return self.get_best_models_for_regime(regime, prediction_date)
    
    def load_model_and_scaler(self, model_id):
        """Load LSTM model and its scaler"""
        try:
            # Convert model_id to properly formatted string
            if isinstance(model_id, (int, np.integer)):
                model_id_str = f"{model_id:05d}"
            else:
                model_id_str = str(model_id).strip()
                if '.' in model_id_str:
                    model_num = int(float(model_id_str))
                    model_id_str = f"{model_num:05d}"
                elif model_id_str.isdigit() and len(model_id_str) < 5:
                    model_id_str = f"{int(model_id_str):05d}"
            
            # Load model
            model_path = models_dir / f'lstm_stock_model_{model_id_str}.keras'
            if not model_path.exists():
                print(f"Model file not found: {model_path}")
                return None, None
            
            model = keras.models.load_model(model_path)
            
            # Load scaler
            scaler_path = models_dir / f'scaler_params_{model_id_str}.json'
            if not scaler_path.exists():
                print(f"Scaler file not found: {scaler_path}")
                return None, None
            
            with open(scaler_path, 'r') as f:
                scaler_params = json.load(f)
            
            scaler = MinMaxScaler()
            scaler.data_min_ = np.array(scaler_params['Min'])
            scaler.data_max_ = np.array(scaler_params['Max'])
            scaler.scale_ = 1.0 / (scaler.data_max_ - scaler.data_min_)
            scaler.min_ = -scaler.data_min_ * scaler.scale_
            
            return model, scaler
            
        except Exception as e:
            print(f"Error loading model {model_id}: {str(e)}")
            return None, None
    
    def create_sequences(self, df, seq_length, feature_cols, target_col):
        """Create sequences for prediction"""
        if len(df) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
        features = df[feature_cols].values
        targets = df[target_col].values
        days = df['TradingDay'].values
        ms = df['TradingMsOfDay'].values
        
        X = []
        y = []
        days_out = []
        ms_out = []
        
        i = 0
        while i <= len(df) - seq_length:
            # Check if the sequence is consecutive
            is_consecutive = True
            current_day = days[i]
            current_ms = ms[i]
            
            for j in range(1, seq_length):
                if days[i + j] != current_day or ms[i + j] != current_ms + j * 60000:
                    is_consecutive = False
                    break
            
            if is_consecutive:
                seq = features[i:i + seq_length]
                label = targets[i + seq_length - 1]
                X.append(seq)
                y.append(label)
                days_out.append(days[i + seq_length - 1])
                ms_out.append(ms[i + seq_length - 1])
                i += 1
            else:
                i += 1
        
        return np.array(X), np.array(y), np.array(days_out), np.array(ms_out)
    
    def make_predictions(self, model, scaler, data, seq_length, feature_cols, target_col):
        """Make predictions using the model"""
        X, y_actual, days, ms = self.create_sequences(data, seq_length, feature_cols, target_col)
        
        if len(X) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Scale features
        num_features = len(feature_cols)
        X_reshaped = X.reshape(-1, num_features)
        X_scaled = scaler.transform(X_reshaped).reshape(X.shape)
        
        # Make predictions
        predictions = model.predict(X_scaled, verbose=0).flatten()
        
        return predictions, y_actual, days, ms
    
    def run_forecasting(self, start_date, end_date, period='daily', method='hmm', model_selection='historical'):
        """Run the complete forecasting process"""
        print(f"Running forecasting from {start_date} to {end_date}")
        print(f"Period: {period}, Method: {method}, Model selection: {model_selection}")
        
        # Load all required data
        self.load_data()
        
        # Load daily/weekly best models data if needed
        if model_selection in ['daily', 'weekly']:
            self.load_best_models_data(model_selection)
        
        self.prepare_regime_history()
        
        # Train regime prediction model if using HMM
        if method == 'hmm':
            self.train_hmm_model()
        
        # Generate date range
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        if period == 'daily':
            date_range = pd.date_range(start_dt, end_dt, freq='D')
        else:  # weekly
            date_range = pd.date_range(start_dt, end_dt, freq='W-MON')
        
        # Results storage
        all_results = []
        
        for current_date in date_range:
            current_date_str = current_date.strftime('%Y%m%d')
            current_date_int = int(current_date_str)
            
            print(f"\nProcessing date: {current_date_str}")
            
            # Get data for the appropriate period
            if period == 'daily':
                # For daily period, get data for just this day
                period_data = self.trading_data[self.trading_data['TradingDay'] == current_date_int]
                period_description = f"day {current_date_str}"
            else:  # weekly
                # For weekly period, get data for the entire week (Monday to Friday)
                week_start = current_date
                week_end = week_start + timedelta(days=6)  # Sunday
                week_start_int = int(week_start.strftime('%Y%m%d'))
                week_end_int = int(week_end.strftime('%Y%m%d'))
                
                period_data = self.trading_data[
                    (self.trading_data['TradingDay'] >= week_start_int) & 
                    (self.trading_data['TradingDay'] <= week_end_int)
                ]
                period_description = f"week {week_start.strftime('%Y%m%d')} to {week_end.strftime('%Y%m%d')}"
            
            if len(period_data) == 0:
                print(f"No trading data for {period_description}, skipping...")
                continue
            
            print(f"Processing {period_description}: {len(period_data):,} records")
            
            # Predict regime for this period
            if method == 'hmm':
                predicted_regime = self.predict_regime_hmm(current_date_str)
            else:
                predicted_regime = self.predict_regime_naive(current_date_str)
            
            print(f"Predicted regime: {predicted_regime}")
            
            # Get best models for this regime using selected strategy
            best_upside, best_downside = self.get_best_models(predicted_regime, current_date_str, model_selection)
            
            if best_upside is None or best_downside is None:
                print(f"No valid models found for regime {predicted_regime} on {current_date_str}")
                continue
            
            print(f"Selected upside model: {best_upside['model_id']} (rank: {best_upside['model_regime_rank']})")
            print(f"Selected downside model: {best_downside['model_id']} (rank: {best_downside['model_regime_rank']})")
            
            # Load models and make predictions
            upside_model, upside_scaler = self.load_model_and_scaler(best_upside['model_id'])
            downside_model, downside_scaler = self.load_model_and_scaler(best_downside['model_id'])
            
            if upside_model is None or downside_model is None:
                print(f"Failed to load models for {current_date_str}")
                continue
            
            # Get model parameters
            upside_model_info = self.model_log[self.model_log['model_id'] == f"{best_upside['model_id']:05d}"].iloc[0]
            downside_model_info = self.model_log[self.model_log['model_id'] == f"{best_downside['model_id']:05d}"].iloc[0]
            
            upside_seq_length = int(upside_model_info['seq_length'])
            downside_seq_length = int(downside_model_info['seq_length'])
            upside_label_number = int(upside_model_info['label_number'])
            downside_label_number = int(downside_model_info['label_number'])
            
            # Define feature columns
            feature_cols = period_data.columns[5:49]  # Features: columns 6 to 49
            upside_target_col = f'Label_{upside_label_number}'
            downside_target_col = f'Label_{downside_label_number}'
            
            # Make predictions
            upside_predictions, upside_actual, days, ms = self.make_predictions(
                upside_model, upside_scaler, period_data, upside_seq_length, feature_cols, upside_target_col
            )
            
            downside_predictions, downside_actual, _, _ = self.make_predictions(
                downside_model, downside_scaler, period_data, downside_seq_length, feature_cols, downside_target_col
            )
            
            # Align predictions if sequence lengths are different
            min_length = min(len(upside_predictions), len(downside_predictions))
            if min_length == 0:
                print(f"No valid predictions for {current_date_str}")
                continue
            
            # Truncate to common length
            upside_predictions = upside_predictions[:min_length]
            downside_predictions = downside_predictions[:min_length]
            upside_actual = upside_actual[:min_length]
            days = days[:min_length]
            ms = ms[:min_length]
            
            # Store results
            for i in range(min_length):
                result = {
                    'trading_day': int(days[i]),
                    'ms_of_day': int(ms[i]),
                    'actual': float(upside_actual[i]),
                    'upside_predict': float(upside_predictions[i]),
                    'upside_model_id': best_upside['model_id'],
                    'upside_regime': int(predicted_regime),
                    'upside_rank_of_model': int(best_upside['model_regime_rank']),
                    'downside_predict': float(downside_predictions[i]),
                    'downside_model_id': best_downside['model_id'],
                    'downside_regime': int(predicted_regime),
                    'downside_rank_of_model': int(best_downside['model_regime_rank'])
                }
                all_results.append(result)
            
            print(f"Generated {min_length} predictions for {current_date_str}")
        
        # Save results
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Reorder columns
            column_order = [
                'trading_day', 'ms_of_day', 'actual', 
                'upside_predict', 'upside_model_id', 'upside_regime', 'upside_rank_of_model',
                'downside_predict', 'downside_model_id', 'downside_regime', 'downside_rank_of_model'
            ]
            results_df = results_df[column_order]
            
            output_file = output_dir / f'regime_forecast_{method}_{period}_{start_date}_{end_date}.csv'
            results_df.to_csv(output_file, index=False)
            print(f"\nSaved {len(all_results)} predictions to {output_file}")
            
            # Print summary statistics
            print(f"\nSummary:")
            print(f"Total predictions: {len(all_results)}")
            print(f"Unique trading days: {results_df['trading_day'].nunique()}")
            print(f"Regime distribution: {dict(results_df['upside_regime'].value_counts().sort_index())}")
            
            return results_df
        else:
            print("No predictions generated")
            return None

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Regime Prediction and LSTM Forecasting')
    parser.add_argument('--period', type=str, choices=['daily', 'weekly'], default='daily',
                       help='Forecasting period: daily (predict each day) or weekly (predict entire weeks)')
    parser.add_argument('--method', type=str, choices=['hmm', 'naive'], default='hmm',
                       help='Regime prediction method (default: hmm)')
    parser.add_argument('--start_date', type=str, required=True,
                       help='Start date for forecasting (YYYYMMDD)')
    parser.add_argument('--end_date', type=str, required=True,
                       help='End date for forecasting (YYYYMMDD)')
    parser.add_argument('--model_selection', type=str, 
                       choices=['historical', 'daily', 'weekly'], default='historical',
                       help='Model selection strategy: historical (best from all time), daily (best from yesterday), weekly (best from last week)')
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.start_date, '%Y%m%d')
        datetime.strptime(args.end_date, '%Y%m%d')
    except ValueError:
        raise ValueError("Dates must be in YYYYMMDD format")
    
    # Create forecaster and run
    forecaster = RegimePredictionForecaster()
    results = forecaster.run_forecasting(
        start_date=args.start_date,
        end_date=args.end_date,
        period=args.period,
        method=args.method,
        model_selection=args.model_selection
    )
    
    if results is not None:
        print(f"\nForecasting completed successfully!")
    else:
        print(f"\nForecasting completed with no results.")

if __name__ == "__main__":
    main()
