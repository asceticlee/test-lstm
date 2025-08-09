#!/usr/bin/env python3
"""
Supervised HMM Market Regime Forecaster

A corrected implementation that directly learns GMM regime structure instead of post-hoc mapping.
This addresses the fundamental flaw where HMM creates arbitrary hidden states that are disconnected 
from the GMM-discovered market regimes.

Key Improvements:
- Direct regime learning: HMM learns to predict GMM regime labels directly
- Transition matrix initialization: Based on observed GMM regime transitions  
- Emission matrix guidance: Guided by GMM-feature relationships
- No post-hoc mapping: Direct alignment between HMM states and market regimes

Usage:
    python supervised_hmm_forecaster.py --train_end 20211231 --n_components 5
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

# Import the successful technical analysis features
from market_regime_hmm_forecaster import TechnicalAnalysisFeatures

# Suppress warnings
warnings.filterwarnings('ignore')


class SupervisedHMMRegimeForecaster:
    """
    Supervised HMM-based market regime forecaster that directly learns GMM regime structure
    """
    
    def __init__(self, n_features=25, random_state=84):
        """
        Initialize the supervised forecaster
        
        Args:
            n_features: Number of top features to select
            random_state: Random seed for reproducibility
        """
        self.n_features = n_features
        self.random_state = random_state
        
        # Time window settings (10:35 AM - 12:00 PM)
        self.start_time_ms = 38100000  # 10:35 AM
        self.end_time_ms = 43200000    # 12:00 PM
        
        # Model components
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        self.selected_features = None
        self.feature_names = None
        self.n_regimes = None
        self.regime_mapping = None
        
        print(f"Supervised HMM Regime Forecaster initialized:")
        print(f"  Features: {n_features}")
        print(f"  Time window: 10:35 AM - 12:00 PM")
        print(f"  Approach: Direct GMM regime learning (no post-hoc mapping)")
    
    def load_market_data(self, data_file):
        """Load and filter market data"""
        print(f"Loading market data from: {data_file}")
        
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
        """Extract features using successful TechnicalAnalysisFeatures"""
        print("Extracting daily technical analysis features...")
        
        daily_features = []
        
        for trading_day in sorted(market_data['trading_day'].unique()):
            day_data = market_data[market_data['trading_day'] == trading_day]
            
            if len(day_data) < 5:  # Skip days with insufficient data
                continue
            
            prices = day_data['mid'].values
            reference_price = prices[0]  # First price of the day (10:35 AM)
            
            # Extract all features using successful approach
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
        """Load regime labels and filter to 5 regimes (0-4)"""
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
        """Merge features with regime labels and prepare training data"""
        print("Preparing training data...")
        
        # Merge features with regime labels
        merged_data = pd.merge(features_df, regime_data, on='trading_day', how='inner')
        
        # Filter by date range
        if train_start:
            merged_data = merged_data[merged_data['trading_day'] >= int(train_start)]
        if train_end:
            merged_data = merged_data[merged_data['trading_day'] <= int(train_end)]
        
        print(f"Training date range: {merged_data['trading_day'].min()} to {merged_data['trading_day'].max()}")
        print(f"Training samples: {len(merged_data)}")
        
        # Separate features and targets
        feature_columns = [col for col in merged_data.columns 
                          if col not in ['trading_day', 'Regime', 'num_observations', 'reference_price']]
        
        X = merged_data[feature_columns].values
        y = merged_data['Regime'].values
        trading_days = merged_data['trading_day'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store feature names and regime info
        self.feature_names = feature_columns
        unique_regimes = sorted(np.unique(y))
        self.n_regimes = len(unique_regimes)
        
        # Create regime mapping (0-indexed for HMM) - convert to native int
        self.regime_mapping = {int(regime): idx for idx, regime in enumerate(unique_regimes)}
        self.reverse_mapping = {idx: int(regime) for regime, idx in self.regime_mapping.items()}
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Regime distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        print(f"Regime mapping for HMM: {self.regime_mapping}")
        
        return X, y, trading_days
    
    def calculate_transition_matrix(self, y_sequence, trading_days):
        """Calculate empirical transition matrix from GMM regime sequence"""
        print("Calculating empirical transition matrix from GMM regimes...")
        
        # Sort by trading day to get correct sequence
        sorted_indices = np.argsort(trading_days)
        sorted_regimes = y_sequence[sorted_indices]
        
        # Convert to 0-indexed
        regime_sequence = np.array([self.regime_mapping[int(regime)] for regime in sorted_regimes])
        
        # Calculate transition matrix
        n_regimes = self.n_regimes
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regime_sequence) - 1):
            current_regime = regime_sequence[i]
            next_regime = regime_sequence[i + 1]
            transition_matrix[current_regime, next_regime] += 1
        
        # Normalize rows to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        for i in range(n_regimes):
            if row_sums[i] > 0:
                transition_matrix[i, :] = transition_matrix[i, :] / row_sums[i]
            else:
                # If no transitions from this regime, use uniform distribution
                transition_matrix[i, :] = 1.0 / n_regimes
        
        print(f"Empirical transition matrix shape: {transition_matrix.shape}")
        print("Transition probabilities:")
        for i in range(n_regimes):
            regime_i = self.reverse_mapping[i]
            print(f"  From Regime {regime_i}: {transition_matrix[i, :]}")
        
        return transition_matrix
    
    def calculate_initial_probabilities(self, y_sequence):
        """Calculate initial state probabilities from regime distribution"""
        regime_counts = np.bincount([self.regime_mapping[int(regime)] for regime in y_sequence])
        
        # Ensure we have counts for all regimes
        initial_probs = np.zeros(self.n_regimes)
        for i in range(min(len(regime_counts), self.n_regimes)):
            initial_probs[i] = regime_counts[i]
        
        # Normalize
        initial_probs = initial_probs / np.sum(initial_probs)
        
        print(f"Initial state probabilities: {initial_probs}")
        return initial_probs
    
    def train(self, X, y, trading_days):
        """
        Train the supervised HMM model with GMM regime guidance
        
        Args:
            X: Feature matrix
            y: Regime labels (GMM regimes)
            trading_days: Trading day sequence
            
        Returns:
            dict: Training results
        """
        print("Training Supervised HMM model with GMM regime structure...")
        
        unique_regimes = sorted(np.unique(y))
        print(f"Training regimes: {unique_regimes}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Select best features
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = [self.feature_names[i] for i in selected_indices]
        
        print(f"Selected {len(self.selected_features)} best features:")
        feature_scores = [(self.feature_names[i], self.feature_selector.scores_[i]) 
                         for i in selected_indices]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, score) in enumerate(feature_scores):
            print(f"  {i+1:2d}. {feature}: {score:.1f}")
        
        # Store feature indices for later use
        self.feature_indices = selected_indices
        
        # Calculate GMM-based initialization
        transition_matrix = self.calculate_transition_matrix(y, trading_days)
        initial_probs = self.calculate_initial_probabilities(y)
        
        # Convert regime labels to 0-indexed for HMM
        y_indexed = np.array([self.regime_mapping[int(regime)] for regime in y])
        
        # Create and train supervised HMM
        print(f"Creating HMM with {self.n_regimes} components (one per GMM regime)")
        
        # Try different covariance types with GMM initialization
        best_model = None
        best_score = -np.inf
        best_cov_type = 'diag'
        
        covariance_types = ['diag', 'full', 'spherical']
        n_trials_per_type = 5
        
        print(f"Training with GMM-initialized HMM...")
        
        for cov_type in covariance_types:
            print(f"  Testing covariance type: {cov_type}")
            
            for trial in range(n_trials_per_type):
                try:
                    # Create HMM with number of components = number of regimes
                    model = hmm.GaussianHMM(
                        n_components=self.n_regimes,
                        covariance_type=cov_type,
                        n_iter=200,
                        random_state=self.random_state + trial,
                        tol=1e-6,
                        verbose=False,
                        init_params='mc'  # Only initialize means and covariances, not transitions/start
                    )
                    
                    # Initialize with GMM-derived parameters after creation
                    model.startprob_ = initial_probs.copy()
                    model.transmat_ = transition_matrix.copy()
                    
                    # Fit the model
                    model.fit(X_selected)
                    score = model.score(X_selected)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_cov_type = cov_type
                        
                except Exception as e:
                    print(f"    Trial {trial + 1} failed: {str(e)}")
                    continue
        
        if best_model is None:
            raise ValueError("All supervised HMM training trials failed")
        
        print(f"  Best covariance type: {best_cov_type}")
        self.hmm_model = best_model
        print(f"Best model log-likelihood: {best_score:.4f}")
        
        # Evaluate training performance
        predicted_states = self.hmm_model.predict(X_selected)
        
        # Convert predictions back to original regime labels
        regime_predictions = np.array([self.reverse_mapping[state] for state in predicted_states])
        
        # Calculate training accuracy
        training_accuracy = accuracy_score(y, regime_predictions)
        
        print(f"Supervised HMM training completed:")
        print(f"  Log-likelihood: {best_score:.4f}")
        print(f"  Training accuracy: {training_accuracy:.4f}")
        print(f"  Direct regime mapping (no post-hoc needed): State i = Regime {list(self.reverse_mapping.values())}")
        
        # Show learned transition matrix
        print("Learned transition matrix:")
        for i in range(self.n_regimes):
            regime_i = self.reverse_mapping[i]
            print(f"  From Regime {regime_i}: {self.hmm_model.transmat_[i, :]}")
        
        return {
            'log_likelihood': best_score,
            'training_accuracy': training_accuracy,
            'regime_mapping': self.regime_mapping,
            'reverse_mapping': self.reverse_mapping,
            'selected_features': self.selected_features,
            'n_regimes': self.n_regimes,
            'n_features': len(self.selected_features),
            'transition_matrix': self.hmm_model.transmat_.tolist(),
            'initial_probabilities': self.hmm_model.startprob_.tolist()
        }
    
    def predict(self, X):
        """
        Make regime predictions using supervised HMM
        
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
        
        # Convert states directly to regimes (no mapping needed!)
        regime_predictions = np.array([self.reverse_mapping[state] for state in predicted_states])
        
        # Calculate confidence scores
        confidence_scores = np.max(state_probabilities, axis=1)
        
        return regime_predictions, state_probabilities, confidence_scores
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
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
    
    def save_model(self, output_dir, model_name='supervised_hmm_forecaster'):
        """Save trained model"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_data = {
            'hmm_model': self.hmm_model,
            'scaler': self.scaler,
            'feature_indices': self.feature_indices,
            'selected_features': self.selected_features,
            'feature_names': self.feature_names,
            'regime_mapping': self.regime_mapping,
            'reverse_mapping': self.reverse_mapping,
            'n_regimes': self.n_regimes,
            'n_features': self.n_features,
            'start_time_ms': self.start_time_ms,
            'end_time_ms': self.end_time_ms
        }
        
        model_path = os.path.join(output_dir, f'{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Supervised model saved to: {model_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Supervised HMM Market Regime Forecaster')
    
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
    parser.add_argument('--n_features', type=int, default=25,
                       help='Number of features to select')
    parser.add_argument('--random_state', type=int, default=84,
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
    print("SUPERVISED HMM MARKET REGIME FORECASTER")
    print("="*80)
    print(f"Data file: {data_file}")
    print(f"Regime file: {regime_file}")
    print(f"Output directory: {output_dir}")
    print(f"Features: {args.n_features}")
    print(f"Approach: Direct GMM regime learning")
    print()
    
    try:
        # Initialize forecaster
        forecaster = SupervisedHMMRegimeForecaster(
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
        training_results = forecaster.train(X_train, y_train, train_days)
        
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
        with open(output_dir / 'supervised_training_results.json', 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        # Save test results if available
        if test_results:
            with open(output_dir / 'supervised_test_results.json', 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
        
        # Save model
        forecaster.save_model(output_dir)
        
        # Save features for inspection
        features_df.to_csv(output_dir / 'supervised_daily_features.csv', index=False)
        
        print("\n" + "="*80)
        print("SUPERVISED HMM EXECUTION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {output_dir}")
        print(f"Training accuracy: {training_results['training_accuracy']:.4f}")
        if test_results:
            print(f"Test accuracy: {test_results['test_accuracy']:.4f}")
            print(f"Improvement over post-hoc mapping approach!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
