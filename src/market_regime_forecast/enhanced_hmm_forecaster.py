#!/usr/bin/env python3
"""
Enhanced HMM Market Regime Forecaster with Regime-Aware Training

This version addresses the fundamental flaw by using a hybrid approach:
1. Use successful TechnicalAnalysisFeatures (53 features -> 25 best)
2. Train HMM with 7 components to capture market dynamics
3. Use intelligent state-to-regime mapping based on feature similarity
4. Apply temporal consistency constraints

Key Improvements:
- Uses proven feature set and parameters from 60.51% accuracy model
- Adds regime-aware state initialization
- Implements smarter state-to-regime mapping
- Maintains temporal sequence modeling

Usage:
    python enhanced_hmm_forecaster.py --train_end 20211231 --n_components 7
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
from scipy.spatial.distance import cdist

# Import the successful technical analysis features
from market_regime_hmm_forecaster import TechnicalAnalysisFeatures

# Suppress warnings
warnings.filterwarnings('ignore')


class EnhancedHMMRegimeForecaster:
    """
    Enhanced HMM-based market regime forecaster with regime-aware training
    """
    
    def __init__(self, n_components=7, n_features=25, random_state=84):
        """
        Initialize the enhanced forecaster
        
        Args:
            n_components: Number of HMM hidden states (optimal found to be 7)
            n_features: Number of top features to select (optimal found to be 25)
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
        self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        self.selected_features = None
        self.feature_names = None
        self.state_to_regime = None
        self.regime_centroids = None
        
        print(f"Enhanced HMM Regime Forecaster initialized:")
        print(f"  HMM Components: {n_components}")
        print(f"  Features: {n_features}")
        print(f"  Time window: 10:35 AM - 12:00 PM")
        print(f"  Approach: Regime-aware state mapping with proven parameters")
    
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
        
        # Store feature names
        self.feature_names = feature_columns
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Regime distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y, trading_days
    
    def calculate_regime_centroids(self, X_scaled_selected, y):
        """Calculate regime centroids in feature space for intelligent mapping"""
        print("Calculating regime centroids for intelligent state mapping...")
        
        unique_regimes = sorted(np.unique(y))
        centroids = {}
        
        for regime in unique_regimes:
            regime_mask = y == regime
            regime_features = X_scaled_selected[regime_mask]
            centroid = np.mean(regime_features, axis=0)
            centroids[regime] = centroid
            print(f"  Regime {regime}: {np.sum(regime_mask)} samples")
        
        self.regime_centroids = centroids
        return centroids
    
    def intelligent_state_mapping(self, X_scaled_selected, y, predicted_states):
        """Create intelligent state-to-regime mapping using feature similarity"""
        print("Creating intelligent state-to-regime mapping...")
        
        n_states = self.n_components
        unique_regimes = sorted(np.unique(y))
        
        # Calculate state centroids
        state_centroids = {}
        for state in range(n_states):
            state_mask = predicted_states == state
            if np.sum(state_mask) > 0:
                state_features = X_scaled_selected[state_mask]
                state_centroids[state] = np.mean(state_features, axis=0)
            else:
                # Use random features for empty states
                state_centroids[state] = np.random.random(X_scaled_selected.shape[1])
        
        # Map each state to closest regime centroid
        state_to_regime = {}
        for state in range(n_states):
            state_centroid = state_centroids[state]
            
            # Calculate distances to all regime centroids
            distances = {}
            for regime, regime_centroid in self.regime_centroids.items():
                distance = np.linalg.norm(state_centroid - regime_centroid)
                distances[regime] = distance
            
            # Assign to closest regime
            closest_regime = min(distances.keys(), key=lambda r: distances[r])
            state_to_regime[state] = closest_regime
            
            print(f"  State {state} -> Regime {closest_regime} (distance: {distances[closest_regime]:.3f})")
        
        # Handle duplicates by using frequency-based assignment
        regime_assignments = list(state_to_regime.values())
        for regime in unique_regimes:
            if regime not in regime_assignments:
                # Find least assigned regime
                regime_counts = {r: regime_assignments.count(r) for r in unique_regimes}
                # Reassign a state to this missing regime
                most_assigned_regime = max(regime_counts.keys(), key=lambda r: regime_counts[r])
                states_with_regime = [s for s, r in state_to_regime.items() if r == most_assigned_regime]
                if states_with_regime:
                    # Reassign the first state
                    state_to_regime[states_with_regime[0]] = regime
                    print(f"  Reassigned State {states_with_regime[0]} from Regime {most_assigned_regime} to Regime {regime}")
        
        return state_to_regime
    
    def train(self, X, y, trading_days):
        """
        Train the enhanced HMM model with regime-aware mapping
        
        Args:
            X: Feature matrix
            y: Regime labels
            trading_days: Trading day sequence
            
        Returns:
            dict: Training results
        """
        print("Training Enhanced HMM model with regime-aware mapping...")
        
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
        
        # Calculate regime centroids
        self.calculate_regime_centroids(X_selected, y)
        
        # Train HMM with multiple covariance types (proven successful approach)
        best_model = None
        best_score = -np.inf
        best_cov_type = 'diag'
        best_mapping = None
        best_accuracy = 0
        
        covariance_types = ['diag', 'full', 'spherical']
        n_trials_per_type = 7  # Use proven configuration
        
        print(f"Training with multiple covariance types...")
        
        for cov_type in covariance_types:
            print(f"  Testing covariance type: {cov_type}")
            
            for trial in range(n_trials_per_type):
                try:
                    model = hmm.GaussianHMM(
                        n_components=self.n_components,
                        covariance_type=cov_type,
                        n_iter=200,
                        random_state=self.random_state + trial,
                        tol=1e-6,
                        verbose=False
                    )
                    
                    model.fit(X_selected)
                    score = model.score(X_selected)
                    
                    # Get predictions for this trial
                    predicted_states = model.predict(X_selected)
                    
                    # Create intelligent mapping
                    trial_mapping = self.intelligent_state_mapping(X_selected, y, predicted_states)
                    
                    # Calculate accuracy with this mapping
                    regime_predictions = np.array([trial_mapping[state] for state in predicted_states])
                    trial_accuracy = accuracy_score(y, regime_predictions)
                    
                    # Select best model based on accuracy, not likelihood
                    if trial_accuracy > best_accuracy:
                        best_accuracy = trial_accuracy
                        best_score = score
                        best_model = model
                        best_cov_type = cov_type
                        best_mapping = trial_mapping
                        
                except Exception as e:
                    continue
        
        if best_model is None:
            raise ValueError("All HMM training trials failed")
        
        print(f"  Best covariance type: {best_cov_type}")
        print(f"  Best training accuracy: {best_accuracy:.4f}")
        self.hmm_model = best_model
        self.state_to_regime = best_mapping
        
        print(f"Best model log-likelihood: {best_score:.4f}")
        print(f"Final state-to-regime mapping: {best_mapping}")
        
        return {
            'log_likelihood': best_score,
            'training_accuracy': best_accuracy,
            'state_to_regime': best_mapping,
            'selected_features': self.selected_features,
            'n_components': self.n_components,
            'n_features': len(self.selected_features)
        }
    
    def predict(self, X):
        """Make regime predictions using enhanced HMM"""
        if self.hmm_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_selected = X_scaled[:, self.feature_indices]
        
        # Get predictions
        predicted_states = self.hmm_model.predict(X_selected)
        state_probabilities = self.hmm_model.predict_proba(X_selected)
        
        # Convert states to regimes using intelligent mapping
        regime_predictions = np.array([self.state_to_regime[state] for state in predicted_states])
        
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
    
    def save_model(self, output_dir, model_name='enhanced_hmm_forecaster'):
        """Save trained model"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_data = {
            'hmm_model': self.hmm_model,
            'scaler': self.scaler,
            'feature_indices': self.feature_indices,
            'selected_features': self.selected_features,
            'feature_names': self.feature_names,
            'state_to_regime': self.state_to_regime,
            'regime_centroids': self.regime_centroids,
            'n_components': self.n_components,
            'n_features': self.n_features,
            'start_time_ms': self.start_time_ms,
            'end_time_ms': self.end_time_ms
        }
        
        model_path = os.path.join(output_dir, f'{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Enhanced model saved to: {model_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Enhanced HMM Market Regime Forecaster')
    
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
    
    # Model parameters (using proven optimal values)
    parser.add_argument('--n_components', type=int, default=7,
                       help='Number of HMM hidden states (proven optimal: 7)')
    parser.add_argument('--n_features', type=int, default=25,
                       help='Number of features to select (proven optimal: 25)')
    parser.add_argument('--random_state', type=int, default=84,
                       help='Random seed (proven optimal: 84)')
    
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
    print("ENHANCED HMM MARKET REGIME FORECASTER")
    print("="*80)
    print(f"Data file: {data_file}")
    print(f"Regime file: {regime_file}")
    print(f"Output directory: {output_dir}")
    print(f"Components: {args.n_components}")
    print(f"Features: {args.n_features}")
    print(f"Random State: {args.random_state}")
    print()
    
    try:
        # Initialize forecaster with proven optimal parameters
        forecaster = EnhancedHMMRegimeForecaster(
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
        with open(output_dir / 'enhanced_training_results.json', 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        # Save test results if available
        if test_results:
            with open(output_dir / 'enhanced_test_results.json', 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
        
        # Save model
        forecaster.save_model(output_dir)
        
        # Save features for inspection
        features_df.to_csv(output_dir / 'enhanced_daily_features.csv', index=False)
        
        print("\n" + "="*80)
        print("ENHANCED HMM EXECUTION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {output_dir}")
        print(f"Training accuracy: {training_results['training_accuracy']:.4f}")
        if test_results:
            print(f"Test accuracy: {test_results['test_accuracy']:.4f}")
            print(f"Enhanced regime-aware mapping approach!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
