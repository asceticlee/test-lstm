#!/usr/bin/env python3
"""
Market Regime Prediction Accuracy Test

This script tests how accurately we can predict the final market regime 
as trading data accumulates throughout the day from 10:35 AM to 12:00 PM.

The goal is to understand at what point during the trading day we can 
reliably predict the final regime classification, which is crucial for 
real-time trading decisions.

Usage:
    python regime_prediction_accuracy_test.py [--test_intervals N] [--min_duration M]
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent
sys.path.insert(0, str(src_dir))

# Machine Learning imports
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Import our statistical features module and GMM clusterer
from market_data_stat.statistical_features import StatisticalFeatureExtractor
from market_regime.gmm_regime_clustering import GMMRegimeClusterer

class RegimePredictionAccuracyTester:
    """
    Test regime prediction accuracy as trading day data accumulates
    """
    
    def __init__(self, data_path='data/history_spot_quote.csv', 
                 reference_model_path='../../market_regime/gmm/daily',
                 output_dir='../../market_regime/prediction_accuracy',
                 trading_start_ms=38100000, trading_end_ms=43200000):
        """
        Initialize the accuracy tester
        
        Args:
            data_path: Path to history_spot_quote.csv
            reference_model_path: Path to trained GMM model directory
            output_dir: Directory to save test results
            trading_start_ms: Start of trading period (10:35 AM)
            trading_end_ms: End of trading period (12:00 PM)
        """
        # Get the absolute path to the script directory
        script_dir = Path(__file__).parent.absolute()
        
        # Resolve paths relative to script directory
        if not os.path.isabs(data_path):
            self.data_path = script_dir / ".." / ".." / data_path
        else:
            self.data_path = Path(data_path)
        
        if not os.path.isabs(reference_model_path):
            self.reference_model_path = script_dir / reference_model_path
        else:
            self.reference_model_path = Path(reference_model_path)
            
        if not os.path.isabs(output_dir):
            self.output_dir = script_dir / output_dir
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trading_start_ms = trading_start_ms
        self.trading_end_ms = trading_end_ms
        
        # Load reference models and data
        self.reference_gmm = None
        self.reference_scaler = None
        self.reference_pca = None
        self.reference_regimes = None
        self.raw_data = None
        
        # Statistical feature extractor
        self.feature_extractor = StatisticalFeatureExtractor()
        
        # Results storage
        self.accuracy_results = []
        self.reference_feature_names = []
        
        print(f"Regime Prediction Accuracy Tester initialized")
        print(f"Data path: {self.data_path}")
        print(f"Reference model path: {self.reference_model_path}")
        print(f"Trading period: {self.ms_to_time(trading_start_ms)} to {self.ms_to_time(trading_end_ms)}")
        print(f"Output directory: {self.output_dir}")
    
    def ms_to_time(self, ms):
        """Convert milliseconds of day to readable time format"""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        return f"{hours:02d}:{minutes:02d}"
    
    def load_reference_models(self):
        """Load the trained GMM model, scaler, and PCA from reference directory"""
        print("Loading reference GMM models...")
        
        # Load model files
        model_file = self.reference_model_path / 'gmm_model.pkl'
        scaler_file = self.reference_model_path / 'feature_scaler.pkl'
        pca_file = self.reference_model_path / 'pca_model.pkl'
        regime_file = self.reference_model_path / 'daily_regime_assignments.csv'
        
        if not all([f.exists() for f in [model_file, scaler_file, pca_file, regime_file]]):
            raise FileNotFoundError(f"Reference model files not found in {self.reference_model_path}")
        
        # Load models
        self.reference_gmm = joblib.load(model_file)
        self.reference_scaler = joblib.load(scaler_file)
        self.reference_pca = joblib.load(pca_file)
        
        # Load clustering info to get feature names
        clustering_info_file = self.reference_model_path / 'clustering_info.json'
        if clustering_info_file.exists():
            import json
            with open(clustering_info_file, 'r') as f:
                clustering_info = json.load(f)
                self.reference_feature_names = clustering_info.get('feature_names', [])
                print(f"Loaded reference feature names: {len(self.reference_feature_names)} features")
        else:
            self.reference_feature_names = []
            print("Warning: No clustering info found - will attempt to infer feature names")
        
        # Load reference regime assignments (ground truth)
        self.reference_regimes = pd.read_csv(regime_file)
        print(f"Loaded reference regime assignments for {len(self.reference_regimes)} days")
        
        # Display regime distribution
        regime_counts = self.reference_regimes['Regime'].value_counts().sort_index()
        print("Reference regime distribution:")
        for regime, count in regime_counts.items():
            percentage = count / len(self.reference_regimes) * 100
            print(f"  Regime {regime}: {count} days ({percentage:.1f}%)")
        
        return True
    
    def load_trading_data(self):
        """Load and filter trading data"""
        print("Loading trading data...")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.raw_data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.raw_data):,} rows of raw data")
        
        # Filter for trading period
        self.raw_data = self.raw_data[
            (self.raw_data['ms_of_day'] >= self.trading_start_ms) &
            (self.raw_data['ms_of_day'] <= self.trading_end_ms)
        ].copy()
        
        # Sort by date and time
        self.raw_data = self.raw_data.sort_values(['trading_day', 'ms_of_day']).reset_index(drop=True)
        
        print(f"Filtered to {len(self.raw_data):,} rows for trading period")
        print(f"Date range: {self.raw_data['trading_day'].min()} to {self.raw_data['trading_day'].max()}")
        
        return self.raw_data
    
    def extract_partial_day_features(self, end_time_ms, reference_time_ms=None):
        """
        Extract features for each day using data up to end_time_ms
        
        Args:
            end_time_ms: End time for partial day analysis
            reference_time_ms: Reference time for relative price calculation
            
        Returns:
            DataFrame with features for each trading day using partial data
        """
        if reference_time_ms is None:
            reference_time_ms = self.trading_start_ms
            
        # Filter data up to end_time_ms
        partial_data = self.raw_data[
            self.raw_data['ms_of_day'] <= end_time_ms
        ].copy()
        
        if len(partial_data) == 0:
            return None
        
        # Extract features using the same method as reference model
        price_column = 'mid'
        volume_column = 'volume' if 'volume' in partial_data.columns else None
        
        daily_features = self.feature_extractor.extract_daily_features(
            daily_data=partial_data,
            price_column=price_column,
            volume_column=volume_column,
            reference_time_ms=reference_time_ms,
            trading_day_column='trading_day',
            time_column='ms_of_day',
            use_relative=True,
            include_overnight_gap=False
        )
        
        # Add metadata
        daily_features['FromMsOfDay'] = self.trading_start_ms
        daily_features['ToMsOfDay'] = end_time_ms
        daily_features['time_range_minutes'] = (end_time_ms - self.trading_start_ms) / 60000
        
        return daily_features
    
    def predict_regimes_for_partial_data(self, partial_features):
        """
        Predict regimes using partial day features and reference models
        
        Args:
            partial_features: DataFrame with partial day features
            
        Returns:
            numpy array of predicted regime labels
        """
        if partial_features is None or len(partial_features) == 0:
            return np.array([])
        
        # Get feature columns (same as reference model)
        metadata_cols = ['trading_day', 'FromMsOfDay', 'ToMsOfDay', 'reference_time_ms', 
                        'reference_price', 'num_observations', 'time_range_minutes']
        
        # Get available feature columns
        available_features = [col for col in partial_features.columns if col not in metadata_cols]
        
        # If we have reference feature names, use only those
        if len(self.reference_feature_names) > 0:
            # Only use features that were in the original training
            valid_features = []
            for feature in self.reference_feature_names:
                if feature in available_features:
                    valid_features.append(feature)
                else:
                    print(f"Warning: Reference feature '{feature}' not found in partial data")
            
            # Only include features that exist in both
            features_to_use = valid_features
        else:
            # Fallback to using all available features
            features_to_use = available_features
        
        if len(features_to_use) == 0:
            print("Error: No valid features found for prediction")
            return np.array([])
        
        # Get feature matrix
        X = partial_features[features_to_use].copy()
        X = X.fillna(0)
        
        # Handle case where we have fewer features than expected
        if len(self.reference_feature_names) > 0 and len(features_to_use) < len(self.reference_feature_names):
            # Create a full feature matrix with missing features as zeros
            full_X = np.zeros((len(X), len(self.reference_feature_names)))
            for i, feature in enumerate(self.reference_feature_names):
                if feature in features_to_use:
                    feature_idx = features_to_use.index(feature)
                    full_X[:, i] = X.iloc[:, feature_idx]
            X_array = full_X
        else:
            X_array = X.values
        
        try:
            # Scale features using reference scaler
            X_scaled = self.reference_scaler.transform(X_array)
            
            # Apply PCA transformation
            X_pca = self.reference_pca.transform(X_scaled)
            
            # Predict regimes
            predicted_regimes = self.reference_gmm.predict(X_pca)
            
            return predicted_regimes
            
        except Exception as e:
            print(f"Warning: Could not predict regimes for partial data: {e}")
            # Return random predictions as fallback
            n_regimes = len(self.reference_regimes['Regime'].unique())
            return np.random.randint(0, n_regimes, size=len(partial_features))
    
    def test_prediction_accuracy_at_time(self, end_time_ms, min_observations=10):
        """
        Test prediction accuracy using data up to end_time_ms
        
        Args:
            end_time_ms: End time for partial day analysis
            min_observations: Minimum observations required per day
            
        Returns:
            Dictionary with accuracy metrics
        """
        time_str = self.ms_to_time(end_time_ms)
        duration_minutes = (end_time_ms - self.trading_start_ms) / 60000
        
        print(f"Testing prediction accuracy at {time_str} ({duration_minutes:.1f} minutes into trading)")
        
        # Extract features using partial day data
        partial_features = self.extract_partial_day_features(end_time_ms)
        
        if partial_features is None or len(partial_features) == 0:
            print(f"  No data available at {time_str}")
            return None
        
        # Filter days with sufficient observations
        sufficient_data = partial_features[
            partial_features['num_observations'] >= min_observations
        ].copy()
        
        if len(sufficient_data) == 0:
            print(f"  No days with sufficient data at {time_str} (min {min_observations} observations)")
            return None
        
        print(f"  Testing on {len(sufficient_data)} days with sufficient data")
        
        # Predict regimes
        predicted_regimes = self.predict_regimes_for_partial_data(sufficient_data)
        
        if len(predicted_regimes) == 0:
            print(f"  Could not generate predictions at {time_str}")
            return None
        
        # Get ground truth regimes for the same days
        test_days = sufficient_data['trading_day'].values
        true_regimes = []
        
        for day in test_days:
            day_regime = self.reference_regimes[
                self.reference_regimes['trading_day'] == day
            ]['Regime']
            
            if len(day_regime) > 0:
                true_regimes.append(day_regime.iloc[0])
            else:
                # Skip days not in reference data
                continue
        
        true_regimes = np.array(true_regimes)
        
        # Align predictions with true regimes
        if len(predicted_regimes) != len(true_regimes):
            min_len = min(len(predicted_regimes), len(true_regimes))
            predicted_regimes = predicted_regimes[:min_len]
            true_regimes = true_regimes[:min_len]
        
        if len(true_regimes) == 0:
            print(f"  No matching reference data at {time_str}")
            return None
        
        # Calculate accuracy metrics
        accuracy = accuracy_score(true_regimes, predicted_regimes)
        
        # Get unique regimes for detailed analysis
        unique_regimes = np.unique(np.concatenate([true_regimes, predicted_regimes]))
        
        # Calculate per-regime accuracy
        regime_accuracies = {}
        for regime in unique_regimes:
            regime_mask = (true_regimes == regime)
            if np.sum(regime_mask) > 0:
                regime_predictions = predicted_regimes[regime_mask]
                regime_accuracy = np.mean(regime_predictions == regime)
                regime_accuracies[f'regime_{regime}_accuracy'] = regime_accuracy
        
        # Create confusion matrix
        conf_matrix = confusion_matrix(true_regimes, predicted_regimes, labels=unique_regimes)
        
        result = {
            'end_time_ms': end_time_ms,
            'end_time_str': time_str,
            'duration_minutes': duration_minutes,
            'completion_percentage': (duration_minutes / ((self.trading_end_ms - self.trading_start_ms) / 60000)) * 100,
            'n_test_days': len(true_regimes),
            'overall_accuracy': accuracy,
            'n_regimes_detected': len(unique_regimes),
            **regime_accuracies
        }
        
        print(f"  Overall accuracy: {accuracy:.3f} ({len(true_regimes)} days tested)")
        print(f"  Regimes detected: {len(unique_regimes)} unique regimes")
        
        return result
    
    def run_accuracy_test(self, test_intervals=20, min_duration_minutes=5):
        """
        Run prediction accuracy test across multiple time points
        
        Args:
            test_intervals: Number of time points to test
            min_duration_minutes: Minimum duration from start before testing
        """
        print("Starting regime prediction accuracy test...")
        print(f"Testing {test_intervals} time intervals with minimum {min_duration_minutes} minutes duration")
        
        # Calculate test times
        min_duration_ms = min_duration_minutes * 60000
        start_test_time = self.trading_start_ms + min_duration_ms
        
        test_times = np.linspace(start_test_time, self.trading_end_ms, test_intervals)
        test_times = test_times.astype(int)
        
        # Run tests at each time point
        self.accuracy_results = []
        
        for i, end_time_ms in enumerate(test_times):
            print(f"\n--- Test {i+1}/{len(test_times)} ---")
            
            result = self.test_prediction_accuracy_at_time(end_time_ms)
            
            if result is not None:
                self.accuracy_results.append(result)
        
        print(f"\nCompleted accuracy testing with {len(self.accuracy_results)} successful tests")
        
        return self.accuracy_results
    
    def analyze_and_save_results(self):
        """Analyze and save the accuracy test results"""
        if len(self.accuracy_results) == 0:
            print("No results to analyze")
            return
        
        print("Analyzing accuracy test results...")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.accuracy_results)
        
        # Save detailed results
        results_file = self.output_dir / 'regime_prediction_accuracy_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Saved detailed results to: {results_file}")
        
        # Create summary analysis
        summary_stats = {
            'test_intervals': len(results_df),
            'min_accuracy': results_df['overall_accuracy'].min(),
            'max_accuracy': results_df['overall_accuracy'].max(),
            'final_accuracy': results_df['overall_accuracy'].iloc[-1] if len(results_df) > 0 else None,
            'mean_accuracy': results_df['overall_accuracy'].mean(),
            'accuracy_at_50_percent': None,
            'accuracy_at_75_percent': None,
            'accuracy_at_90_percent': None
        }
        
        # Find accuracy at specific completion percentages
        for target_pct in [50, 75, 90]:
            closest_idx = np.argmin(np.abs(results_df['completion_percentage'] - target_pct))
            if np.abs(results_df.iloc[closest_idx]['completion_percentage'] - target_pct) < 10:  # Within 10%
                summary_stats[f'accuracy_at_{target_pct}_percent'] = results_df.iloc[closest_idx]['overall_accuracy']
        
        # Print summary
        print("\n" + "="*60)
        print("ACCURACY TEST SUMMARY")
        print("="*60)
        print(f"Total test intervals: {summary_stats['test_intervals']}")
        print(f"Accuracy range: {summary_stats['min_accuracy']:.3f} to {summary_stats['max_accuracy']:.3f}")
        print(f"Mean accuracy: {summary_stats['mean_accuracy']:.3f}")
        print(f"Final accuracy (end of day): {summary_stats['final_accuracy']:.3f}")
        
        if summary_stats['accuracy_at_50_percent'] is not None:
            print(f"Accuracy at ~50% completion: {summary_stats['accuracy_at_50_percent']:.3f}")
        if summary_stats['accuracy_at_75_percent'] is not None:
            print(f"Accuracy at ~75% completion: {summary_stats['accuracy_at_75_percent']:.3f}")
        if summary_stats['accuracy_at_90_percent'] is not None:
            print(f"Accuracy at ~90% completion: {summary_stats['accuracy_at_90_percent']:.3f}")
        
        # Save summary
        summary_file = self.output_dir / 'accuracy_test_summary.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"\nSaved summary to: {summary_file}")
        
        # Create visualization if possible
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            # Plot accuracy over completion percentage
            plt.subplot(2, 1, 1)
            plt.plot(results_df['completion_percentage'], results_df['overall_accuracy'], 'b-o', markersize=4)
            plt.xlabel('Trading Day Completion (%)')
            plt.ylabel('Prediction Accuracy')
            plt.title('Regime Prediction Accuracy vs Trading Day Progress')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random baseline (50%)')
            plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect accuracy')
            plt.legend()
            
            # Plot number of test days over time
            plt.subplot(2, 1, 2)
            plt.plot(results_df['completion_percentage'], results_df['n_test_days'], 'g-s', markersize=4)
            plt.xlabel('Trading Day Completion (%)')
            plt.ylabel('Number of Test Days')
            plt.title('Number of Days with Sufficient Data')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = self.output_dir / 'accuracy_over_time.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved accuracy plot to: {plot_file}")
            
        except ImportError:
            print("matplotlib not available - skipping plot generation")
        except Exception as e:
            print(f"Could not create plot: {e}")
        
        return summary_stats

def main():
    parser = argparse.ArgumentParser(description='Market Regime Prediction Accuracy Test')
    parser.add_argument('--data_path', default='data/history_spot_quote.csv',
                       help='Path to trading data CSV file')
    parser.add_argument('--reference_model_path', default='../../market_regime/gmm/daily',
                       help='Path to reference GMM model directory')
    parser.add_argument('--output_dir', default='../../market_regime/prediction_accuracy',
                       help='Output directory for test results')
    parser.add_argument('--test_intervals', type=int, default=20,
                       help='Number of time points to test throughout the day')
    parser.add_argument('--min_duration', type=int, default=5,
                       help='Minimum duration (minutes) from start before testing')
    parser.add_argument('--trading_start', default='10:35',
                       help='Trading start time (HH:MM format)')
    parser.add_argument('--trading_end', default='12:00',
                       help='Trading end time (HH:MM format)')
    
    args = parser.parse_args()
    
    # Convert times to milliseconds
    def time_to_ms(time_str):
        hours, minutes = map(int, time_str.split(':'))
        return hours * 3600000 + minutes * 60000
    
    trading_start_ms = time_to_ms(args.trading_start)
    trading_end_ms = time_to_ms(args.trading_end)
    
    # Initialize tester
    tester = RegimePredictionAccuracyTester(
        data_path=args.data_path,
        reference_model_path=args.reference_model_path,
        output_dir=args.output_dir,
        trading_start_ms=trading_start_ms,
        trading_end_ms=trading_end_ms
    )
    
    try:
        # Load reference models and data
        tester.load_reference_models()
        tester.load_trading_data()
        
        # Run accuracy test
        results = tester.run_accuracy_test(
            test_intervals=args.test_intervals,
            min_duration_minutes=args.min_duration
        )
        
        # Analyze and save results
        summary = tester.analyze_and_save_results()
        
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
