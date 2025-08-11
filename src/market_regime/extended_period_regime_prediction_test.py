#!/usr/bin/env python3
"""
Extended Period Market Regime Prediction Test

This script tests how accurately we can predict the market regime (originally defined for 10:35-12:00)
using data from an earlier start time (e.g., 9:30 AM) to provide more trading time.

The key difference from the original test:
- Original: Test prediction accuracy using partial data within the same period (10:35-12:00)
- Extended: Test prediction accuracy using data from BEFORE the regime period (9:30-12:00)

This allows us to predict the 10:35-12:00 regime classification using earlier market data,
potentially giving traders more time to execute strategies.

Usage:
    python extended_period_regime_prediction_test.py --extended_start 09:30 --test_intervals 25
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

# Import our statistical features module
from market_data_stat.statistical_features import StatisticalFeatureExtractor

class ExtendedPeriodRegimePredictionTester:
    """
    Test regime prediction accuracy using extended data collection periods
    """
    
    def __init__(self, data_path='data/history_spot_quote.csv', 
                 reference_model_path='../../market_regime/gmm/daily',
                 output_dir='../../market_regime/extended_prediction_accuracy',
                 regime_start_ms=38100000, regime_end_ms=43200000,
                 extended_start_ms=34200000):
        """
        Initialize the extended period tester
        
        Args:
            data_path: Path to history_spot_quote.csv
            reference_model_path: Path to trained GMM model directory
            output_dir: Directory to save test results
            regime_start_ms: Original regime period start (10:35 AM = 38100000)
            regime_end_ms: Original regime period end (12:00 PM = 43200000)
            extended_start_ms: Extended data collection start (9:30 AM = 34200000)
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
        
        self.regime_start_ms = regime_start_ms
        self.regime_end_ms = regime_end_ms
        self.extended_start_ms = extended_start_ms
        
        # Load reference models and data
        self.reference_gmm = None
        self.reference_scaler = None
        self.reference_pca = None
        self.reference_regimes = None
        self.reference_feature_names = []
        self.raw_data = None
        
        # Statistical feature extractor
        self.feature_extractor = StatisticalFeatureExtractor()
        
        # Results storage
        self.accuracy_results = []
        
        print(f"Extended Period Regime Prediction Tester initialized")
        print(f"Data path: {self.data_path}")
        print(f"Reference model path: {self.reference_model_path}")
        print(f"Original regime period: {self.ms_to_time(regime_start_ms)} to {self.ms_to_time(regime_end_ms)}")
        print(f"Extended data collection: {self.ms_to_time(extended_start_ms)} to {self.ms_to_time(regime_end_ms)}")
        print(f"Additional trading time: {(regime_start_ms - extended_start_ms) / 60000:.1f} minutes")
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
        """Load and filter trading data for extended period"""
        print("Loading extended trading data...")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.raw_data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.raw_data):,} rows of raw data")
        
        # Filter for extended period (from extended_start to regime_end)
        self.raw_data = self.raw_data[
            (self.raw_data['ms_of_day'] >= self.extended_start_ms) &
            (self.raw_data['ms_of_day'] <= self.regime_end_ms)
        ].copy()
        
        # Sort by date and time
        self.raw_data = self.raw_data.sort_values(['trading_day', 'ms_of_day']).reset_index(drop=True)
        
        print(f"Filtered to {len(self.raw_data):,} rows for extended period")
        print(f"Date range: {self.raw_data['trading_day'].min()} to {self.raw_data['trading_day'].max()}")
        
        # Calculate additional time gained
        original_duration = (self.regime_end_ms - self.regime_start_ms) / 60000
        extended_duration = (self.regime_end_ms - self.extended_start_ms) / 60000
        additional_time = extended_duration - original_duration
        
        print(f"Original regime period: {original_duration:.1f} minutes")
        print(f"Extended collection period: {extended_duration:.1f} minutes")
        print(f"Additional time for trading: {additional_time:.1f} minutes")
        
        return self.raw_data
    
    def extract_extended_features(self, end_time_ms, reference_time_ms=None):
        """
        Extract features using extended data period (from extended_start to end_time_ms)
        
        Args:
            end_time_ms: End time for feature extraction
            reference_time_ms: Reference time for relative price calculation
            
        Returns:
            DataFrame with features for each trading day using extended data
        """
        if reference_time_ms is None:
            # Use the extended start time as reference for consistency
            reference_time_ms = self.extended_start_ms
            
        # Filter data from extended_start to end_time_ms
        extended_data = self.raw_data[
            (self.raw_data['ms_of_day'] >= self.extended_start_ms) &
            (self.raw_data['ms_of_day'] <= end_time_ms)
        ].copy()
        
        if len(extended_data) == 0:
            return None
        
        # Extract features using the same method as reference model
        price_column = 'mid'
        volume_column = 'volume' if 'volume' in extended_data.columns else None
        
        daily_features = self.feature_extractor.extract_daily_features(
            daily_data=extended_data,
            price_column=price_column,
            volume_column=volume_column,
            reference_time_ms=reference_time_ms,
            trading_day_column='trading_day',
            time_column='ms_of_day',
            use_relative=True,
            include_overnight_gap=False
        )
        
        # Add metadata
        daily_features['FromMsOfDay'] = self.extended_start_ms
        daily_features['ToMsOfDay'] = end_time_ms
        daily_features['time_range_minutes'] = (end_time_ms - self.extended_start_ms) / 60000
        daily_features['regime_coverage'] = min(1.0, max(0.0, 
            (min(end_time_ms, self.regime_end_ms) - max(self.extended_start_ms, self.regime_start_ms)) / 
            (self.regime_end_ms - self.regime_start_ms)))
        
        return daily_features
    
    def predict_regimes_for_extended_data(self, extended_features):
        """
        Predict regimes using extended period features and reference models
        
        Args:
            extended_features: DataFrame with extended period features
            
        Returns:
            numpy array of predicted regime labels
        """
        if extended_features is None or len(extended_features) == 0:
            return np.array([])
        
        # Get feature columns (same as reference model)
        metadata_cols = ['trading_day', 'FromMsOfDay', 'ToMsOfDay', 'reference_time_ms', 
                        'reference_price', 'num_observations', 'time_range_minutes', 'regime_coverage']
        
        # Get available feature columns
        available_features = [col for col in extended_features.columns if col not in metadata_cols]
        
        # If we have reference feature names, use only those
        if len(self.reference_feature_names) > 0:
            # Only use features that were in the original training
            valid_features = []
            for feature in self.reference_feature_names:
                if feature in available_features:
                    valid_features.append(feature)
                else:
                    # For extended period, some features might not be available
                    pass
            
            features_to_use = valid_features
        else:
            features_to_use = available_features
        
        if len(features_to_use) == 0:
            print("Error: No valid features found for prediction")
            return np.array([])
        
        # Get feature matrix
        X = extended_features[features_to_use].copy()
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
            print(f"Warning: Could not predict regimes for extended data: {e}")
            # Return random predictions as fallback
            n_regimes = len(self.reference_regimes['Regime'].unique())
            return np.random.randint(0, n_regimes, size=len(extended_features))
    
    def test_extended_prediction_accuracy_at_time(self, end_time_ms, min_observations=20):
        """
        Test prediction accuracy using extended data up to end_time_ms
        
        Args:
            end_time_ms: End time for extended analysis
            min_observations: Minimum observations required per day
            
        Returns:
            Dictionary with accuracy metrics
        """
        time_str = self.ms_to_time(end_time_ms)
        extended_duration_minutes = (end_time_ms - self.extended_start_ms) / 60000
        
        # Calculate how much of the original regime period is covered
        regime_coverage = min(1.0, max(0.0, 
            (min(end_time_ms, self.regime_end_ms) - max(self.extended_start_ms, self.regime_start_ms)) / 
            (self.regime_end_ms - self.regime_start_ms)))
        
        # Calculate additional trading time available
        if end_time_ms <= self.regime_start_ms:
            additional_trading_time = (self.regime_end_ms - self.regime_start_ms) / 60000  # Full regime period available
        else:
            additional_trading_time = max(0, (self.regime_end_ms - end_time_ms) / 60000)
        
        print(f"Testing extended prediction at {time_str}")
        print(f"  Extended data duration: {extended_duration_minutes:.1f} minutes")
        print(f"  Regime period coverage: {regime_coverage:.1%}")
        print(f"  Additional trading time: {additional_trading_time:.1f} minutes")
        
        # Extract features using extended period data
        extended_features = self.extract_extended_features(end_time_ms)
        
        if extended_features is None or len(extended_features) == 0:
            print(f"  No data available for extended period ending at {time_str}")
            return None
        
        # Filter days with sufficient observations
        sufficient_data = extended_features[
            extended_features['num_observations'] >= min_observations
        ].copy()
        
        if len(sufficient_data) == 0:
            print(f"  No days with sufficient data at {time_str} (min {min_observations} observations)")
            return None
        
        print(f"  Testing on {len(sufficient_data)} days with sufficient data")
        
        # Predict regimes
        predicted_regimes = self.predict_regimes_for_extended_data(sufficient_data)
        
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
        
        result = {
            'end_time_ms': end_time_ms,
            'end_time_str': time_str,
            'extended_duration_minutes': extended_duration_minutes,
            'regime_coverage': regime_coverage,
            'additional_trading_time_minutes': additional_trading_time,
            'n_test_days': len(true_regimes),
            'overall_accuracy': accuracy,
            'n_regimes_detected': len(unique_regimes),
            **regime_accuracies
        }
        
        print(f"  Overall accuracy: {accuracy:.3f} ({len(true_regimes)} days tested)")
        print(f"  Regimes detected: {len(unique_regimes)} unique regimes")
        print(f"  Additional trading time: {additional_trading_time:.1f} minutes")
        
        return result
    
    def run_extended_accuracy_test(self, test_intervals=25, min_duration_minutes=5):
        """
        Run extended prediction accuracy test across multiple time points
        
        Args:
            test_intervals: Number of time points to test
            min_duration_minutes: Minimum duration from extended start before testing
        """
        print("Starting extended regime prediction accuracy test...")
        print(f"Testing {test_intervals} time intervals with minimum {min_duration_minutes} minutes duration")
        
        # Calculate test times from extended_start to regime_end
        min_duration_ms = min_duration_minutes * 60000
        start_test_time = self.extended_start_ms + min_duration_ms
        
        test_times = np.linspace(start_test_time, self.regime_end_ms, test_intervals)
        test_times = test_times.astype(int)
        
        # Add key milestone times if not already included
        key_times = [self.regime_start_ms]  # Add the original regime start time
        for key_time in key_times:
            if key_time not in test_times:
                test_times = np.append(test_times, key_time)
        
        test_times = np.unique(np.sort(test_times))
        
        # Run tests at each time point
        self.accuracy_results = []
        
        for i, end_time_ms in enumerate(test_times):
            print(f"\n--- Extended Test {i+1}/{len(test_times)} ---")
            
            result = self.test_extended_prediction_accuracy_at_time(end_time_ms)
            
            if result is not None:
                self.accuracy_results.append(result)
        
        print(f"\nCompleted extended accuracy testing with {len(self.accuracy_results)} successful tests")
        
        return self.accuracy_results
    
    def analyze_and_save_extended_results(self):
        """Analyze and save the extended accuracy test results"""
        if len(self.accuracy_results) == 0:
            print("No results to analyze")
            return
        
        print("Analyzing extended accuracy test results...")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.accuracy_results)
        
        # Save detailed results
        results_file = self.output_dir / 'extended_regime_prediction_accuracy_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Saved detailed results to: {results_file}")
        
        # Create summary analysis
        # Find results at key time points
        regime_start_results = results_df[results_df['end_time_ms'] == self.regime_start_ms]
        early_results = results_df[results_df['end_time_ms'] < self.regime_start_ms]
        
        summary_stats = {
            'test_intervals': len(results_df),
            'extended_start_time': self.ms_to_time(self.extended_start_ms),
            'regime_start_time': self.ms_to_time(self.regime_start_ms),
            'regime_end_time': self.ms_to_time(self.regime_end_ms),
            'additional_trading_time_minutes': (self.regime_start_ms - self.extended_start_ms) / 60000,
            'min_accuracy': results_df['overall_accuracy'].min(),
            'max_accuracy': results_df['overall_accuracy'].max(),
            'final_accuracy': results_df['overall_accuracy'].iloc[-1] if len(results_df) > 0 else None,
            'mean_accuracy': results_df['overall_accuracy'].mean(),
            'accuracy_at_regime_start': regime_start_results['overall_accuracy'].iloc[0] if len(regime_start_results) > 0 else None,
            'best_early_accuracy': early_results['overall_accuracy'].max() if len(early_results) > 0 else None,
            'best_early_time': None,
            'viable_for_trading': False
        }
        
        # Find best early prediction
        if len(early_results) > 0:
            best_early_idx = early_results['overall_accuracy'].idxmax()
            best_early_row = results_df.loc[best_early_idx]
            summary_stats['best_early_time'] = best_early_row['end_time_str']
            summary_stats['best_early_additional_time'] = best_early_row['additional_trading_time_minutes']
            
            # Consider viable if early accuracy is reasonable (>= 50%) with good trading time (>= 30 min)
            if (best_early_row['overall_accuracy'] >= 0.5 and 
                best_early_row['additional_trading_time_minutes'] >= 30):
                summary_stats['viable_for_trading'] = True
        
        # Print summary
        print("\n" + "="*70)
        print("EXTENDED REGIME PREDICTION ANALYSIS")
        print("="*70)
        print(f"Extended data collection: {summary_stats['extended_start_time']} to {summary_stats['regime_end_time']}")
        print(f"Original regime period: {summary_stats['regime_start_time']} to {summary_stats['regime_end_time']}")
        print(f"Additional trading time gained: {summary_stats['additional_trading_time_minutes']:.1f} minutes")
        print(f"Total test intervals: {summary_stats['test_intervals']}")
        
        print(f"\n📊 ACCURACY SUMMARY:")
        print(f"Accuracy range: {summary_stats['min_accuracy']:.3f} to {summary_stats['max_accuracy']:.3f}")
        print(f"Mean accuracy: {summary_stats['mean_accuracy']:.3f}")
        print(f"Final accuracy (end of extended period): {summary_stats['final_accuracy']:.3f}")
        
        if summary_stats['accuracy_at_regime_start'] is not None:
            print(f"Accuracy at original regime start: {summary_stats['accuracy_at_regime_start']:.3f}")
        
        if summary_stats['best_early_accuracy'] is not None:
            print(f"Best early prediction accuracy: {summary_stats['best_early_accuracy']:.3f} at {summary_stats['best_early_time']}")
            print(f"Trading time available: {summary_stats['best_early_additional_time']:.1f} minutes")
        
        print(f"\n🚀 TRADING VIABILITY:")
        if summary_stats['viable_for_trading']:
            print("✅ EXTENDED PERIOD STRATEGY IS VIABLE!")
            print(f"   - Best early prediction: {summary_stats['best_early_accuracy']:.1%} at {summary_stats['best_early_time']}")
            print(f"   - Additional trading time: {summary_stats['best_early_additional_time']:.0f} minutes")
        else:
            print("⚠️  EXTENDED PERIOD STRATEGY HAS LIMITED BENEFIT")
            if summary_stats['best_early_accuracy'] is not None:
                if summary_stats['best_early_accuracy'] < 0.5:
                    print("   - Early prediction accuracy too low (<50%)")
                else:
                    print("   - Insufficient additional trading time")
            
        # Save summary
        summary_file = self.output_dir / 'extended_accuracy_test_summary.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"\nSaved summary to: {summary_file}")
        
        # Create visualization if possible
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Accuracy over time
            plt.subplot(2, 2, 1)
            plt.plot(results_df['extended_duration_minutes'], results_df['overall_accuracy'], 'b-o', markersize=4)
            plt.axvline(x=(self.regime_start_ms - self.extended_start_ms)/60000, color='r', linestyle='--', 
                       label='Original regime start')
            plt.xlabel('Extended Data Duration (minutes)')
            plt.ylabel('Prediction Accuracy')
            plt.title('Extended Period: Accuracy vs Data Duration')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 2: Accuracy vs additional trading time
            plt.subplot(2, 2, 2)
            plt.plot(results_df['additional_trading_time_minutes'], results_df['overall_accuracy'], 'g-s', markersize=4)
            plt.xlabel('Additional Trading Time (minutes)')
            plt.ylabel('Prediction Accuracy')
            plt.title('Trade-off: Accuracy vs Trading Time')
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Regime coverage vs accuracy
            plt.subplot(2, 2, 3)
            plt.plot(results_df['regime_coverage'], results_df['overall_accuracy'], 'r-^', markersize=4)
            plt.xlabel('Regime Period Coverage')
            plt.ylabel('Prediction Accuracy')
            plt.title('Accuracy vs Regime Period Coverage')
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Number of test days over time
            plt.subplot(2, 2, 4)
            plt.plot(results_df['extended_duration_minutes'], results_df['n_test_days'], 'm-d', markersize=4)
            plt.xlabel('Extended Data Duration (minutes)')
            plt.ylabel('Number of Test Days')
            plt.title('Data Availability')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = self.output_dir / 'extended_accuracy_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved extended analysis plot to: {plot_file}")
            
        except ImportError:
            print("matplotlib not available - skipping plot generation")
        except Exception as e:
            print(f"Could not create plot: {e}")
        
        return summary_stats

def time_to_ms(time_str):
    """Convert time string (HH:MM) to milliseconds of day"""
    hours, minutes = map(int, time_str.split(':'))
    return hours * 3600000 + minutes * 60000

def main():
    parser = argparse.ArgumentParser(description='Extended Period Market Regime Prediction Test')
    parser.add_argument('--data_path', default='data/history_spot_quote.csv',
                       help='Path to trading data CSV file')
    parser.add_argument('--reference_model_path', default='../../market_regime/gmm/daily',
                       help='Path to reference GMM model directory')
    parser.add_argument('--output_dir', default='../../market_regime/extended_prediction_accuracy',
                       help='Output directory for test results')
    parser.add_argument('--extended_start', default='09:30',
                       help='Extended data collection start time (HH:MM format)')
    parser.add_argument('--regime_start', default='10:35',
                       help='Original regime period start time (HH:MM format)')
    parser.add_argument('--regime_end', default='12:00',
                       help='Original regime period end time (HH:MM format)')
    parser.add_argument('--test_intervals', type=int, default=25,
                       help='Number of time points to test')
    parser.add_argument('--min_duration', type=int, default=5,
                       help='Minimum duration (minutes) from extended start before testing')
    
    args = parser.parse_args()
    
    # Convert times to milliseconds
    extended_start_ms = time_to_ms(args.extended_start)
    regime_start_ms = time_to_ms(args.regime_start)
    regime_end_ms = time_to_ms(args.regime_end)
    
    # Initialize tester
    tester = ExtendedPeriodRegimePredictionTester(
        data_path=args.data_path,
        reference_model_path=args.reference_model_path,
        output_dir=args.output_dir,
        regime_start_ms=regime_start_ms,
        regime_end_ms=regime_end_ms,
        extended_start_ms=extended_start_ms
    )
    
    try:
        # Load reference models and data
        tester.load_reference_models()
        tester.load_trading_data()
        
        # Run extended accuracy test
        results = tester.run_extended_accuracy_test(
            test_intervals=args.test_intervals,
            min_duration_minutes=args.min_duration
        )
        
        # Analyze and save results
        summary = tester.analyze_and_save_extended_results()
        
        print("\n" + "="*70)
        print("EXTENDED PERIOD TEST COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
