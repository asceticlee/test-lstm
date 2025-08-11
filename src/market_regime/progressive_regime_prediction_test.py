#!/usr/bin/env python3
"""
Progressive Market Regime Prediction Test (9:30 Start)

This script tests how accurately we can predict the final market regime 
(clustered on 9:30-12:00 data) using progressively longer periods starting 
from 9:30 AM with different end times.

Tests periods like:
- 9:30-10:00 vs 9:30-12:00 regime
- 9:30-10:30 vs 9:30-12:00 regime  
- 9:30-11:00 vs 9:30-12:00 regime
- 9:30-11:30 vs 9:30-12:00 regime
- 9:30-12:00 vs 9:30-12:00 regime (should be 100%)

Usage:
    python progressive_regime_prediction_test.py [--test_intervals N]
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

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import our statistical features module
from market_data_stat.statistical_features import StatisticalFeatureExtractor

class ProgressiveRegimePredictionTester:
    """
    Test market regime prediction accuracy using different end times from 9:30 AM start
    """
    
    def __init__(self, data_path='data/history_spot_quote.csv', 
                 regime_assignments_path='../../market_regime/gmm/daily/daily_regime_assignments.csv',
                 models_path='../../market_regime/gmm/daily',
                 output_dir='../../market_regime/progressive_prediction_test'):
        """
        Initialize the progressive prediction tester
        
        Args:
            data_path: Path to history_spot_quote.csv
            regime_assignments_path: Path to daily regime assignments from GMM clustering (9:30-12:00)
            models_path: Path to saved GMM models
            output_dir: Directory to save test results
        """
        # Get the absolute path to the script directory
        script_dir = Path(__file__).parent.absolute()
        
        # Handle both absolute and relative paths
        self.data_path = Path(data_path) if Path(data_path).is_absolute() else script_dir / ".." / ".." / data_path
        self.regime_assignments_path = Path(regime_assignments_path) if Path(regime_assignments_path).is_absolute() else script_dir / regime_assignments_path
        self.models_path = Path(models_path) if Path(models_path).is_absolute() else script_dir / models_path
        self.output_dir = Path(output_dir) if Path(output_dir).is_absolute() else script_dir / output_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Time constants (9:30 AM to 12:00 PM)
        self.start_time_ms = 34200000  # 9:30 AM
        self.full_end_time_ms = 43200000  # 12:00 PM
        
        # Load existing models and data
        self.load_existing_models()
        self.load_regime_assignments()
        self.load_raw_data()
        
        # Statistical feature extractor
        self.feature_extractor = StatisticalFeatureExtractor()
        
        print(f"Progressive Regime Prediction Tester initialized")
        print(f"Data path: {self.data_path}")
        print(f"Base period: 9:30 AM to 12:00 PM")
        print(f"Regime assignments: {self.regime_assignments_path}")
        print(f"Models path: {self.models_path}")
        print(f"Output directory: {self.output_dir}")
    
    def load_existing_models(self):
        """Load the trained GMM model, scaler, and PCA from the clustering"""
        print("Loading existing GMM models (trained on 9:30-12:00)...")
        
        try:
            self.gmm_model = joblib.load(self.models_path / 'gmm_model.pkl')
            self.scaler = joblib.load(self.models_path / 'feature_scaler.pkl')
            self.pca = joblib.load(self.models_path / 'pca_model.pkl')
            
            # Load clustering info
            import json
            with open(self.models_path / 'clustering_info.json', 'r') as f:
                self.clustering_info = json.load(f)
            
            self.feature_names = self.clustering_info['feature_names']
            self.reference_time_ms = self.clustering_info['reference_time_ms']
            
            # Get overnight feature settings from the original clustering
            self.include_overnight_gap = self.clustering_info.get('include_overnight_gap', False)
            self.overnight_indicators = self.clustering_info.get('overnight_indicators', False)
            
            print(f"Loaded models trained with {self.clustering_info['n_regimes']} regimes")
            print(f"Training period: {self.ms_to_time(self.clustering_info['trading_start_ms'])} to {self.ms_to_time(self.clustering_info['trading_end_ms'])}")
            print(f"Reference time: {self.ms_to_time(self.reference_time_ms)}")
            print(f"Feature count: {len(self.feature_names)}")
            print(f"Include overnight gap: {self.include_overnight_gap}")
            print(f"Overnight indicators: {self.overnight_indicators}")
            
        except Exception as e:
            raise FileNotFoundError(f"Could not load existing models: {e}")
    
    def load_regime_assignments(self):
        """Load the daily regime assignments from GMM clustering (9:30-12:00)"""
        print("Loading regime assignments...")
        
        if not self.regime_assignments_path.exists():
            raise FileNotFoundError(f"Regime assignments file not found: {self.regime_assignments_path}")
        
        self.regime_assignments = pd.read_csv(self.regime_assignments_path)
        print(f"Loaded regime assignments for {len(self.regime_assignments)} trading days")
        
        # Display regime distribution
        regime_counts = self.regime_assignments['Regime'].value_counts().sort_index()
        print("True regime distribution (9:30-12:00 clustering):")
        for regime, count in regime_counts.items():
            percentage = count / len(self.regime_assignments) * 100
            print(f"  Regime {regime}: {count} days ({percentage:.1f}%)")
    
    def load_raw_data(self):
        """Load the raw trading data"""
        print("Loading raw trading data...")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.raw_data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.raw_data):,} rows of raw data")
        
        # Filter for the time range we're interested in (9:30-12:00)
        self.raw_data = self.raw_data[
            (self.raw_data['ms_of_day'] >= self.start_time_ms) &
            (self.raw_data['ms_of_day'] <= self.full_end_time_ms)
        ].copy()
        
        print(f"Filtered to {len(self.raw_data):,} rows for 9:30-12:00 period")
        
        # Sort by date and time
        self.raw_data = self.raw_data.sort_values(['trading_day', 'ms_of_day']).reset_index(drop=True)
        
        # Get unique trading days
        unique_days = self.raw_data['trading_day'].unique()
        print(f"Found {len(unique_days)} unique trading days")
    
    def ms_to_time(self, ms):
        """Convert milliseconds of day to readable time format"""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        return f"{hours:02d}:{minutes:02d}"
    
    def time_to_ms(self, time_str):
        """Convert time string to milliseconds of day"""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 3600000 + minutes * 60000
    
    def calculate_overnight_indicators(self, daily_data):
        """
        Calculate overnight-specific technical indicators (not basic gap features)
        
        Basic gap features are handled by StatisticalFeatureExtractor.
        This method only adds advanced overnight indicators like gap fill analysis, opening patterns, etc.
        
        Args:
            daily_data: DataFrame with trading data including previous day's close
            
        Returns:
            Dictionary with overnight indicator features
        """
        overnight_features = {}
        
        try:
            # Get the first price of current day (market open)
            first_price = daily_data['mid'].iloc[0] if len(daily_data) > 0 else np.nan
            
            # Get previous day's close - simple approach using prev_close column
            if 'prev_close' in daily_data.columns:
                prev_close = daily_data['prev_close'].iloc[0]
            else:
                prev_close = np.nan
            
            # Calculate basic overnight gap for use in indicators
            if not np.isnan(prev_close) and not np.isnan(first_price) and prev_close != 0:
                overnight_gap = (first_price - prev_close) / prev_close
            else:
                overnight_gap = 0
            
            # Calculate overnight-specific technical indicators
            prices = daily_data['mid'].values
            if len(prices) > 10 and not np.isnan(prev_close) and prev_close != 0:  # Need sufficient data
                
                # Gap fill analysis: does price return to previous close level?
                gap_direction = 1 if first_price > prev_close else -1
                
                # Check if gap gets filled during the day
                if gap_direction > 0:  # Gap up - check if price goes back down to prev_close
                    gap_filled = np.any(prices <= prev_close)
                else:  # Gap down - check if price goes back up to prev_close
                    gap_filled = np.any(prices >= prev_close)
                
                overnight_features['gap_filled'] = 1 if gap_filled else 0
                
                # Time to gap fill (in 5-minute intervals from start)
                if gap_filled:
                    if gap_direction > 0:
                        fill_idx = np.argmax(prices <= prev_close)
                    else:
                        fill_idx = np.argmax(prices >= prev_close)
                    overnight_features['gap_fill_time_minutes'] = fill_idx * 5  # Assuming 5-min intervals
                else:
                    overnight_features['gap_fill_time_minutes'] = 999  # Not filled
                
                # Opening direction (first 30 minutes): does it continue gap direction?
                first_30min_data = prices[:min(6, len(prices))]  # First 6 intervals (30 min)
                if len(first_30min_data) > 1:
                    opening_move = (first_30min_data[-1] - first_price) / first_price if first_price != 0 else 0
                    overnight_features['opening_direction'] = opening_move
                    overnight_features['opening_momentum'] = abs(opening_move)
                    
                    # Gap continuation: same direction as overnight gap
                    gap_continuation = (overnight_gap * opening_move > 0) if overnight_gap != 0 else False
                    overnight_features['gap_continuation'] = 1 if gap_continuation else 0
            else:
                # Default values when insufficient data
                overnight_features['gap_filled'] = 0
                overnight_features['gap_fill_time_minutes'] = 0
                overnight_features['opening_direction'] = 0
                overnight_features['opening_momentum'] = 0
                overnight_features['gap_continuation'] = 0
                
        except Exception as e:
            print(f"Warning: Error calculating overnight indicator features: {e}")
            # Ensure all expected features exist with default values
            overnight_indicator_features = [
                'gap_filled', 'gap_fill_time_minutes', 'opening_direction',
                'opening_momentum', 'gap_continuation'
            ]
            for feature in overnight_indicator_features:
                if feature not in overnight_features:
                    overnight_features[feature] = 0
        
        return overnight_features
    
    def extract_features_for_period(self, end_time_ms):
        """Extract features for each trading day using 9:30 to end_time_ms"""
        print(f"Extracting features for period 9:30 to {self.ms_to_time(end_time_ms)}...")
        
        # Filter data for this time period (always start from 9:30)
        period_data = self.raw_data[
            (self.raw_data['ms_of_day'] >= self.start_time_ms) &
            (self.raw_data['ms_of_day'] <= end_time_ms)
        ].copy()
        
        if len(period_data) == 0:
            print(f"No data available for period")
            return None
        
        # Extract features using the same methodology as the original clustering
        daily_features = self.feature_extractor.extract_daily_features(
            daily_data=period_data,
            price_column='mid',
            volume_column='volume' if 'volume' in period_data.columns else None,
            reference_time_ms=self.reference_time_ms,  # Use same reference time as original (9:30)
            trading_day_column='trading_day',
            time_column='ms_of_day',
            use_relative=True,
            include_overnight_gap=self.include_overnight_gap  # Use same setting as training
        )
        
        # Add overnight indicator features if they were used in training
        if self.overnight_indicators:
            # Group data by trading day and calculate overnight indicator features
            overnight_features_list = []
            daily_groups = period_data.groupby('trading_day')
            
            for trading_day, day_data in daily_groups:
                # Calculate overnight indicator features for this day
                overnight_features = self.calculate_overnight_indicators(day_data)
                overnight_features['trading_day'] = trading_day
                overnight_features_list.append(overnight_features)
            
            # Convert to DataFrame and merge with existing features
            if overnight_features_list:
                overnight_df = pd.DataFrame(overnight_features_list)
                
                # Merge with existing daily features
                daily_features = daily_features.merge(
                    overnight_df, 
                    on='trading_day', 
                    how='left'
                )
                
                # Fill any missing overnight features with zeros
                overnight_columns = [col for col in overnight_df.columns if col != 'trading_day']
                for col in overnight_columns:
                    if col in daily_features.columns:
                        daily_features[col] = daily_features[col].fillna(0)
        
        if daily_features is None or len(daily_features) == 0:
            print(f"No features extracted for period")
            return None
        
        print(f"Extracted features for {len(daily_features)} trading days")
        return daily_features
    
    def predict_regimes_for_period(self, daily_features):
        """Predict regimes using the trained model for the given features"""
        if daily_features is None or len(daily_features) == 0:
            return np.array([])
        
        # Select feature columns that match the trained model
        available_features = [f for f in self.feature_names if f in daily_features.columns]
        missing_features = [f for f in self.feature_names if f not in daily_features.columns]
        
        if missing_features:
            print(f"Warning: Missing features for this period: {len(missing_features)} features")
            print(f"Available features: {len(available_features)}/{len(self.feature_names)}")
        
        # Get feature matrix
        X = daily_features[available_features].copy()
        
        # Handle missing features by adding zeros
        for feature in missing_features:
            X[feature] = 0
        
        # Reorder columns to match training order
        X = X[self.feature_names]
        
        # Handle missing values
        X = X.fillna(0)
        
        try:
            # Apply the same preprocessing as training
            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled)
            
            # Predict regimes
            predicted_regimes = self.gmm_model.predict(X_pca)
            
            return predicted_regimes
            
        except Exception as e:
            print(f"Error predicting regimes: {e}")
            return np.array([])
    
    def test_prediction_accuracy_for_end_time(self, end_time_ms):
        """Test prediction accuracy using 9:30 to end_time_ms vs full day regime"""
        period_minutes = (end_time_ms - self.start_time_ms) / 60000
        
        print(f"\nTesting period: 9:30 to {self.ms_to_time(end_time_ms)} ({period_minutes:.0f} minutes)")
        
        try:
            # Extract features for this period
            period_features = self.extract_features_for_period(end_time_ms)
            
            if period_features is None or len(period_features) == 0:
                print(f"  No features available for this period")
                return None
            
            # Only test days that exist in both datasets
            common_days = set(period_features['trading_day']) & set(self.regime_assignments['trading_day'])
            
            if len(common_days) == 0:
                print(f"  No common trading days found")
                return None
            
            # Filter both datasets to common days
            period_features_filtered = period_features[period_features['trading_day'].isin(common_days)].copy()
            regime_assignments_filtered = self.regime_assignments[self.regime_assignments['trading_day'].isin(common_days)].copy()
            
            # Sort both by trading_day to ensure alignment
            period_features_filtered = period_features_filtered.sort_values('trading_day').reset_index(drop=True)
            regime_assignments_filtered = regime_assignments_filtered.sort_values('trading_day').reset_index(drop=True)
            
            # Predict regimes for this period
            predicted_regimes = self.predict_regimes_for_period(period_features_filtered)
            
            if len(predicted_regimes) == 0:
                print(f"  No predictions generated")
                return None
            
            true_regimes = regime_assignments_filtered['Regime'].values
            
            # Calculate accuracy
            accuracy = accuracy_score(true_regimes, predicted_regimes)
            
            # Calculate per-regime accuracy
            regime_accuracies = {}
            unique_regimes = np.unique(true_regimes)
            
            for regime in unique_regimes:
                regime_mask = true_regimes == regime
                if np.sum(regime_mask) > 0:
                    regime_predictions = predicted_regimes[regime_mask]
                    regime_accuracy = np.mean(regime_predictions == regime)
                    regime_accuracies[f'regime_{regime}_accuracy'] = regime_accuracy
                    regime_accuracies[f'regime_{regime}_count'] = np.sum(regime_mask)
            
            # Create confusion matrix for detailed analysis
            conf_matrix = confusion_matrix(true_regimes, predicted_regimes)
            
            # Store results
            result = {
                'end_time': self.ms_to_time(end_time_ms),
                'end_time_ms': end_time_ms,
                'period_minutes': period_minutes,
                'completion_percentage': (period_minutes / 150) * 100,  # 150 min = 9:30-12:00
                'num_days': len(common_days),
                'overall_accuracy': accuracy,
                'unique_regimes_predicted': len(np.unique(predicted_regimes)),
                'unique_regimes_true': len(unique_regimes),
                **regime_accuracies
            }
            
            print(f"  Overall accuracy: {accuracy:.3f} ({len(common_days)} days)")
            print(f"  Regimes predicted: {len(np.unique(predicted_regimes))}, True regimes: {len(unique_regimes)}")
            
            # Print per-regime accuracy
            for regime in unique_regimes:
                if f'regime_{regime}_accuracy' in regime_accuracies:
                    acc = regime_accuracies[f'regime_{regime}_accuracy']
                    count = regime_accuracies[f'regime_{regime}_count']
                    print(f"  Regime {regime}: {acc:.3f} accuracy ({count} days)")
            
            return result
            
        except Exception as e:
            print(f"  Error testing period: {e}")
            return None
    
    def test_progressive_accuracy(self, test_intervals=15):
        """Test prediction accuracy for different end times with specified intervals"""
        print(f"Testing progressive prediction accuracy with {test_intervals}-minute intervals...")
        
        # Generate test end times every N minutes from 9:30
        test_end_times = []
        current_time = self.start_time_ms + (test_intervals * 60000)  # First test after N minutes
        
        while current_time <= self.full_end_time_ms:
            test_end_times.append(current_time)
            current_time += (test_intervals * 60000)
        
        # Ensure we include the full period (12:00)
        if test_end_times[-1] != self.full_end_time_ms:
            test_end_times.append(self.full_end_time_ms)
        
        print(f"Testing {len(test_end_times)} time periods:")
        for end_time in test_end_times:
            period_minutes = (end_time - self.start_time_ms) / 60000
            print(f"  9:30 to {self.ms_to_time(end_time)} ({period_minutes:.0f} minutes)")
        
        # Test each period
        results = []
        
        for end_time in test_end_times:
            result = self.test_prediction_accuracy_for_end_time(end_time)
            if result is not None:
                results.append(result)
        
        # Convert results to DataFrame
        self.accuracy_results = pd.DataFrame(results)
        
        print(f"\nCompleted testing with {len(results)} successful periods")
        
        return self.accuracy_results
    
    def analyze_accuracy_progression(self):
        """Analyze how accuracy improves with longer periods"""
        print("\nAnalyzing accuracy progression...")
        
        if self.accuracy_results is None or len(self.accuracy_results) == 0:
            print("No accuracy results to analyze")
            return
        
        # Find key thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        threshold_times = {}
        
        for threshold in thresholds:
            achieving_periods = self.accuracy_results[self.accuracy_results['overall_accuracy'] >= threshold]
            if len(achieving_periods) > 0:
                first_achievement = achieving_periods.iloc[0]
                threshold_times[threshold] = {
                    'time': first_achievement['end_time'],
                    'period_minutes': first_achievement['period_minutes'],
                    'accuracy': first_achievement['overall_accuracy']
                }
        
        print("\nAccuracy Milestones:")
        print("-" * 60)
        for threshold in sorted(threshold_times.keys()):
            info = threshold_times[threshold]
            print(f"{threshold*100:3.0f}% accuracy: First achieved at {info['time']} ({info['period_minutes']:.0f} min period) with {info['accuracy']:.3f}")
        
        # Find best accuracy and when achieved
        best_accuracy_idx = self.accuracy_results['overall_accuracy'].idxmax()
        best_result = self.accuracy_results.iloc[best_accuracy_idx]
        
        print(f"\nBest accuracy: {best_result['overall_accuracy']:.3f} at {best_result['end_time']} ({best_result['period_minutes']:.0f} min period)")
        
        # Calculate accuracy improvement rates
        if len(self.accuracy_results) > 1:
            accuracy_diffs = np.diff(self.accuracy_results['overall_accuracy'])
            time_diffs = np.diff(self.accuracy_results['period_minutes'])
            improvement_rates = accuracy_diffs / time_diffs
            
            print(f"\nAverage accuracy improvement rate: {np.mean(improvement_rates):.4f} per minute")
            print(f"Best improvement rate: {np.max(improvement_rates):.4f} per minute")
        
        return threshold_times
    
    def create_visualizations(self):
        """Create comprehensive visualization plots"""
        print("Creating visualizations...")
        
        if self.accuracy_results is None or len(self.accuracy_results) == 0:
            print("No results to visualize")
            return
        
        # Create comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall accuracy progression
        ax1.plot(self.accuracy_results['period_minutes'], self.accuracy_results['overall_accuracy'], 
                'b-o', linewidth=3, markersize=8, label='Accuracy')
        ax1.set_xlabel('Period Length (minutes from 9:30)')
        ax1.set_ylabel('Overall Accuracy')
        ax1.set_title('Regime Prediction Accuracy vs Period Length\n(9:30 Start, Progressive End Times)')
        ax1.grid(True, alpha=0.3)
        
        # Add horizontal lines for key thresholds
        for threshold in [0.5, 0.7, 0.9]:
            ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label=f'{threshold*100:.0f}%')
        
        # Annotate key points
        for i, row in self.accuracy_results.iterrows():
            if i % max(1, len(self.accuracy_results)//6) == 0:  # Annotate every few points
                ax1.annotate(f"{row['end_time']}\n{row['overall_accuracy']:.3f}", 
                           (row['period_minutes'], row['overall_accuracy']),
                           textcoords="offset points", xytext=(0,15), ha='center', fontsize=9)
        
        ax1.legend()
        
        # 2. Accuracy by completion percentage
        ax2.plot(self.accuracy_results['completion_percentage'], self.accuracy_results['overall_accuracy'], 
                'g-s', linewidth=3, markersize=8)
        ax2.set_xlabel('Trading Day Completion (%)')
        ax2.set_ylabel('Overall Accuracy')
        ax2.set_title('Accuracy vs Day Completion Percentage')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect (100%)')
        ax2.legend()
        
        # 3. Number of test days over time
        ax3.plot(self.accuracy_results['period_minutes'], self.accuracy_results['num_days'], 
                'purple', marker='d', linewidth=2, markersize=6)
        ax3.set_xlabel('Period Length (minutes)')
        ax3.set_ylabel('Number of Test Days')
        ax3.set_title('Data Availability vs Period Length')
        ax3.grid(True, alpha=0.3)
        
        # 4. Accuracy improvement rate
        if len(self.accuracy_results) > 1:
            accuracy_diff = np.diff(self.accuracy_results['overall_accuracy'])
            period_diff = np.diff(self.accuracy_results['period_minutes'])
            improvement_rate = accuracy_diff / period_diff
            
            ax4.plot(self.accuracy_results['period_minutes'][1:], improvement_rate, 
                    'red', marker='o', linewidth=2, markersize=6)
            ax4.set_xlabel('Period Length (minutes)')
            ax4.set_ylabel('Accuracy Improvement per Minute')
            ax4.set_title('Marginal Accuracy Improvement Rate')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'progressive_accuracy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed accuracy table visualization
        plt.figure(figsize=(14, 10))
        
        # Bar chart showing accuracy for each time period
        bars = plt.bar(range(len(self.accuracy_results)), self.accuracy_results['overall_accuracy'], 
                      color=plt.cm.RdYlGn(self.accuracy_results['overall_accuracy']))
        
        plt.xlabel('Test Period')
        plt.ylabel('Overall Accuracy')
        plt.title('Progressive Regime Prediction Accuracy (9:30 AM Start)\nComparing Different End Times vs Full Day (9:30-12:00) Regime')
        
        # Customize x-axis
        xtick_labels = [f"{row['end_time']}\n({row['period_minutes']:.0f}min)" 
                       for _, row in self.accuracy_results.iterrows()]
        plt.xticks(range(len(self.accuracy_results)), xtick_labels, rotation=45, ha='right')
        
        # Add accuracy values on bars
        for i, (bar, row) in enumerate(zip(bars, self.accuracy_results.itertuples())):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f"{row.overall_accuracy:.3f}", ha='center', va='bottom', fontsize=9)
        
        # Add horizontal reference lines
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Match (100%)')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Baseline (50%)')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_progressive_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualizations to: {self.output_dir}")
    
    def save_results(self):
        """Save test results to CSV files"""
        print("Saving progressive accuracy test results...")
        
        if self.accuracy_results is None or len(self.accuracy_results) == 0:
            print("No results to save")
            return
        
        # Save main results
        results_file = self.output_dir / 'progressive_regime_prediction_results.csv'
        self.accuracy_results.to_csv(results_file, index=False)
        print(f"Saved accuracy results to: {results_file}")
        
        # Create summary report
        summary_lines = [
            "Progressive Market Regime Prediction Test Summary",
            "=" * 60,
            f"Test date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Base period: 9:30 AM to 12:00 PM (150 minutes)",
            f"Reference regime: Full day clustering (9:30-12:00)",
            f"Test strategy: Progressive end times from 9:30 start",
            f"Number of test periods: {len(self.accuracy_results)}",
            f"Number of trading days: {self.accuracy_results['num_days'].iloc[0] if len(self.accuracy_results) > 0 else 0}",
            "",
            "Progressive Accuracy Results:",
            "-" * 40,
        ]
        
        for _, row in self.accuracy_results.iterrows():
            summary_lines.append(f"9:30 to {row['end_time']} ({row['period_minutes']:3.0f} min, {row['completion_percentage']:5.1f}%): {row['overall_accuracy']:.3f}")
        
        # Add key findings
        if len(self.accuracy_results) > 0:
            best_idx = self.accuracy_results['overall_accuracy'].idxmax()
            best_result = self.accuracy_results.iloc[best_idx]
            
            summary_lines.extend([
                "",
                "Key Findings:",
                "-" * 20,
                f"Best accuracy: {best_result['overall_accuracy']:.3f} at {best_result['end_time']} ({best_result['period_minutes']:.0f} min)",
                f"Final accuracy (12:00): {self.accuracy_results['overall_accuracy'].iloc[-1]:.3f} (should be 1.000)",
            ])
            
            # Find when key thresholds are first achieved
            for threshold in [0.5, 0.7, 0.8, 0.9]:
                achieving = self.accuracy_results[self.accuracy_results['overall_accuracy'] >= threshold]
                if len(achieving) > 0:
                    first = achieving.iloc[0]
                    summary_lines.append(f"{threshold*100:.0f}% accuracy first achieved: {first['end_time']} ({first['period_minutes']:.0f} min, {first['completion_percentage']:.1f}% complete)")
                else:
                    summary_lines.append(f"{threshold*100:.0f}% accuracy: Never achieved in test periods")
            
            # Calculate accuracy growth rates
            if len(self.accuracy_results) > 1:
                total_improvement = self.accuracy_results['overall_accuracy'].iloc[-1] - self.accuracy_results['overall_accuracy'].iloc[0]
                total_time = self.accuracy_results['period_minutes'].iloc[-1] - self.accuracy_results['period_minutes'].iloc[0]
                avg_rate = total_improvement / total_time if total_time > 0 else 0
                
                summary_lines.extend([
                    "",
                    "Improvement Analysis:",
                    "-" * 20,
                    f"Total accuracy improvement: {total_improvement:.3f}",
                    f"Total time span: {total_time:.0f} minutes",
                    f"Average improvement rate: {avg_rate:.4f} per minute",
                ])
        
        # Save summary
        summary_file = self.output_dir / 'progressive_accuracy_summary.txt'
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"Saved summary report to: {summary_file}")
    
    def run_progressive_test(self, test_intervals=15):
        """Run the complete progressive accuracy testing pipeline"""
        print("Starting progressive regime prediction accuracy test...")
        print(f"Testing periods from 9:30 AM with {test_intervals}-minute intervals")
        print("Comparing against full day regime clustering (9:30-12:00)")
        
        # Step 1: Test progressive accuracy
        results = self.test_progressive_accuracy(test_intervals)
        
        if results is None or len(results) == 0:
            print("No results generated. Check data and model files.")
            return None
        
        # Step 2: Analyze accuracy progression
        self.analyze_accuracy_progression()
        
        # Step 3: Create visualizations
        self.create_visualizations()
        
        # Step 4: Save results
        self.save_results()
        
        print(f"\nProgressive accuracy test completed. Results saved to: {self.output_dir}")
        
        return self.accuracy_results

def main():
    parser = argparse.ArgumentParser(description='Progressive Market Regime Prediction Test (9:30 Start)')
    parser.add_argument('--test_intervals', type=int, default=15,
                       help='Test interval in minutes (default: 15)')
    parser.add_argument('--data_path', default='data/history_spot_quote.csv',
                       help='Path to trading data CSV file')
    parser.add_argument('--regime_assignments', default='../../market_regime/gmm/daily/daily_regime_assignments.csv',
                       help='Path to regime assignments CSV (9:30-12:00 clustering)')
    parser.add_argument('--models_path', default='../../market_regime/gmm/daily',
                       help='Path to trained models directory')
    parser.add_argument('--output_dir', default='../../market_regime/progressive_prediction_test',
                       help='Output directory for test results')
    
    args = parser.parse_args()
    
    tester = ProgressiveRegimePredictionTester(
        data_path=args.data_path,
        regime_assignments_path=args.regime_assignments,
        models_path=args.models_path,
        output_dir=args.output_dir
    )
    
    results = tester.run_progressive_test(args.test_intervals)
    
    print("\n" + "="*70)
    print("PROGRESSIVE ACCURACY TEST SUMMARY")
    print("="*70)
    
    if results is not None and len(results) > 0:
        print(f"Test intervals: {args.test_intervals} minutes")
        print(f"Number of test periods: {len(results)}")
        print(f"Trading days tested: {results['num_days'].iloc[0]}")
        
        print("\nProgressive Accuracy Results:")
        print("-" * 50)
        for _, row in results.iterrows():
            print(f"9:30 to {row['end_time']} ({row['period_minutes']:3.0f} min, {row['completion_percentage']:5.1f}%): {row['overall_accuracy']:.3f}")
        
        # Best result
        best_idx = results['overall_accuracy'].idxmax()
        best = results.iloc[best_idx]
        print(f"\nBest accuracy: {best['overall_accuracy']:.3f} at {best['end_time']} ({best['period_minutes']:.0f} minutes)")
        
        # Final result (should be 100%)
        final_result = results.iloc[-1]
        print(f"Final accuracy (12:00): {final_result['overall_accuracy']:.3f} (should be 1.000)")
        
        # Key thresholds
        print("\nAccuracy Milestones:")
        for threshold in [0.5, 0.7, 0.8, 0.9]:
            achieving = results[results['overall_accuracy'] >= threshold]
            if len(achieving) > 0:
                first = achieving.iloc[0]
                print(f"  {threshold*100:.0f}% accuracy: {first['end_time']} ({first['period_minutes']:.0f} min, {first['completion_percentage']:.1f}% complete)")
            else:
                print(f"  {threshold*100:.0f}% accuracy: Never achieved")
    else:
        print("No results generated")

if __name__ == "__main__":
    main()
