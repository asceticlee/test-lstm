#!/usr/bin/env python3
"""
Optimal Start Time Analysis for Regime Prediction

This script tests multiple start times and compares their accuracy growth curves
to find the optimal balance between forecast accuracy and trading feasibility.

Tests various start times from 9:00 AM to 10:30 AM and analyzes:
1. How accuracy grows over time for each start time
2. Maximum achievable accuracy for each start time
3. Time required to reach acceptable accuracy thresholds
4. Optimal start time for different trading requirements

Usage:
    python optimal_start_time_analysis.py --accuracy_threshold 0.5 --min_trading_time 30
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
from sklearn.metrics import accuracy_score
import joblib

# Import our statistical features module
from market_data_stat.statistical_features import StatisticalFeatureExtractor

class OptimalStartTimeAnalyzer:
    """
    Analyze optimal start times for regime prediction by comparing accuracy growth curves
    """
    
    def __init__(self, data_path='data/history_spot_quote.csv', 
                 reference_model_path='../../market_regime/gmm/daily',
                 output_dir='../../market_regime/optimal_start_time_analysis',
                 regime_start_ms=38100000, regime_end_ms=43200000):
        """
        Initialize the optimal start time analyzer
        
        Args:
            data_path: Path to history_spot_quote.csv
            reference_model_path: Path to trained GMM model directory
            output_dir: Directory to save analysis results
            regime_start_ms: Original regime period start (10:35 AM)
            regime_end_ms: Regime period end (12:00 PM)
        """
        # Get the absolute path to the script directory
        script_dir = Path(__file__).parent.absolute()
        
        # Resolve paths
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
        self.all_start_time_results = {}
        self.comparison_results = []
        
        print(f"Optimal Start Time Analyzer initialized")
        print(f"Regime period: {self.ms_to_time(regime_start_ms)} to {self.ms_to_time(regime_end_ms)}")
        print(f"Output directory: {self.output_dir}")
    
    def ms_to_time(self, ms):
        """Convert milliseconds of day to readable time format"""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        return f"{hours:02d}:{minutes:02d}"
    
    def time_to_ms(self, time_str):
        """Convert time string (HH:MM) to milliseconds of day"""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 3600000 + minutes * 60000
    
    def load_reference_models_and_data(self):
        """Load reference models and trading data"""
        print("Loading reference models and data...")
        
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
        
        # Load feature names
        clustering_info_file = self.reference_model_path / 'clustering_info.json'
        if clustering_info_file.exists():
            import json
            with open(clustering_info_file, 'r') as f:
                clustering_info = json.load(f)
                self.reference_feature_names = clustering_info.get('feature_names', [])
        
        # Load reference regime assignments
        self.reference_regimes = pd.read_csv(regime_file)
        
        # Load trading data
        self.raw_data = pd.read_csv(self.data_path)
        # Filter for extended period (we'll use the widest range possible)
        earliest_start = self.time_to_ms("09:00")  # 9:00 AM
        self.raw_data = self.raw_data[
            (self.raw_data['ms_of_day'] >= earliest_start) &
            (self.raw_data['ms_of_day'] <= self.regime_end_ms)
        ].copy()
        self.raw_data = self.raw_data.sort_values(['trading_day', 'ms_of_day']).reset_index(drop=True)
        
        print(f"Loaded {len(self.reference_regimes)} reference regime assignments")
        print(f"Loaded {len(self.raw_data):,} rows of trading data")
        
        return True
    
    def analyze_start_time_accuracy_curve(self, start_time_ms, test_intervals=20):
        """
        Analyze accuracy growth curve for a specific start time
        
        Args:
            start_time_ms: Start time in milliseconds
            test_intervals: Number of test points from start to end
            
        Returns:
            DataFrame with accuracy progression
        """
        start_time_str = self.ms_to_time(start_time_ms)
        print(f"\n📊 Analyzing accuracy curve for {start_time_str} start...")
        
        # Create test time points from start_time to regime_end
        test_times = np.linspace(start_time_ms, self.regime_end_ms, test_intervals)
        test_times = test_times.astype(int)
        
        results = []
        
        for i, end_time_ms in enumerate(test_times):
            end_time_str = self.ms_to_time(end_time_ms)
            
            # Calculate metrics
            duration_minutes = (end_time_ms - start_time_ms) / 60000
            regime_coverage = self.calculate_regime_coverage(start_time_ms, end_time_ms)
            time_to_regime_start = max(0, (self.regime_start_ms - end_time_ms) / 60000)
            time_from_regime_start = max(0, (end_time_ms - self.regime_start_ms) / 60000)
            
            # Extract features and predict
            accuracy = self.predict_accuracy_at_time(start_time_ms, end_time_ms)
            
            if accuracy is not None:
                result = {
                    'start_time_ms': start_time_ms,
                    'start_time_str': start_time_str,
                    'end_time_ms': end_time_ms,
                    'end_time_str': end_time_str,
                    'duration_minutes': duration_minutes,
                    'regime_coverage': regime_coverage,
                    'time_to_regime_start': time_to_regime_start,
                    'time_from_regime_start': time_from_regime_start,
                    'accuracy': accuracy,
                    'additional_trading_time': (self.regime_start_ms - start_time_ms) / 60000
                }
                results.append(result)
                
                if i % 5 == 0:  # Progress update
                    print(f"  {end_time_str}: {accuracy:.3f} accuracy ({duration_minutes:.1f}min data)")
        
        results_df = pd.DataFrame(results)
        print(f"Completed analysis for {start_time_str}: {len(results_df)} data points")
        
        return results_df
    
    def calculate_regime_coverage(self, start_time_ms, end_time_ms):
        """Calculate what percentage of the original regime period is covered"""
        overlap_start = max(start_time_ms, self.regime_start_ms)
        overlap_end = min(end_time_ms, self.regime_end_ms)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_duration = overlap_end - overlap_start
        regime_duration = self.regime_end_ms - self.regime_start_ms
        
        return overlap_duration / regime_duration
    
    def predict_accuracy_at_time(self, start_time_ms, end_time_ms, min_observations=20):
        """Predict regime accuracy using data from start_time to end_time"""
        
        # Filter data for the time period
        period_data = self.raw_data[
            (self.raw_data['ms_of_day'] >= start_time_ms) &
            (self.raw_data['ms_of_day'] <= end_time_ms)
        ].copy()
        
        if len(period_data) == 0:
            return None
        
        # Extract features
        try:
            daily_features = self.feature_extractor.extract_daily_features(
                daily_data=period_data,
                price_column='mid',
                volume_column='volume' if 'volume' in period_data.columns else None,
                reference_time_ms=start_time_ms,
                trading_day_column='trading_day',
                time_column='ms_of_day',
                use_relative=True,
                include_overnight_gap=False
            )
            
            # Filter days with sufficient observations
            sufficient_data = daily_features[
                daily_features['num_observations'] >= min_observations
            ].copy()
            
            if len(sufficient_data) == 0:
                return None
            
            # Predict regimes
            predicted_regimes = self.predict_regimes_for_features(sufficient_data)
            
            if len(predicted_regimes) == 0:
                return None
            
            # Get ground truth
            test_days = sufficient_data['trading_day'].values
            true_regimes = []
            
            for day in test_days:
                day_regime = self.reference_regimes[
                    self.reference_regimes['trading_day'] == day
                ]['Regime']
                
                if len(day_regime) > 0:
                    true_regimes.append(day_regime.iloc[0])
            
            true_regimes = np.array(true_regimes)
            
            if len(true_regimes) == 0:
                return None
            
            # Align predictions and calculate accuracy
            min_len = min(len(predicted_regimes), len(true_regimes))
            predicted_regimes = predicted_regimes[:min_len]
            true_regimes = true_regimes[:min_len]
            
            accuracy = accuracy_score(true_regimes, predicted_regimes)
            return accuracy
            
        except Exception as e:
            # Return None for failed predictions
            return None
    
    def predict_regimes_for_features(self, features):
        """Predict regimes using the reference model"""
        metadata_cols = ['trading_day', 'FromMsOfDay', 'ToMsOfDay', 'reference_time_ms', 
                        'reference_price', 'num_observations', 'time_range_minutes']
        
        available_features = [col for col in features.columns if col not in metadata_cols]
        
        if len(self.reference_feature_names) > 0:
            valid_features = [f for f in self.reference_feature_names if f in available_features]
            features_to_use = valid_features
        else:
            features_to_use = available_features
        
        if len(features_to_use) == 0:
            return np.array([])
        
        X = features[features_to_use].copy()
        X = X.fillna(0)
        
        # Handle missing features
        if len(self.reference_feature_names) > 0 and len(features_to_use) < len(self.reference_feature_names):
            full_X = np.zeros((len(X), len(self.reference_feature_names)))
            for i, feature in enumerate(self.reference_feature_names):
                if feature in features_to_use:
                    feature_idx = features_to_use.index(feature)
                    full_X[:, i] = X.iloc[:, feature_idx]
            X_array = full_X
        else:
            X_array = X.values
        
        try:
            X_scaled = self.reference_scaler.transform(X_array)
            X_pca = self.reference_pca.transform(X_scaled)
            predicted_regimes = self.reference_gmm.predict(X_pca)
            return predicted_regimes
        except:
            return np.array([])
    
    def run_comprehensive_start_time_analysis(self, start_times=None, test_intervals=20):
        """
        Run comprehensive analysis across multiple start times
        
        Args:
            start_times: List of start times to test (default: 9:00-10:30 in 15min intervals)
            test_intervals: Number of test points for each start time
        """
        if start_times is None:
            # Default: Test from 9:00 to 10:30 in 15-minute intervals
            start_times = [
                "09:00", "09:15", "09:30", "09:45", 
                "10:00", "10:15", "10:30"
            ]
        
        print("="*80)
        print("COMPREHENSIVE START TIME ANALYSIS")
        print("="*80)
        print(f"Testing {len(start_times)} start times with {test_intervals} intervals each")
        print(f"Start times: {', '.join(start_times)}")
        
        # Run analysis for each start time
        for start_time_str in start_times:
            start_time_ms = self.time_to_ms(start_time_str)
            
            # Skip if start time is after regime start
            if start_time_ms >= self.regime_start_ms:
                print(f"⚠️  Skipping {start_time_str} (after regime start)")
                continue
            
            try:
                results_df = self.analyze_start_time_accuracy_curve(start_time_ms, test_intervals)
                self.all_start_time_results[start_time_str] = results_df
            except Exception as e:
                print(f"❌ Failed to analyze {start_time_str}: {e}")
        
        print(f"\n✅ Completed analysis for {len(self.all_start_time_results)} start times")
        
        return self.all_start_time_results
    
    def compare_start_times(self, accuracy_thresholds=[0.3, 0.4, 0.5, 0.6]):
        """Compare different start times across various metrics"""
        print("\n📊 COMPARING START TIME PERFORMANCE...")
        
        comparison_results = []
        
        for start_time_str, results_df in self.all_start_time_results.items():
            if len(results_df) == 0:
                continue
            
            start_time_ms = self.time_to_ms(start_time_str)
            additional_trading_time = (self.regime_start_ms - start_time_ms) / 60000
            
            # Basic metrics
            max_accuracy = results_df['accuracy'].max()
            final_accuracy = results_df['accuracy'].iloc[-1] if len(results_df) > 0 else 0
            
            # Find when different accuracy thresholds are reached
            threshold_times = {}
            threshold_trading_times = {}
            
            for threshold in accuracy_thresholds:
                threshold_rows = results_df[results_df['accuracy'] >= threshold]
                if len(threshold_rows) > 0:
                    first_time = threshold_rows.iloc[0]
                    threshold_times[f'time_to_{threshold:.0%}'] = first_time['end_time_str']
                    # Calculate remaining trading time when threshold is reached
                    remaining_time = (self.regime_end_ms - first_time['end_time_ms']) / 60000
                    threshold_trading_times[f'trading_time_at_{threshold:.0%}'] = remaining_time
                else:
                    threshold_times[f'time_to_{threshold:.0%}'] = 'Never'
                    threshold_trading_times[f'trading_time_at_{threshold:.0%}'] = 0
            
            # Find best early prediction (before regime start)
            early_results = results_df[results_df['end_time_ms'] <= self.regime_start_ms]
            best_early_accuracy = early_results['accuracy'].max() if len(early_results) > 0 else 0
            
            # Accuracy at regime start
            regime_start_results = results_df[results_df['end_time_ms'] >= self.regime_start_ms]
            accuracy_at_regime_start = regime_start_results['accuracy'].iloc[0] if len(regime_start_results) > 0 else 0
            
            result = {
                'start_time': start_time_str,
                'additional_trading_time': additional_trading_time,
                'max_accuracy': max_accuracy,
                'final_accuracy': final_accuracy,
                'best_early_accuracy': best_early_accuracy,
                'accuracy_at_regime_start': accuracy_at_regime_start,
                **threshold_times,
                **threshold_trading_times
            }
            
            comparison_results.append(result)
        
        self.comparison_results = pd.DataFrame(comparison_results)
        
        return self.comparison_results
    
    def find_optimal_start_times(self, min_accuracy=0.4, min_trading_time=30):
        """
        Find optimal start times based on accuracy and trading time requirements
        
        Args:
            min_accuracy: Minimum acceptable accuracy
            min_trading_time: Minimum required trading time (minutes)
        """
        print(f"\n🎯 FINDING OPTIMAL START TIMES")
        print(f"Requirements: ≥{min_accuracy:.0%} accuracy, ≥{min_trading_time} min trading time")
        
        optimal_scenarios = []
        
        for start_time_str, results_df in self.all_start_time_results.items():
            if len(results_df) == 0:
                continue
            
            # Find when min_accuracy is first reached
            accuracy_rows = results_df[results_df['accuracy'] >= min_accuracy]
            
            if len(accuracy_rows) > 0:
                first_accurate_time = accuracy_rows.iloc[0]
                remaining_trading_time = (self.regime_end_ms - first_accurate_time['end_time_ms']) / 60000
                
                if remaining_trading_time >= min_trading_time:
                    scenario = {
                        'start_time': start_time_str,
                        'achieves_accuracy_at': first_accurate_time['end_time_str'],
                        'accuracy_achieved': first_accurate_time['accuracy'],
                        'data_duration_needed': first_accurate_time['duration_minutes'],
                        'remaining_trading_time': remaining_trading_time,
                        'additional_prep_time': first_accurate_time['additional_trading_time'],
                        'feasibility_score': first_accurate_time['accuracy'] * remaining_trading_time / 100
                    }
                    optimal_scenarios.append(scenario)
        
        if len(optimal_scenarios) > 0:
            optimal_df = pd.DataFrame(optimal_scenarios)
            optimal_df = optimal_df.sort_values('feasibility_score', ascending=False)
            
            print(f"\n✅ VIABLE SCENARIOS ({len(optimal_df)} found):")
            print("-" * 80)
            print("Start | Accuracy At | Accuracy | Trading Time | Prep Time | Score")
            print("-" * 80)
            
            for _, row in optimal_df.iterrows():
                print(f"{row['start_time']:5} | {row['achieves_accuracy_at']:11} | "
                      f"{row['accuracy_achieved']:7.1%} | {row['remaining_trading_time']:11.1f}min | "
                      f"{row['additional_prep_time']:8.1f}min | {row['feasibility_score']:5.1f}")
            
            return optimal_df
        else:
            print("❌ No scenarios meet the requirements")
            return pd.DataFrame()
    
    def save_results_and_visualizations(self):
        """Save all results and create visualizations"""
        print("\n💾 Saving results and creating visualizations...")
        
        # Save individual start time results
        for start_time_str, results_df in self.all_start_time_results.items():
            filename = f"accuracy_curve_{start_time_str.replace(':', '')}.csv"
            filepath = self.output_dir / filename
            results_df.to_csv(filepath, index=False)
        
        # Save comparison results
        if len(self.comparison_results) > 0:
            comparison_file = self.output_dir / "start_time_comparison.csv"
            self.comparison_results.to_csv(comparison_file, index=False)
        
        # Create comprehensive visualization
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Accuracy curves over time
            ax1 = axes[0, 0]
            for start_time_str, results_df in self.all_start_time_results.items():
                if len(results_df) > 0:
                    # Convert end times to hours for better readability
                    time_hours = results_df['end_time_ms'] / 3600000
                    ax1.plot(time_hours, results_df['accuracy'], 'o-', label=start_time_str, markersize=3)
            
            ax1.axvline(x=self.regime_start_ms/3600000, color='red', linestyle='--', alpha=0.7, label='Regime Start')
            ax1.axvline(x=self.regime_end_ms/3600000, color='green', linestyle='--', alpha=0.7, label='Regime End')
            ax1.set_xlabel('Time of Day (Hours)')
            ax1.set_ylabel('Prediction Accuracy')
            ax1.set_title('Accuracy Growth Curves by Start Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Maximum accuracy by start time
            ax2 = axes[0, 1]
            if len(self.comparison_results) > 0:
                ax2.bar(self.comparison_results['start_time'], self.comparison_results['max_accuracy'])
                ax2.set_xlabel('Start Time')
                ax2.set_ylabel('Maximum Accuracy')
                ax2.set_title('Maximum Achievable Accuracy')
                ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Trading time vs accuracy trade-off
            ax3 = axes[1, 0]
            if len(self.comparison_results) > 0:
                scatter = ax3.scatter(self.comparison_results['additional_trading_time'], 
                                    self.comparison_results['max_accuracy'],
                                    c=self.comparison_results['best_early_accuracy'], 
                                    cmap='viridis', s=100)
                ax3.set_xlabel('Additional Trading Time (minutes)')
                ax3.set_ylabel('Maximum Accuracy')
                ax3.set_title('Trading Time vs Accuracy Trade-off')
                plt.colorbar(scatter, ax=ax3, label='Best Early Accuracy')
                
                # Add labels for each point
                for i, row in self.comparison_results.iterrows():
                    ax3.annotate(row['start_time'], 
                               (row['additional_trading_time'], row['max_accuracy']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Plot 4: Time to reach different accuracy thresholds
            ax4 = axes[1, 1]
            if len(self.comparison_results) > 0:
                thresholds = ['trading_time_at_30%', 'trading_time_at_40%', 'trading_time_at_50%', 'trading_time_at_60%']
                threshold_labels = ['30%', '40%', '50%', '60%']
                
                x = np.arange(len(self.comparison_results))
                width = 0.2
                
                for i, (thresh, label) in enumerate(zip(thresholds, threshold_labels)):
                    if thresh in self.comparison_results.columns:
                        values = self.comparison_results[thresh].fillna(0)
                        ax4.bar(x + i*width, values, width, label=label)
                
                ax4.set_xlabel('Start Time')
                ax4.set_ylabel('Remaining Trading Time (minutes)')
                ax4.set_title('Trading Time Available at Accuracy Thresholds')
                ax4.set_xticks(x + width * 1.5)
                ax4.set_xticklabels(self.comparison_results['start_time'], rotation=45)
                ax4.legend()
            
            plt.tight_layout()
            plot_file = self.output_dir / 'comprehensive_start_time_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Visualization saved to: {plot_file}")
            
        except ImportError:
            print("matplotlib not available - skipping visualization")
        except Exception as e:
            print(f"Could not create visualization: {e}")
        
        print(f"📁 All results saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Optimal Start Time Analysis for Regime Prediction')
    parser.add_argument('--data_path', default='data/history_spot_quote.csv',
                       help='Path to trading data CSV file')
    parser.add_argument('--reference_model_path', default='../../market_regime/gmm/daily',
                       help='Path to reference GMM model directory')
    parser.add_argument('--output_dir', default='../../market_regime/optimal_start_time_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--start_times', nargs='+', 
                       default=['09:00', '09:15', '09:30', '09:45', '10:00', '10:15', '10:30'],
                       help='Start times to test (HH:MM format)')
    parser.add_argument('--test_intervals', type=int, default=20,
                       help='Number of test intervals for each start time')
    parser.add_argument('--accuracy_threshold', type=float, default=0.4,
                       help='Minimum acceptable accuracy for optimal scenarios')
    parser.add_argument('--min_trading_time', type=int, default=30,
                       help='Minimum required trading time (minutes)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = OptimalStartTimeAnalyzer(
        data_path=args.data_path,
        reference_model_path=args.reference_model_path,
        output_dir=args.output_dir
    )
    
    try:
        # Load models and data
        analyzer.load_reference_models_and_data()
        
        # Run comprehensive analysis
        all_results = analyzer.run_comprehensive_start_time_analysis(
            start_times=args.start_times,
            test_intervals=args.test_intervals
        )
        
        # Compare start times
        comparison = analyzer.compare_start_times()
        
        # Find optimal scenarios
        optimal_scenarios = analyzer.find_optimal_start_times(
            min_accuracy=args.accuracy_threshold,
            min_trading_time=args.min_trading_time
        )
        
        # Save results and visualizations
        analyzer.save_results_and_visualizations()
        
        print("\n" + "="*80)
        print("OPTIMAL START TIME ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
