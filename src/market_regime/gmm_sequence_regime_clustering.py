#!/usr/bin/env python3
"""
GMM-based Market Regime Sequence Clustering

This script uses Gaussian Mixture Models to cluster market regimes based on 
sliding 30-minute windows of intraday statistical features. Starting from 10:05 AM 
as reference point, it analyzes 10:06-10:35 (30 minutes), then slides the window 
by 1 minute: 10:06 reference for 10:07-10:36, and so on until 11:30 reference 
for 11:31-12:00, creating minute-by-minute regime classifications.

Features analyzed (via statistical_features module):
- Volatility measures (rolling std, range volatility, realized volatility)
- Momentum indicators (ROC, directional momentum) 
- Trend characteristics (linear trend, trend strength, reversals)
- Price action patterns (jumps, microstructure)
- Statistical properties (skewness, kurtosis, autocorrelation)

Usage:
    python gmm_sequence_regime_clustering.py [--n_regimes N] [--window_minutes 30]
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
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import joblib

# Technical analysis
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

# Import our statistical features module
from market_data_stat.statistical_features import StatisticalFeatureExtractor

class GMMSequenceRegimeClusterer:
    """
    Gaussian Mixture Model based market regime clustering for minute-by-minute sequences
    using sliding 30-minute windows
    """
    
    def __init__(self, data_path, output_dir='../../market_regime/gmm/sequence', 
                 window_minutes=30, sequence_start_ms=36300000, sequence_end_ms=43200000):
        """
        Initialize the sequence regime clusterer
        
        Args:
            data_path: Path to history_spot_quote.csv
            output_dir: Directory to save results (relative to script location)
            window_minutes: Size of sliding window in minutes (default: 30)
            sequence_start_ms: Start of sequence analysis (10:05 AM = 36300000) - first reference point
            sequence_end_ms: End of sequence analysis (12:00 PM = 43200000) - last analysis point
        """
        # Get the absolute path to the script directory
        script_dir = Path(__file__).parent.absolute()
        
        # Resolve data path relative to script directory
        if not os.path.isabs(data_path):
            self.data_path = script_dir / ".." / ".." / data_path
        else:
            self.data_path = Path(data_path)
        
        # Resolve output directory relative to script directory
        if not os.path.isabs(output_dir):
            self.output_dir = script_dir / output_dir
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.window_minutes = window_minutes
        self.window_ms = window_minutes * 60000  # Convert to milliseconds
        self.sequence_start_ms = sequence_start_ms  # 10:05 AM - first reference point
        self.sequence_end_ms = sequence_end_ms      # 12:00 PM - last analysis point
        
        # Calculate the range for sequence analysis
        # Last reference point is when we can still get a full 30-minute window
        self.last_reference_ms = self.sequence_end_ms - self.window_ms  # 11:30 AM for 30-min window ending at 12:00
        
        # Model components
        self.scaler = StandardScaler()
        self.gmm_model = None
        self.pca = None
        
        # Statistical feature extractor
        self.feature_extractor = StatisticalFeatureExtractor()
        
        # Data storage
        self.raw_data = None
        self.sequence_features = None  # Features for all sequences
        self.regime_labels = None
        self.regime_summary = None
        
        print(f"GMM Sequence Regime Clusterer initialized")
        print(f"Data path: {self.data_path}")
        print(f"Window size: {window_minutes} minutes")
        print(f"Reference sequence: {self.ms_to_time(sequence_start_ms)} to {self.ms_to_time(self.last_reference_ms)}")
        print(f"Analysis windows: {self.ms_to_time(sequence_start_ms + 60000)} to {self.ms_to_time(sequence_end_ms)}")
        print(f"Output directory: {self.output_dir}")
    
    def ms_to_time(self, ms):
        """Convert milliseconds of day to readable time format"""
        # Handle NaN or None values
        if pd.isna(ms) or ms is None:
            return "00:00"
        
        # Ensure ms is an integer
        try:
            ms = int(ms)
        except (ValueError, TypeError):
            return "00:00"
        
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        return f"{int(hours):02d}:{int(minutes):02d}"
    
    def load_and_filter_data(self):
        """Load trading data from history_spot_quote.csv and filter for sequence analysis period"""
        print("Loading and filtering trading data for sequence analysis...")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        self.raw_data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.raw_data):,} rows of raw data")
        
        # Filter for extended period to accommodate sliding windows
        # Need data from sequence_start to sequence_end for full analysis
        self.raw_data = self.raw_data[
            (self.raw_data['ms_of_day'] >= self.sequence_start_ms) &
            (self.raw_data['ms_of_day'] <= self.sequence_end_ms)
        ].copy()
        print(f"Filtered to {len(self.raw_data):,} rows for sequence period")
        
        # Sort by date and time
        self.raw_data = self.raw_data.sort_values(['trading_day', 'ms_of_day']).reset_index(drop=True)
        
        # Get unique trading days
        unique_days = self.raw_data['trading_day'].unique()
        print(f"Found {len(unique_days)} unique trading days")
        
        return self.raw_data
    
    def extract_sequence_features(self):
        """Extract features for each sliding 30-minute window sequence"""
        print("Extracting sequence features using sliding windows...")
        print(f"Window size: {self.window_minutes} minutes")
        
        all_sequence_features = []
        
        # Use mid price and check for volume columns
        price_column = 'mid'
        volume_column = 'volume' if 'volume' in self.raw_data.columns else None
        
        # Get unique trading days
        trading_days = sorted(self.raw_data['trading_day'].unique())
        
        total_sequences = 0
        
        for day in trading_days:
            day_data = self.raw_data[self.raw_data['trading_day'] == day].copy()
            
            if len(day_data) == 0:
                continue
                
            print(f"Processing {day}...")
            
            # Generate sliding windows for this day
            # Reference points: 10:05, 10:06, 10:07, ..., 11:30
            reference_times = range(self.sequence_start_ms, self.last_reference_ms + 60000, 60000)
            
            day_sequences = 0
            
            for ref_time_ms in reference_times:
                # Analysis window: reference_time + 1 minute to reference_time + window_minutes + 1 minute
                window_start_ms = ref_time_ms + 60000  # Start 1 minute after reference
                window_end_ms = ref_time_ms + self.window_ms + 60000  # End window_minutes + 1 after reference
                
                # Filter data for this window
                window_data = day_data[
                    (day_data['ms_of_day'] >= window_start_ms) &
                    (day_data['ms_of_day'] <= window_end_ms)
                ].copy()
                
                # Need sufficient data points for analysis
                if len(window_data) < 5:  # Minimum data points
                    continue
                
                try:
                    # Extract features for this window using the reference time
                    window_features = self.feature_extractor.extract_daily_features(
                        daily_data=window_data,
                        price_column=price_column,
                        volume_column=volume_column,
                        reference_time_ms=ref_time_ms,
                        trading_day_column='trading_day',
                        time_column='ms_of_day',
                        use_relative=True,
                        include_overnight_gap=False
                    )
                    
                    if len(window_features) > 0:
                        # Debug: Check for NaN values before adding metadata
                        if window_features.isna().any().any():
                            print(f"  Warning: NaN values found in features for {day} at {self.ms_to_time(ref_time_ms)}")
                            print(f"  NaN columns: {window_features.columns[window_features.isna().any()].tolist()}")
                        
                        # Add sequence metadata
                        for idx in window_features.index:
                            window_features.loc[idx, 'sequence_reference_time'] = int(ref_time_ms)
                            window_features.loc[idx, 'window_start_ms'] = int(window_start_ms)
                            window_features.loc[idx, 'window_end_ms'] = int(window_end_ms)
                            window_features.loc[idx, 'window_minutes'] = self.window_minutes
                            window_features.loc[idx, 'sequence_id'] = f"{day}_{self.ms_to_time(ref_time_ms)}"
                        
                        # Debug: Check for NaN values after adding metadata
                        nan_cols_after = ['sequence_reference_time', 'window_start_ms', 'window_end_ms'] 
                        for col in nan_cols_after:
                            if col in window_features.columns and window_features[col].isna().any():
                                print(f"  Warning: NaN found in {col} after metadata addition")
                                print(f"  Values: {window_features[col].unique()}")
                        
                        all_sequence_features.append(window_features)
                        day_sequences += 1
                        total_sequences += 1
                        
                except Exception as e:
                    print(f"  Warning: Failed to extract features for {day} at {self.ms_to_time(ref_time_ms)}: {e}")
                    continue
            
            if day_sequences > 0:
                print(f"  Extracted {day_sequences} sequences for {day}")
        
        if len(all_sequence_features) == 0:
            raise ValueError("No valid sequences extracted")
        
        # Combine all sequence features
        self.sequence_features = pd.concat(all_sequence_features, ignore_index=True)
        
        # Debug: Check for NaN values after concatenation
        print("Debug: Checking for NaN values after concatenation...")
        time_cols = ['sequence_reference_time', 'window_start_ms', 'window_end_ms']
        for col in time_cols:
            if col in self.sequence_features.columns:
                nan_count = self.sequence_features[col].isna().sum()
                if nan_count > 0:
                    print(f"  {col}: {nan_count} NaN values out of {len(self.sequence_features)}")
                    print(f"  Sample NaN indices: {self.sequence_features[self.sequence_features[col].isna()].index.tolist()[:5]}")
        
        print(f"Total sequences extracted: {total_sequences}")
        print(f"Feature dimensions: {self.sequence_features.shape[1] - 11} features per sequence")  # Subtract metadata columns
        
        return self.sequence_features
    
    def prepare_features_for_clustering(self):
        """Prepare and normalize features for GMM clustering"""
        print("Preparing sequence features for clustering...")
        
        # Select feature columns (exclude metadata)
        metadata_cols = [
            'trading_day', 'reference_time_ms', 'reference_price', 'num_observations', 
            'sequence_reference_time', 'window_start_ms', 'window_end_ms', 
            'window_minutes', 'sequence_id'
        ]
        feature_cols = [col for col in self.sequence_features.columns if col not in metadata_cols]
        
        # Get feature matrix
        X = self.sequence_features[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Remove features with zero variance
        feature_variances = X.var()
        zero_var_features = feature_variances[feature_variances == 0].index
        if len(zero_var_features) > 0:
            print(f"Removing {len(zero_var_features)} zero-variance features: {list(zero_var_features)}")
            X = X.drop(columns=zero_var_features)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction (optional)
        explained_variance_threshold = 0.95
        self.pca = PCA(n_components=explained_variance_threshold, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        
        print(f"Original features: {X.shape[1]}")
        print(f"PCA features: {X_pca.shape[1]} (explains {explained_variance_threshold*100}% variance)")
        
        self.feature_names = X.columns.tolist()
        return X_scaled, X_pca
    
    def fit_gmm_model(self, X, n_components_range=range(2, 8), use_pca=True):
        """Fit GMM model with optimal number of components"""
        print("Fitting GMM model for sequence clustering...")
        
        # Select features for clustering
        features_to_use = X if not use_pca else self.pca.transform(self.scaler.transform(
            self.sequence_features[self.feature_names].fillna(0)
        ))
        
        best_n_components = 2
        best_score = -np.inf
        best_model = None
        
        scores = {}
        
        # Test different numbers of components
        for n_components in n_components_range:
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    max_iter=200,
                    random_state=42,
                    n_init=5
                )
                
                labels = gmm.fit_predict(features_to_use)
                
                # Calculate evaluation metrics
                if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
                    silhouette = silhouette_score(features_to_use, labels)
                    calinski = calinski_harabasz_score(features_to_use, labels)
                    bic = gmm.bic(features_to_use)
                    aic = gmm.aic(features_to_use)
                    
                    # Combined score (higher silhouette, lower BIC)
                    combined_score = silhouette - (bic / 10000)  # Scale BIC down
                    
                    scores[n_components] = {
                        'silhouette': silhouette,
                        'calinski': calinski,
                        'bic': bic,
                        'aic': aic,
                        'combined': combined_score,
                        'model': gmm
                    }
                    
                    print(f"  {n_components} components: Silhouette={silhouette:.3f}, BIC={bic:.1f}, Combined={combined_score:.3f}")
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_n_components = n_components
                        best_model = gmm
                        
            except Exception as e:
                print(f"  Failed to fit {n_components} components: {e}")
        
        if best_model is None:
            raise ValueError("Failed to fit any GMM model")
        
        self.gmm_model = best_model
        print(f"Selected {best_n_components} components (best combined score: {best_score:.3f})")
        
        # Get final labels
        self.regime_labels = self.gmm_model.predict(features_to_use)
        
        return scores
    
    def analyze_regimes(self):
        """Analyze characteristics of each identified regime"""
        print("Analyzing sequence regime characteristics...")
        
        # Add regime labels to sequence features
        self.sequence_features['Regime'] = self.regime_labels
        
        # Calculate regime summary statistics
        regime_summaries = []
        
        for regime in sorted(np.unique(self.regime_labels)):
            regime_data = self.sequence_features[self.sequence_features['Regime'] == regime]
            
            # Basic regime info
            regime_summary = {
                'Regime': regime,
                'num_sequences': len(regime_data),
                'percentage': len(regime_data) / len(self.sequence_features) * 100,
            }
            
            # Calculate statistics for each feature
            for feature in self.feature_names:
                if feature in regime_data.columns:
                    values = regime_data[feature].dropna()
                    if len(values) > 0:
                        regime_summary.update({
                            f'{feature}_mean': np.mean(values),
                            f'{feature}_std': np.std(values),
                            f'{feature}_median': np.median(values),
                            f'{feature}_min': np.min(values),
                            f'{feature}_max': np.max(values),
                        })
            
            regime_summaries.append(regime_summary)
        
        self.regime_summary = pd.DataFrame(regime_summaries)
        
        # Print regime distribution
        print("\nSequence Regime Distribution:")
        for regime in sorted(np.unique(self.regime_labels)):
            count = np.sum(self.regime_labels == regime)
            percentage = count / len(self.regime_labels) * 100
            print(f"  Regime {regime}: {count} sequences ({percentage:.1f}%)")
        
        return self.regime_summary
    
    def save_results(self):
        """Save clustering results to CSV files"""
        print("Saving sequence clustering results...")
        
        # 1. Sequence regime assignments with features
        sequence_output = self.sequence_features[[
            'trading_day', 'sequence_id', 'sequence_reference_time', 'window_start_ms', 
            'window_end_ms', 'window_minutes', 'reference_time_ms', 'reference_price', 'Regime'
        ] + self.feature_names].copy()
        
        # Debug: Check data types
        print(f"Debug: sequence_reference_time dtype: {sequence_output['sequence_reference_time'].dtype}")
        print(f"Debug: window_start_ms dtype: {sequence_output['window_start_ms'].dtype}")
        print(f"Debug: window_end_ms dtype: {sequence_output['window_end_ms'].dtype}")
        
        # Ensure integer types for time columns
        sequence_output['sequence_reference_time'] = sequence_output['sequence_reference_time'].astype('Int64')
        sequence_output['window_start_ms'] = sequence_output['window_start_ms'].astype('Int64')
        sequence_output['window_end_ms'] = sequence_output['window_end_ms'].astype('Int64')
        
        # Add readable time columns
        sequence_output['reference_time_readable'] = sequence_output['sequence_reference_time'].apply(lambda x: self.ms_to_time(x))
        sequence_output['window_start_readable'] = sequence_output['window_start_ms'].apply(lambda x: self.ms_to_time(x))
        sequence_output['window_end_readable'] = sequence_output['window_end_ms'].apply(lambda x: self.ms_to_time(x))
        
        sequence_output_file = self.output_dir / 'sequence_regime_assignments.csv'
        sequence_output.to_csv(sequence_output_file, index=False)
        print(f"Saved sequence regime assignments to: {sequence_output_file}")
        
        # 2. Regime characteristics summary
        regime_output_file = self.output_dir / 'regime_characteristics.csv'
        self.regime_summary.to_csv(regime_output_file, index=False)
        print(f"Saved regime characteristics to: {regime_output_file}")
        
        # 3. Save model and scaler for future use
        model_file = self.output_dir / 'gmm_model.pkl'
        scaler_file = self.output_dir / 'feature_scaler.pkl'
        pca_file = self.output_dir / 'pca_model.pkl'
        
        joblib.dump(self.gmm_model, model_file)
        joblib.dump(self.scaler, scaler_file)
        joblib.dump(self.pca, pca_file)
        
        print(f"Saved models to: {model_file}, {scaler_file}, {pca_file}")
        
        # 4. Create minute-by-minute regime timeline
        timeline_data = []
        for _, row in sequence_output.iterrows():
            timeline_data.append({
                'trading_day': row['trading_day'],
                'reference_time_ms': row['sequence_reference_time'],
                'reference_time': row['reference_time_readable'], 
                'regime': row['Regime'],
                'sequence_id': row['sequence_id']
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        timeline_file = self.output_dir / 'minute_by_minute_regimes.csv'
        timeline_df.to_csv(timeline_file, index=False)
        print(f"Saved minute-by-minute regime timeline to: {timeline_file}")
        
        # 5. Save feature names and configuration for reference
        feature_info = {
            'feature_names': self.feature_names,
            'window_minutes': self.window_minutes,
            'sequence_start_ms': self.sequence_start_ms,
            'sequence_end_ms': self.sequence_end_ms,
            'first_reference_ms': self.sequence_start_ms,
            'last_reference_ms': self.last_reference_ms,
            'n_regimes': len(np.unique(self.regime_labels)),
            'total_sequences': len(self.sequence_features),
            'note': f'Sliding {self.window_minutes}-minute windows with 1-minute steps. Reference times from {self.ms_to_time(self.sequence_start_ms)} to {self.ms_to_time(self.last_reference_ms)}. Analysis windows from {self.ms_to_time(self.sequence_start_ms + 60000)} to {self.ms_to_time(self.sequence_end_ms)}.'
        }
        
        feature_info_file = self.output_dir / 'clustering_info.json'
        import json
        with open(feature_info_file, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"Saved clustering info to: {feature_info_file}")
    
    def create_visualizations(self):
        """Create visualization plots for sequence regime analysis"""
        print("Creating sequence regime visualizations...")
        
        try:
            # 1. Sequence regime distribution plot
            plt.figure(figsize=(10, 6))
            regime_counts = pd.Series(self.regime_labels).value_counts().sort_index()
            regime_counts.plot(kind='bar')
            plt.title('Sequence Market Regime Distribution')
            plt.xlabel('Regime')
            plt.ylabel('Number of Sequences')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'sequence_regime_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Regime timeline plot (sample day)
            if len(self.sequence_features) > 0:
                sample_day = self.sequence_features['trading_day'].iloc[0]
                day_data = self.sequence_features[self.sequence_features['trading_day'] == sample_day].copy()
                
                if len(day_data) > 0:
                    plt.figure(figsize=(15, 6))
                    times = [self.ms_to_time(t) for t in day_data['sequence_reference_time']]
                    regimes = day_data['Regime']
                    
                    # Create timeline plot
                    colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(regimes))))
                    for i, regime in enumerate(sorted(np.unique(regimes))):
                        regime_mask = regimes == regime
                        plt.scatter(np.array(times)[regime_mask], [regime] * sum(regime_mask), 
                                  c=[colors[i]], label=f'Regime {regime}', s=50, alpha=0.7)
                    
                    plt.xlabel('Time')
                    plt.ylabel('Regime')
                    plt.title(f'Intraday Regime Timeline - {sample_day}')
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / f'regime_timeline_{sample_day}.png', dpi=300, bbox_inches='tight')
                    plt.close()
            
            # 3. Feature importance heatmap (top features by regime)
            if len(self.feature_names) > 0:
                # Select top features based on variance across regimes
                feature_variance_across_regimes = []
                for feature in self.feature_names[:20]:  # Limit to top 20 for readability
                    if feature in self.sequence_features.columns:
                        regime_means = []
                        for regime in sorted(np.unique(self.regime_labels)):
                            regime_data = self.sequence_features[self.sequence_features['Regime'] == regime]
                            if len(regime_data) > 0 and feature in regime_data.columns:
                                regime_means.append(regime_data[feature].mean())
                            else:
                                regime_means.append(0)
                        feature_variance_across_regimes.append((feature, np.var(regime_means)))
                
                # Sort by variance and take top features
                feature_variance_across_regimes.sort(key=lambda x: x[1], reverse=True)
                top_features = [f[0] for f in feature_variance_across_regimes[:15]]
                
                # Create heatmap data
                heatmap_data = []
                for regime in sorted(np.unique(self.regime_labels)):
                    regime_data = self.sequence_features[self.sequence_features['Regime'] == regime]
                    regime_row = []
                    for feature in top_features:
                        if feature in regime_data.columns and len(regime_data) > 0:
                            regime_row.append(regime_data[feature].mean())
                        else:
                            regime_row.append(0)
                    heatmap_data.append(regime_row)
                
                # Normalize for heatmap
                heatmap_df = pd.DataFrame(heatmap_data, 
                                        index=[f'Regime {i}' for i in sorted(np.unique(self.regime_labels))],
                                        columns=top_features)
                
                # Normalize each feature (column) to 0-1 scale
                heatmap_df_norm = (heatmap_df - heatmap_df.min()) / (heatmap_df.max() - heatmap_df.min())
                
                plt.figure(figsize=(15, 8))
                sns.heatmap(heatmap_df_norm, annot=False, cmap='RdYlBu_r', center=0.5)
                plt.title('Sequence Regime Characteristics Heatmap (Normalized Features)')
                plt.xlabel('Features')
                plt.ylabel('Regimes')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'sequence_regime_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Saved visualizations to: {self.output_dir}")
            
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
    
    def run_clustering(self, n_components_range=range(2, 8), use_pca=True):
        """Run the complete sequence clustering pipeline"""
        print("Starting GMM sequence regime clustering pipeline...")
        print(f"Sliding window: {self.window_minutes} minutes")
        print(f"Reference sequence: {self.ms_to_time(self.sequence_start_ms)} to {self.ms_to_time(self.last_reference_ms)}")
        print(f"Analysis windows: {self.ms_to_time(self.sequence_start_ms + 60000)} to {self.ms_to_time(self.sequence_end_ms)}")
        
        # Step 1: Load and filter data
        self.load_and_filter_data()
        
        # Step 2: Extract sequence features using sliding windows
        self.extract_sequence_features()
        
        # Step 3: Prepare features for clustering
        X_scaled, X_pca = self.prepare_features_for_clustering()
        
        # Step 4: Fit GMM model
        scores = self.fit_gmm_model(X_pca if use_pca else X_scaled, n_components_range, use_pca)
        
        # Step 5: Analyze regimes
        self.analyze_regimes()
        
        # Step 6: Save results
        self.save_results()
        
        # Step 7: Create visualizations
        self.create_visualizations()
        
        return scores

# Utility function for time conversion
def time_to_ms(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 3600000 + minutes * 60000

def main():
    parser = argparse.ArgumentParser(description='GMM-based Market Regime Sequence Clustering')
    parser.add_argument('--data_path', default='data/history_spot_quote.csv', 
                       help='Path to trading data CSV file (relative to project root)')
    parser.add_argument('--n_regimes', type=int, default=None,
                       help='Number of regimes (if not specified, will optimize)')
    parser.add_argument('--output_dir', default='../../market_regime/gmm/sequence',
                       help='Output directory for results (relative to script)')
    parser.add_argument('--window_minutes', type=int, default=30,
                       help='Size of sliding window in minutes')
    parser.add_argument('--sequence_start', default='10:05',
                       help='First reference time (HH:MM format)')
    parser.add_argument('--sequence_end', default='12:00', 
                       help='Last analysis time (HH:MM format)')
    parser.add_argument('--use_pca', action='store_true', default=True,
                       help='Use PCA for dimensionality reduction')
    args = parser.parse_args()

    sequence_start_ms = time_to_ms(args.sequence_start)
    sequence_end_ms = time_to_ms(args.sequence_end)

    clusterer = GMMSequenceRegimeClusterer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        window_minutes=args.window_minutes,
        sequence_start_ms=sequence_start_ms,
        sequence_end_ms=sequence_end_ms
    )

    if args.n_regimes is not None:
        n_components_range = [args.n_regimes]
    else:
        n_components_range = range(2, 8)

    scores = clusterer.run_clustering(n_components_range, args.use_pca)

    print("\n" + "="*60)
    print("SEQUENCE CLUSTERING SUMMARY")
    print("="*60)
    print(f"Data file: {clusterer.data_path}")
    print(f"Window size: {args.window_minutes} minutes")
    print(f"Reference sequence: {args.sequence_start} to {clusterer.ms_to_time(clusterer.last_reference_ms)}")
    print(f"Analysis windows: {clusterer.ms_to_time(sequence_start_ms + 60000)} to {args.sequence_end}")
    print(f"Total sequences: {len(clusterer.sequence_features)}")
    print(f"Number of regimes: {len(np.unique(clusterer.regime_labels))}")
    print(f"Features extracted: {len(clusterer.feature_names)}")
    print(f"Output directory: {clusterer.output_dir}")
    print(f"Note: Each sequence uses {args.window_minutes}-minute sliding window with 1-minute steps")

    print("\nSequence Regime Distribution:")
    for regime in sorted(np.unique(clusterer.regime_labels)):
        count = np.sum(clusterer.regime_labels == regime)
        percentage = count / len(clusterer.regime_labels) * 100
        print(f"  Regime {regime}: {count:4d} sequences ({percentage:5.1f}%)")

    # Show sample sequences per day
    unique_days = clusterer.sequence_features['trading_day'].unique()
    sequences_per_day = len(clusterer.sequence_features) / len(unique_days)
    print(f"\nAverage sequences per day: {sequences_per_day:.1f}")

if __name__ == "__main__":
    main()
