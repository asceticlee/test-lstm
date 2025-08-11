#!/usr/bin/env python3
"""
GMM-based Market Regime Clustering

This script uses Gaussian Mixture Models to cluster market regimes based on 
intraday statistical features during trading hours (10:35 AM to 12:00 PM).
The analysis uses 10:35 AM as both the start time and reference point for 
relative price calculations.

Features analyzed (via statistical_features module):
- Volatility measures (rolling std, range volatility, realized volatility)
- Momentum indicators (ROC, directional momentum)
- Trend characteristics (linear trend, trend strength, reversals)
- Price action patterns (jumps, microstructure)
- Statistical properties (skewness, kurtosis, autocorrelation)

Usage:
    python gmm_regime_clustering.py [--n_regimes N] [--trading_start HH:MM]
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

class GMMRegimeClusterer:
    """
    Gaussian Mixture Model based market regime clustering for intraday periods
    """
    
    def __init__(self, data_path, output_dir='../../market_regime/gmm/daily', 
                 trading_start_ms=38100000, trading_end_ms=43200000, 
                 include_overnight_gap=False, overnight_indicators=False):
        """
        Initialize the regime clusterer
        
        Args:
            data_path: Path to trainingData.csv
            output_dir: Directory to save results (relative to script location)
            trading_start_ms: Start of trading period (10:35 AM = 38100000) - also reference time
            trading_end_ms: End of trading period (12:00 PM = 43200000)
            include_overnight_gap: Whether to include overnight gap features
            overnight_indicators: Whether to include overnight-specific technical indicators
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
        
        self.trading_start_ms = trading_start_ms
        self.trading_end_ms = trading_end_ms
        self.include_overnight_gap = include_overnight_gap
        self.overnight_indicators = overnight_indicators
        
        # Model components
        self.scaler = StandardScaler()
        self.gmm_model = None
        self.pca = None
        
        # Statistical feature extractor
        self.feature_extractor = StatisticalFeatureExtractor()
        
        # Data storage
        self.raw_data = None
        self.daily_features = None
        self.regime_labels = None
        self.regime_summary = None
        
        print(f"GMM Regime Clusterer initialized")
        print(f"Data path: {self.data_path}")
        print(f"Trading period: {self.ms_to_time(trading_start_ms)} to {self.ms_to_time(trading_end_ms)}")
        print(f"Include overnight gap: {include_overnight_gap}")
        print(f"Overnight indicators: {overnight_indicators}")
        print(f"Output directory: {self.output_dir}")
    
    def ms_to_time(self, ms):
        """Convert milliseconds of day to readable time format"""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        return f"{hours:02d}:{minutes:02d}"
    
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
    
    def load_and_filter_data(self):
        """Load trading data from history_spot_quote.csv and filter for specified time period"""
        print("Loading and filtering trading data from history_spot_quote.csv ...")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        self.raw_data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.raw_data):,} rows of raw data")
        # Filter for trading period
        self.raw_data = self.raw_data[
            (self.raw_data['ms_of_day'] >= self.trading_start_ms) &
            (self.raw_data['ms_of_day'] <= self.trading_end_ms)
        ].copy()
        print(f"Filtered to {len(self.raw_data):,} rows for trading period")
        # Sort by date and time
        self.raw_data = self.raw_data.sort_values(['trading_day', 'ms_of_day']).reset_index(drop=True)
        # Get unique trading days
        unique_days = self.raw_data['trading_day'].unique()
        print(f"Found {len(unique_days)} unique trading days")
        return self.raw_data
    
    def extract_daily_features(self, reference_time_ms=None):
        """Extract features for each trading day using statistical features module"""
        # Use trading_start_ms as reference time if not specified
        if reference_time_ms is None:
            reference_time_ms = self.trading_start_ms
            
        print(f"Extracting daily statistical features with reference time: {self.ms_to_time(reference_time_ms)}...")
        print(f"Analysis period: {self.ms_to_time(self.trading_start_ms)} to {self.ms_to_time(self.trading_end_ms)}")
        print(f"Include overnight gap: {self.include_overnight_gap}")
        print(f"Overnight indicators: {self.overnight_indicators}")
        
        # Use mid price and check for volume columns from history_spot_quote.csv
        price_column = 'mid'  # history_spot_quote.csv has mid prices
        volume_column = 'volume' if 'volume' in self.raw_data.columns else None
        
        # Use the statistical features module to extract daily features
        self.daily_features = self.feature_extractor.extract_daily_features(
            daily_data=self.raw_data,
            price_column=price_column,
            volume_column=volume_column,
            reference_time_ms=reference_time_ms,
            trading_day_column='trading_day',
            time_column='ms_of_day',
            use_relative=True,
            include_overnight_gap=self.include_overnight_gap
        )
        
        # Add overnight-specific technical indicators if enabled (gap fill, opening direction, etc.)
        if self.overnight_indicators:
            print("Adding overnight-specific technical indicator features...")
            
            # Calculate overnight indicator features for each trading day
            overnight_features_list = []
            
            # Group data by trading day
            daily_groups = self.raw_data.groupby('trading_day')
            
            for trading_day, day_data in daily_groups:
                # Calculate overnight indicator features for this day
                overnight_features = self.calculate_overnight_indicators(day_data)
                overnight_features['trading_day'] = trading_day
                overnight_features_list.append(overnight_features)
            
            # Convert to DataFrame and merge with existing features
            if overnight_features_list:
                overnight_df = pd.DataFrame(overnight_features_list)
                
                # Merge with existing daily features
                self.daily_features = self.daily_features.merge(
                    overnight_df, 
                    on='trading_day', 
                    how='left'
                )
                
                # Fill any missing overnight features with zeros
                overnight_columns = [col for col in overnight_df.columns if col != 'trading_day']
                for col in overnight_columns:
                    if col in self.daily_features.columns:
                        self.daily_features[col] = self.daily_features[col].fillna(0)
                
                print(f"Added {len(overnight_columns)} overnight indicator features")
        
        # Add additional metadata
        self.daily_features['FromMsOfDay'] = self.trading_start_ms
        self.daily_features['ToMsOfDay'] = self.trading_end_ms
        self.daily_features['time_range_minutes'] = (self.trading_end_ms - self.trading_start_ms) / 60000
        
        # Count total features (excluding metadata)
        metadata_cols = ['trading_day', 'FromMsOfDay', 'ToMsOfDay', 'reference_time_ms', 'reference_price', 'num_observations', 'time_range_minutes']
        feature_cols = [col for col in self.daily_features.columns if col not in metadata_cols]
        
        print(f"Extracted features for {len(self.daily_features)} trading days")
        print(f"Total feature dimensions: {len(feature_cols)} features per day")
        
        if self.include_overnight_gap or self.overnight_indicators:
            overnight_cols = [col for col in feature_cols if 'overnight' in col or 'gap' in col or 'opening' in col or 'vwap' in col]
            print(f"  - Standard features: {len(feature_cols) - len(overnight_cols)}")
            print(f"  - Overnight features: {len(overnight_cols)}")
        
        return self.daily_features
    
    def prepare_features_for_clustering(self):
        """Prepare and normalize features for GMM clustering"""
        print("Preparing features for clustering...")
        
        # Select feature columns (exclude metadata)
        metadata_cols = ['trading_day', 'FromMsOfDay', 'ToMsOfDay', 'reference_time_ms', 'reference_price', 'num_observations', 'time_range_minutes']
        feature_cols = [col for col in self.daily_features.columns if col not in metadata_cols]
        
        # Get feature matrix
        X = self.daily_features[feature_cols].copy()
        
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
        print("Fitting GMM model...")
        
        # Select features for clustering
        features_to_use = X if not use_pca else self.pca.transform(self.scaler.transform(
            self.daily_features[self.feature_names].fillna(0)
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
        print("Analyzing regime characteristics...")
        
        # Add regime labels to daily features
        self.daily_features['Regime'] = self.regime_labels
        
        # Calculate regime summary statistics
        regime_summaries = []
        
        for regime in sorted(np.unique(self.regime_labels)):
            regime_data = self.daily_features[self.daily_features['Regime'] == regime]
            
            # Basic regime info
            regime_summary = {
                'Regime': regime,
                'num_days': len(regime_data),
                'percentage': len(regime_data) / len(self.daily_features) * 100,
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
        print("\nRegime Distribution:")
        for regime in sorted(np.unique(self.regime_labels)):
            count = np.sum(self.regime_labels == regime)
            percentage = count / len(self.regime_labels) * 100
            print(f"  Regime {regime}: {count} days ({percentage:.1f}%)")
        
        return self.regime_summary
    
    def save_results(self):
        """Save clustering results to CSV files"""
        print("Saving results...")
        
        # 1. Daily regime assignments with features
        daily_output = self.daily_features[['trading_day', 'FromMsOfDay', 'ToMsOfDay', 'reference_time_ms', 'reference_price', 'Regime'] + self.feature_names].copy()
        daily_output_file = self.output_dir / 'daily_regime_assignments.csv'
        daily_output.to_csv(daily_output_file, index=False)
        print(f"Saved daily regime assignments to: {daily_output_file}")
        
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
        
        # 4. Save feature names for reference
        feature_info = {
            'feature_names': self.feature_names,
            'trading_start_ms': self.trading_start_ms,
            'trading_end_ms': self.trading_end_ms,
            'reference_time_ms': self.trading_start_ms,  # Reference time same as start time
            'include_overnight_gap': self.include_overnight_gap,
            'overnight_indicators': self.overnight_indicators,
            'n_regimes': len(np.unique(self.regime_labels)),
            'total_days': len(self.daily_features),
            'note': f'All price-based features calculated as percentage change from reference time ({self.ms_to_time(self.trading_start_ms)}). Overnight gaps {"included" if self.include_overnight_gap else "excluded"}. Overnight indicators {"enabled" if self.overnight_indicators else "disabled"}.'
        }
        
        feature_info_file = self.output_dir / 'clustering_info.json'
        import json
        with open(feature_info_file, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"Saved clustering info to: {feature_info_file}")
    
    def create_visualizations(self):
        """Create visualization plots for regime analysis"""
        print("Creating visualizations...")
        
        try:
            # 1. Regime distribution plot
            plt.figure(figsize=(10, 6))
            regime_counts = pd.Series(self.regime_labels).value_counts().sort_index()
            regime_counts.plot(kind='bar')
            plt.title('Market Regime Distribution')
            plt.xlabel('Regime')
            plt.ylabel('Number of Days')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'regime_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Feature importance heatmap (top features by regime)
            if len(self.feature_names) > 0:
                # Select top features based on variance across regimes
                feature_variance_across_regimes = []
                for feature in self.feature_names[:20]:  # Limit to top 20 for readability
                    if feature in self.daily_features.columns:
                        regime_means = []
                        for regime in sorted(np.unique(self.regime_labels)):
                            regime_data = self.daily_features[self.daily_features['Regime'] == regime]
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
                    regime_data = self.daily_features[self.daily_features['Regime'] == regime]
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
                plt.title('Regime Characteristics Heatmap (Normalized Features)')
                plt.xlabel('Features')
                plt.ylabel('Regimes')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'regime_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Saved visualizations to: {self.output_dir}")
            
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
    
    def run_clustering(self, n_components_range=range(2, 8), use_pca=True, reference_time_ms=None):
        """Run the complete clustering pipeline"""
        # Use trading_start_ms as reference time if not specified
        if reference_time_ms is None:
            reference_time_ms = self.trading_start_ms
            
        print("Starting GMM regime clustering pipeline...")
        print(f"Using relative prices with reference time: {self.ms_to_time(reference_time_ms)}")
        print(f"Analysis window: {self.ms_to_time(self.trading_start_ms)} to {self.ms_to_time(self.trading_end_ms)}")
        
        # Step 1: Load and filter data
        self.load_and_filter_data()
        
        # Step 2: Extract features with relative prices
        self.extract_daily_features(reference_time_ms)
        
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
    parser = argparse.ArgumentParser(description='GMM-based Market Regime Clustering')
    parser.add_argument('--data_path', default='data/history_spot_quote.csv', 
                       help='Path to trading data CSV file (relative to project root)')
    parser.add_argument('--n_regimes', type=int, default=None,
                       help='Number of regimes (if not specified, will optimize)')
    parser.add_argument('--output_dir', default='../../market_regime/gmm/daily',
                       help='Output directory for results (relative to script)')
    parser.add_argument('--trading_start', default='10:35',
                       help='Trading start time (HH:MM format) - also used as reference time')
    parser.add_argument('--trading_end', default='12:00', 
                       help='Trading end time (HH:MM format)')
    parser.add_argument('--reference_time', default=None,
                       help='Reference time for relative price calculations (HH:MM format) - defaults to trading_start')
    parser.add_argument('--use_pca', action='store_true', default=True,
                       help='Use PCA for dimensionality reduction')
    parser.add_argument('--include_overnight_gap', action='store_true', default=False,
                       help='Include overnight gap features in regime clustering')
    parser.add_argument('--overnight_indicators', action='store_true', default=False,
                       help='Include overnight-specific technical indicators')
    args = parser.parse_args()

    trading_start_ms = time_to_ms(args.trading_start)
    trading_end_ms = time_to_ms(args.trading_end)
    if args.reference_time is not None:
        reference_time_ms = time_to_ms(args.reference_time)
    else:
        reference_time_ms = trading_start_ms

    clusterer = GMMRegimeClusterer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        trading_start_ms=trading_start_ms,
        trading_end_ms=trading_end_ms,
        include_overnight_gap=args.include_overnight_gap,
        overnight_indicators=args.overnight_indicators
    )

    if args.n_regimes is not None:
        n_components_range = [args.n_regimes]
    else:
        n_components_range = range(2, 8)

    scores = clusterer.run_clustering(n_components_range, args.use_pca, reference_time_ms)

    print("\n" + "="*60)
    print("CLUSTERING SUMMARY")
    print("="*60)
    print(f"Data file: {clusterer.data_path}")
    print(f"Analysis window: {args.trading_start} to {args.trading_end}")
    print(f"Reference time: {args.reference_time if args.reference_time else args.trading_start} (for relative price calculation)")
    print(f"Include overnight gap: {args.include_overnight_gap}")
    print(f"Overnight indicators: {args.overnight_indicators}")
    print(f"Total trading days: {len(clusterer.daily_features)}")
    print(f"Number of regimes: {len(np.unique(clusterer.regime_labels))}")
    print(f"Features extracted: {len(clusterer.feature_names)}")
    print(f"Output directory: {clusterer.output_dir}")
    print(f"Note: All price features calculated as % change from {args.reference_time if args.reference_time else args.trading_start}")

    print("\nRegime Distribution:")
    for regime in sorted(np.unique(clusterer.regime_labels)):
        count = np.sum(clusterer.regime_labels == regime)
        percentage = count / len(clusterer.regime_labels) * 100
        print(f"  Regime {regime}: {count:3d} days ({percentage:5.1f}%)")

if __name__ == "__main__":
    main()
