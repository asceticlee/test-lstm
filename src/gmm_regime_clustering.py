#!/usr/bin/env python3
"""
GMM-based Market Regime Clustering

This script uses Gaussian Mixture Models to cluster market regimes based on 
intraday statistical features during trading hours (10:36 AM to 12:00 PM).

Features analyzed:
- Volatility measures (rolling std, range volatility, GARCH-like)
- Momentum indicators (ROC, directional momentum)
- Trend characteristics (linear trend, trend strength, reversals)
- Price action patterns (jumps, microstructure)
- Statistical properties (skewness, kurtosis, autocorrelation)

Usage:
    python gmm_regime_clustering.py [--n_regimes N] [--output_dir DIR] [--retrain]
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

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

class GMMRegimeClusterer:
    """
    Gaussian Mixture Model based market regime clustering for intraday periods
    """
    
    def __init__(self, data_path, output_dir='regime_clustering_results', 
                 trading_start_ms=38160000, trading_end_ms=43200000):
        """
        Initialize the regime clusterer
        
        Args:
            data_path: Path to trainingData.csv
            output_dir: Directory to save results
            trading_start_ms: Start of trading period (10:36 AM = 38160000)
            trading_end_ms: End of trading period (12:00 PM = 43200000)
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.trading_start_ms = trading_start_ms
        self.trading_end_ms = trading_end_ms
        
        # Model components
        self.scaler = StandardScaler()
        self.gmm_model = None
        self.pca = None
        
        # Data storage
        self.raw_data = None
        self.daily_features = None
        self.regime_labels = None
        self.regime_summary = None
        
        print(f"GMM Regime Clusterer initialized")
        print(f"Trading period: {self.ms_to_time(trading_start_ms)} to {self.ms_to_time(trading_end_ms)}")
        print(f"Output directory: {self.output_dir}")
    
    def ms_to_time(self, ms):
        """Convert milliseconds of day to readable time format"""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        return f"{hours:02d}:{minutes:02d}"
    
    def load_and_filter_data(self):
        """Load trading data and filter for specified time period"""
        print("Loading and filtering trading data...")
        
        # Load data
        self.raw_data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.raw_data):,} rows of raw data")
        
        # Filter for trading period
        self.raw_data = self.raw_data[
            (self.raw_data['TradingMsOfDay'] >= self.trading_start_ms) &
            (self.raw_data['TradingMsOfDay'] <= self.trading_end_ms)
        ].copy()
        
        print(f"Filtered to {len(self.raw_data):,} rows for trading period")
        
        # Sort by date and time
        self.raw_data = self.raw_data.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
        
        # Get unique trading days
        unique_days = self.raw_data['TradingDay'].unique()
        print(f"Found {len(unique_days)} unique trading days")
        
        return self.raw_data
    
    def calculate_technical_indicators(self, prices, volumes=None):
        """Calculate various technical indicators for a price series"""
        prices = np.array(prices)
        n = len(prices)
        
        if n < 2:
            return {}
        
        # Price changes and returns
        price_changes = np.diff(prices)
        returns = price_changes / prices[:-1]
        
        # Basic statistics
        features = {
            # Price level features
            'price_mean': np.mean(prices),
            'price_std': np.std(prices),
            'price_min': np.min(prices),
            'price_max': np.max(prices),
            'price_range': np.max(prices) - np.min(prices),
            'price_range_pct': (np.max(prices) - np.min(prices)) / np.mean(prices) * 100,
            
            # Return/change features
            'return_mean': np.mean(returns) if len(returns) > 0 else 0,
            'return_std': np.std(returns) if len(returns) > 0 else 0,
            'return_skewness': stats.skew(returns) if len(returns) > 2 else 0,
            'return_kurtosis': stats.kurtosis(returns) if len(returns) > 3 else 0,
            
            # Volatility measures
            'realized_volatility': np.sqrt(np.sum(returns**2)) if len(returns) > 0 else 0,
            'price_change_std': np.std(price_changes) if len(price_changes) > 0 else 0,
            'max_price_change': np.max(np.abs(price_changes)) if len(price_changes) > 0 else 0,
        }
        
        # Rolling statistics (if enough data)
        if n >= 5:
            rolling_std = pd.Series(prices).rolling(window=5, min_periods=1).std()
            features.update({
                'rolling_vol_mean': np.mean(rolling_std),
                'rolling_vol_std': np.std(rolling_std),
                'rolling_vol_max': np.max(rolling_std),
            })
        
        # Momentum features
        if n >= 10:
            short_ma = pd.Series(prices).rolling(window=5).mean()
            long_ma = pd.Series(prices).rolling(window=10).mean()
            momentum = short_ma - long_ma
            
            features.update({
                'momentum_mean': np.nanmean(momentum),
                'momentum_std': np.nanstd(momentum),
                'momentum_final': momentum.iloc[-1] if not pd.isna(momentum.iloc[-1]) else 0,
            })
        
        # Trend analysis
        if n >= 3:
            # Linear trend
            x = np.arange(n)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
            
            features.update({
                'trend_slope': slope,
                'trend_r_squared': r_value**2,
                'trend_p_value': p_value,
                'trend_strength': abs(slope) * (r_value**2),  # Combined measure
            })
            
            # Trend direction consistency
            price_directions = np.sign(price_changes)
            direction_changes = np.sum(np.diff(price_directions) != 0)
            features['direction_changes'] = direction_changes
            features['direction_consistency'] = 1 - (direction_changes / max(1, len(price_changes)-1))
        
        # Peak and trough analysis
        if n >= 5:
            try:
                peaks, _ = find_peaks(prices, distance=2)
                troughs, _ = find_peaks(-prices, distance=2)
                
                features.update({
                    'num_peaks': len(peaks),
                    'num_troughs': len(troughs),
                    'peak_trough_ratio': len(peaks) / max(1, len(troughs)),
                    'reversal_frequency': (len(peaks) + len(troughs)) / n * 100,
                })
            except:
                features.update({
                    'num_peaks': 0,
                    'num_troughs': 0,
                    'peak_trough_ratio': 1,
                    'reversal_frequency': 0,
                })
        
        # Autocorrelation (if enough data)
        if n >= 10:
            try:
                autocorr_1 = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
                features['autocorr_lag1'] = autocorr_1 if not np.isnan(autocorr_1) else 0
            except:
                features['autocorr_lag1'] = 0
        
        # Jump detection (large price movements)
        if len(returns) > 0:
            return_threshold = 2 * np.std(returns) if np.std(returns) > 0 else 0.001
            jumps = np.abs(returns) > return_threshold
            features.update({
                'num_jumps': np.sum(jumps),
                'jump_frequency': np.sum(jumps) / len(returns) * 100,
                'max_jump_size': np.max(np.abs(returns)) if len(returns) > 0 else 0,
            })
        
        # Volume features (if available)
        if volumes is not None and len(volumes) > 0:
            volumes = np.array(volumes)
            features.update({
                'volume_mean': np.mean(volumes),
                'volume_std': np.std(volumes),
                'volume_skewness': stats.skew(volumes) if len(volumes) > 2 else 0,
                'price_volume_corr': np.corrcoef(prices, volumes)[0, 1] if len(prices) == len(volumes) else 0,
            })
        
        return features
    
    def extract_daily_features(self):
        """Extract features for each trading day"""
        print("Extracting daily statistical features...")
        
        daily_features_list = []
        
        for trading_day in sorted(self.raw_data['TradingDay'].unique()):
            day_data = self.raw_data[self.raw_data['TradingDay'] == trading_day].copy()
            
            if len(day_data) < 5:  # Skip days with insufficient data
                continue
            
            # Extract price and volume data
            prices = day_data['Mid'].values if 'Mid' in day_data.columns else day_data.iloc[:, 2].values
            
            # Try to find volume column
            volumes = None
            volume_cols = ['Volume', 'volume', 'Vol', 'vol']
            for col in volume_cols:
                if col in day_data.columns:
                    volumes = day_data[col].values
                    break
            
            # Calculate all technical features
            features = self.calculate_technical_indicators(prices, volumes)
            
            # Add metadata
            features.update({
                'TradingDay': trading_day,
                'FromMsOfDay': self.trading_start_ms,
                'ToMsOfDay': self.trading_end_ms,
                'num_observations': len(day_data),
                'time_range_minutes': (self.trading_end_ms - self.trading_start_ms) / 60000,
            })
            
            daily_features_list.append(features)
        
        self.daily_features = pd.DataFrame(daily_features_list)
        print(f"Extracted features for {len(self.daily_features)} trading days")
        print(f"Feature dimensions: {self.daily_features.shape[1] - 4} features per day")  # Subtract metadata columns
        
        return self.daily_features
    
    def prepare_features_for_clustering(self):
        """Prepare and normalize features for GMM clustering"""
        print("Preparing features for clustering...")
        
        # Select feature columns (exclude metadata)
        metadata_cols = ['TradingDay', 'FromMsOfDay', 'ToMsOfDay', 'num_observations', 'time_range_minutes']
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
        daily_output = self.daily_features[['TradingDay', 'FromMsOfDay', 'ToMsOfDay', 'Regime'] + self.feature_names].copy()
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
            'n_regimes': len(np.unique(self.regime_labels)),
            'total_days': len(self.daily_features)
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
    
    def run_clustering(self, n_components_range=range(2, 8), use_pca=True):
        """Run the complete clustering pipeline"""
        print("Starting GMM regime clustering pipeline...")
        
        # Step 1: Load and filter data
        self.load_and_filter_data()
        
        # Step 2: Extract features
        self.extract_daily_features()
        
        # Step 3: Prepare features
        X_scaled, X_pca = self.prepare_features_for_clustering()
        
        # Step 4: Fit GMM
        scores = self.fit_gmm_model(X_scaled, n_components_range, use_pca)
        
        # Step 5: Analyze regimes
        self.analyze_regimes()
        
        # Step 6: Save results
        self.save_results()
        
        # Step 7: Create visualizations
        self.create_visualizations()
        
        print("\nClustering completed successfully!")
        print(f"Results saved to: {self.output_dir}")
        
        return scores

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='GMM-based Market Regime Clustering')
    parser.add_argument('--data_path', default='data/trainingData.csv', 
                       help='Path to training data CSV file')
    parser.add_argument('--n_regimes', type=int, default=None,
                       help='Number of regimes (if not specified, will optimize)')
    parser.add_argument('--output_dir', default='regime_clustering_results',
                       help='Output directory for results')
    parser.add_argument('--trading_start', default='10:36',
                       help='Trading start time (HH:MM format)')
    parser.add_argument('--trading_end', default='12:00', 
                       help='Trading end time (HH:MM format)')
    parser.add_argument('--use_pca', action='store_true', default=True,
                       help='Use PCA for dimensionality reduction')
    
    args = parser.parse_args()
    
    # Convert time strings to milliseconds
    def time_to_ms(time_str):
        hours, minutes = map(int, time_str.split(':'))
        return hours * 3600000 + minutes * 60000
    
    trading_start_ms = time_to_ms(args.trading_start)
    trading_end_ms = time_to_ms(args.trading_end)
    
    # Create clusterer
    clusterer = GMMRegimeClusterer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        trading_start_ms=trading_start_ms,
        trading_end_ms=trading_end_ms
    )
    
    # Determine regime range
    if args.n_regimes is not None:
        n_components_range = [args.n_regimes]
    else:
        n_components_range = range(2, 8)
    
    # Run clustering
    scores = clusterer.run_clustering(n_components_range, args.use_pca)
    
    # Print final summary
    print("\n" + "="*60)
    print("CLUSTERING SUMMARY")
    print("="*60)
    print(f"Data file: {args.data_path}")
    print(f"Trading period: {args.trading_start} to {args.trading_end}")
    print(f"Total trading days: {len(clusterer.daily_features)}")
    print(f"Number of regimes: {len(np.unique(clusterer.regime_labels))}")
    print(f"Features extracted: {len(clusterer.feature_names)}")
    print(f"Output directory: {clusterer.output_dir}")
    
    # Print regime distribution
    print("\nRegime Distribution:")
    for regime in sorted(np.unique(clusterer.regime_labels)):
        count = np.sum(clusterer.regime_labels == regime)
        percentage = count / len(clusterer.regime_labels) * 100
        print(f"  Regime {regime}: {count:3d} days ({percentage:5.1f}%)")

if __name__ == "__main__":
    main()
