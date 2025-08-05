#!/usr/bin/env python3
"""
Market Regime Classification using Gaussian Mixture Model

This script analyzes market data to classify different market regimes on a weekly basis.
The classification helps identify different market regimes to apply specialized LSTM models.

Features:
- Weekly market regime classification
- Feature engineering for volatility, trend, and momentum indicators
- GMM-based clustering to identify regimes
- Regime performance analysis for model selection
- Saves regime labels for training data splits

Usage:
    python market_regime_classifier.py [--retrain] [--weeks N]
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
import joblib

# Visualization
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_palette("husl")
except ImportError:
    print("Seaborn not available, using matplotlib defaults")
    sns = None

# Set style
plt.style.use('default')


class MarketRegimeClassifier:
    """
    Classifies market regimes using weekly aggregated features and GMM
    """
    
    def __init__(self, data_path, output_dir='regime_analysis'):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model components
        self.scaler = StandardScaler()
        self.clusterer = None
        self.gmm_classifier = None
        self.pca = None
        
        # Data storage
        self.raw_data = None
        self.weekly_features = None
        self.regime_labels = None
        
        print(f"Market Regime Classifier initialized")
        print(f"Output directory: {self.output_dir}")
        
    def load_data(self):
        """Load and preprocess the trading data"""
        print("Loading trading data...")
        
        # Load data
        self.raw_data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.raw_data):,} rows of data")
        
        # Convert TradingDay to datetime
        self.raw_data['Date'] = pd.to_datetime(self.raw_data['TradingDay'], format='%Y%m%d')
        
        # Create custom Sunday-to-Saturday weeks
        # Find the Sunday for each date (day 0 = Monday, 6 = Sunday)
        self.raw_data['DayOfWeek'] = self.raw_data['Date'].dt.dayofweek
        # Calculate days since Sunday (Sunday = 0, Monday = 1, ..., Saturday = 6)
        days_since_sunday = (self.raw_data['DayOfWeek'] + 1) % 7
        # Find the Sunday for each date
        week_start = self.raw_data['Date'] - pd.to_timedelta(days_since_sunday, unit='D')
        week_end = week_start + pd.to_timedelta(6, unit='D')
        
        # Create week identifier as "YYYY-MM-DD/YYYY-MM-DD" format
        self.raw_data['Week'] = week_start.dt.strftime('%Y-%m-%d') + '/' + week_end.dt.strftime('%Y-%m-%d')
        self.raw_data['Week_Start_Date'] = week_start
        self.raw_data['Week_End_Date'] = week_end
        
        # Get date range
        start_date = self.raw_data['Date'].min()
        end_date = self.raw_data['Date'].max()
        print(f"Data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Using Sunday-to-Saturday weeks")
        
        unique_weeks = self.raw_data['Week'].nunique()
        print(f"Total weeks: {unique_weeks}")
        
    def engineer_weekly_features(self):
        """Create weekly aggregated features for regime classification"""
        print("Engineering weekly features...")
        
        weekly_list = []
        
        for week, week_data in self.raw_data.groupby('Week'):
            if len(week_data) < 10:  # Skip weeks with insufficient data
                continue
                
            # Get the week start and end dates from the data
            week_start = week_data['Week_Start_Date'].iloc[0]
            week_end = week_data['Week_End_Date'].iloc[0]
            
            features = {
                'Week': week,
                'Date': week_data['Date'].iloc[0],
                'Week_Start': week_data['Date'].min(),
                'Week_End': week_data['Date'].max(),
                'Week_Start_Sunday': week_start,
                'Week_End_Saturday': week_end,
                'Trading_Days': week_data['Date'].nunique(),
                'Total_Minutes': len(week_data)
            }
            
            # Price-based features
            mid_prices = week_data['Mid'].values
            features['Price_Mean'] = np.mean(mid_prices)
            features['Price_Std'] = np.std(mid_prices)
            features['Price_Range'] = np.max(mid_prices) - np.min(mid_prices)
            features['Price_CV'] = features['Price_Std'] / features['Price_Mean'] if features['Price_Mean'] != 0 else 0
            
            # Returns analysis
            returns = np.diff(mid_prices) / mid_prices[:-1]
            features['Return_Mean'] = np.mean(returns)
            features['Return_Std'] = np.std(returns)
            features['Return_Skew'] = pd.Series(returns).skew()
            features['Return_Kurt'] = pd.Series(returns).kurtosis()
            features['Return_Min'] = np.min(returns)
            features['Return_Max'] = np.max(returns)
            
            # Volatility measures
            features['Realized_Vol'] = np.sqrt(np.sum(returns**2))
            features['High_Low_Vol'] = features['Price_Range'] / features['Price_Mean']
            
            # Trend indicators
            features['Week_Return'] = (mid_prices[-1] - mid_prices[0]) / mid_prices[0]
            features['Trend_Strength'] = abs(features['Week_Return'])
            
            # Up/Down minutes ratio
            up_moves = np.sum(returns > 0)
            down_moves = np.sum(returns < 0)
            features['Up_Down_Ratio'] = up_moves / (down_moves + 1e-8)
            features['Up_Percentage'] = up_moves / len(returns)
            
            # Bid-Ask spread analysis
            spreads = week_data['Ask'] - week_data['Bid']
            features['Spread_Mean'] = np.mean(spreads)
            features['Spread_Std'] = np.std(spreads)
            features['Spread_Max'] = np.max(spreads)
            
            # ROC features aggregation
            roc_cols = [col for col in week_data.columns if 'ROC' in col]
            for col in roc_cols:
                features[f'{col}_Mean'] = week_data[col].mean()
                features[f'{col}_Std'] = week_data[col].std()
                features[f'{col}_Max'] = week_data[col].max()
                features[f'{col}_Min'] = week_data[col].min()
            
            # Price difference features
            diff_cols = [col for col in week_data.columns if 'PriceDiff' in col]
            for col in diff_cols:
                features[f'{col}_Mean'] = week_data[col].mean()
                features[f'{col}_Std'] = week_data[col].std()
                
            # Momentum indicators
            features['Momentum_1min'] = week_data['ROC_01min:ROC'].mean()
            features['Momentum_5min'] = week_data['ROC_05min:ROC'].mean()
            features['Momentum_10min'] = week_data['ROC_10min:ROC'].mean()
            
            # Label performance (for regime quality assessment)
            label_cols = [col for col in week_data.columns if col.startswith('Label_')]
            for label_col in label_cols[:10]:  # Use first 10 labels for analysis
                label_values = week_data[label_col].values
                features[f'{label_col}_Mean'] = np.mean(label_values)
                features[f'{label_col}_Std'] = np.std(label_values)
                features[f'{label_col}_Positive_Ratio'] = np.mean(label_values > 0)
                
            weekly_list.append(features)
        
        self.weekly_features = pd.DataFrame(weekly_list)
        print(f"Created {len(self.weekly_features)} weekly feature vectors")
        print(f"Feature dimensions: {self.weekly_features.shape[1] - 6} features")  # Exclude metadata columns
        
        # Save weekly features
        output_file = self.output_dir / 'weekly_features.csv'
        self.weekly_features.to_csv(output_file, index=False)
        print(f"Saved weekly features to {output_file}")
        
    def determine_optimal_clusters(self, max_clusters=8, min_regime_size=3):
        """Determine optimal number of regimes using multiple metrics for GMM"""
        print("Determining optimal number of regimes using GMM...")
        print(f"Minimum regime size constraint: {min_regime_size} weeks")
        
        # Get feature columns (exclude metadata and datetime columns)
        feature_cols = [col for col in self.weekly_features.columns 
                       if col not in ['Week', 'Date', 'Week_Start', 'Week_End', 'Week_Start_Sunday', 'Week_End_Saturday', 'Trading_Days', 'Total_Minutes']]
        
        X = self.weekly_features[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Dimensionality reduction for better clustering
        self.pca = PCA(n_components=min(20, X_scaled.shape[1]))
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Test different numbers of clusters with minimum size constraint
        bic_scores = []
        aic_scores = []
        silhouette_scores = []
        log_likelihoods = []
        valid_clusters = []
        
        # Calculate theoretical maximum clusters based on minimum size
        n_weeks = len(X_pca)
        theoretical_max = min(max_clusters, n_weeks // min_regime_size)
        
        cluster_range = range(2, theoretical_max + 1)
        
        for n_clusters in cluster_range:
            # Fit GMM with multiple random initializations to avoid poor local minima
            best_gmm = None
            best_bic = float('inf')
            
            for init_trial in range(5):  # Try 5 different initializations
                try:
                    gmm = GaussianMixture(
                        n_components=n_clusters, 
                        random_state=42 + init_trial, 
                        covariance_type='full',
                        max_iter=200,
                        n_init=3
                    )
                    labels = gmm.fit_predict(X_pca)
                    
                    # Check minimum regime size constraint
                    regime_counts = np.bincount(labels)
                    min_regime_count = np.min(regime_counts)
                    
                    if min_regime_count >= min_regime_size:
                        current_bic = gmm.bic(X_pca)
                        if current_bic < best_bic:
                            best_bic = current_bic
                            best_gmm = gmm
                            
                except Exception as e:
                    print(f"Warning: GMM fitting failed for {n_clusters} clusters, trial {init_trial}: {e}")
                    continue
            
            if best_gmm is not None:
                labels = best_gmm.predict(X_pca)
                regime_counts = np.bincount(labels)
                
                # Calculate metrics
                bic = best_gmm.bic(X_pca)
                aic = best_gmm.aic(X_pca)
                log_likelihood = best_gmm.score(X_pca)
                silhouette_avg = silhouette_score(X_pca, labels)
                
                bic_scores.append(bic)
                aic_scores.append(aic)
                log_likelihoods.append(log_likelihood)
                silhouette_scores.append(silhouette_avg)
                valid_clusters.append(n_clusters)
                
                print(f"  {n_clusters} clusters: min regime size = {np.min(regime_counts)}, BIC = {bic:.1f}")
            else:
                print(f"  {n_clusters} clusters: Failed to meet minimum size constraint")
        
        if not valid_clusters:
            print(f"Warning: No valid clustering solutions found with min_regime_size={min_regime_size}")
            print("Falling back to less restrictive clustering...")
            # Fallback with reduced constraint
            return self.determine_optimal_clusters(max_clusters, min_regime_size=1)
        
        # Update cluster_range to only include valid clusters
        cluster_range = valid_clusters
            
        # Plot cluster analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(cluster_range, bic_scores, 'bo-')
        axes[0, 0].set_title('BIC Score (lower is better)')
        axes[0, 0].set_xlabel('Number of Regimes')
        axes[0, 0].set_ylabel('BIC Score')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(cluster_range, aic_scores, 'ro-')
        axes[0, 1].set_title('AIC Score (lower is better)')
        axes[0, 1].set_xlabel('Number of Regimes')
        axes[0, 1].set_ylabel('AIC Score')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(cluster_range, log_likelihoods, 'go-')
        axes[1, 0].set_title('Log Likelihood (higher is better)')
        axes[1, 0].set_xlabel('Number of Regimes')
        axes[1, 0].set_ylabel('Log Likelihood')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(cluster_range, silhouette_scores, 'mo-')
        axes[1, 1].set_title('Silhouette Score')
        axes[1, 1].set_xlabel('Number of Regimes')
        axes[1, 1].set_ylabel('Silhouette Score')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Select optimal number based on BIC (lower is better)
        optimal_idx = np.argmin(bic_scores)
        optimal_clusters = list(cluster_range)[optimal_idx]
        
        print(f"Optimal number of regimes: {optimal_clusters}")
        print(f"BIC score: {bic_scores[optimal_idx]:.3f}")
        print(f"AIC score: {aic_scores[optimal_idx]:.3f}")
        print(f"Silhouette score: {silhouette_scores[optimal_idx]:.3f}")
        
        return optimal_clusters, X_pca
    
    def consolidate_small_regimes(self, min_regime_size=3):
        """Consolidate regimes that are too small by merging them with nearest neighbors"""
        print(f"Consolidating regimes smaller than {min_regime_size} weeks...")
        
        if self.weekly_features is None or 'Regime' not in self.weekly_features.columns:
            print("Warning: No regime assignments found. Run classify_regimes() first.")
            return
        
        # Count regime sizes
        regime_counts = self.weekly_features['Regime'].value_counts().sort_index()
        small_regimes = regime_counts[regime_counts < min_regime_size].index.tolist()
        
        if not small_regimes:
            print("No small regimes found. No consolidation needed.")
            return
        
        print(f"Found {len(small_regimes)} regimes with < {min_regime_size} weeks: {small_regimes}")
        
        # Get feature data for regime centers
        feature_cols = [col for col in self.weekly_features.columns 
                       if col not in ['Week', 'Date', 'Week_Start', 'Week_End', 'Week_Start_Sunday', 'Week_End_Saturday', 'Trading_Days', 'Total_Minutes', 'Regime', 'Regime_Probability']]
        
        # Calculate regime centers
        regime_centers = {}
        for regime in self.weekly_features['Regime'].unique():
            regime_data = self.weekly_features[self.weekly_features['Regime'] == regime]
            regime_centers[regime] = regime_data[feature_cols].mean().values
        
        # Merge small regimes with their nearest neighbors
        for small_regime in small_regimes:
            if small_regime not in regime_centers:
                continue
                
            small_center = regime_centers[small_regime]
            
            # Find nearest large regime
            min_distance = float('inf')
            nearest_regime = None
            
            for target_regime, target_center in regime_centers.items():
                if (target_regime != small_regime and 
                    target_regime not in small_regimes and
                    regime_counts[target_regime] >= min_regime_size):
                    
                    distance = np.linalg.norm(small_center - target_center)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_regime = target_regime
            
            if nearest_regime is not None:
                # Merge small regime into nearest regime
                mask = self.weekly_features['Regime'] == small_regime
                self.weekly_features.loc[mask, 'Regime'] = nearest_regime
                print(f"  Merged regime {small_regime} ({regime_counts[small_regime]} weeks) into regime {nearest_regime}")
            else:
                print(f"  Warning: Could not find suitable merge target for regime {small_regime}")
        
        # Relabel regimes to be consecutive starting from 0
        unique_regimes = sorted(self.weekly_features['Regime'].unique())
        regime_mapping = {old_regime: new_regime for new_regime, old_regime in enumerate(unique_regimes)}
        
        self.weekly_features['Regime'] = self.weekly_features['Regime'].map(regime_mapping)
        
        # Print final regime distribution
        final_counts = self.weekly_features['Regime'].value_counts().sort_index()
        print(f"Final regime distribution:")
        for regime, count in final_counts.items():
            print(f"  Regime {regime}: {count} weeks ({count/len(self.weekly_features)*100:.1f}%)")
        
        return regime_mapping
        
    def classify_regimes(self, n_regimes=None, min_regime_size=3):
        """Classify market regimes using Gaussian Mixture Model"""
        print("Classifying market regimes using GMM...")
        
        # Determine optimal clusters if not specified
        if n_regimes is None:
            n_regimes, X_pca = self.determine_optimal_clusters(min_regime_size=min_regime_size)
        else:
            feature_cols = [col for col in self.weekly_features.columns 
                           if col not in ['Week', 'Date', 'Week_Start', 'Week_End', 'Week_Start_Sunday', 'Week_End_Saturday', 'Trading_Days', 'Total_Minutes']]
            X = self.weekly_features[feature_cols].fillna(0)
            X_scaled = self.scaler.fit_transform(X)
            self.pca = PCA(n_components=min(20, X_scaled.shape[1]))
            X_pca = self.pca.fit_transform(X_scaled)
        
        # Fit Gaussian Mixture Model with multiple attempts to avoid small clusters
        best_gmm = None
        best_score = float('-inf')
        best_labels = None
        
        for attempt in range(10):  # Multiple attempts with different initializations
            try:
                gmm = GaussianMixture(
                    n_components=n_regimes, 
                    random_state=42 + attempt, 
                    covariance_type='full',
                    max_iter=300,
                    n_init=5,
                    init_params='k-means++'  # Better initialization
                )
                labels = gmm.fit_predict(X_pca)
                
                # Check if all regimes meet minimum size requirement
                regime_counts = np.bincount(labels)
                min_count = np.min(regime_counts)
                
                # Score based on likelihood and minimum regime size
                likelihood_score = gmm.score(X_pca)
                size_penalty = -100 if min_count < min_regime_size else 0
                total_score = likelihood_score + size_penalty
                
                if total_score > best_score:
                    best_score = total_score
                    best_gmm = gmm
                    best_labels = labels
                    
                print(f"  Attempt {attempt + 1}: min regime size = {min_count}, score = {likelihood_score:.2f}")
                
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {e}")
                continue
        
        if best_gmm is None:
            raise ValueError("Failed to fit GMM after multiple attempts")
        
        self.gmm_classifier = best_gmm
        regime_labels = best_labels
        
        # Get probabilities for each regime assignment
        regime_probabilities = self.gmm_classifier.predict_proba(X_pca)
        
        # Add regime labels to weekly features
        self.weekly_features['Regime'] = regime_labels
        self.weekly_features['Regime_Probability'] = np.max(regime_probabilities, axis=1)
        
        # Check and consolidate small regimes if necessary
        regime_counts = self.weekly_features['Regime'].value_counts()
        small_regimes = regime_counts[regime_counts < min_regime_size]
        
        if len(small_regimes) > 0:
            print(f"Found {len(small_regimes)} regimes with < {min_regime_size} weeks. Consolidating...")
            self.consolidate_small_regimes(min_regime_size)
        
        # Analyze regime characteristics
        self.analyze_regimes()
        
        # Save regime assignments
        self.save_regime_assignments()
        
        # Store the fitted GMM for future predictions
        self.clusterer = self.gmm_classifier  # For compatibility
        
    def analyze_regimes(self):
        """Analyze characteristics of each regime"""
        print("Analyzing regime characteristics...")
        
        regime_summary = []
        
        for regime in sorted(self.weekly_features['Regime'].unique()):
            regime_data = self.weekly_features[self.weekly_features['Regime'] == regime]
            
            summary = {
                'Regime': regime,
                'Weeks': len(regime_data),
                'Percentage': len(regime_data) / len(self.weekly_features) * 100,
                'Avg_Weekly_Return': regime_data['Week_Return'].mean(),
                'Volatility': regime_data['Return_Std'].mean(),
                'Trend_Strength': regime_data['Trend_Strength'].mean(),
                'Up_Down_Ratio': regime_data['Up_Down_Ratio'].mean(),
                'Spread_Mean': regime_data['Spread_Mean'].mean(),
                'Avg_Probability': regime_data['Regime_Probability'].mean(),
            }
            
            # Label performance analysis
            for i in range(1, 11):  # Analyze first 10 labels
                col = f'Label_{i}_Mean'
                if col in regime_data.columns:
                    summary[f'Label_{i}_Avg'] = regime_data[col].mean()
                    summary[f'Label_{i}_Pos_Ratio'] = regime_data[f'Label_{i}_Positive_Ratio'].mean()
            
            regime_summary.append(summary)
        
        self.regime_summary = pd.DataFrame(regime_summary)
        
        # Save regime summary
        output_file = self.output_dir / 'regime_summary.csv'
        self.regime_summary.to_csv(output_file, index=False)
        
        # Print summary
        print("\nRegime Summary:")
        print("=" * 80)
        for _, row in self.regime_summary.iterrows():
            print(f"Regime {int(row['Regime'])}: {row['Weeks']} weeks ({row['Percentage']:.1f}%)")
            print(f"  Weekly Return: {row['Avg_Weekly_Return']:.4f}")
            print(f"  Volatility: {row['Volatility']:.4f}")
            print(f"  Trend Strength: {row['Trend_Strength']:.4f}")
            print(f"  Up/Down Ratio: {row['Up_Down_Ratio']:.2f}")
            print(f"  Avg Assignment Probability: {row['Avg_Probability']:.3f}")
            print()
        
    def evaluate_gmm_quality(self, X_pca):
        """Evaluate the quality of GMM clustering"""
        print("Evaluating GMM clustering quality...")
        
        if self.gmm_classifier is None:
            print("Warning: GMM not fitted yet.")
            return
        
        # Calculate model metrics
        bic = self.gmm_classifier.bic(X_pca)
        aic = self.gmm_classifier.aic(X_pca)
        log_likelihood = self.gmm_classifier.score(X_pca)
        
        # Get regime assignments
        regime_labels = self.gmm_classifier.predict(X_pca)
        regime_probs = self.gmm_classifier.predict_proba(X_pca)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_pca, regime_labels)
        
        print(f"GMM Clustering Quality:")
        print(f"  BIC Score: {bic:.3f}")
        print(f"  AIC Score: {aic:.3f}")
        print(f"  Log Likelihood: {log_likelihood:.3f}")
        print(f"  Silhouette Score: {silhouette_avg:.3f}")
        print(f"  Average Assignment Probability: {np.mean(np.max(regime_probs, axis=1)):.3f}")
        
        # Plot regime assignment probabilities
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Histogram of assignment probabilities
        plt.subplot(2, 2, 1)
        max_probs = np.max(regime_probs, axis=1)
        plt.hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Regime Assignment Probabilities')
        plt.xlabel('Maximum Probability')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(max_probs), color='red', linestyle='--', label=f'Mean: {np.mean(max_probs):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Regime assignment probabilities over time
        plt.subplot(2, 2, 2)
        timeline_data = self.weekly_features.copy()
        timeline_data['Date'] = pd.to_datetime(timeline_data['Date'])
        timeline_data = timeline_data.sort_values('Date')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(regime_labels))))
        for i, regime in enumerate(sorted(np.unique(regime_labels))):
            regime_mask = timeline_data['Regime'] == regime
            if np.any(regime_mask):
                plt.scatter(timeline_data.loc[regime_mask, 'Date'], 
                           timeline_data.loc[regime_mask, 'Regime_Probability'], 
                           c=[colors[i]], label=f'Regime {regime}', alpha=0.7)
        
        plt.title('Regime Assignment Probabilities Over Time')
        plt.xlabel('Date')
        plt.ylabel('Assignment Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Plot 3: Covariance matrices visualization (first few components)
        n_components = min(4, self.gmm_classifier.n_components)
        for i in range(n_components):
            plt.subplot(2, 2, 3 if i < 2 else 4)
            if i == 0:
                # Show mean vectors
                means = self.gmm_classifier.means_
                plt.scatter(means[:, 0], means[:, 1], s=100, c=colors[:len(means)], 
                           marker='x', linewidths=3)
                plt.title('GMM Component Centers (PC1 vs PC2)')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.grid(True, alpha=0.3)
            elif i == 1:
                # Show weights
                weights = self.gmm_classifier.weights_
                plt.pie(weights, labels=[f'Regime {j}' for j in range(len(weights))], 
                       autopct='%1.1f%%', colors=colors[:len(weights)])
                plt.title('GMM Component Weights')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gmm_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_regime_assignments(self):
        """Save regime assignments for use in LSTM training"""
        print("Saving regime assignments...")
        
        # Create detailed regime mapping
        paradigm_mapping = []
        
        for _, week_row in self.weekly_features.iterrows():
            # Get all days in this week from raw data using the Week identifier
            week_data = self.raw_data[self.raw_data['Week'] == week_row['Week']]
            
            for _, day_row in week_data.iterrows():
                paradigm_mapping.append({
                    'TradingDay': int(day_row['TradingDay']),
                    'Date': day_row['Date'].strftime('%Y-%m-%d'),
                    'Week': week_row['Week'],
                    'Regime': int(week_row['Regime'])
                })
        
        paradigm_df = pd.DataFrame(paradigm_mapping)
        
        # Save detailed mapping
        output_file = self.output_dir / 'regime_assignments.csv'
        paradigm_df.to_csv(output_file, index=False)
        print(f"Saved regime assignments to {output_file}")
        
        # Save weekly summary with Sunday-Saturday information
        weekly_output = self.output_dir / 'weekly_regimes.csv'
        weekly_paradigm = self.weekly_features[['Week', 'Date', 'Week_Start', 'Week_End', 'Week_Start_Sunday', 'Week_End_Saturday', 'Regime']].copy()
        weekly_paradigm.to_csv(weekly_output, index=False)
        print(f"Saved weekly paradigms to {weekly_output}")
        
    def save_models(self):
        """Save trained models for future use"""
        print("Saving trained models...")
        
        model_files = {
            'scaler': self.scaler,
            'gmm_classifier': self.gmm_classifier,
            'pca': self.pca
        }
        
        for name, model in model_files.items():
            if model is not None:
                model_path = self.output_dir / f'{name}.joblib'
                joblib.dump(model, model_path)
                print(f"Saved {name} to {model_path}")
        
    def load_models(self):
        """Load previously trained models"""
        print("Loading trained models...")
        
        model_files = ['scaler', 'gmm_classifier', 'pca']
        
        for name in model_files:
            model_path = self.output_dir / f'{name}.joblib'
            if model_path.exists():
                setattr(self, name, joblib.load(model_path))
                print(f"Loaded {name} from {model_path}")
                
                # For backward compatibility
                if name == 'gmm_classifier':
                    self.clusterer = self.gmm_classifier
            else:
                print(f"Model file {model_path} not found")
                
    def predict_regime(self, week_features):
        """Predict regime for new weekly features"""
        if self.gmm_classifier is None:
            raise ValueError("GMM classifier not trained. Run classify_regimes() first.")
            
        # Prepare features
        feature_cols = [col for col in week_features.columns 
                       if col not in ['Week', 'Date', 'Week_Start', 'Week_End', 'Week_Start_Sunday', 'Week_End_Saturday', 'Trading_Days', 'Total_Minutes']]
        
        X = week_features[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # Predict regime
        regime = self.gmm_classifier.predict(X_pca)
        probability = self.gmm_classifier.predict_proba(X_pca)
        
        return regime[0], probability[0]
        
    def visualize_regimes(self):
        """Create visualizations of regime analysis"""
        print("Creating regime visualizations...")
        
        # Time series of regimes
        plt.figure(figsize=(15, 8))
        
        # Plot 1: Regime timeline
        plt.subplot(2, 2, 1)
        regime_timeline = self.weekly_features.copy()
        regime_timeline['Date'] = pd.to_datetime(regime_timeline['Date'])
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(regime_timeline['Regime'].unique())))
        
        for i, regime in enumerate(sorted(regime_timeline['Regime'].unique())):
            regime_data = regime_timeline[regime_timeline['Regime'] == regime]
            plt.scatter(regime_data['Date'], regime_data['Week_Return'], 
                       c=[colors[i]], label=f'Regime {regime}', alpha=0.7)
        
        plt.title('Market Regimes Over Time')
        plt.xlabel('Date')
        plt.ylabel('Weekly Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Volatility vs Return by regime
        plt.subplot(2, 2, 2)
        for i, regime in enumerate(sorted(regime_timeline['Regime'].unique())):
            regime_data = regime_timeline[regime_timeline['Regime'] == regime]
            plt.scatter(regime_data['Return_Std'], regime_data['Week_Return'], 
                       c=[colors[i]], label=f'Regime {regime}', alpha=0.7)
        
        plt.title('Volatility vs Return by Regime')
        plt.xlabel('Weekly Volatility')
        plt.ylabel('Weekly Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Regime distribution
        plt.subplot(2, 2, 3)
        regime_counts = regime_timeline['Regime'].value_counts().sort_index()
        plt.pie(regime_counts.values, labels=[f'Regime {i}' for i in regime_counts.index], 
                autopct='%1.1f%%', colors=colors[:len(regime_counts)])
        plt.title('Regime Distribution')
        
        # Plot 4: Assignment probability distribution by regime
        plt.subplot(2, 2, 4)
        for regime in sorted(regime_timeline['Regime'].unique()):
            regime_data = regime_timeline[regime_timeline['Regime'] == regime]
            plt.hist(regime_data['Regime_Probability'], alpha=0.5, label=f'Regime {regime}', bins=15)
        
        plt.title('Assignment Probability Distribution by Regime')
        plt.xlabel('Assignment Probability')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'regime_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation heatmap for each regime (if seaborn available)
        if sns is not None:
            for regime in sorted(self.weekly_features['Regime'].unique()):
                regime_data = self.weekly_features[self.weekly_features['Regime'] == regime]
                
                # Select key features for correlation
                key_features = ['Week_Return', 'Return_Std', 'Trend_Strength', 'Up_Down_Ratio', 
                               'Realized_Vol', 'Spread_Mean', 'Momentum_1min', 'Momentum_5min']
                
                available_features = [f for f in key_features if f in regime_data.columns]
                
                if len(available_features) > 1:
                    corr_matrix = regime_data[available_features].corr()
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                               square=True, fmt='.2f')
                    plt.title(f'Feature Correlation - Regime {regime}')
                    plt.tight_layout()
                    plt.savefig(self.output_dir / f'regime_{regime}_correlation.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        
        print(f"Visualizations saved to {self.output_dir}/")
        
    def run_analysis(self, n_regimes=None, min_regime_size=3, create_plots=True):
        """Run complete regime analysis"""
        print("Starting Market Regime Analysis with GMM")
        print("=" * 50)
        print(f"Minimum regime size: {min_regime_size} weeks")
        
        # Load and process data
        self.load_data()
        self.engineer_weekly_features()
        
        # Classify regimes using GMM
        self.classify_regimes(n_regimes, min_regime_size)
        
        # Evaluate GMM quality
        feature_cols = [col for col in self.weekly_features.columns 
                       if col not in ['Week', 'Date', 'Week_Start', 'Week_End', 'Week_Start_Sunday', 'Week_End_Saturday', 'Trading_Days', 'Total_Minutes', 'Regime', 'Regime_Probability']]
        X = self.weekly_features[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        self.evaluate_gmm_quality(X_pca)
        
        # Save models and results
        self.save_models()
        
        # Create visualizations
        if create_plots:
            self.visualize_regimes()
        
        print("\nAnalysis complete!")
        print(f"Results saved to: {self.output_dir}/")
        print(f"Use regime assignments for specialized LSTM training.")
        
        return self.regime_summary


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Regime Classification')
    parser.add_argument('--data', type=str, default='data/trainingData.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--output', type=str, default='regime_analysis',
                       help='Output directory for results')
    parser.add_argument('--regimes', type=int, default=None,
                       help='Number of regimes (auto-detect if not specified)')
    parser.add_argument('--min-size', type=int, default=3,
                       help='Minimum number of weeks per regime (default: 3)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Get project root and data path
    script_dir = Path(__file__).parent.absolute()
    if script_dir.name == 'src':
        project_root = script_dir.parent
    else:
        project_root = script_dir
        
    data_path = project_root / args.data
    output_dir = project_root / args.output
    
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    # Run analysis
    classifier = MarketRegimeClassifier(data_path, output_dir)
    regime_summary = classifier.run_analysis(
        n_regimes=args.regimes,
        min_regime_size=args.min_size,
        create_plots=not args.no_plots
    )
    
    print("\nRegime Analysis Summary:")
    print(regime_summary)
    

if __name__ == "__main__":
    main()
