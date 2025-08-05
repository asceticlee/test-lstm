#!/usr/bin/env python3
"""
Market Paradigm Classification using Gradient Boosting Machine

This script analyzes market data to classify different market paradigms on a weekly basis.
The classification helps identify different market regimes to apply specialized LSTM models.

Features:
- Weekly market paradigm classification
- Feature engineering for volatility, trend, and momentum indicators
- GBM-based unsupervised clustering to identify paradigms
- Paradigm performance analysis for model selection
- Saves paradigm labels for training data splits

Usage:
    python market_paradigm_classifier.py [--retrain] [--weeks N]
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
from sklearn.ensemble import GradientBoostingClassifier
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


class MarketParadigmClassifier:
    """
    Classifies market paradigms using weekly aggregated features and GBM
    """
    
    def __init__(self, data_path, output_dir='paradigm_analysis'):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model components
        self.scaler = StandardScaler()
        self.clusterer = None
        self.gbm_classifier = None
        self.pca = None
        
        # Data storage
        self.raw_data = None
        self.weekly_features = None
        self.paradigm_labels = None
        
        print(f"Market Paradigm Classifier initialized")
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
        """Create weekly aggregated features for paradigm classification"""
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
            
            # Label performance (for paradigm quality assessment)
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
        
    def determine_optimal_clusters(self, max_clusters=8):
        """Determine optimal number of paradigms using multiple metrics"""
        print("Determining optimal number of paradigms...")
        
        # Get feature columns (exclude metadata and datetime columns)
        feature_cols = [col for col in self.weekly_features.columns 
                       if col not in ['Week', 'Date', 'Week_Start', 'Week_End', 'Week_Start_Sunday', 'Week_End_Saturday', 'Trading_Days', 'Total_Minutes']]
        
        X = self.weekly_features[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Dimensionality reduction for better clustering
        self.pca = PCA(n_components=min(20, X_scaled.shape[1]))
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Test different numbers of clusters
        silhouette_scores = []
        calinski_scores = []
        inertias = []
        
        cluster_range = range(2, min(max_clusters + 1, len(X_pca) // 4))
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_pca)
            
            silhouette_avg = silhouette_score(X_pca, labels)
            calinski_score = calinski_harabasz_score(X_pca, labels)
            
            silhouette_scores.append(silhouette_avg)
            calinski_scores.append(calinski_score)
            inertias.append(kmeans.inertia_)
            
        # Plot cluster analysis
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].plot(cluster_range, silhouette_scores, 'bo-')
        axes[0].set_title('Silhouette Score')
        axes[0].set_xlabel('Number of Paradigms')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].grid(True)
        
        axes[1].plot(cluster_range, calinski_scores, 'ro-')
        axes[1].set_title('Calinski-Harabasz Score')
        axes[1].set_xlabel('Number of Paradigms')
        axes[1].set_ylabel('CH Score')
        axes[1].grid(True)
        
        axes[2].plot(cluster_range, inertias, 'go-')
        axes[2].set_title('Elbow Method')
        axes[2].set_xlabel('Number of Paradigms')
        axes[2].set_ylabel('Inertia')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Select optimal number based on silhouette score
        optimal_idx = np.argmax(silhouette_scores)
        optimal_clusters = list(cluster_range)[optimal_idx]
        
        print(f"Optimal number of paradigms: {optimal_clusters}")
        print(f"Silhouette score: {silhouette_scores[optimal_idx]:.3f}")
        
        return optimal_clusters, X_pca
        
    def classify_paradigms(self, n_paradigms=None):
        """Classify market paradigms using K-means clustering"""
        print("Classifying market paradigms...")
        
        # Determine optimal clusters if not specified
        if n_paradigms is None:
            n_paradigms, X_pca = self.determine_optimal_clusters()
        else:
            feature_cols = [col for col in self.weekly_features.columns 
                           if col not in ['Week', 'Date', 'Week_Start', 'Week_End', 'Week_Start_Sunday', 'Week_End_Saturday', 'Trading_Days', 'Total_Minutes']]
            X = self.weekly_features[feature_cols].fillna(0)
            X_scaled = self.scaler.fit_transform(X)
            self.pca = PCA(n_components=min(20, X_scaled.shape[1]))
            X_pca = self.pca.fit_transform(X_scaled)
        
        # Fit K-means clustering
        self.clusterer = KMeans(n_clusters=n_paradigms, random_state=42, n_init=10)
        paradigm_labels = self.clusterer.fit_predict(X_pca)
        
        # Add paradigm labels to weekly features
        self.weekly_features['Paradigm'] = paradigm_labels
        
        # Analyze paradigm characteristics
        self.analyze_paradigms()
        
        # Train GBM classifier for new data prediction
        self.train_gbm_classifier(X_pca, paradigm_labels)
        
        # Save paradigm assignments
        self.save_paradigm_assignments()
        
    def analyze_paradigms(self):
        """Analyze characteristics of each paradigm"""
        print("Analyzing paradigm characteristics...")
        
        paradigm_summary = []
        
        for paradigm in sorted(self.weekly_features['Paradigm'].unique()):
            paradigm_data = self.weekly_features[self.weekly_features['Paradigm'] == paradigm]
            
            summary = {
                'Paradigm': paradigm,
                'Weeks': len(paradigm_data),
                'Percentage': len(paradigm_data) / len(self.weekly_features) * 100,
                'Avg_Weekly_Return': paradigm_data['Week_Return'].mean(),
                'Volatility': paradigm_data['Return_Std'].mean(),
                'Trend_Strength': paradigm_data['Trend_Strength'].mean(),
                'Up_Down_Ratio': paradigm_data['Up_Down_Ratio'].mean(),
                'Spread_Mean': paradigm_data['Spread_Mean'].mean(),
            }
            
            # Label performance analysis
            for i in range(1, 11):  # Analyze first 10 labels
                col = f'Label_{i}_Mean'
                if col in paradigm_data.columns:
                    summary[f'Label_{i}_Avg'] = paradigm_data[col].mean()
                    summary[f'Label_{i}_Pos_Ratio'] = paradigm_data[f'Label_{i}_Positive_Ratio'].mean()
            
            paradigm_summary.append(summary)
        
        self.paradigm_summary = pd.DataFrame(paradigm_summary)
        
        # Save paradigm summary
        output_file = self.output_dir / 'paradigm_summary.csv'
        self.paradigm_summary.to_csv(output_file, index=False)
        
        # Print summary
        print("\nParadigm Summary:")
        print("=" * 80)
        for _, row in self.paradigm_summary.iterrows():
            print(f"Paradigm {int(row['Paradigm'])}: {row['Weeks']} weeks ({row['Percentage']:.1f}%)")
            print(f"  Weekly Return: {row['Avg_Weekly_Return']:.4f}")
            print(f"  Volatility: {row['Volatility']:.4f}")
            print(f"  Trend Strength: {row['Trend_Strength']:.4f}")
            print(f"  Up/Down Ratio: {row['Up_Down_Ratio']:.2f}")
            print()
        
    def train_gbm_classifier(self, X_pca, paradigm_labels):
        """Train GBM classifier for paradigm prediction on new data"""
        print("Training GBM classifier for paradigm prediction...")
        
        # Check class distribution for stratification
        unique_labels, label_counts = np.unique(paradigm_labels, return_counts=True)
        min_class_size = np.min(label_counts)
        
        # Split data for training (stratify only if all classes have at least 2 samples)
        if min_class_size >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X_pca, paradigm_labels, test_size=0.2, random_state=42, stratify=paradigm_labels
            )
        else:
            print(f"Warning: Some paradigms have too few samples ({min_class_size}). Using random split instead of stratified.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_pca, paradigm_labels, test_size=0.2, random_state=42
            )
        
        # Train GBM classifier
        self.gbm_classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.gbm_classifier.fit(X_train, y_train)
        
        # Evaluate classifier
        train_score = self.gbm_classifier.score(X_train, y_train)
        test_score = self.gbm_classifier.score(X_test, y_test)
        
        print(f"GBM Classifier Performance:")
        print(f"  Training Accuracy: {train_score:.3f}")
        print(f"  Test Accuracy: {test_score:.3f}")
        
        # Feature importance
        feature_importance = self.gbm_classifier.feature_importances_
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        indices = np.argsort(feature_importance)[::-1][:20]
        plt.bar(range(len(indices)), feature_importance[indices])
        plt.title('Top 20 PCA Components for Paradigm Classification')
        plt.xlabel('PCA Component')
        plt.ylabel('Importance')
        plt.xticks(range(len(indices)), [f'PC{i+1}' for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_paradigm_assignments(self):
        """Save paradigm assignments for use in LSTM training"""
        print("Saving paradigm assignments...")
        
        # Create detailed paradigm mapping
        paradigm_mapping = []
        
        for _, week_row in self.weekly_features.iterrows():
            # Get all days in this week from raw data using the Week identifier
            week_data = self.raw_data[self.raw_data['Week'] == week_row['Week']]
            
            for _, day_row in week_data.iterrows():
                paradigm_mapping.append({
                    'TradingDay': int(day_row['TradingDay']),
                    'Date': day_row['Date'].strftime('%Y-%m-%d'),
                    'Week': week_row['Week'],
                    'Paradigm': int(week_row['Paradigm'])
                })
        
        paradigm_df = pd.DataFrame(paradigm_mapping)
        
        # Save detailed mapping
        output_file = self.output_dir / 'paradigm_assignments.csv'
        paradigm_df.to_csv(output_file, index=False)
        print(f"Saved paradigm assignments to {output_file}")
        
        # Save weekly summary with Sunday-Saturday information
        weekly_output = self.output_dir / 'weekly_paradigms.csv'
        weekly_paradigm = self.weekly_features[['Week', 'Date', 'Week_Start', 'Week_End', 'Week_Start_Sunday', 'Week_End_Saturday', 'Paradigm']].copy()
        weekly_paradigm.to_csv(weekly_output, index=False)
        print(f"Saved weekly paradigms to {weekly_output}")
        
    def save_models(self):
        """Save trained models for future use"""
        print("Saving trained models...")
        
        model_files = {
            'scaler': self.scaler,
            'clusterer': self.clusterer,
            'gbm_classifier': self.gbm_classifier,
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
        
        model_files = ['scaler', 'clusterer', 'gbm_classifier', 'pca']
        
        for name in model_files:
            model_path = self.output_dir / f'{name}.joblib'
            if model_path.exists():
                setattr(self, name, joblib.load(model_path))
                print(f"Loaded {name} from {model_path}")
            else:
                print(f"Model file {model_path} not found")
                
    def predict_paradigm(self, week_features):
        """Predict paradigm for new weekly features"""
        if self.gbm_classifier is None:
            raise ValueError("GBM classifier not trained. Run classify_paradigms() first.")
            
        # Prepare features
        feature_cols = [col for col in week_features.columns 
                       if col not in ['Week', 'Date', 'Week_Start', 'Week_End', 'Week_Start_Sunday', 'Week_End_Saturday', 'Trading_Days', 'Total_Minutes']]
        
        X = week_features[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # Predict paradigm
        paradigm = self.gbm_classifier.predict(X_pca)
        probability = self.gbm_classifier.predict_proba(X_pca)
        
        return paradigm[0], probability[0]
        
    def visualize_paradigms(self):
        """Create visualizations of paradigm analysis"""
        print("Creating paradigm visualizations...")
        
        # Time series of paradigms
        plt.figure(figsize=(15, 8))
        
        # Plot 1: Paradigm timeline
        plt.subplot(2, 2, 1)
        paradigm_timeline = self.weekly_features.copy()
        paradigm_timeline['Date'] = pd.to_datetime(paradigm_timeline['Date'])
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(paradigm_timeline['Paradigm'].unique())))
        
        for i, paradigm in enumerate(sorted(paradigm_timeline['Paradigm'].unique())):
            paradigm_data = paradigm_timeline[paradigm_timeline['Paradigm'] == paradigm]
            plt.scatter(paradigm_data['Date'], paradigm_data['Week_Return'], 
                       c=[colors[i]], label=f'Paradigm {paradigm}', alpha=0.7)
        
        plt.title('Market Paradigms Over Time')
        plt.xlabel('Date')
        plt.ylabel('Weekly Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Volatility vs Return by paradigm
        plt.subplot(2, 2, 2)
        for i, paradigm in enumerate(sorted(paradigm_timeline['Paradigm'].unique())):
            paradigm_data = paradigm_timeline[paradigm_timeline['Paradigm'] == paradigm]
            plt.scatter(paradigm_data['Return_Std'], paradigm_data['Week_Return'], 
                       c=[colors[i]], label=f'Paradigm {paradigm}', alpha=0.7)
        
        plt.title('Volatility vs Return by Paradigm')
        plt.xlabel('Weekly Volatility')
        plt.ylabel('Weekly Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Paradigm distribution
        plt.subplot(2, 2, 3)
        paradigm_counts = paradigm_timeline['Paradigm'].value_counts().sort_index()
        plt.pie(paradigm_counts.values, labels=[f'Paradigm {i}' for i in paradigm_counts.index], 
                autopct='%1.1f%%', colors=colors[:len(paradigm_counts)])
        plt.title('Paradigm Distribution')
        
        # Plot 4: Weekly return distribution by paradigm
        plt.subplot(2, 2, 4)
        for paradigm in sorted(paradigm_timeline['Paradigm'].unique()):
            paradigm_data = paradigm_timeline[paradigm_timeline['Paradigm'] == paradigm]
            plt.hist(paradigm_data['Week_Return'], alpha=0.5, label=f'Paradigm {paradigm}', bins=20)
        
        plt.title('Weekly Return Distribution by Paradigm')
        plt.xlabel('Weekly Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'paradigm_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation heatmap for each paradigm (if seaborn available)
        if sns is not None:
            for paradigm in sorted(self.weekly_features['Paradigm'].unique()):
                paradigm_data = self.weekly_features[self.weekly_features['Paradigm'] == paradigm]
                
                # Select key features for correlation
                key_features = ['Week_Return', 'Return_Std', 'Trend_Strength', 'Up_Down_Ratio', 
                               'Realized_Vol', 'Spread_Mean', 'Momentum_1min', 'Momentum_5min']
                
                available_features = [f for f in key_features if f in paradigm_data.columns]
                
                if len(available_features) > 1:
                    corr_matrix = paradigm_data[available_features].corr()
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                               square=True, fmt='.2f')
                    plt.title(f'Feature Correlation - Paradigm {paradigm}')
                    plt.tight_layout()
                    plt.savefig(self.output_dir / f'paradigm_{paradigm}_correlation.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        
        print(f"Visualizations saved to {self.output_dir}/")
        
    def run_analysis(self, n_paradigms=None, create_plots=True):
        """Run complete paradigm analysis"""
        print("Starting Market Paradigm Analysis")
        print("=" * 50)
        
        # Load and process data
        self.load_data()
        self.engineer_weekly_features()
        
        # Classify paradigms
        self.classify_paradigms(n_paradigms)
        
        # Save models and results
        self.save_models()
        
        # Create visualizations
        if create_plots:
            self.visualize_paradigms()
        
        print("\nAnalysis complete!")
        print(f"Results saved to: {self.output_dir}/")
        print(f"Use paradigm assignments for specialized LSTM training.")
        
        return self.paradigm_summary


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Paradigm Classification')
    parser.add_argument('--data', type=str, default='data/trainingData.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--output', type=str, default='paradigm_analysis',
                       help='Output directory for results')
    parser.add_argument('--paradigms', type=int, default=None,
                       help='Number of paradigms (auto-detect if not specified)')
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
    classifier = MarketParadigmClassifier(data_path, output_dir)
    paradigm_summary = classifier.run_analysis(
        n_paradigms=args.paradigms,
        create_plots=not args.no_plots
    )
    
    print("\nParadigm Analysis Summary:")
    print(paradigm_summary)
    

if __name__ == "__main__":
    main()
