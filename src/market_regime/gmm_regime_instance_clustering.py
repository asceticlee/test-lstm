#!/usr/bin/env python3
"""
GMM Regime Instance Clustering

This script provides real-time regime classification for market data instances.
It loads the pre-trained GMM model and applies it to new data points using the same
technical indicators as the original gmm_regime_clustering.py.

Main Function:
    classify_regime(overnight_gap, quote_data) -> int
    
    Given overnight gap and quote data, returns the regime classification (0-4).

Usage:
    from gmm_regime_instance_clustering import GMMRegimeInstanceClassifier
    
    classifier = GMMRegimeInstanceClassifier()
    regime = classifier.classify_regime(overnight_gap, quote_data)
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import warnings
import json
import joblib
warnings.filterwarnings('ignore')

# Add the src directory to the path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent
sys.path.insert(0, str(src_dir))

# Import our statistical features module
from market_data_stat.statistical_features import StatisticalFeatureExtractor

class GMMRegimeInstanceClassifier:
    """
    Real-time regime classification using pre-trained GMM model
    """
    
    def __init__(self, model_dir='../../market_regime/gmm/daily'):
        """
        Initialize the regime classifier with pre-trained models
        
        Args:
            model_dir: Directory containing the trained models and configuration
        """
        # Get the absolute path to the script directory
        script_dir = Path(__file__).parent.absolute()
        
        # Resolve model directory relative to script directory
        if not os.path.isabs(model_dir):
            self.model_dir = script_dir / model_dir
        else:
            self.model_dir = Path(model_dir)
        
        # Load configuration
        self._load_configuration()
        
        # Load pre-trained models
        self._load_models()
        
        # Initialize feature extractor with same parameters
        self.feature_extractor = StatisticalFeatureExtractor()
        
        print(f"GMM Regime Instance Classifier initialized")
        print(f"Model directory: {self.model_dir}")
        print(f"Trading period: {self.ms_to_time(self.trading_start_ms)} to {self.ms_to_time(self.trading_end_ms)}")
        print(f"Reference time: {self.ms_to_time(self.reference_time_ms)}")
        print(f"Number of regimes: {self.n_regimes}")
        print(f"Feature dimensions: {len(self.feature_names)}")
    
    def ms_to_time(self, ms):
        """Convert milliseconds of day to readable time format"""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        return f"{hours:02d}:{minutes:02d}"
    
    def _load_configuration(self):
        """Load the clustering configuration from JSON file"""
        config_file = self.model_dir / 'clustering_info.json'
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Store configuration parameters
        self.feature_names = config['feature_names']
        self.trading_start_ms = config['trading_start_ms']
        self.trading_end_ms = config['trading_end_ms']
        self.reference_time_ms = config['reference_time_ms']
        self.include_overnight_gap = config['include_overnight_gap']
        self.overnight_indicators = config.get('overnight_indicators', False)
        self.n_regimes = config['n_regimes']
        
        print(f"Loaded configuration: {len(self.feature_names)} features, {self.n_regimes} regimes")
    
    def _load_models(self):
        """Load the pre-trained GMM model, scaler, and PCA"""
        # Load GMM model
        gmm_file = self.model_dir / 'gmm_model.pkl'
        if not gmm_file.exists():
            raise FileNotFoundError(f"GMM model file not found: {gmm_file}")
        self.gmm_model = joblib.load(gmm_file)
        
        # Load feature scaler
        scaler_file = self.model_dir / 'feature_scaler.pkl'
        if not scaler_file.exists():
            raise FileNotFoundError(f"Feature scaler file not found: {scaler_file}")
        self.scaler = joblib.load(scaler_file)
        
        # Load PCA model
        pca_file = self.model_dir / 'pca_model.pkl'
        if not pca_file.exists():
            raise FileNotFoundError(f"PCA model file not found: {pca_file}")
        self.pca = joblib.load(pca_file)
        
        print("Loaded pre-trained models: GMM, scaler, and PCA")
    
    def _prepare_quote_data(self, quote_data, overnight_gap):
        """
        Prepare quote data in the format expected by the feature extractor
        
        Args:
            quote_data: List of dict with keys ['ms_of_day', 'bid', 'ask', 'mid'] or DataFrame
            overnight_gap: Overnight gap value
            
        Returns:
            pd.DataFrame: Prepared data frame
        """
        # Convert to DataFrame if needed
        if isinstance(quote_data, list):
            df = pd.DataFrame(quote_data)
        else:
            df = quote_data.copy()
        
        # Ensure required columns exist
        required_cols = ['ms_of_day', 'mid']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in quote_data")
        
        # Filter for trading period
        df = df[
            (df['ms_of_day'] >= self.trading_start_ms) &
            (df['ms_of_day'] <= self.trading_end_ms)
        ].copy()
        
        if len(df) < 5:
            raise ValueError(f"Insufficient data points: {len(df)}. Need at least 5 data points in trading period.")
        
        # Sort by time
        df = df.sort_values('ms_of_day').reset_index(drop=True)
        
        # Add trading day (dummy value for feature extraction)
        df['trading_day'] = 20250815  # Use current date as dummy
        
        # Add previous close for overnight gap calculation
        # Calculate previous close from overnight gap: first_price = prev_close + gap
        first_price = df['mid'].iloc[0]
        prev_close = first_price - overnight_gap
        df['prev_close'] = prev_close
        
        return df
    
    def extract_features_for_instance(self, quote_data, overnight_gap):
        """
        Extract the same features used in training for a single instance
        
        Args:
            quote_data: List of dict or DataFrame with quote data
            overnight_gap: Overnight gap value
            
        Returns:
            np.array: Feature vector for the instance
        """
        # Prepare data
        df = self._prepare_quote_data(quote_data, overnight_gap)
        
        # Extract prices
        prices = df['mid'].values
        volumes = None  # history_spot_quote.csv doesn't have volume
        
        # Find reference price (same logic as original)
        reference_price = None
        reference_rows = df[df['ms_of_day'] == self.reference_time_ms]
        if len(reference_rows) > 0:
            reference_price = reference_rows['mid'].iloc[0]
        else:
            # Find closest time before reference
            before_ref = df[df['ms_of_day'] <= self.reference_time_ms]
            if len(before_ref) > 0:
                closest_idx = before_ref['ms_of_day'].idxmax()
                reference_price = before_ref.loc[closest_idx, 'mid']
            else:
                reference_price = prices[0]
        
        # Calculate features using the same extractor
        features = self.feature_extractor.calculate_all_features(
            prices=prices, 
            volumes=volumes, 
            reference_price=reference_price, 
            use_relative=True,
            include_volume=False
        )
        
        # Add overnight gap feature if enabled
        if self.include_overnight_gap:
            features['overnight_gap_absolute'] = overnight_gap
        
        # Create feature vector in the same order as training
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                # Handle missing features with default values
                print(f"Warning: Feature '{feature_name}' not found, using 0")
                feature_vector.append(0.0)
        
        return np.array(feature_vector).reshape(1, -1)
    
    def classify_regime(self, overnight_gap, quote_data):
        """
        Classify the regime for given overnight gap and quote data
        
        Args:
            overnight_gap: Overnight gap value (absolute price difference)
            quote_data: Quote data as list of dict or DataFrame
                       Must contain columns: 'ms_of_day', 'mid'
                       Optional columns: 'bid', 'ask'
                       
        Returns:
            int: Regime classification (0 to n_regimes-1)
            
        Example:
            quote_data = [
                {'ms_of_day': 34200000, 'bid': 323.53, 'ask': 323.54, 'mid': 323.535},
                {'ms_of_day': 34260000, 'bid': 323.72, 'ask': 323.73, 'mid': 323.725},
                # ... more data points
            ]
            regime = classifier.classify_regime(-1.5, quote_data)
        """
        try:
            # Extract features
            feature_vector = self.extract_features_for_instance(quote_data, overnight_gap)
            
            # Handle missing values (fill with 0)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply scaling (same as training)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Apply PCA (same as training)
            feature_vector_pca = self.pca.transform(feature_vector_scaled)
            
            # Predict regime using GMM model
            regime = self.gmm_model.predict(feature_vector_pca)[0]
            
            return int(regime)
            
        except Exception as e:
            print(f"Error in regime classification: {e}")
            raise
    
    def classify_regime_with_probabilities(self, overnight_gap, quote_data):
        """
        Classify regime and return probabilities for all regimes
        
        Args:
            overnight_gap: Overnight gap value
            quote_data: Quote data as list of dict or DataFrame
            
        Returns:
            tuple: (predicted_regime, probabilities_array)
        """
        try:
            # Extract features
            feature_vector = self.extract_features_for_instance(quote_data, overnight_gap)
            
            # Handle missing values
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply scaling and PCA
            feature_vector_scaled = self.scaler.transform(feature_vector)
            feature_vector_pca = self.pca.transform(feature_vector_scaled)
            
            # Get prediction and probabilities
            regime = self.gmm_model.predict(feature_vector_pca)[0]
            probabilities = self.gmm_model.predict_proba(feature_vector_pca)[0]
            
            return int(regime), probabilities
            
        except Exception as e:
            print(f"Error in regime classification with probabilities: {e}")
            raise

def create_sample_quote_data(trading_start_ms, trading_end_ms, base_price=323.5, num_points=30):
    """
    Create sample quote data for testing
    
    Args:
        trading_start_ms: Start time in milliseconds
        trading_end_ms: End time in milliseconds  
        base_price: Base price for simulation
        num_points: Number of data points
        
    Returns:
        list: Sample quote data
    """
    np.random.seed(42)
    
    time_points = np.linspace(trading_start_ms, trading_end_ms, num_points)
    
    quote_data = []
    current_price = base_price
    
    for ms_of_day in time_points:
        # Simulate price movement
        price_change = np.random.normal(0, 0.001) * current_price
        current_price += price_change
        
        spread = 0.01  # 1 cent spread
        bid = current_price - spread/2
        ask = current_price + spread/2
        mid = current_price
        
        quote_data.append({
            'ms_of_day': int(ms_of_day),
            'bid': bid,
            'ask': ask,
            'mid': mid
        })
    
    return quote_data

# Example usage and testing
if __name__ == "__main__":
    # Initialize classifier
    try:
        classifier = GMMRegimeInstanceClassifier()
        
        # Create sample data for testing
        sample_data = create_sample_quote_data(
            classifier.trading_start_ms, 
            classifier.trading_end_ms,
            base_price=323.5,
            num_points=50
        )
        
        # Test classification
        overnight_gap = -1.5  # Example overnight gap
        regime = classifier.classify_regime(overnight_gap, sample_data)
        regime_with_probs, probabilities = classifier.classify_regime_with_probabilities(
            overnight_gap, sample_data
        )
        
        print(f"\nSample classification results:")
        print(f"Overnight gap: {overnight_gap}")
        print(f"Data points: {len(sample_data)}")
        print(f"Predicted regime: {regime}")
        print(f"Regime probabilities: {probabilities}")
        print(f"Most likely regime: {regime_with_probs}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
