#!/usr/bin/env python3
"""
Model Trading Weighter - Fixed Version

This script provides functionality to calculate weighted scores for LSTM trading models
based on their performance metrics and select the best performing model for trading.

The weighter analyzes both daily performance and regime-based performance data to
determine the optimal model, trading direction (upside/downside), and threshold.

Main function:
    get_best_trading_model(trading_day, market_regime, weighting_array)

Returns:
    dict: Best model information with model_id, direction, threshold, and score

Usage:
    from model_trading_weighter_fixed import ModelTradingWeighter
    
    weighter = ModelTradingWeighter()
    result = weighter.get_best_trading_model(
        trading_day="20250707",
        market_regime=3,
        weighting_array=np.array([0.1, 0.3, 0.7, -0.5, 0.65, -0.2, ...])  # 1368 weights
    )
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class ModelTradingWeighter:
    """
    A class to calculate weighted scores for trading models based on performance metrics.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the ModelTradingWeighter.
        
        Args:
            data_dir: Base directory containing model performance data.
                     If None, uses current directory.
        """
        if data_dir is None:
            data_dir = os.getcwd()
        
        self.data_dir = data_dir
        self.daily_performance_dir = os.path.join(data_dir, "model_performance", "daily_performance")
        self.regime_performance_dir = os.path.join(data_dir, "model_performance", "daily_regime_performance")
        
        # Cache for loaded data
        self._daily_cache = {}
        self._regime_cache = {}
    
    def _load_daily_performance(self, trading_day: str) -> Optional[pd.DataFrame]:
        """Load daily performance data for a specific trading day."""
        if trading_day in self._daily_cache:
            return self._daily_cache[trading_day]
        
        filename = f"trading_day_{trading_day}_performance.csv"
        filepath = os.path.join(self.daily_performance_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: Daily performance file not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            self._daily_cache[trading_day] = df
            return df
        except Exception as e:
            print(f"Error loading daily performance file {filepath}: {e}")
            return None
    
    def _load_regime_performance(self, trading_day: str, market_regime: int) -> Optional[pd.DataFrame]:
        """Load regime performance data for a specific trading day and market regime."""
        cache_key = f"{trading_day}_{market_regime}"
        if cache_key in self._regime_cache:
            return self._regime_cache[cache_key]
        
        filename = f"trading_day_{trading_day}_regime_performance.csv"
        filepath = os.path.join(self.regime_performance_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: Regime performance file not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            # Filter for the specific market regime
            regime_data = df[df['Regime'] == market_regime].copy()
            
            if regime_data.empty:
                print(f"Warning: No data found for market regime {market_regime} on trading day {trading_day}")
                return None
            
            self._regime_cache[cache_key] = regime_data
            return regime_data
        except Exception as e:
            print(f"Error loading regime performance file {filepath}: {e}")
            return None
    
    def get_metric_columns_info(self) -> Dict:
        """
        Get information about metric columns structure.
        
        Returns:
            Dict with daily_metrics, regime_metrics, total_metrics
        """
        # Try to load a sample file to get actual column structure
        sample_files = [f for f in os.listdir(self.daily_performance_dir) if f.endswith('.csv')]
        if not sample_files:
            raise FileNotFoundError("No daily performance files found")
        
        # Load sample daily file
        sample_daily = pd.read_csv(os.path.join(self.daily_performance_dir, sample_files[0]))
        daily_metrics = len(sample_daily.columns) - 2  # Subtract ModelID and TradingDay
        
        # Load sample regime file
        regime_files = [f for f in os.listdir(self.regime_performance_dir) if f.endswith('.csv')]
        if not regime_files:
            raise FileNotFoundError("No regime performance files found")
        
        sample_regime = pd.read_csv(os.path.join(self.regime_performance_dir, regime_files[0]))
        regime_metrics = len(sample_regime.columns) - 2  # Subtract ModelID and Regime
        
        return {
            'daily_metrics': daily_metrics,
            'regime_metrics': regime_metrics,
            'total_metrics': daily_metrics + regime_metrics,
            'daily_columns': [col for col in sample_daily.columns if col not in ['ModelID', 'TradingDay']],
            'regime_columns': [col for col in sample_regime.columns if col not in ['ModelID', 'Regime']]
        }
    
    def calculate_model_score(self, model_id: str, daily_data: pd.DataFrame, 
                            regime_data: pd.DataFrame, weighting_array: np.ndarray) -> float:
        """
        Calculate weighted score for a specific model using all available metrics.
        
        Args:
            model_id: Model ID to evaluate
            daily_data: Daily performance DataFrame
            regime_data: Regime performance DataFrame  
            weighting_array: Array of weights for all metrics
            
        Returns:
            Total weighted score for the model
        """
        # Get model rows
        daily_row = daily_data[daily_data['ModelID'] == model_id]
        regime_row = regime_data[regime_data['ModelID'] == model_id]
        
        if daily_row.empty or regime_row.empty:
            return float('-inf')  # Model not found
        
        daily_row = daily_row.iloc[0]
        regime_row = regime_row.iloc[0]
        
        # Get metric columns info
        info = self.get_metric_columns_info()
        
        # Validate weighting array length
        if len(weighting_array) != info['total_metrics']:
            raise ValueError(
                f"Weighting array length ({len(weighting_array)}) does not match "
                f"total metrics ({info['total_metrics']}). Expected {info['daily_metrics']} "
                f"daily + {info['regime_metrics']} regime metrics."
            )
        
        # Combine all metric values
        all_values = []
        
        # Add daily metrics
        for col in info['daily_columns']:
            value = daily_row.get(col, 0.0)
            if pd.isna(value):
                value = 0.0
            all_values.append(float(value))
        
        # Add regime metrics
        for col in info['regime_columns']:
            value = regime_row.get(col, 0.0)
            if pd.isna(value):
                value = 0.0
            all_values.append(float(value))
        
        # Calculate weighted score
        all_values = np.array(all_values)
        total_score = np.dot(all_values, weighting_array)
        
        return total_score
    
    def get_best_trading_model(self, trading_day: str, market_regime: int, 
                             weighting_array: np.ndarray) -> Dict:
        """
        Find the best trading model based on weighted performance scores.
        
        Args:
            trading_day: Trading day in format 'YYYYMMDD'
            market_regime: Market regime (0, 1, 2, 3, or 4)
            weighting_array: Array of weights for all performance metrics (length 1368)
            
        Returns:
            Dict with keys: model_id, score, direction, threshold, details
        """
        # Load performance data
        daily_data = self._load_daily_performance(trading_day)
        regime_data = self._load_regime_performance(trading_day, market_regime)
        
        if daily_data is None or regime_data is None:
            raise ValueError(f"Could not load performance data for trading day {trading_day} and regime {market_regime}")
        
        # Find common models
        daily_models = set(daily_data['ModelID'].unique())
        regime_models = set(regime_data['ModelID'].unique())
        common_models = daily_models.intersection(regime_models)
        
        if not common_models:
            raise ValueError(f"No common models found between daily and regime data")
        
        print(f"Evaluating {len(common_models)} models for trading day {trading_day}, market regime {market_regime}")
        
        # Calculate scores for all models
        model_scores = {}
        for model_id in common_models:
            try:
                score = self.calculate_model_score(model_id, daily_data, regime_data, weighting_array)
                model_scores[model_id] = score
            except Exception as e:
                print(f"Warning: Error evaluating model {model_id}: {e}")
                continue
        
        if not model_scores:
            raise ValueError("No valid model combinations found")
        
        # Find best model
        best_model_id = max(model_scores.keys(), key=lambda x: model_scores[x])
        best_score = model_scores[best_model_id]
        
        # For now, return basic info (direction and threshold logic can be added later)
        return {
            'model_id': best_model_id,
            'score': best_score,
            'direction': 'up',  # Default - can be enhanced with actual analysis
            'threshold': 0.5,   # Default - can be enhanced with actual analysis
            'details': f"Best model from {len(model_scores)} evaluated models"
        }


def get_best_trading_model(trading_day: str, market_regime: int, weighting_array: np.ndarray) -> Dict:
    """
    Convenience function to get the best trading model.
    
    Args:
        trading_day: Trading day in format 'YYYYMMDD'
        market_regime: Market regime (0, 1, 2, 3, or 4)
        weighting_array: Array of weights for all performance metrics
        
    Returns:
        Dict with best model information
    """
    weighter = ModelTradingWeighter()
    return weighter.get_best_trading_model(trading_day, market_regime, weighting_array)


if __name__ == "__main__":
    # Example usage
    weighter = ModelTradingWeighter()
    
    # Get metric info
    info = weighter.get_metric_columns_info()
    print(f"Metrics structure:")
    print(f"- Daily metrics: {info['daily_metrics']}")
    print(f"- Regime metrics: {info['regime_metrics']}")
    print(f"- Total metrics: {info['total_metrics']}")
    
    # Create a simple weighting array (equal weights)
    total_metrics = info['total_metrics']
    equal_weights = np.ones(total_metrics) / total_metrics
    
    # Test with sample data
    try:
        result = weighter.get_best_trading_model("20250707", 0, equal_weights)
        print(f"\nBest model result: {result}")
    except Exception as e:
        print(f"Error: {e}")
