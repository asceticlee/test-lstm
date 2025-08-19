#!/usr/bin/env python3
"""
Model Trading Weighter

This script provides functionality to calculate weighted scores for LSTM trading models
based on their performance metrics and select the best performing model for trading.

The weighter analyzes both daily performance and regime-based performance data to
determine the optimal model, trading direction (upside/downside), and threshold.

Main function:
    get_best_trading_model(trading_day, market_regime, weighting_array)

Returns:
    tuple: (model_id, direction, threshold, total_score)
    - model_id: string (e.g., "00001")
    - direction: int (1 for upside, -1 for downside)
    - threshold: float (0.0 to 0.8)
    - total_score: float (weighted score)

Usage:
    from model_trading_weighter import get_best_trading_model
    
    model_id, direction, threshold, score = get_best_trading_model(
        trading_day=20250707,
        market_regime=3,
        weighting_array=[0.1, 0.3, 0.7, -0.5, 0.65, -0.2, ...]
    )
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

# Define available thresholds and timeframes
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
DOWNSIDE_THRESHOLDS = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]

# Define timeframes available in the performance files
DAILY_TIMEFRAMES = [
    'daily', '2day', '3day', '1week', '2week', '4week', '8week', 
    '13week', '26week', '52week', 'from_begin'
]

REGIME_TIMEFRAMES = [
    '1day', '2day', '3day', '4day', '5day', '10day', '20day', '30day'
]

class ModelTradingWeighter:
    """
    Main class for calculating weighted scores and selecting best trading models.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize the weighter with project paths.
        
        Args:
            project_root: Optional path to project root. If None, auto-detects from script location.
        """
        if project_root is None:
            # Auto-detect project root (assuming script is in src/model_trading/)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.dirname(os.path.dirname(script_dir))
        else:
            self.project_root = project_root
            
        self.daily_performance_dir = os.path.join(
            self.project_root, 'model_performance', 'daily_performance'
        )
        self.regime_performance_dir = os.path.join(
            self.project_root, 'model_performance', 'daily_regime_performance'
        )
    
    def load_daily_performance(self, trading_day: int) -> pd.DataFrame:
        """
        Load daily performance data for a specific trading day.
        
        Args:
            trading_day: Trading day in YYYYMMDD format
            
        Returns:
            DataFrame with daily performance data
            
        Raises:
            FileNotFoundError: If the performance file doesn't exist
        """
        filename = f"trading_day_{trading_day}_performance.csv"
        filepath = os.path.join(self.daily_performance_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Daily performance file not found: {filepath}")
            
        return pd.read_csv(filepath)
    
    def load_regime_performance(self, trading_day: int, market_regime: int) -> pd.DataFrame:
        """
        Load regime performance data for a specific trading day and market regime.
        
        Args:
            trading_day: Trading day in YYYYMMDD format
            market_regime: Market regime ID
            
        Returns:
            DataFrame with regime performance data filtered for the specified regime
            
        Raises:
            FileNotFoundError: If the regime performance file doesn't exist
        """
        filename = f"trading_day_{trading_day}_regime_performance.csv"
        filepath = os.path.join(self.regime_performance_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Regime performance file not found: {filepath}")
            
        df = pd.read_csv(filepath)
        # Filter for the specific market regime
        regime_data = df[df['Regime'] == market_regime].copy()
        
        if regime_data.empty:
            raise ValueError(f"No data found for market regime {market_regime} on trading day {trading_day}")
            
        return regime_data
    
    def get_column_names(self) -> Tuple[List[str], List[str]]:
        """
        Generate column names for performance metrics.
        
        Returns:
            Tuple of (daily_columns, regime_columns) lists
        """
        daily_columns = []
        regime_columns = []
        
        # Daily performance columns
        for timeframe in DAILY_TIMEFRAMES:
            for direction in ['up', 'down']:
                for metric in ['acc', 'num', 'den', 'pnl']:
                    for threshold in THRESHOLDS:
                        col_name = f"{timeframe}_{direction}_{metric}_thr_{threshold}"
                        daily_columns.append(col_name)
        
        # Regime performance columns  
        for timeframe in REGIME_TIMEFRAMES:
            for direction in ['up', 'down']:
                for metric in ['acc', 'num', 'den', 'pnl']:
                    for threshold in THRESHOLDS:
                        col_name = f"{timeframe}_{direction}_{metric}_thr_{threshold}"
                        regime_columns.append(col_name)
        
        return daily_columns, regime_columns
    
    def calculate_model_score(self, daily_row: pd.Series, regime_row: pd.Series, 
                            model_id: str, direction: int, threshold: float, 
                            weighting_array: List[float], verbose: bool = False) -> float:
        """
        Calculate weighted score for a specific model, direction, and threshold.
        
        Args:
            daily_row: Row from daily performance data for the model
            regime_row: Row from regime performance data for the model
            model_id: Model ID string
            direction: 1 for upside, -1 for downside
            threshold: Trading threshold (0.0 to 0.8)
            weighting_array: Array of weights to apply to different metrics
            verbose: If True, print detailed scoring breakdown
            
        Returns:
            Total weighted score
        """
        # Get all available performance columns
        daily_columns, regime_columns = self.get_column_names()
        
        # Combine daily and regime data
        all_values = []
        field_names = []
        
        # Add daily performance values
        direction_str = 'up' if direction == 1 else 'down'
        
        for timeframe in DAILY_TIMEFRAMES:
            for metric in ['acc', 'num', 'den', 'pnl']:
                col_name = f"{timeframe}_{direction_str}_{metric}_thr_{threshold}"
                if col_name in daily_row.index:
                    value = daily_row[col_name]
                    # Handle NaN values
                    if pd.isna(value):
                        value = 0.0
                    all_values.append(float(value))
                    field_names.append(f"Daily_{col_name}")
                else:
                    all_values.append(0.0)
                    field_names.append(f"Daily_{col_name}")
        
        # Add regime performance values
        for timeframe in REGIME_TIMEFRAMES:
            for metric in ['acc', 'num', 'den', 'pnl']:
                col_name = f"{timeframe}_{direction_str}_{metric}_thr_{threshold}"
                if col_name in regime_row.index:
                    value = regime_row[col_name]
                    # Handle NaN values
                    if pd.isna(value):
                        value = 0.0
                    all_values.append(float(value))
                    field_names.append(f"Regime_{col_name}")
                else:
                    all_values.append(0.0)
                    field_names.append(f"Regime_{col_name}")
        
        # Ensure weighting array matches the number of values
        if len(weighting_array) != len(all_values):
            raise ValueError(
                f"Weighting array length ({len(weighting_array)}) does not match "
                f"number of performance metrics ({len(all_values)})"
            )
        
        # Calculate weighted score
        total_score = 0.0
        
        if verbose:
            print(f"\nScoring breakdown for Model {model_id}, Direction: {direction}, Threshold: {threshold}")
            print("=" * 80)
        
        for i, (value, weight, field_name) in enumerate(zip(all_values, weighting_array, field_names)):
            weighted_value = value * weight
            total_score += weighted_value
            
            if verbose:
                print(f"{field_name:40} = {value:10.6f} × {weight:8.4f} = {weighted_value:12.6f}")
        
        if verbose:
            print("=" * 80)
            print(f"{'Total Score':40} = {total_score:12.6f}")
            print("=" * 80)
        
        return total_score
    
    def get_best_trading_model(self, trading_day: int, market_regime: int, 
                             weighting_array: List[float], verbose: bool = False) -> Tuple[str, int, float, float]:
        """
        Find the best trading model based on weighted performance scores.
        
        Args:
            trading_day: Trading day in YYYYMMDD format
            market_regime: Market regime ID
            weighting_array: Array of weights for different performance metrics
            verbose: If True, print detailed scoring for the best model
            
        Returns:
            Tuple of (model_id, direction, threshold, total_score)
            - model_id: Best model ID (e.g., "00001")
            - direction: 1 for upside, -1 for downside
            - threshold: Optimal threshold (0.0 to 0.8)
            - total_score: Weighted score of the best model
            
        Raises:
            FileNotFoundError: If required performance files don't exist
            ValueError: If no data found or weighting array length mismatch
        """
        # Load performance data
        daily_df = self.load_daily_performance(trading_day)
        regime_df = self.load_regime_performance(trading_day, market_regime)
        
        # Merge the data on ModelID to ensure we have both daily and regime data
        merged_df = daily_df.merge(
            regime_df, 
            left_on='ModelID', 
            right_on='ModelID', 
            how='inner'
        )
        
        if merged_df.empty:
            raise ValueError(f"No models have both daily and regime data for trading day {trading_day}, regime {market_regime}")
        
        best_score = float('-inf')
        best_model = None
        best_direction = None
        best_threshold = None
        
        print(f"Evaluating {len(merged_df)} models for trading day {trading_day}, market regime {market_regime}")
        print(f"Testing {len(THRESHOLDS)} thresholds × 2 directions = {len(THRESHOLDS) * 2} combinations per model")
        print(f"Total combinations to evaluate: {len(merged_df) * len(THRESHOLDS) * 2}")
        
        # Iterate through all models, directions, and thresholds
        for idx, row in merged_df.iterrows():
            model_id = row['ModelID']
            
            for direction in [1, -1]:  # 1 for upside, -1 for downside
                for threshold in THRESHOLDS:
                    try:
                        # Get daily and regime rows
                        daily_row = daily_df[daily_df['ModelID'] == model_id].iloc[0]
                        regime_row = regime_df[regime_df['ModelID'] == model_id].iloc[0]
                        
                        # Calculate score for this combination
                        score = self.calculate_model_score(
                            daily_row, regime_row, model_id, direction, threshold, 
                            weighting_array, verbose=False
                        )
                        
                        # Check if this is the best score so far
                        if score > best_score:
                            best_score = score
                            best_model = model_id
                            best_direction = direction
                            best_threshold = threshold
                            
                    except Exception as e:
                        print(f"Warning: Error evaluating model {model_id}, direction {direction}, threshold {threshold}: {e}")
                        continue
        
        if best_model is None:
            raise ValueError("No valid model combinations found")
        
        # Print results
        print(f"\nBest trading model found:")
        print(f"Model ID: {best_model}")
        print(f"Direction: {'Upside' if best_direction == 1 else 'Downside'} ({best_direction})")
        print(f"Threshold: {best_threshold}")
        print(f"Total Score: {best_score:.6f}")
        
        # Show detailed breakdown for the best model if verbose
        if verbose:
            daily_row = daily_df[daily_df['ModelID'] == best_model].iloc[0]
            regime_row = regime_df[regime_df['ModelID'] == best_model].iloc[0]
            
            self.calculate_model_score(
                daily_row, regime_row, best_model, best_direction, best_threshold,
                weighting_array, verbose=True
            )
        
        return best_model, best_direction, best_threshold, best_score


def get_best_trading_model(trading_day: int, market_regime: int, 
                         weighting_array: List[float], verbose: bool = False) -> Tuple[str, int, float, float]:
    """
    Convenience function to get the best trading model.
    
    Args:
        trading_day: Trading day in YYYYMMDD format
        market_regime: Market regime ID
        weighting_array: Array of weights for different performance metrics
        verbose: If True, print detailed scoring for the best model
        
    Returns:
        Tuple of (model_id, direction, threshold, total_score)
    """
    weighter = ModelTradingWeighter()
    return weighter.get_best_trading_model(trading_day, market_regime, weighting_array, verbose)


def get_expected_weighting_array_length() -> int:
    """
    Get the expected length of the weighting array.
    
    Returns:
        Integer representing the number of performance metrics
    """
    # Calculate total number of metrics
    total_metrics = 0
    
    # Daily performance: timeframes × directions × metrics × thresholds
    total_metrics += len(DAILY_TIMEFRAMES) * 2 * 4 * len(THRESHOLDS)
    
    # Regime performance: timeframes × directions × metrics × thresholds  
    total_metrics += len(REGIME_TIMEFRAMES) * 2 * 4 * len(THRESHOLDS)
    
    return total_metrics


def print_weighting_array_structure():
    """
    Print the structure of the weighting array to help users understand the format.
    """
    print("Weighting Array Structure:")
    print("=" * 50)
    
    index = 0
    
    print("\nDAILY PERFORMANCE METRICS:")
    print("-" * 30)
    for timeframe in DAILY_TIMEFRAMES:
        for direction in ['up', 'down']:
            for metric in ['acc', 'num', 'den', 'pnl']:
                for threshold in THRESHOLDS:
                    col_name = f"Daily_{timeframe}_{direction}_{metric}_thr_{threshold}"
                    print(f"Index {index:3d}: {col_name}")
                    index += 1
    
    print(f"\nREGIME PERFORMANCE METRICS:")
    print("-" * 30)
    for timeframe in REGIME_TIMEFRAMES:
        for direction in ['up', 'down']:
            for metric in ['acc', 'num', 'den', 'pnl']:
                for threshold in THRESHOLDS:
                    col_name = f"Regime_{timeframe}_{direction}_{metric}_thr_{threshold}"
                    print(f"Index {index:3d}: {col_name}")
                    index += 1
    
    print(f"\nTotal expected weighting array length: {index}")


if __name__ == "__main__":
    """
    Example usage and testing
    """
    if len(sys.argv) == 1:
        # Print structure information
        print("Model Trading Weighter")
        print("=" * 50)
        print_weighting_array_structure()
        
        expected_length = get_expected_weighting_array_length()
        print(f"\nTo use this module, create a weighting array of length {expected_length}")
        print("\nExample usage:")
        print("```python")
        print("from model_trading_weighter import get_best_trading_model")
        print()
        print("# Create example weighting array (all equal weights)")
        print(f"weighting_array = [1.0] * {expected_length}")
        print()
        print("# Get best model")
        print("model_id, direction, threshold, score = get_best_trading_model(")
        print("    trading_day=20250707,")
        print("    market_regime=3,")
        print("    weighting_array=weighting_array,")
        print("    verbose=True")
        print(")")
        print("```")
        
    elif len(sys.argv) == 4:
        # Example execution with command line arguments
        try:
            trading_day = int(sys.argv[1])
            market_regime = int(sys.argv[2])
            
            # Parse weighting array (comma-separated values)
            weights_str = sys.argv[3]
            weighting_array = [float(x.strip()) for x in weights_str.split(',')]
            
            print(f"Testing with trading_day={trading_day}, market_regime={market_regime}")
            print(f"Weighting array length: {len(weighting_array)}")
            
            model_id, direction, threshold, score = get_best_trading_model(
                trading_day, market_regime, weighting_array, verbose=True
            )
            
            print(f"\nResult: Model {model_id}, Direction {direction}, Threshold {threshold}, Score {score}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: python model_trading_weighter.py <trading_day> <market_regime> <comma_separated_weights>")
    
    else:
        print("Usage:")
        print("  python model_trading_weighter.py")
        print("    Show weighting array structure")
        print()
        print("  python model_trading_weighter.py <trading_day> <market_regime> <comma_separated_weights>")
        print("    Example: python model_trading_weighter.py 20250707 3 '0.1,0.3,0.7,-0.5,0.65,-0.2,...'")
