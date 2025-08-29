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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

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
        self.model_predictions_dir = os.path.join(data_dir, "model_predictions")
        
        # Cache for loaded data
        self._daily_cache = {}
        self._regime_cache = {}
    
    def wilson_score_accuracy(self, k: float, n: float, z: float = 1.96) -> float:
        """
        Calculate Wilson score interval point estimate for adjusted accuracy.
        
        This method provides a statistically sound way to compare accuracies with different
        sample sizes. It penalizes models with high accuracy but very few trades.
        
        Args:
            k: Number of correct trades (numerator)
            n: Total number of trades (denominator)
            z: Z-score for confidence interval (1.96 for 95% CI)
            
        Returns:
            Adjusted accuracy using Wilson score interval
            
        Formula:
            hat_p = (k + z²/2) / (n + z²)
            
        Example:
            Model 1: k=1, n=1 (100% raw) → Wilson ≈ 0.603
            Model 2: k=5, n=7 (71.4% raw) → Wilson ≈ 0.638
        """
        if n <= 0:
            return 0.0
        
        z_squared = z * z
        numerator = k + (z_squared / 2)
        denominator = n + z_squared
        
        return numerator / denominator
    
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
            Dict with daily_metrics, regime_metrics, total_metrics, and all column info
        """
        # Try to load a sample file to get actual column structure
        sample_files = [f for f in os.listdir(self.daily_performance_dir) if f.endswith('.csv')]
        if not sample_files:
            raise FileNotFoundError("No daily performance files found")
        
        # Load sample daily file
        sample_daily = pd.read_csv(os.path.join(self.daily_performance_dir, sample_files[0]))
        
        # Load sample regime file
        regime_files = [f for f in os.listdir(self.regime_performance_dir) if f.endswith('.csv')]
        if not regime_files:
            raise FileNotFoundError("No regime performance files found")
        
        sample_regime = pd.read_csv(os.path.join(self.regime_performance_dir, regime_files[0]))
        
        # Get original columns (excluding num/den, keep acc and pnl)
        daily_original_cols = [col for col in sample_daily.columns 
                              if col not in ['ModelID', 'TradingDay'] 
                              and not ('_num_' in col or '_den_' in col)]
        
        regime_original_cols = [col for col in sample_regime.columns 
                               if col not in ['ModelID', 'Regime'] 
                               and not ('_num_' in col or '_den_' in col)]
        
        # Create extended columns: for each acc column, add ws_acc column
        # For each pnl column, add ppt (pnl per trade) column
        daily_extended_cols = []
        for col in daily_original_cols:
            daily_extended_cols.append(col)  # Original acc or pnl column
            if '_acc_' in col:
                # Add Wilson scored version
                ws_col = col.replace('_acc_', '_ws_acc_')
                daily_extended_cols.append(ws_col)
            elif '_pnl_' in col:
                # Add PnL per trade version (pnl / den)
                ppt_col = col.replace('_pnl_', '_ppt_')
                daily_extended_cols.append(ppt_col)
        
        regime_extended_cols = []
        for col in regime_original_cols:
            regime_extended_cols.append(col)  # Original acc or pnl column
            if '_acc_' in col:
                # Add Wilson scored version
                ws_col = col.replace('_acc_', '_ws_acc_')
                regime_extended_cols.append(ws_col)
            elif '_pnl_' in col:
                # Add PnL per trade version (pnl / den)
                ppt_col = col.replace('_pnl_', '_ppt_')
                regime_extended_cols.append(ppt_col)
        
        return {
            'daily_metrics': len(daily_extended_cols),
            'regime_metrics': len(regime_extended_cols),
            'total_metrics': len(daily_extended_cols) + len(regime_extended_cols),
            'daily_columns': daily_extended_cols,
            'regime_columns': regime_extended_cols,
            'original_daily_columns': [col for col in sample_daily.columns if col not in ['ModelID', 'TradingDay']],
            'original_regime_columns': [col for col in sample_regime.columns if col not in ['ModelID', 'Regime']]
        }
    
    def get_threshold_direction_columns(self, threshold: float, direction: str) -> List[str]:
        """
        Get the column names for a specific threshold and direction combination.
        
        Args:
            threshold: Threshold value (negative for downside, positive for upside)
            direction: Direction ('up' or 'down')
            
        Returns:
            List of column names for this threshold+direction combination
        """
        # Load sample files to get column structure
        sample_files = [f for f in os.listdir(self.daily_performance_dir) if f.endswith('.csv')]
        sample_daily = pd.read_csv(os.path.join(self.daily_performance_dir, sample_files[0]))
        
        regime_files = [f for f in os.listdir(self.regime_performance_dir) if f.endswith('.csv')]
        sample_regime = pd.read_csv(os.path.join(self.regime_performance_dir, regime_files[0]))
        
        columns = []
        # Use absolute threshold value for column matching since CSV stores all thresholds as positive
        abs_threshold = abs(threshold)
        threshold_str = f"_thr_{abs_threshold}"
        direction_str = f"_{direction}_"
        
        # Get daily columns for this threshold+direction
        for col in sample_daily.columns:
            if col not in ['ModelID', 'TradingDay'] and direction_str in col and threshold_str in col:
                columns.append(col)
        
        # Get regime columns for this threshold+direction  
        for col in sample_regime.columns:
            if col not in ['ModelID', 'Regime'] and direction_str in col and threshold_str in col:
                columns.append(col)
                
        return columns
    
    def get_all_threshold_direction_combinations(self) -> List[Tuple[float, str]]:
        """
        Get all threshold-direction combinations used in the analysis.
        
        Modified to enforce minimum thresholds:
        - Upside: minimum 0.1 (removes weak 0.0 signals)
        - Downside: maximum -0.1 (removes weak 0.0 signals)
        
        Note: Downside thresholds are stored as positive values in CSV files,
        but returned as negative values here for logical consistency.
        
        Returns:
            List of (threshold, direction) tuples
        """
        # Use positive threshold values that match CSV column names
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Removed 0.0
        
        combinations = []
        
        # Add upside combinations (threshold > 0, direction = "up")
        for threshold in thresholds:
            combinations.append((threshold, "up"))
        
        # Add downside combinations (threshold stored as negative for logical consistency)
        for threshold in thresholds:
            combinations.append((-threshold, "down"))  # Convert to negative for downside
        
        return combinations
    
    def _get_threshold_direction_columns_cached(self, threshold: float, direction: str,
                                               daily_data: pd.DataFrame, regime_data: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Get columns for threshold+direction combination using cached data (no disk I/O).
        
        Args:
            threshold: Threshold value (negative for downside, positive for upside)
            direction: Direction ('up' or 'down') 
            daily_data: Pre-loaded daily DataFrame
            regime_data: Pre-loaded regime DataFrame
            
        Returns:
            List of (source, column_name) tuples for this combination
        """
        columns = []
        # Use absolute threshold value for column matching since CSV stores all thresholds as positive
        abs_threshold = abs(threshold)
        threshold_str = f"_thr_{abs_threshold}"
        direction_str = f"_{direction}_"
        
        # Get daily columns for this threshold+direction
        for col in daily_data.columns:
            if col not in ['ModelID', 'TradingDay'] and threshold_str in col and direction_str in col:
                if '_acc_' in col:
                    columns.append(('daily', col))  # Original accuracy
                    columns.append(('daily', col.replace('_acc_', '_ws_acc_')))  # Wilson score accuracy
                elif '_pnl_' in col:
                    columns.append(('daily', col))  # Original PnL
                    columns.append(('daily', col.replace('_pnl_', '_ppt_')))  # PnL per trade
        
        # Get regime columns for this threshold+direction
        for col in regime_data.columns:
            if col not in ['ModelID', 'Regime'] and threshold_str in col and direction_str in col:
                if '_acc_' in col:
                    columns.append(('regime', col))  # Original accuracy
                    columns.append(('regime', col.replace('_acc_', '_ws_acc_')))  # Wilson score accuracy
                elif '_pnl_' in col:
                    columns.append(('regime', col))  # Original PnL
                    columns.append(('regime', col.replace('_pnl_', '_ppt_')))  # PnL per trade
                
        return columns
    
    def _calculate_combination_score_optimized(self, model_id: int, daily_data: pd.DataFrame, 
                                             regime_data: pd.DataFrame, threshold: float, 
                                             direction: str, weighting_array: np.ndarray,
                                             column_cache: Dict) -> float:
        """
        Calculate score using pre-cached column mappings (no disk I/O).
        
        Args:
            model_id: Model ID to evaluate
            daily_data: Pre-loaded daily DataFrame
            regime_data: Pre-loaded regime DataFrame
            threshold: Threshold value
            direction: Direction ('up' or 'down')
            weighting_array: 76-element weighting array
            column_cache: Pre-computed column mappings
            
        Returns:
            Score for this combination
        """
        # Get model rows
        daily_row = daily_data[daily_data['ModelID'] == model_id]
        regime_row = regime_data[regime_data['ModelID'] == model_id]
        
        if daily_row.empty or regime_row.empty:
            return float('-inf')
            
        daily_row = daily_row.iloc[0]
        regime_row = regime_row.iloc[0]
        
        # Get pre-cached columns for this threshold+direction combination
        columns = column_cache[(threshold, direction)]
        
        if len(columns) != len(weighting_array):
            raise ValueError(
                f"Column count ({len(columns)}) doesn't match weighting array length ({len(weighting_array)}) "
                f"for threshold {threshold}, direction {direction}"
            )
        
        # Extract metric values efficiently
        metric_values = []
        for source, col in columns:
            if source == 'daily':
                # Daily metric
                if '_ws_acc_' in col:
                    # Calculate Wilson score accuracy on-the-fly
                    base_col = col.replace('_ws_acc_', '_acc_')
                    num_col = col.replace('_ws_acc_', '_num_')
                    den_col = col.replace('_ws_acc_', '_den_')
                    
                    k = daily_row.get(num_col, 0.0)
                    n = daily_row.get(den_col, 0.0)
                    if pd.isna(k): k = 0.0
                    if pd.isna(n): n = 0.0
                    
                    wilson_acc = self.wilson_score_accuracy(k, n)
                    metric_values.append(wilson_acc)
                    
                elif '_ppt_' in col:
                    # Calculate PnL per trade on-the-fly
                    pnl_col = col.replace('_ppt_', '_pnl_')
                    den_col = col.replace('_ppt_', '_den_')
                    
                    pnl = daily_row.get(pnl_col, 0.0)
                    den = daily_row.get(den_col, 0.0)
                    if pd.isna(pnl): pnl = 0.0
                    if pd.isna(den): den = 0.0
                    
                    ppt = pnl / den if den > 0 else 0.0
                    metric_values.append(ppt)
                    
                else:
                    # Regular daily metric
                    value = daily_row.get(col, 0.0)
                    if pd.isna(value): value = 0.0
                    metric_values.append(value)
            else:
                # Regime metric
                if '_ws_acc_' in col:
                    # Calculate Wilson score accuracy on-the-fly
                    base_col = col.replace('_ws_acc_', '_acc_')
                    num_col = col.replace('_ws_acc_', '_num_')
                    den_col = col.replace('_ws_acc_', '_den_')
                    
                    k = regime_row.get(num_col, 0.0)
                    n = regime_row.get(den_col, 0.0)
                    if pd.isna(k): k = 0.0
                    if pd.isna(n): n = 0.0
                    
                    wilson_acc = self.wilson_score_accuracy(k, n)
                    metric_values.append(wilson_acc)
                    
                elif '_ppt_' in col:
                    # Calculate PnL per trade on-the-fly
                    pnl_col = col.replace('_ppt_', '_pnl_')
                    den_col = col.replace('_ppt_', '_den_')
                    
                    pnl = regime_row.get(pnl_col, 0.0)
                    den = regime_row.get(den_col, 0.0)
                    if pd.isna(pnl): pnl = 0.0
                    if pd.isna(den): den = 0.0
                    
                    ppt = pnl / den if den > 0 else 0.0
                    metric_values.append(ppt)
                    
                else:
                    # Regular regime metric
                    value = regime_row.get(col, 0.0)
                    if pd.isna(value): value = 0.0
                    metric_values.append(value)
        
        # Calculate weighted score
        metric_values = np.array(metric_values)
        score = np.dot(metric_values, weighting_array)
        
        return score
    
    def get_best_trading_model_batch_vectorized(self, trading_day: str, market_regime: int, 
                                              weighting_arrays: List[np.ndarray], show_metrics: bool = False) -> List[Dict]:
        """
        Highly optimized vectorized version that processes all combinations at once.
        
        Args:
            trading_day: The trading day (format: YYYYMMDD)
            market_regime: Market regime identifier (0-3)
            weighting_arrays: List of weight arrays (each should be 76 elements)
            show_metrics: If True, include detailed metrics breakdown in results
            
        Returns:
            List of dicts: Best model info for each weight array
        """
        print(f"Vectorized batch evaluation: Finding best models for day {trading_day}, regime {market_regime}, {len(weighting_arrays)} weight arrays")
        
        # Load data once
        daily_data = self._load_daily_performance(trading_day)
        regime_data = self._load_regime_performance(trading_day, market_regime)
        
        if daily_data is None or regime_data is None:
            raise ValueError(f"Could not load performance data for trading day {trading_day} and regime {market_regime}")
        
        available_models = daily_data['ModelID'].unique().tolist()
        threshold_dir_combos = self.get_all_threshold_direction_combinations()
        print(f"Found {len(available_models)} models, {len(threshold_dir_combos)} threshold-direction combinations")
        
        # Pre-compute all metric data for all models and combinations
        num_models = len(available_models)
        num_combos = len(threshold_dir_combos)
        num_weights = len(weighting_arrays)
        
        # Pre-allocate result arrays
        all_scores = np.full((num_weights, num_models, num_combos), float('-inf'))
        
        # Get all metric values for all combinations at once
        for model_idx, model_id in enumerate(available_models):
            daily_row = daily_data[daily_data['ModelID'] == model_id]
            regime_row = regime_data[regime_data['ModelID'] == model_id]
            
            if daily_row.empty or regime_row.empty:
                continue  # Skip invalid models
                
            daily_row = daily_row.iloc[0]
            regime_row = regime_row.iloc[0]
            
            for combo_idx, (threshold, direction) in enumerate(threshold_dir_combos):
                # Get columns for this combination
                columns = self._get_threshold_direction_columns_cached(
                    threshold, direction, daily_data, regime_data
                )
                
                if len(columns) != 76:
                    continue  # Skip invalid combinations
                
                # Extract metric values for this model-combination
                metric_values = []
                for source, col in columns:
                    if source == 'daily':
                        data_row = daily_row
                    else:
                        data_row = regime_row
                        
                    if '_ws_acc_' in col:
                        # Wilson score accuracy
                        acc_col = col.replace('_ws_acc_', '_acc_')
                        num_col = col.replace('_ws_acc_', '_num_')
                        den_col = col.replace('_ws_acc_', '_den_')
                        
                        k = data_row.get(num_col, 0.0)
                        n = data_row.get(den_col, 0.0)
                        if pd.isna(k): k = 0.0
                        if pd.isna(n): n = 0.0
                        
                        value = self.wilson_score_accuracy(k, n)
                        
                    elif '_ppt_' in col:
                        # PnL per trade
                        pnl_col = col.replace('_ppt_', '_pnl_')
                        den_col = pnl_col.replace('_pnl_', '_den_')
                        
                        pnl = data_row.get(pnl_col, 0.0)
                        den = data_row.get(den_col, 0.0)
                        if pd.isna(pnl): pnl = 0.0
                        if pd.isna(den): den = 0.0
                        
                        value = pnl / den if den > 0 else 0.0
                        
                    else:
                        # Regular metric
                        value = data_row.get(col, 0.0)
                        if pd.isna(value): value = 0.0
                        
                    metric_values.append(value)
                
                # Convert to numpy array for vectorized computation
                metric_array = np.array(metric_values)
                
                # Calculate scores for all weight arrays at once using vectorized operations
                for weight_idx, weighting_array in enumerate(weighting_arrays):
                    score = np.dot(metric_array, weighting_array)
                    all_scores[weight_idx, model_idx, combo_idx] = score
        
        # Find best combinations for each weight array
        results = []
        
        for weight_idx, weighting_array in enumerate(weighting_arrays):
            # Find the best score across all models and combinations
            weight_scores = all_scores[weight_idx]
            max_indices = np.unravel_index(np.argmax(weight_scores), weight_scores.shape)
            best_model_idx, best_combo_idx = max_indices
            
            best_score = weight_scores[best_model_idx, best_combo_idx]
            best_model = available_models[best_model_idx]
            best_threshold, best_direction = threshold_dir_combos[best_combo_idx]
            
            # Count valid combinations
            valid_mask = weight_scores != float('-inf')
            valid_combinations = np.sum(valid_mask)
            total_combinations = weight_scores.size
            
            result = {
                'model_id': best_model,
                'score': best_score,
                'direction': best_direction,
                'threshold': best_threshold,
                'trading_day': trading_day,
                'market_regime': market_regime,
                'weight_array_index': weight_idx
            }
            
            # If metrics breakdown is requested, calculate and include it
            if show_metrics:
                try:
                    # Get the best model's detailed metrics
                    daily_row = daily_data[daily_data['ModelID'] == best_model].iloc[0]
                    regime_row = regime_data[regime_data['ModelID'] == best_model].iloc[0]
                    
                    # Get columns for the best combination
                    best_columns = self._get_threshold_direction_columns_cached(
                        best_threshold, best_direction, daily_data, regime_data
                    )
                    
                    # Extract detailed metrics
                    metric_details = []
                    metric_values = []
                    
                    for i, (source, col) in enumerate(best_columns):
                        # Determine data source and get the appropriate row
                        if source == 'daily':
                            data_row = daily_row
                            data_source = "daily"
                        else:
                            data_row = regime_row
                            data_source = "regime"
                            
                        if '_ws_acc_' in col:
                            # Calculate Wilson scored accuracy
                            acc_col = col.replace('_ws_acc_', '_acc_')
                            num_col = acc_col.replace('_acc_', '_num_')
                            den_col = acc_col.replace('_acc_', '_den_')
                            
                            k = data_row.get(num_col, 0.0)
                            n = data_row.get(den_col, 0.0)
                            if pd.isna(k): k = 0.0
                            if pd.isna(n): n = 0.0
                            
                            value = self.wilson_score_accuracy(k, n)
                            metric_values.append(value)
                            
                        elif '_ppt_' in col:
                            # Calculate PnL per trade
                            pnl_col = col.replace('_ppt_', '_pnl_')
                            den_col = pnl_col.replace('_pnl_', '_den_')
                            
                            pnl = data_row.get(pnl_col, 0.0)
                            den = data_row.get(den_col, 0.0)
                            if pd.isna(pnl): pnl = 0.0
                            if pd.isna(den): den = 0.0
                            
                            value = pnl / den if den > 0 else 0.0
                            metric_values.append(value)
                            
                        else:
                            # Regular metric
                            value = data_row.get(col, 0.0)
                            if pd.isna(value): value = 0.0
                            metric_values.append(value)
                        
                        # Store metric details
                        weight = weighting_array[i]
                        weighted_value = value * weight
                        
                        metric_details.append({
                            'index': i + 1,
                            'column_name': col,
                            'data_source': data_source,
                            'weight': weight,
                            'value': value,
                            'weighted_value': weighted_value
                        })
                    
                    # Add metrics breakdown to result
                    result['metrics_breakdown'] = {
                        'total_metrics': len(metric_details),
                        'metrics': metric_details,
                        'total_score_verification': sum(m['weighted_value'] for m in metric_details)
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not generate metrics breakdown for weight array {weight_idx}: {e}")
            
            results.append(result)
        
        print(f"Vectorized batch evaluation complete: Processed {len(weighting_arrays)} weight arrays")
        return results


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
    
    # Test with sample data using the only remaining method
    try:
        result = weighter.get_best_trading_model_batch_vectorized("20250707", 0, [equal_weights])
        print(f"\nBest model result: {result[0]}")
    except Exception as e:
        print(f"Error: {e}")
