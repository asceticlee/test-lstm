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
            threshold: Threshold value (e.g., 0.0, 0.3, etc.)
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
        threshold_str = f"_thr_{threshold}"
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
        
        Returns:
            List of (threshold, direction) tuples
        """
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        directions = ['up', 'down']
        
        combinations = []
        for threshold in thresholds:
            for direction in directions:
                combinations.append((threshold, direction))
        
        return combinations
    
    def _get_threshold_direction_columns_cached(self, threshold: float, direction: str,
                                               daily_data: pd.DataFrame, regime_data: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Get columns for threshold+direction combination using cached data (no disk I/O).
        
        Args:
            threshold: Threshold value
            direction: Direction ('up' or 'down') 
            daily_data: Pre-loaded daily DataFrame
            regime_data: Pre-loaded regime DataFrame
            
        Returns:
            List of (source, column_name) tuples for this combination
        """
        columns = []
        threshold_str = f"_thr_{threshold}"
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
    
    def calculate_combination_score(self, model_id: str, daily_data: pd.DataFrame, 
                                  regime_data: pd.DataFrame, threshold: float, 
                                  direction: str, weighting_array: np.ndarray) -> float:
        """
        Calculate score for a specific model+threshold+direction combination.
        
        Args:
            model_id: Model ID to evaluate
            daily_data: Daily performance DataFrame
            regime_data: Regime performance DataFrame
            threshold: Threshold value
            direction: Direction ('up' or 'down')
            weighting_array: 76-element weighting array
            
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
        
        # Get columns for this threshold+direction combination
        columns = self.get_threshold_direction_columns(threshold, direction)
        
        if len(columns) != len(weighting_array):
            raise ValueError(
                f"Column count ({len(columns)}) doesn't match weighting array length ({len(weighting_array)}) "
                f"for threshold {threshold}, direction {direction}"
            )
        
        # Extract metric values
        metric_values = []
        for col in columns:
            if col.startswith(('daily_', '2day_', '3day_', '1week_', '2week_', '4week_', 
                              '8week_', '13week_', '26week_', '52week_', 'from_begin_')):
                # Daily metric
                value = daily_row.get(col, 0.0)
            else:
                # Regime metric  
                value = regime_row.get(col, 0.0)
                
            if pd.isna(value):
                value = 0.0
            metric_values.append(float(value))
        
        # Calculate weighted score
        metric_values = np.array(metric_values)
        score = np.dot(metric_values, weighting_array)
        
        return score
    
    def get_best_trading_model(self, trading_day: str, market_regime: int, 
                              weighting_array: np.ndarray, show_metrics: bool = False) -> Dict:
        """
        Find the best trading model (optimized single-threaded version).
        
        Args:
            trading_day: The trading day (format: YYYYMMDD)
            market_regime: Market regime identifier (0-3)
            weighting_array: Array of 76 weights for different metrics
            show_metrics: If True, include detailed metrics breakdown in results
            
        Returns:
            dict: Best model info with model_id, score, direction, threshold
                  If show_metrics=True, also includes metrics breakdown
        """
        print(f"Standard evaluation: Finding best model for day {trading_day}, regime {market_regime}")
        
        # Load data once at the beginning
        daily_data = self._load_daily_performance(trading_day)
        regime_data = self._load_regime_performance(trading_day, market_regime)
        
        if daily_data is None or regime_data is None:
            raise ValueError(f"Could not load performance data for trading day {trading_day} and regime {market_regime}")
        
        # Get model list from daily data (optimization #1: use data instead of file system)
        available_models = daily_data['ModelID'].unique().tolist()
        print(f"Found {len(available_models)} models in daily data")
        
        # Pre-cache all threshold+direction column mappings (optimization #2: avoid repeated disk I/O)
        threshold_dir_combos = self.get_all_threshold_direction_combinations()
        column_cache = {}
        for threshold, direction in threshold_dir_combos:
            column_cache[(threshold, direction)] = self._get_threshold_direction_columns_cached(
                threshold, direction, daily_data, regime_data
            )
        
        best_score = float('-inf')
        best_model = None
        best_direction = None 
        best_threshold = None
        
        total_combinations = 0
        valid_combinations = 0
        
        # Iterate through models from data (not file system)
        for model_id in available_models:
            # Iterate through all threshold-direction combinations
            for threshold, direction in threshold_dir_combos:
                total_combinations += 1
                
                try:
                    score = self._calculate_combination_score_optimized(
                        model_id, daily_data, regime_data, 
                        threshold, direction, weighting_array, column_cache
                    )
                    
                    if score is not None and score != float('-inf'):
                        valid_combinations += 1
                        if score > best_score:
                            best_score = score
                            best_model = model_id
                            best_direction = direction
                            best_threshold = threshold
                            
                except Exception as e:
                    continue
        
        if best_model is None:
            raise ValueError(f"No valid combinations found. Checked {total_combinations} combinations, {valid_combinations} were valid.")
        
        print(f"Standard evaluation complete: {valid_combinations}/{total_combinations} valid combinations")
        
        # Prepare result dictionary
        result = {
            'model_id': best_model,
            'score': best_score,
            'direction': best_direction,
            'threshold': best_threshold,
            'trading_day': trading_day,
            'market_regime': market_regime
        }
        
        # If metrics breakdown is requested, calculate and include it
        if show_metrics:
            try:
                # Get the best model's detailed metrics
                daily_row = daily_data[daily_data['ModelID'] == best_model].iloc[0]
                regime_row = regime_data[regime_data['ModelID'] == best_model].iloc[0]
                
                # Get columns for the best combination
                best_columns = column_cache[(best_threshold, best_direction)]
                
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
                print(f"Warning: Could not generate metrics breakdown: {e}")
                result['metrics_breakdown'] = None
        
        return result
    
    def get_best_trading_model_batch(self, trading_day: str, market_regime: int, 
                                   weighting_arrays: List[np.ndarray], show_metrics: bool = False) -> List[Dict]:
        """
        Find the best trading model for multiple weight arrays in a single batch operation.
        This method loads the data only once and processes all weight arrays efficiently.
        
        Args:
            trading_day: The trading day (format: YYYYMMDD)
            market_regime: Market regime identifier (0-3)
            weighting_arrays: List of weight arrays (each should be 76 elements)
            show_metrics: If True, include detailed metrics breakdown in results
            
        Returns:
            List of dicts: Best model info for each weight array with model_id, score, direction, threshold
                          If show_metrics=True, also includes metrics breakdown
        """
        import time
        start_time = time.time()
        
        print(f"Batch evaluation: Finding best models for day {trading_day}, regime {market_regime}, {len(weighting_arrays)} weight arrays")
        
        # Timing: Data loading
        data_load_start = time.time()
        daily_data = self._load_daily_performance(trading_day)
        regime_data = self._load_regime_performance(trading_day, market_regime)
        data_load_time = time.time() - data_load_start
        print(f"  Data loading took: {data_load_time:.3f} seconds")
        
        if daily_data is None or regime_data is None:
            raise ValueError(f"Could not load performance data for trading day {trading_day} and regime {market_regime}")
        
        # Get model list from daily data
        available_models = daily_data['ModelID'].unique().tolist()
        print(f"Found {len(available_models)} models in daily data")
        
        # Timing: Column cache preparation
        cache_prep_start = time.time()
        threshold_dir_combos = self.get_all_threshold_direction_combinations()
        column_cache = {}
        for threshold, direction in threshold_dir_combos:
            column_cache[(threshold, direction)] = self._get_threshold_direction_columns_cached(
                threshold, direction, daily_data, regime_data
            )
        cache_prep_time = time.time() - cache_prep_start
        print(f"  Column cache preparation took: {cache_prep_time:.3f} seconds")
        print(f"  Processing {len(threshold_dir_combos)} threshold-direction combinations")
        
        # Process each weight array
        results = []
        total_weight_processing_time = 0
        
        for weight_idx, weighting_array in enumerate(weighting_arrays):
            weight_start_time = time.time()
            print(f"  Processing weight array {weight_idx + 1}/{len(weighting_arrays)}")
            
            # Validate weight array length
            if len(weighting_array) != 76:
                raise ValueError(f"Weight array {weight_idx + 1} has length {len(weighting_array)}, expected 76")
            
            best_score = float('-inf')
            best_model = None
            best_direction = None 
            best_threshold = None
            
            total_combinations = 0
            valid_combinations = 0
            
            # Timing: Model evaluation loop
            model_eval_start = time.time()
            
            # Track detailed timing for every 1000 combinations
            combinations_processed = 0
            detailed_timing_interval = 1000
            last_detailed_time = model_eval_start
            
            # Iterate through models and combinations (using pre-loaded data)
            for model_id in available_models:
                for threshold, direction in threshold_dir_combos:
                    total_combinations += 1
                    combinations_processed += 1
                    
                    try:
                        score = self._calculate_combination_score_optimized(
                            model_id, daily_data, regime_data, 
                            threshold, direction, weighting_array, column_cache
                        )
                        
                        if score is not None and score != float('-inf'):
                            valid_combinations += 1
                            if score > best_score:
                                best_score = score
                                best_model = model_id
                                best_direction = direction
                                best_threshold = threshold
                                
                    except Exception as e:
                        continue
                    
                    # Print detailed timing every 1000 combinations
                    if combinations_processed % detailed_timing_interval == 0:
                        current_time = time.time()
                        interval_time = current_time - last_detailed_time
                        avg_per_combination = interval_time / detailed_timing_interval
                        print(f"      Processed {combinations_processed}/{len(available_models) * len(threshold_dir_combos)} combinations "
                              f"(last {detailed_timing_interval} took {interval_time:.3f}s, {avg_per_combination*1000:.2f}ms per combination)")
                        last_detailed_time = current_time
            
            model_eval_time = time.time() - model_eval_start
            avg_combination_time = model_eval_time / total_combinations if total_combinations > 0 else 0
            print(f"    Model evaluation took: {model_eval_time:.3f} seconds ({total_combinations} combinations, {avg_combination_time*1000:.2f}ms per combination)")
            
            if best_model is None:
                raise ValueError(f"No valid combinations found for weight array {weight_idx + 1}. Checked {total_combinations} combinations, {valid_combinations} were valid.")
            
            # Timing: Result preparation
            result_prep_start = time.time()
            
            # Prepare result dictionary for this weight array
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
                    best_columns = column_cache[(best_threshold, best_direction)]
                    
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
                        'metrics': metric_details,
                        'total_score': best_score,
                        'total_combinations_checked': total_combinations,
                        'valid_combinations': valid_combinations
                    }
                    
                except Exception as e:
                    print(f"    Warning: Could not generate metrics breakdown for weight array {weight_idx + 1}: {e}")
            
            result_prep_time = time.time() - result_prep_start
            print(f"    Result preparation took: {result_prep_time:.3f} seconds")
            
            results.append(result)
            
            weight_total_time = time.time() - weight_start_time
            total_weight_processing_time += weight_total_time
            print(f"    Weight {weight_idx + 1} total time: {weight_total_time:.3f} seconds")
            print(f"    Best model: {best_model}, direction: {best_direction}, threshold: {best_threshold}, score: {best_score:.4f}")
        
        total_time = time.time() - start_time
        avg_weight_time = total_weight_processing_time / len(weighting_arrays)
        
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
        import time
        start_time = time.time()
        
        print(f"Vectorized batch evaluation: Finding best models for day {trading_day}, regime {market_regime}, {len(weighting_arrays)} weight arrays")
        
        # Load data once
        data_load_start = time.time()
        daily_data = self._load_daily_performance(trading_day)
        regime_data = self._load_regime_performance(trading_day, market_regime)
        data_load_time = time.time() - data_load_start
        print(f"  Data loading took: {data_load_time:.3f} seconds")
        
        if daily_data is None or regime_data is None:
            raise ValueError(f"Could not load performance data for trading day {trading_day} and regime {market_regime}")
        
        available_models = daily_data['ModelID'].unique().tolist()
        threshold_dir_combos = self.get_all_threshold_direction_combinations()
        print(f"Found {len(available_models)} models, {len(threshold_dir_combos)} threshold-direction combinations")
        
        # Pre-compute all metric data for all models and combinations
        vectorize_start = time.time()
        
        # Create arrays to hold all metric values for vectorized computation
        num_models = len(available_models)
        num_combos = len(threshold_dir_combos)
        num_weights = len(weighting_arrays)
        
        print(f"  Pre-computing metrics for {num_models * num_combos} combinations...")
        
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
        
        vectorize_time = time.time() - vectorize_start
        print(f"  Vectorized computation took: {vectorize_time:.3f} seconds")
        
        # Find best combinations for each weight array
        results_start = time.time()
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
            
            results.append(result)
            print(f"  Weight {weight_idx + 1}: Best model {best_model}, direction: {best_direction}, threshold: {best_threshold}, score: {best_score:.4f}")
        
        results_time = time.time() - results_start
        total_time = time.time() - start_time
        
        print(f"Vectorized batch evaluation complete: Processed {len(weighting_arrays)} weight arrays")
        print(f"VECTORIZED TIMING SUMMARY:")
        print(f"  Data loading: {data_load_time:.3f}s")
        print(f"  Vectorized computation: {vectorize_time:.3f}s") 
        print(f"  Results extraction: {results_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Speedup vs original: {9.9/total_time:.1f}x faster")
        
        return results
    
    def get_best_trading_model_fast(self, trading_day: str, market_regime: int, 
                                   weighting_array: np.ndarray, use_gpu: bool = False,
                                   n_workers: int = None, show_metrics: bool = False) -> Dict:
        """
        Fast version using vectorized operations and parallel processing.
        
        Args:
            trading_day: Trading day in format 'YYYYMMDD'
            market_regime: Market regime (0, 1, 2, 3, or 4)
            weighting_array: Array of 76 weights for performance metrics
            use_gpu: Whether to use GPU acceleration (requires cupy)
            n_workers: Number of parallel workers (default: CPU count)
            show_metrics: If True, include detailed metrics breakdown in results
            
        Returns:
            Dict with best model information
        """
        # Load performance data
        daily_data = self._load_daily_performance(trading_day)
        regime_data = self._load_regime_performance(trading_day, market_regime)
        
        if daily_data is None or regime_data is None:
            raise ValueError(f"Could not load performance data for trading day {trading_day} and regime {market_regime}")
        
        # Validate weighting array
        if len(weighting_array) != 76:
            raise ValueError(f"Weighting array must have exactly 76 elements, got {len(weighting_array)}")
        
        # Find common models
        daily_models = set(daily_data['ModelID'].unique())
        regime_models = set(regime_data['ModelID'].unique())
        common_models = list(daily_models.intersection(regime_models))
        
        if not common_models:
            raise ValueError("No common models found between daily and regime data")
        
        # Define combinations to evaluate
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        directions = ['up', 'down']
        
        total_combinations = len(common_models) * len(thresholds) * len(directions)
        print(f"Fast evaluation: {len(common_models)} models × {len(thresholds)} thresholds × {len(directions)} directions = {total_combinations} combinations")
        
        if use_gpu:
            return self._evaluate_with_gpu(daily_data, regime_data, common_models, 
                                         thresholds, directions, weighting_array, show_metrics)
        else:
            return self._evaluate_with_parallel_cpu(daily_data, regime_data, common_models,
                                                  thresholds, directions, weighting_array, n_workers, show_metrics)
    
    def _evaluate_with_parallel_cpu(self, daily_data: pd.DataFrame, regime_data: pd.DataFrame,
                                  common_models: List[str], thresholds: List[float], 
                                  directions: List[str], weighting_array: np.ndarray,
                                  n_workers: int = None, show_metrics: bool = False) -> Dict:
        """
        Evaluate combinations using parallel CPU processing.
        """
        if n_workers is None:
            n_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overwhelming system
        
        print(f"Using {n_workers} parallel workers")
        
        # Pre-extract all needed data into numpy arrays for faster access
        model_data_cache = self._prepare_model_data_cache(daily_data, regime_data, common_models)
        
        # Create all combinations
        combinations = [
            (model_id, threshold, direction)
            for model_id in common_models
            for threshold in thresholds
            for direction in directions
        ]
        
        # Create partial function with cached data
        eval_func = partial(
            self._evaluate_single_combination_vectorized,
            model_data_cache=model_data_cache,
            weighting_array=weighting_array
        )
        
        # Evaluate in parallel
        best_score = float('-inf')
        best_result = None
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all combinations
            futures = [executor.submit(eval_func, combo) for combo in combinations]
            
            # Collect results as they complete
            for i, future in enumerate(futures):
                try:
                    score, model_id, threshold, direction = future.result()
                    if score > best_score:
                        best_score = score
                        best_result = (model_id, threshold, direction)
                        
                    # Progress reporting
                    if (i + 1) % 100 == 0:
                        print(f"Evaluated {i + 1}/{len(combinations)} combinations...")
                        
                except Exception as e:
                    print(f"Error in combination {combinations[i]}: {e}")
                    continue
        
        if best_result is None:
            raise ValueError("No valid combinations found")
        
        model_id, threshold, direction = best_result
        
        # Prepare result dictionary
        result = {
            'model_id': model_id,
            'score': best_score,
            'direction': direction,
            'threshold': threshold,
            'details': f"Fast evaluation: best from {len(combinations)} combinations (parallel CPU)"
        }
        
        # If metrics breakdown is requested, calculate and include it
        if show_metrics:
            try:
                # Get the best model's detailed metrics using the same logic as standard method
                daily_row = daily_data[daily_data['ModelID'] == model_id].iloc[0]
                regime_row = regime_data[regime_data['ModelID'] == model_id].iloc[0]
                
                # Get columns for the best combination
                threshold_dir_combos = self.get_all_threshold_direction_combinations()
                column_cache = {}
                for thr, dir in threshold_dir_combos:
                    column_cache[(thr, dir)] = self._get_threshold_direction_columns_cached(
                        thr, dir, daily_data, regime_data
                    )
                
                best_columns = column_cache[(threshold, direction)]
                
                # Extract detailed metrics
                metric_details = []
                
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
                        
                    elif '_ppt_' in col:
                        # Calculate PnL per trade
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
                print(f"Warning: Could not generate metrics breakdown: {e}")
                result['metrics_breakdown'] = None
        
        return result
    
    def _prepare_model_data_cache(self, daily_data: pd.DataFrame, regime_data: pd.DataFrame,
                                common_models: List[str]) -> Dict:
        """
        Pre-extract and cache all model data as numpy arrays for faster access.
        """
        cache = {}
        
        for model_id in common_models:
            daily_row = daily_data[daily_data['ModelID'] == model_id]
            regime_row = regime_data[regime_data['ModelID'] == model_id]
            
            if daily_row.empty or regime_row.empty:
                continue
                
            daily_row = daily_row.iloc[0]
            regime_row = regime_row.iloc[0]
            
            # Extract all relevant columns as numpy arrays
            cache[model_id] = {
                'daily': daily_row.to_dict(),
                'regime': regime_row.to_dict()
            }
        
        return cache
    
    def _evaluate_single_combination_vectorized(self, combination: Tuple[str, float, str],
                                              model_data_cache: Dict,
                                              weighting_array: np.ndarray) -> Tuple[float, str, float, str]:
        """
        Vectorized evaluation of a single model+threshold+direction combination.
        """
        model_id, threshold, direction = combination
        
        if model_id not in model_data_cache:
            return float('-inf'), model_id, threshold, direction
        
        try:
            # Get cached data
            daily_data_dict = model_data_cache[model_id]['daily']
            regime_data_dict = model_data_cache[model_id]['regime']
            
            # Get column names for this threshold+direction - need to create dummy DataFrames for the cached method
            # This is a bit inefficient but ensures consistency
            daily_dummy = pd.DataFrame([daily_data_dict])
            regime_dummy = pd.DataFrame([regime_data_dict])
            
            # Get columns with source information
            columns_with_source = self._get_threshold_direction_columns_cached(threshold, direction, daily_dummy, regime_dummy)
            
            # Extract values vectorized
            values = np.zeros(76, dtype=np.float64)
            
            for i, (source, col_name) in enumerate(columns_with_source):
                if i >= 76:  # Safety check
                    break
                    
                if source == 'daily':
                    data_source = daily_data_dict
                else:  # regime
                    data_source = regime_data_dict
                
                # Handle different metric types
                if '_ws_acc_' in col_name:
                    # Wilson scored accuracy
                    acc_col = col_name.replace('_ws_acc_', '_acc_')
                    num_col = acc_col.replace('_acc_', '_num_')
                    den_col = acc_col.replace('_acc_', '_den_')
                    
                    k = data_source.get(num_col, 0.0)
                    n = data_source.get(den_col, 0.0)
                    
                    if pd.isna(k): k = 0.0
                    if pd.isna(n): n = 0.0
                    
                    values[i] = self.wilson_score_accuracy(k, n)
                    
                elif '_ppt_' in col_name:
                    # PnL per trade
                    pnl_col = col_name.replace('_ppt_', '_pnl_')
                    den_col = pnl_col.replace('_pnl_', '_den_')
                    
                    pnl = data_source.get(pnl_col, 0.0)
                    den = data_source.get(den_col, 0.0)
                    
                    if pd.isna(pnl): pnl = 0.0
                    if pd.isna(den): den = 0.0
                    
                    values[i] = pnl / den if den > 0 else 0.0
                    
                else:
                    # Direct value (acc or pnl)
                    value = data_source.get(col_name, 0.0)
                    if pd.isna(value): value = 0.0
                    values[i] = float(value)
            
            # Vectorized score calculation
            score = np.dot(values, weighting_array)
            return score, model_id, threshold, direction
            
        except Exception as e:
            return float('-inf'), model_id, threshold, direction
    
    def _evaluate_with_gpu(self, daily_data: pd.DataFrame, regime_data: pd.DataFrame,
                          common_models: List[str], thresholds: List[float], 
                          directions: List[str], weighting_array: np.ndarray, show_metrics: bool = False) -> Dict:
        """
        Evaluate combinations using GPU acceleration (requires cupy).
        """
        try:
            import cupy as cp
            print("Using GPU acceleration with CuPy")
        except ImportError:
            print("CuPy not available, falling back to parallel CPU")
            return self._evaluate_with_parallel_cpu(daily_data, regime_data, common_models,
                                                  thresholds, directions, weighting_array)
        
        # Convert weighting array to GPU
        gpu_weights = cp.asarray(weighting_array)
        
        # Pre-extract all data
        model_data_cache = self._prepare_model_data_cache(daily_data, regime_data, common_models)
        
        # Batch process combinations
        batch_size = 1000  # Process in batches to manage GPU memory
        best_score = float('-inf')
        best_result = None
        
        combinations = [
            (model_id, threshold, direction)
            for model_id in common_models
            for threshold in thresholds
            for direction in directions
        ]
        
        total_combinations = len(combinations)
        print(f"Processing {total_combinations} combinations in batches of {batch_size}")
        
        for batch_start in range(0, total_combinations, batch_size):
            batch_end = min(batch_start + batch_size, total_combinations)
            batch_combinations = combinations[batch_start:batch_end]
            
            # Prepare batch data
            batch_values = np.zeros((len(batch_combinations), 76), dtype=np.float64)
            valid_mask = np.ones(len(batch_combinations), dtype=bool)
            
            for i, (model_id, threshold, direction) in enumerate(batch_combinations):
                try:
                    if model_id not in model_data_cache:
                        valid_mask[i] = False
                        continue
                    
                    # Extract values for this combination
                    daily_data_dict = model_data_cache[model_id]['daily']
                    regime_data_dict = model_data_cache[model_id]['regime']
                    columns = self.get_threshold_direction_columns(threshold, direction)
                    
                    for j, (file_type, col_name) in enumerate(columns):
                        if j >= 76:
                            break
                            
                        data_source = daily_data_dict if file_type == 'daily' else regime_data_dict
                        
                        if '_ws_acc_' in col_name:
                            acc_col = col_name.replace('_ws_acc_', '_acc_')
                            num_col = acc_col.replace('_acc_', '_num_')
                            den_col = acc_col.replace('_acc_', '_den_')
                            
                            k = data_source.get(num_col, 0.0)
                            n = data_source.get(den_col, 0.0)
                            if pd.isna(k): k = 0.0
                            if pd.isna(n): n = 0.0
                            
                            batch_values[i, j] = self.wilson_score_accuracy(k, n)
                            
                        elif '_ppt_' in col_name:
                            pnl_col = col_name.replace('_ppt_', '_pnl_')
                            den_col = pnl_col.replace('_pnl_', '_den_')
                            
                            pnl = data_source.get(pnl_col, 0.0)
                            den = data_source.get(den_col, 0.0)
                            if pd.isna(pnl): pnl = 0.0
                            if pd.isna(den): den = 0.0
                            
                            batch_values[i, j] = pnl / den if den > 0 else 0.0
                            
                        else:
                            value = data_source.get(col_name, 0.0)
                            if pd.isna(value): value = 0.0
                            batch_values[i, j] = float(value)
                            
                except Exception as e:
                    valid_mask[i] = False
                    continue
            
            # GPU computation
            gpu_batch = cp.asarray(batch_values[valid_mask])
            if len(gpu_batch) > 0:
                gpu_scores = cp.dot(gpu_batch, gpu_weights)
                scores = cp.asnumpy(gpu_scores)
                
                # Find best in this batch
                valid_combinations = [combo for i, combo in enumerate(batch_combinations) if valid_mask[i]]
                for score, (model_id, threshold, direction) in zip(scores, valid_combinations):
                    if score > best_score:
                        best_score = score
                        best_result = (model_id, threshold, direction)
            
            print(f"Processed batch {batch_start//batch_size + 1}/{(total_combinations + batch_size - 1)//batch_size}")
        
        if best_result is None:
            raise ValueError("No valid combinations found")
        
        model_id, threshold, direction = best_result
        
        # Prepare result dictionary
        result = {
            'model_id': model_id,
            'score': best_score,
            'direction': direction,
            'threshold': threshold,
            'details': f"Fast evaluation: best from {total_combinations} combinations (GPU accelerated)"
        }
        
        # If metrics breakdown is requested, calculate and include it
        if show_metrics:
            try:
                # Get the best model's detailed metrics using the same logic as standard method
                daily_row = daily_data[daily_data['ModelID'] == model_id].iloc[0]
                regime_row = regime_data[regime_data['ModelID'] == model_id].iloc[0]
                
                # Get columns for the best combination
                threshold_dir_combos = self.get_all_threshold_direction_combinations()
                column_cache = {}
                for thr, dir in threshold_dir_combos:
                    column_cache[(thr, dir)] = self._get_threshold_direction_columns_cached(
                        thr, dir, daily_data, regime_data
                    )
                
                best_columns = column_cache[(threshold, direction)]
                
                # Extract detailed metrics
                metric_details = []
                
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
                        
                    elif '_ppt_' in col:
                        # Calculate PnL per trade
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
                print(f"Warning: Could not generate metrics breakdown: {e}")
                result['metrics_breakdown'] = None
        
        return result
    
    def get_best_trading_model_gpu(self, trading_day: str, market_regime: int, 
                                  weighting_array: np.ndarray, show_metrics: bool = False) -> Dict:
        """
        GPU-accelerated version using CuPy for maximum performance.
        
        This method uses GPU vectorization for massive parallel computation
        of all model×threshold×direction combinations simultaneously.
        """
        try:
            import cupy as cp
            print("🚀 Using GPU acceleration with CuPy")
        except ImportError:
            print("⚠️  CuPy not available, falling back to CPU vectorized version")
            return self.get_best_trading_model_fast(trading_day, market_regime, weighting_array)
        
        # Load data
        daily_data = self._load_daily_performance(trading_day)
        regime_data = self._load_regime_performance(trading_day, market_regime)
        
        if daily_data is None or regime_data is None:
            raise ValueError(f"Could not load performance data for trading day {trading_day} and regime {market_regime}")
        
        # Get common models
        common_models = list(set(daily_data['ModelID'].unique()).intersection(
                            set(regime_data['ModelID'].unique())))
        
        if not common_models:
            raise ValueError("No common models found")
        
        # Define all combinations
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        directions = ['up', 'down']
        
        print(f"🔥 GPU processing {len(common_models)} models × {len(thresholds)} thresholds × {len(directions)} directions = {len(common_models) * len(thresholds) * len(directions)} combinations")
        
        # Prepare data matrices on GPU
        num_models = len(common_models)
        num_combinations = len(thresholds) * len(directions)
        
        # Create massive result matrix: [models, threshold×direction combinations]
        scores_gpu = cp.full((num_models, num_combinations), -cp.inf, dtype=cp.float32)
        
        # Get column structure for one threshold-direction combination to determine size
        sample_cols = self.get_threshold_direction_columns(thresholds[0], directions[0])
        num_metrics = len(sample_cols)
        
        if len(weighting_array) != num_metrics:
            raise ValueError(f"Weighting array length ({len(weighting_array)}) doesn't match expected metrics ({num_metrics})")
        
        # Transfer weighting array to GPU
        weights_gpu = cp.array(weighting_array, dtype=cp.float32)
        
        # Process each threshold-direction combination
        combo_idx = 0
        for threshold in thresholds:
            for direction in directions:
                try:
                    # Get columns for this combination using the cached method
                    target_columns_with_source = self._get_threshold_direction_columns_cached(
                        threshold, direction, daily_data, regime_data
                    )
                    
                    # Extract metrics for all models at once
                    model_metrics = np.zeros((num_models, len(target_columns_with_source)), dtype=np.float32)
                    
                    for model_idx, model_id in enumerate(common_models):
                        daily_row = daily_data[daily_data['ModelID'] == model_id]
                        regime_row = regime_data[regime_data['ModelID'] == model_id]
                        
                        if not daily_row.empty and not regime_row.empty:
                            daily_row = daily_row.iloc[0]
                            regime_row = regime_row.iloc[0]
                            
                            # Extract values for all metrics
                            values = []
                            for source, col in target_columns_with_source:
                                if source == 'daily':
                                    # Daily metrics
                                    if '_ws_acc_' in col:
                                        acc_col = col.replace('_ws_acc_', '_acc_')
                                        num_col = acc_col.replace('_acc_', '_num_')
                                        den_col = acc_col.replace('_acc_', '_den_')
                                        k = daily_row.get(num_col, 0.0) or 0.0
                                        n = daily_row.get(den_col, 0.0) or 0.0
                                        values.append(self.wilson_score_accuracy(k, n))
                                    elif '_ppt_' in col:
                                        pnl_col = col.replace('_ppt_', '_pnl_')
                                        den_col = pnl_col.replace('_pnl_', '_den_')
                                        pnl = daily_row.get(pnl_col, 0.0) or 0.0
                                        den = daily_row.get(den_col, 0.0) or 0.0
                                        values.append(pnl / den if den > 0 else 0.0)
                                    else:
                                        values.append(daily_row.get(col, 0.0) or 0.0)
                                else:
                                    # Regime metrics  
                                    if '_ws_acc_' in col:
                                        acc_col = col.replace('_ws_acc_', '_acc_')
                                        num_col = acc_col.replace('_acc_', '_num_')
                                        den_col = acc_col.replace('_acc_', '_den_')
                                        k = regime_row.get(num_col, 0.0) or 0.0
                                        n = regime_row.get(den_col, 0.0) or 0.0
                                        values.append(self.wilson_score_accuracy(k, n))
                                    elif '_ppt_' in col:
                                        pnl_col = col.replace('_ppt_', '_pnl_')
                                        den_col = pnl_col.replace('_pnl_', '_den_')
                                        pnl = regime_row.get(pnl_col, 0.0) or 0.0
                                        den = regime_row.get(den_col, 0.0) or 0.0
                                        values.append(pnl / den if den > 0 else 0.0)
                                    else:
                                        values.append(regime_row.get(col, 0.0) or 0.0)
                            
                            model_metrics[model_idx] = values
                    
                    # Transfer to GPU and compute scores vectorized
                    metrics_gpu = cp.array(model_metrics, dtype=cp.float32)
                    
                    # Vectorized dot product: [models, metrics] × [metrics] = [models]
                    combo_scores = cp.dot(metrics_gpu, weights_gpu)
                    
                    # Store results
                    scores_gpu[:, combo_idx] = combo_scores
                    
                except Exception as e:
                    print(f"Error processing threshold {threshold}, direction {direction}: {e}")
                
                combo_idx += 1
        
        # Find global maximum on GPU
        max_flat_idx = cp.argmax(scores_gpu)
        max_score = float(scores_gpu.flatten()[max_flat_idx])
        
        # Convert flat index back to model and combination indices
        max_model_idx = int(max_flat_idx // num_combinations)
        max_combo_idx = int(max_flat_idx % num_combinations)
        
        # Decode combination
        max_threshold_idx = max_combo_idx // len(directions)
        max_direction_idx = max_combo_idx % len(directions)
        
        best_model_id = common_models[max_model_idx]
        best_threshold = thresholds[max_threshold_idx]
        best_direction = directions[max_direction_idx]
        
        # Free GPU memory
        del scores_gpu, metrics_gpu, weights_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        # Prepare result dictionary
        result = {
            'model_id': best_model_id,
            'score': max_score,
            'direction': best_direction,
            'threshold': best_threshold,
            'details': f"🚀 GPU-accelerated best combination: model {best_model_id}, threshold {best_threshold}, {best_direction}side"
        }
        
        # If metrics breakdown is requested, calculate and include it
        if show_metrics:
            try:
                # Get the best model's detailed metrics using the same logic as standard method
                daily_row = daily_data[daily_data['ModelID'] == best_model_id].iloc[0]
                regime_row = regime_data[regime_data['ModelID'] == best_model_id].iloc[0]
                
                # Get columns for the best combination
                threshold_dir_combos = self.get_all_threshold_direction_combinations()
                column_cache = {}
                for thr, dir in threshold_dir_combos:
                    column_cache[(thr, dir)] = self._get_threshold_direction_columns_cached(
                        thr, dir, daily_data, regime_data
                    )
                
                best_columns = column_cache[(best_threshold, best_direction)]
                
                # Extract detailed metrics
                metric_details = []
                
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
                        
                    elif '_ppt_' in col:
                        # Calculate PnL per trade
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
                print(f"Warning: Could not generate metrics breakdown: {e}")
                result['metrics_breakdown'] = None
        
        return result


def get_best_trading_model(trading_day: str, market_regime: int, weighting_array: np.ndarray, 
                          mode: str = 'gpu') -> Dict:
    """
    Convenience function to get the best trading model.
    
    Args:
        trading_day: Trading day in format 'YYYYMMDD'
        market_regime: Market regime (0, 1, 2, 3, or 4)
        weighting_array: Array of weights for 76 performance metrics
        mode: 'gpu' for GPU acceleration, 'fast' for CPU parallel, 'standard' for single-threaded
        
    Returns:
        Dict with best model information
    """
    weighter = ModelTradingWeighter()
    
    if mode == 'gpu':
        return weighter.get_best_trading_model_gpu(trading_day, market_regime, weighting_array)
    elif mode == 'fast':
        return weighter.get_best_trading_model_fast(trading_day, market_regime, weighting_array)
    else:
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
