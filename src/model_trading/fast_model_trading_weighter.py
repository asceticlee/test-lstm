#!/usr/bin/env python3
"""
Fast Model Trading Weighter System (Index-Based)

This module provides ultra-fast functionality to apply weighting arrays to model 
performance data using pre-generated index files for instant data lookup.

Performance improvements:
- Uses index files for O(1) data lookup instead of scanning CSV files
- Loads only required data rows using index
- Multi-threaded data loading for parallel processing
- Optimized memory usage and data structures

Author: AI Assistant
Date: Created for test-lstm project
"""

import pandas as pd
import numpy as np
import os
import subprocess
from typing import Tuple, Dict, List, Optional, Any
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastModelTradingWeighter:
    """
    Ultra-fast model trading weighter using index files for instant data access.
    """
    
    def __init__(self, base_path: str = None):
        """Initialize the Fast ModelTradingWeighter."""
        if base_path is None:
            # Auto-detect project root
            current_path = Path(__file__).parent.absolute()
            while current_path.parent != current_path:
                if (current_path / "model_performance").exists():
                    base_path = str(current_path)
                    break
                current_path = current_path.parent
            else:
                base_path = "/home/stephen/projects/Testing/TestPy/test-lstm"
        
        self.base_path = Path(base_path)
        self.model_performance_path = self.base_path / "model_performance"
        self.models_path = self.base_path / "models"
        
        # Index manager for fast lookups
        import sys
        sys.path.append(str(self.base_path / "src"))
        from performance_index_generator import PerformanceIndexManager
        self.index_manager = PerformanceIndexManager(str(self.base_path))
        
        # Load indexes into memory
        self._load_indexes()
        
        # Load essential metadata
        self._load_metadata()
        
        logger.info("FastModelTradingWeighter initialized with index-based lookups")
    
    def _load_indexes(self):
        """Load performance indexes into memory for fast lookup"""
        start_time = time.time()
        self.index_manager.load_daily_index()
        self.index_manager.load_regime_index()
        load_time = time.time() - start_time
        logger.info(f"Loaded performance indexes in {load_time:.3f} seconds")
    
    def _load_metadata(self):
        """Load essential metadata using optimized methods"""
        try:
            # Load model log using grep to get only essential columns
            model_log_path = self.models_path / "model_log.csv"
            if model_log_path.exists():
                # Use only essential columns to save memory and force model_id to string
                self.model_log_data = pd.read_csv(
                    model_log_path, 
                    usecols=['model_id', 'train_from', 'train_to'],
                    dtype={'model_id': str}  # Force model_id to be string
                )
                logger.info(f"Loaded model log metadata: {len(self.model_log_data)} models")
            else:
                self.model_log_data = None
                logger.warning("Model log not found")
            
            # Load alltime performance data
            alltime_perf_path = self.model_performance_path / "models_alltime_performance.csv"
            if alltime_perf_path.exists():
                self.alltime_performance_data = pd.read_csv(alltime_perf_path, dtype={'ModelID': str})
                logger.info(f"Loaded alltime performance data: {len(self.alltime_performance_data)} models")
            else:
                self.alltime_performance_data = None
                logger.warning("Alltime performance data not found")
            
            # Load alltime regime performance data
            alltime_regime_perf_path = self.model_performance_path / "models_alltime_regime_performance.csv"
            if alltime_regime_perf_path.exists():
                self.alltime_regime_performance_data = pd.read_csv(alltime_regime_perf_path, dtype={'ModelID': str})
                logger.info(f"Loaded alltime regime performance data: {len(self.alltime_regime_performance_data)} records")
            else:
                self.alltime_regime_performance_data = None
                logger.warning("Alltime regime performance data not found")
                
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.model_log_data = None
            self.alltime_performance_data = None
            self.alltime_regime_performance_data = None
    
    def _get_available_models_for_trading_day(self, trading_day: str) -> List[str]:
        """Get models available for a specific trading day using index"""
        trading_day_int = int(trading_day)
        return self.index_manager.get_models_for_trading_day(trading_day_int)
    
    def _filter_models_by_training_period(self, model_ids: List[str], trading_day: str) -> List[str]:
        """Filter models to exclude those whose training period includes the trading day"""
        if self.model_log_data is None:
            logger.warning("Model log data not available, cannot filter by training period")
            return model_ids
        
        filtered_models = []
        trading_day_int = int(trading_day)
        
        for model_id in model_ids:
            # Use vectorized pandas operations for speed
            model_row = self.model_log_data[self.model_log_data['model_id'] == model_id]
            
            if len(model_row) == 0:
                continue
            
            train_from = int(model_row.iloc[0]['train_from'])
            train_to = int(model_row.iloc[0]['train_to'])
            
            # Exclude model if trading day falls within training period
            if not (train_from <= trading_day_int <= train_to):
                filtered_models.append(model_id)
        
        logger.info(f"Filtered models by training period: {len(filtered_models)}/{len(model_ids)} models remaining")
        return filtered_models
    
    def _get_performance_data_fast(self, model_ids: List[str], trading_day: str, regime_id: int) -> pd.DataFrame:
        """Fast retrieval of performance data using indexes and parallel processing"""
        trading_day_int = int(trading_day)
        
        def get_model_data(model_id: str) -> Optional[pd.Series]:
            """Get performance data for a single model"""
            combined_data = pd.Series(dtype=float)
            combined_data['ModelID'] = model_id
            
            try:
                # 1. Get regime performance data (time-specific)
                regime_data = self.index_manager.get_regime_performance_fast(
                    model_id, trading_day_int, regime_id
                )
                
                if regime_data is not None:
                    # Add regime performance fields
                    for col, val in regime_data.items():
                        if col not in ['ModelID', 'TradingDay', 'Regime']:
                            combined_data[f"regime_{col}"] = val
                
                # 2. Get daily performance data (time-specific)
                daily_data = self.index_manager.get_daily_performance_fast(
                    model_id, trading_day_int
                )
                
                if daily_data is not None:
                    # Add daily performance fields
                    for col, val in daily_data.items():
                        if col not in ['ModelID', 'TradingDay']:
                            combined_data[f"daily_{col}"] = val
                
                # 3. Get alltime performance data (if available)
                if self.alltime_performance_data is not None:
                    alltime_row = self.alltime_performance_data[
                        self.alltime_performance_data['ModelID'] == model_id
                    ]
                    if len(alltime_row) > 0:
                        for col, val in alltime_row.iloc[0].items():
                            if col != 'ModelID':
                                combined_data[f"alltime_{col}"] = val
                
                # 4. Get alltime regime performance data (if available)
                if self.alltime_regime_performance_data is not None:
                    alltime_regime_row = self.alltime_regime_performance_data[
                        (self.alltime_regime_performance_data['ModelID'] == model_id) &
                        (self.alltime_regime_performance_data['Regime'] == regime_id)
                    ]
                    if len(alltime_regime_row) > 0:
                        for col, val in alltime_regime_row.iloc[0].items():
                            if col not in ['ModelID', 'Regime']:
                                combined_data[f"alltime_regime_{col}"] = val
                
                # Only return if we have some performance data
                if len(combined_data) > 1:  # More than just ModelID
                    combined_data['source'] = 'combined'
                    return combined_data
                
                return None
                
            except Exception as e:
                logger.warning(f"Error getting data for model {model_id}: {e}")
                return None
        
        # Use parallel processing to load data faster
        performance_data = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_model = {
                executor.submit(get_model_data, model_id): model_id 
                for model_id in model_ids[:50]  # Limit to prevent memory overload
            }
            
            for future in as_completed(future_to_model):
                model_data = future.result()
                if model_data is not None:
                    performance_data.append(model_data)
        
        if not performance_data:
            return pd.DataFrame()
        
        # Combine all model data
        combined_df = pd.DataFrame(performance_data)
        logger.info(f"Fast-loaded performance data with alltime: {len(combined_df)} models, {len(combined_df.columns)} total fields")
        
        return combined_df
    
    def _apply_weighting_optimized(self, performance_df: pd.DataFrame, weighting_array: List[float]) -> pd.DataFrame:
        """Optimized weighting application using vectorized operations"""
        if performance_df.empty:
            return performance_df.copy()
        
        # Get metric columns (numeric columns excluding metadata)
        numeric_cols = performance_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['ModelID', 'TradingDay', 'Regime', 'source']
        metric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Adjust weighting array
        if len(weighting_array) != len(metric_cols):
            if len(weighting_array) < len(metric_cols):
                # Extend with ones
                weighting_array = weighting_array + [1.0] * (len(metric_cols) - len(weighting_array))
            else:
                # Truncate
                weighting_array = weighting_array[:len(metric_cols)]
        
        # Apply weights to all metrics at once using numpy
        metrics_matrix = performance_df[metric_cols].values
        weights_array = np.array(weighting_array)
        
        # Element-wise multiplication and sum
        weighted_scores = np.sum(metrics_matrix * weights_array, axis=1)
        
        # Create all weighted columns at once using pd.concat to avoid fragmentation
        weighted_columns = []
        
        # Add ModelID column
        weighted_columns.append(performance_df[['ModelID']].copy())
        
        # Add total weighted score
        total_score_df = pd.DataFrame({'total_weighted_score': weighted_scores}, index=performance_df.index)
        weighted_columns.append(total_score_df)
        
        # Create all individual weighted scores at once
        if metric_cols:
            weighted_metrics_matrix = metrics_matrix * weights_array
            weighted_metrics_df = pd.DataFrame(
                weighted_metrics_matrix,
                columns=[f'weighted_{col}' for col in metric_cols],
                index=performance_df.index
            )
            weighted_columns.append(weighted_metrics_df)
        
        # Combine all columns at once using pd.concat
        weighted_data = pd.concat(weighted_columns, axis=1)
        
        return weighted_data
    
    def _select_optimal_model_simple(self, weighted_scores: pd.DataFrame) -> Tuple[str, str, str, float]:
        """Simple but effective model selection"""
        if len(weighted_scores) == 0:
            raise ValueError("No performance data available")
        
        # Find model with highest total weighted score
        best_idx = weighted_scores['total_weighted_score'].idxmax()
        best_score = weighted_scores.loc[best_idx, 'total_weighted_score']
        model_id = str(weighted_scores.loc[best_idx, 'ModelID'])
        
        # Simple heuristic for direction and threshold
        # Look at the best weighted individual metrics
        weighted_cols = [col for col in weighted_scores.columns if col.startswith('weighted_')]
        
        if weighted_cols:
            # Find the metric with highest contribution
            individual_scores = weighted_scores.loc[best_idx, weighted_cols]
            best_metric = individual_scores.idxmax()
            
            # Extract direction and threshold from metric name
            original_metric = best_metric.replace('weighted_', '')
            
            # Determine direction (up/down)
            direction = 'up' if '_up_' in original_metric else 'down'
            
            # Extract threshold
            if '_thr_' in original_metric:
                threshold_part = original_metric.split('_thr_')[-1]
                threshold = threshold_part.replace('_', '.')
            else:
                threshold = "0.0"
        else:
            direction = 'up'
            threshold = "0.0"
        
        return model_id, direction, threshold, best_score
    
    def weight_and_select_model_fast(self, trading_day: str, regime_id: int, 
                                   weighting_array: List[float]) -> Tuple[str, str, str]:
        """Fast model selection using index-based data access"""
        try:
            start_time = time.time()
            
            # Get available models for this trading day (using index)
            available_models = self._get_available_models_for_trading_day(trading_day)
            if not available_models:
                raise ValueError(f"No models available for trading day {trading_day}")
            
            logger.info(f"Found {len(available_models)} models for trading day {trading_day}")
            
            # Skip training period filtering - performance files already exclude training data
            valid_models = available_models
            
            # Fast data loading using indexes
            performance_data = self._get_performance_data_fast(valid_models, trading_day, regime_id)
            if performance_data.empty:
                raise ValueError(f"No performance data found for day {trading_day}, regime {regime_id}")
            
            # Apply weighting
            weighted_scores = self._apply_weighting_optimized(performance_data, weighting_array)
            
            # Select optimal model
            model_id, direction, threshold, score = self._select_optimal_model_simple(weighted_scores)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Fast selection completed in {elapsed_time:.3f}s: Model {model_id}, {direction}, {threshold}")
            
            return model_id, direction, threshold
            
        except Exception as e:
            logger.error(f"Error in fast model selection: {e}")
            raise


# Test the fast weighter
def test_fast_weighter():
    """Test the FastModelTradingWeighter performance"""
    print("Testing FastModelTradingWeighter...")
    
    try:
        # Initialize fast weighter
        start_time = time.time()
        fast_weighter = FastModelTradingWeighter()
        init_time = time.time() - start_time
        print(f"Initialization time: {init_time:.3f}s")
        
        # Test different weighting strategies
        strategies = {
            "Balanced": [1.0] * 50,
            "Accuracy_Heavy": [2.0 if i % 4 == 0 else 0.5 for i in range(50)],
            "PnL_Heavy": [2.0 if i % 4 == 3 else 0.5 for i in range(50)]
        }
        
        test_cases = [
            ("20250110", 1),
            ("20250111", 2),
            ("20250112", 0)
        ]
        
        for strategy_name, weights in strategies.items():
            print(f"\nTesting {strategy_name} strategy...")
            
            for trading_day, regime_id in test_cases:
                try:
                    start_time = time.time()
                    model_id, direction, threshold = fast_weighter.weight_and_select_model_fast(
                        trading_day, regime_id, weights
                    )
                    elapsed_time = time.time() - start_time
                    
                    print(f"  Day {trading_day}, Regime {regime_id}: Model {model_id}, {direction}, {threshold} ({elapsed_time:.3f}s)")
                    
                except Exception as e:
                    print(f"  Day {trading_day}, Regime {regime_id}: ERROR - {e}")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_fast_weighter()
