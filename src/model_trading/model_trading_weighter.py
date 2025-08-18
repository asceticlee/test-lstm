#!/usr/bin/env python3
"""
Model Trading Weighter System

This module provides functionality to apply weighting arrays to model performance data
and select optimal trading models based on regime, trading day, and weighted performance metrics.

Author: AI Assistant
Date: Created for test-lstm project
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List, Optional, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTradingWeighter:
    """
    A class for weighting model performance data and selecting optimal trading models.
    
    This class provides functionality to:
    1. Load and filter performance data based on regime and trading day
    2. Apply user-defined weighting arrays to performance metrics
    3. Determine proportional vs inverse impact of metrics on trading performance
    4. Select optimal models based on weighted performance scores
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize the ModelTradingWeighter.
        
        Args:
            base_path: Base path to the test-lstm project directory. If None, auto-detect.
        """
        if base_path is None:
            # Auto-detect project root by looking for key directories
            current_path = Path(__file__).parent.absolute()
            # Navigate up to find the project root (where model_performance directory exists)
            while current_path.parent != current_path:  # Stop at filesystem root
                if (current_path / "model_performance").exists():
                    base_path = str(current_path)
                    break
                current_path = current_path.parent
            else:
                # Fallback to default path
                base_path = "/home/stephen/projects/Testing/TestPy/test-lstm"
        
        self.base_path = Path(base_path)
        self.model_performance_path = self.base_path / "model_performance"
        self.models_path = self.base_path / "models"
        
        # Data containers
        self.daily_performance_data = {}
        self.regime_performance_data = {}
        self.alltime_performance_data = None
        self.alltime_regime_performance_data = None
        self.model_log_data = None
        
        # Performance metric patterns and their impact direction
        self.metric_impact_mapping = self._define_metric_impact_mapping()
        
        # Load base data
        self._load_base_data()
    
    def _define_metric_impact_mapping(self) -> Dict[str, str]:
        """
        Define whether metrics have proportional or inverse impact on trading performance.
        
        Returns:
            Dictionary mapping metric patterns to their impact type ('proportional' or 'inverse')
        """
        # Based on analysis of performance metrics:
        # - Accuracy (acc) metrics: Higher is better -> proportional impact
        # - Number of trades (num) metrics: More data points for confidence -> proportional impact  
        # - Denominator (den) metrics: More opportunities -> proportional impact
        # - P&L (pnl) metrics: Higher profits -> proportional impact
        
        return {
            '_acc_': 'proportional',  # Accuracy - higher is better
            '_num_': 'proportional',  # Number of trades - more is generally better for confidence
            '_den_': 'proportional',  # Denominator - more opportunities is better
            '_pnl_': 'proportional',  # Profit & Loss - higher is better
        }
    
    def _load_base_data(self):
        """Load all base performance data files."""
        try:
            logger.info("Loading base performance data...")
            
            # Load alltime performance data
            alltime_perf_path = self.model_performance_path / "models_alltime_performance.csv"
            if alltime_perf_path.exists():
                self.alltime_performance_data = pd.read_csv(alltime_perf_path)
                logger.info(f"Loaded alltime performance data: {len(self.alltime_performance_data)} models")
            
            # Load alltime regime performance data
            alltime_regime_perf_path = self.model_performance_path / "models_alltime_regime_performance.csv"
            if alltime_regime_perf_path.exists():
                self.alltime_regime_performance_data = pd.read_csv(alltime_regime_perf_path)
                logger.info(f"Loaded alltime regime performance data: {len(self.alltime_regime_performance_data)} records")
            
            # Load model log data
            model_log_path = self.models_path / "model_log.csv"
            if model_log_path.exists():
                self.model_log_data = pd.read_csv(model_log_path)
                logger.info(f"Loaded model log data: {len(self.model_log_data)} models")
            
        except Exception as e:
            logger.error(f"Error loading base data: {e}")
            raise
    
    def _load_daily_performance_data(self, model_ids: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load daily performance data for specified models.
        
        Args:
            model_ids: List of model IDs to load
            
        Returns:
            Dictionary mapping model_id to daily performance DataFrame
        """
        daily_data = {}
        daily_path = self.model_performance_path / "model_daily_performance"
        
        for model_id in model_ids:
            # Ensure model_id is zero-padded to 5 digits for file naming
            padded_model_id = f"{int(model_id):05d}"
            file_path = daily_path / f"model_{padded_model_id}_daily_performance.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    daily_data[model_id] = df
                except Exception as e:
                    logger.warning(f"Error loading daily data for model {model_id}: {e}")
        
        logger.info(f"Loaded daily performance data for {len(daily_data)} models")
        return daily_data
    
    def _load_regime_performance_data(self, model_ids: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load regime performance data for specified models.
        
        Args:
            model_ids: List of model IDs to load
            
        Returns:
            Dictionary mapping model_id to regime performance DataFrame
        """
        regime_data = {}
        regime_path = self.model_performance_path / "model_regime_performance"
        
        for model_id in model_ids:
            # Ensure model_id is zero-padded to 5 digits for file naming
            padded_model_id = f"{int(model_id):05d}"
            file_path = regime_path / f"model_{padded_model_id}_regime_performance.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    regime_data[model_id] = df
                except Exception as e:
                    logger.warning(f"Error loading regime data for model {model_id}: {e}")
        
        logger.info(f"Loaded regime performance data for {len(regime_data)} models")
        return regime_data
    
    def _get_available_models(self) -> List[str]:
        """
        Get list of available model IDs from the alltime performance data.
        
        Returns:
            List of available model IDs
        """
        if self.alltime_performance_data is not None:
            # Extract model IDs from ModelID column
            return self.alltime_performance_data['ModelID'].astype(str).tolist()
        return []
    
    def _filter_models_by_training_period(self, model_ids: List[str], trading_day: str) -> List[str]:
        """
        Filter models to exclude those whose training period includes the trading day.
        
        Args:
            model_ids: List of model IDs to filter
            trading_day: Trading day in YYYYMMDD format
            
        Returns:
            Filtered list of model IDs
        """
        if self.model_log_data is None:
            logger.warning("Model log data not available, cannot filter by training period")
            return model_ids
        
        filtered_models = []
        trading_day_int = int(trading_day)
        
        for model_id in model_ids:
            # Find model in log data - try both original and zero-padded format
            padded_model_id = f"{int(model_id):05d}"
            model_row = self.model_log_data[self.model_log_data['model_id'] == padded_model_id]
            
            if len(model_row) == 0:
                # Try without padding
                model_row = self.model_log_data[self.model_log_data['model_id'] == model_id]
            
            if len(model_row) == 0:
                # Model not found in log, include it (might be manually created)
                filtered_models.append(model_id)
                continue
            
            train_from = int(model_row.iloc[0]['train_from'])
            train_to = int(model_row.iloc[0]['train_to'])
            
            # Exclude model if trading day falls within training period
            if train_from <= trading_day_int <= train_to:
                logger.debug(f"Excluding model {model_id}: trading day {trading_day} within training period {train_from}-{train_to}")
            else:
                filtered_models.append(model_id)
        
        logger.info(f"Filtered models by training period: {len(filtered_models)}/{len(model_ids)} models remaining")
        return filtered_models
    
    def _get_metric_impact_type(self, metric_name: str) -> str:
        """
        Determine if a metric has proportional or inverse impact on performance.
        
        Args:
            metric_name: Name of the performance metric
            
        Returns:
            'proportional' or 'inverse'
        """
        for pattern, impact_type in self.metric_impact_mapping.items():
            if pattern in metric_name:
                return impact_type
        
        # Default to proportional if pattern not found
        return 'proportional'
    
    def _apply_weighting_to_metrics(self, performance_df, weighting_array):
        """Apply weighting array to performance metrics"""
        if performance_df.empty:
            return performance_df.copy()
        
        # Get metric columns (exclude model_id and metadata columns)
        metric_cols = [col for col in performance_df.columns 
                      if col not in ['ModelID', 'TradingDay', 'Regime']]
        
        # Adjust weighting array to match number of metrics
        if len(weighting_array) != len(metric_cols):
            logging.warning(f"Weighting array length ({len(weighting_array)}) doesn't match metrics ({len(metric_cols)})")
            if len(weighting_array) < len(metric_cols):
                # Extend with 1.0 (neutral weight)
                weighting_array = list(weighting_array) + [1.0] * (len(metric_cols) - len(weighting_array))
                logging.info(f"Extended weighting array from {len(weighting_array) - (len(metric_cols) - len(weighting_array))} to {len(weighting_array)} elements")
            else:
                # Truncate to match
                weighting_array = weighting_array[:len(metric_cols)]
                logging.info(f"Truncated weighting array to {len(weighting_array)} elements")
        
        # Start with core columns
        base_cols = ['ModelID']
        weighted_data = {'ModelID': performance_df['ModelID']}
        
        if 'TradingDay' in performance_df.columns:
            base_cols.append('TradingDay')
            weighted_data['TradingDay'] = performance_df['TradingDay']
        if 'Regime' in performance_df.columns:
            base_cols.append('Regime')
            weighted_data['Regime'] = performance_df['Regime']
        
        # Calculate all weighted metrics at once
        total_weighted_score = pd.Series(0.0, index=performance_df.index)
        
        for i, metric_col in enumerate(metric_cols):
            weight = weighting_array[i]
            metric_values = pd.to_numeric(performance_df[metric_col], errors='coerce').fillna(0)
            
            # Apply impact direction (all metrics are proportional for now)
            impact_direction = self.metric_impact_mapping.get(
                metric_col.split('_')[1] if '_' in metric_col else 'default', 'proportional')
            
            if impact_direction == 'inverse':
                weighted_metric_score = (1 / (metric_values + 1e-6)) * weight
            else:  # proportional
                weighted_metric_score = metric_values * weight
            
            weighted_data[f"weighted_{metric_col}"] = weighted_metric_score
            total_weighted_score += weighted_metric_score
        
        # Add total score
        weighted_data['total_weighted_score'] = total_weighted_score
        
        # Create DataFrame from dictionary to avoid fragmentation
        weighted_scores = pd.DataFrame(weighted_data)
        
        return weighted_scores
    
    def _select_optimal_model(self, weighted_scores: pd.DataFrame) -> Tuple[str, str, str, float]:
        """
        Select optimal model based on weighted scores.
        
        Args:
            weighted_scores: DataFrame with weighted performance scores
            
        Returns:
            Tuple of (model_id, direction, threshold, best_score)
        """
        if len(weighted_scores) == 0:
            raise ValueError("No performance data available for model selection")
        
        # Group by ModelID and aggregate scores (in case same model appears multiple times from different sources)
        model_scores = weighted_scores.groupby('ModelID')['total_weighted_score'].max().reset_index()
        
        # Find the model with highest total weighted score
        best_model_idx = model_scores['total_weighted_score'].idxmax()
        best_model_id = str(model_scores.iloc[best_model_idx]['ModelID'])
        best_score = model_scores.iloc[best_model_idx]['total_weighted_score']
        
        # Get all rows for the best model to analyze metrics
        best_model_rows = weighted_scores[weighted_scores['ModelID'].astype(str) == best_model_id]
        
        # For the best model, find the best direction/threshold combination
        # Look at all weighted metrics to find the highest scoring individual metric
        best_individual_score = 0
        best_direction = 'up'
        best_threshold = '0.0'
        
        for _, row in best_model_rows.iterrows():
            weighted_metric_cols = [col for col in row.index if col.startswith('weighted_alltime_')]
            
            for metric_col in weighted_metric_cols:
                metric_score = row[metric_col]
                if pd.notna(metric_score) and metric_score > best_individual_score:
                    best_individual_score = metric_score
                    
                    # Extract direction and threshold from metric name
                    # Format: weighted_alltime_up_acc_0.1 or weighted_alltime_down_pnl_0.5
                    parts = metric_col.replace('weighted_alltime_', '').split('_')
                    if len(parts) >= 3:
                        direction = parts[0]  # 'up' or 'down'
                        threshold = parts[2]  # '0.1', '0.5', etc.
                        
                        best_direction = direction
                        best_threshold = threshold
        
        return best_model_id, best_direction, best_threshold, best_score
    
    def weight_and_select_model(self, trading_day: str, regime_id: int, 
                              weighting_array: List[float]) -> Tuple[str, str, str]:
        """
        Main function to apply weighting and select optimal trading model.
        
        Args:
            trading_day: Trading day in YYYYMMDD format
            regime_id: Regime ID for filtering (0, 1, 2, 3, 4)
            weighting_array: Array of weights to apply to performance metrics
            
        Returns:
            Tuple of (model_id, upside/downside direction, threshold)
        """
        try:
            logger.info(f"Starting model selection for trading_day={trading_day}, regime_id={regime_id}")
            
            # Get available models
            available_models = self._get_available_models()
            if not available_models:
                raise ValueError("No models available in performance data")
            
            # Filter models by training period
            valid_models = self._filter_models_by_training_period(available_models, trading_day)
            if not valid_models:
                raise ValueError(f"No valid models available for trading day {trading_day}")
            
            # Load daily performance data for valid models
            daily_data = self._load_daily_performance_data(valid_models)
            
            # Load regime performance data for valid models  
            regime_data = self._load_regime_performance_data(valid_models)
            
            # Combine and filter performance data
            combined_performance = []
            
            # Prioritize specific data over general data
            # 1. Daily performance data filtered by trading day (most specific)
            for model_id, df in daily_data.items():
                if 'TradingDay' in df.columns:
                    filtered_df = df[df['TradingDay'] == int(trading_day)].copy()
                    if len(filtered_df) > 0:
                        filtered_df['ModelID'] = model_id
                        filtered_df['DataSource'] = 'daily'
                        combined_performance.append(filtered_df)
            
            # 2. Regime performance data filtered by trading day and regime (regime-specific)
            for model_id, df in regime_data.items():
                if 'TradingDay' in df.columns and 'Regime' in df.columns:
                    filtered_df = df[(df['TradingDay'] == int(trading_day)) & 
                                   (df['Regime'] == regime_id)].copy()
                    if len(filtered_df) > 0:
                        filtered_df['ModelID'] = model_id
                        filtered_df['DataSource'] = 'regime'
                        combined_performance.append(filtered_df)
            
            # 3. Alltime regime performance data (if no specific daily/regime data found)
            models_with_specific_data = set()
            if combined_performance:
                for df in combined_performance:
                    models_with_specific_data.update(df['ModelID'].astype(str).unique())
            
            if self.alltime_regime_performance_data is not None:
                # Only include models that don't already have specific data
                remaining_models = [m for m in valid_models if str(m) not in models_with_specific_data]
                if remaining_models:
                    regime_filtered = self.alltime_regime_performance_data[
                        (self.alltime_regime_performance_data['Regime'] == regime_id) &
                        (self.alltime_regime_performance_data['ModelID'].astype(str).isin(remaining_models))
                    ].copy()
                    if len(regime_filtered) > 0:
                        regime_filtered['DataSource'] = 'alltime_regime'
                        combined_performance.append(regime_filtered)
                        models_with_specific_data.update(regime_filtered['ModelID'].astype(str).unique())
            
            # 4. Alltime performance data (fallback for any remaining models)
            if self.alltime_performance_data is not None:
                remaining_models = [m for m in valid_models if str(m) not in models_with_specific_data]
                if remaining_models:
                    alltime_filtered = self.alltime_performance_data[
                        self.alltime_performance_data['ModelID'].astype(str).isin(remaining_models)
                    ].copy()
                    if len(alltime_filtered) > 0:
                        alltime_filtered['DataSource'] = 'alltime'
                        combined_performance.append(alltime_filtered)
            
            if not combined_performance:
                raise ValueError(f"No performance data found for trading_day={trading_day}, regime_id={regime_id}")
            
            # Combine all performance data
            all_performance = pd.concat(combined_performance, ignore_index=True, sort=False)
            
            logger.info(f"Combined performance data: {len(all_performance)} records from {len(combined_performance)} sources")
            
            # Apply weighting to the combined performance data
            weighted_scores = self._apply_weighting_to_metrics(all_performance, weighting_array)
            
            # Select optimal model
            model_id, direction, threshold, best_score = self._select_optimal_model(weighted_scores)
            
            logger.info(f"Selected model: {model_id}, direction: {direction}, threshold: {threshold}, score: {best_score:.4f}")
            
            return model_id, direction, threshold
            
        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            raise


def main():
    """Example usage of the ModelTradingWeighter."""
    try:
        # Initialize the weighter
        weighter = ModelTradingWeighter()
        
        # Create a reasonable weighting array based on typical performance metrics
        # For simplicity, start with a smaller array that will be extended automatically
        example_weights = [
            1.5,  # accuracy metrics - higher weight
            1.0,  # number metrics - standard weight
            1.0,  # denominator metrics - standard weight  
            2.0,  # P&L metrics - highest weight
        ] * 18  # Repeat pattern to cover more metrics (72 total)
        
        # Example usage
        trading_day = "20250110"
        regime_id = 1
        
        model_id, direction, threshold = weighter.weight_and_select_model(
            trading_day=trading_day,
            regime_id=regime_id, 
            weighting_array=example_weights
        )
        
        print(f"Optimal model selection:")
        print(f"  Model ID: {model_id}")
        print(f"  Direction: {direction}")
        print(f"  Threshold: {threshold}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
