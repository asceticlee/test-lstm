#!/usr/bin/env python3
"""
Performance Index Generator

This module provides functionality to generate and use index files for fast lookup
of model performance data without scanning entire CSV files.

Index formats:
- Daily Performance Index: model_id,trading_day,file_path,row_number
- Regime Performance Index: model_id,trading_day,regime,file_path,row_number

This dramatically speeds up data access by providing direct row lookup.
"""

import os
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class PerformanceIndexManager:
    """Manages performance data index files for fast lookup"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.model_performance_path = self.base_path / "model_performance"
        
        # Index file paths
        self.daily_index_path = self.model_performance_path / "daily_performance_index.csv"
        self.regime_index_path = self.model_performance_path / "regime_performance_index.csv"
        
        # In-memory index caches
        self.daily_index = {}  # (model_id, trading_day) -> (file_path, row_number)
        self.regime_index = {}  # (model_id, trading_day) -> (file_path, start_row, end_row)
        
        # Cache for regime performance data per trading day
        self.regime_data_cache = {}  # (model_id, trading_day) -> DataFrame with all regimes
        
    def generate_daily_performance_index(self, start_model_id: int, end_model_id: int):
        """Generate index for daily performance files"""
        print(f"\nGenerating daily performance index for models {start_model_id:05d} to {end_model_id:05d}...")
        
        daily_performance_dir = self.model_performance_path / "model_daily_performance"
        index_data = []
        
        for model_num in range(start_model_id, end_model_id + 1):
            model_id = f"{model_num:05d}"
            file_path = daily_performance_dir / f"model_{model_id}_daily_performance.csv"
            
            if not file_path.exists():
                continue
                
            try:
                # Read just the TradingDay column to build index
                df = pd.read_csv(file_path, usecols=['TradingDay'])
                
                for row_num, trading_day in enumerate(df['TradingDay'], start=1):  # Start at 1 (skip header)
                    index_data.append([
                        model_id,
                        int(trading_day),
                        str(file_path),
                        row_num
                    ])
                    
                if model_num % 50 == 0:
                    print(f"  Processed {model_num} models...")
                    
            except Exception as e:
                logger.warning(f"Error indexing daily performance for model {model_id}: {e}")
                continue
        
        # Write index file
        with open(self.daily_index_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model_id', 'trading_day', 'file_path', 'row_number'])
            writer.writerows(index_data)
            
        print(f"  Generated daily performance index with {len(index_data):,} entries")
        print(f"  Saved to: {self.daily_index_path}")
        
    def generate_regime_performance_index(self, start_model_id: int, end_model_id: int):
        """Generate index for regime performance files - grouped by trading day for efficient batch loading"""
        print(f"\nGenerating regime performance index for models {start_model_id:05d} to {end_model_id:05d}...")
        
        regime_performance_dir = self.model_performance_path / "model_regime_performance"
        index_data = []
        
        for model_num in range(start_model_id, end_model_id + 1):
            model_id = f"{model_num:05d}"
            file_path = regime_performance_dir / f"model_{model_id}_regime_performance.csv"
            
            if not file_path.exists():
                continue
                
            try:
                # Read file to get trading day information
                df = pd.read_csv(file_path, usecols=['TradingDay'])
                
                # Group consecutive rows by trading day for batch loading
                current_trading_day = None
                start_row = None
                
                for row_idx, trading_day in enumerate(df['TradingDay'], start=1):
                    if trading_day != current_trading_day:
                        # Save previous group if exists
                        if current_trading_day is not None:
                            index_data.append([
                                model_id,
                                int(current_trading_day),
                                str(file_path),
                                start_row,
                                row_idx - 1  # End row of previous group
                            ])
                        
                        # Start new group
                        current_trading_day = trading_day
                        start_row = row_idx
                
                # Save final group
                if current_trading_day is not None:
                    index_data.append([
                        model_id,
                        int(current_trading_day),
                        str(file_path),
                        start_row,
                        len(df)  # End row is last row
                    ])
                    
                if model_num % 50 == 0:
                    print(f"  Processed {model_num} models...")
                    
            except Exception as e:
                logger.warning(f"Error indexing regime performance for model {model_id}: {e}")
                continue
        
        # Write index file
        with open(self.regime_index_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model_id', 'trading_day', 'file_path', 'start_row', 'end_row'])
            writer.writerows(index_data)
            
        print(f"  Generated regime performance index with {len(index_data):,} entries")
        print(f"  Saved to: {self.regime_index_path}")
    
    def load_daily_index(self):
        """Load daily performance index into memory"""
        if not self.daily_index_path.exists():
            logger.warning(f"Daily index file not found: {self.daily_index_path}")
            return
            
        try:
            df = pd.read_csv(self.daily_index_path, dtype={'model_id': str})  # Force model_id to be string
            for _, row in df.iterrows():
                key = (row['model_id'], int(row['trading_day']))
                value = (row['file_path'], int(row['row_number']))
                self.daily_index[key] = value
            
            logger.info(f"Loaded daily index with {len(self.daily_index):,} entries")
        except Exception as e:
            logger.error(f"Error loading daily index: {e}")
    
    def load_regime_index(self):
        """Load regime performance index into memory"""
        if not self.regime_index_path.exists():
            logger.warning(f"Regime index file not found: {self.regime_index_path}")
            return
            
        try:
            df = pd.read_csv(self.regime_index_path, dtype={'model_id': str})  # Force model_id to be string
            for _, row in df.iterrows():
                key = (row['model_id'], int(row['trading_day']))
                value = (row['file_path'], int(row['start_row']), int(row['end_row']))
                self.regime_index[key] = value
            
            logger.info(f"Loaded regime index with {len(self.regime_index):,} entries")
        except Exception as e:
            logger.error(f"Error loading regime index: {e}")
    
    def get_daily_performance_fast(self, model_id: str, trading_day: int) -> Optional[pd.Series]:
        """Fast lookup of daily performance data using index"""
        key = (model_id, trading_day)
        
        if key not in self.daily_index:
            return None
            
        file_path, row_number = self.daily_index[key]
        
        try:
            # Use sed or pandas to get specific row efficiently
            df = pd.read_csv(file_path, skiprows=range(1, row_number), nrows=1)
            return df.iloc[0] if len(df) > 0 else None
        except Exception as e:
            logger.error(f"Error reading daily performance for {model_id}, {trading_day}: {e}")
            return None
    
    def get_regime_performance_fast(self, model_id: str, trading_day: int, regime: int) -> Optional[pd.Series]:
        """Fast lookup of regime performance data using index and caching"""
        # First check if we have the trading day data cached
        cache_key = (model_id, trading_day)
        
        if cache_key not in self.regime_data_cache:
            # Load all regime data for this trading day
            self._load_regime_data_for_trading_day(model_id, trading_day)
        
        # Get the specific regime data from cache
        if cache_key in self.regime_data_cache:
            regime_data = self.regime_data_cache[cache_key]
            regime_rows = regime_data[regime_data['Regime'] == regime]
            if len(regime_rows) > 0:
                return regime_rows.iloc[0]
        
        return None
    
    def _load_regime_data_for_trading_day(self, model_id: str, trading_day: int):
        """Load all regime performance data for a specific model and trading day"""
        key = (model_id, trading_day)
        
        if key not in self.regime_index:
            return
            
        file_path, start_row, end_row = self.regime_index[key]
        
        try:
            # Load all regime data for this trading day efficiently
            df = pd.read_csv(
                file_path, 
                skiprows=range(1, start_row),  # Skip rows before start
                nrows=end_row - start_row + 1  # Read only the needed rows
            )
            
            # Cache the data
            self.regime_data_cache[key] = df
            
        except Exception as e:
            logger.error(f"Error loading regime data for {model_id}, {trading_day}: {e}")
    
    def get_models_for_trading_day(self, trading_day: int) -> List[str]:
        """Get all models that have data for a specific trading day"""
        models = set()
        for model_id, td in self.daily_index.keys():
            if td == trading_day:
                models.add(model_id)  # model_id is already string
        return sorted(list(models))
    
    def get_trading_days_for_model(self, model_id: str) -> List[int]:
        """Get all trading days that have data for a specific model"""
        trading_days = set()
        for mid, td in self.daily_index.keys():
            if mid == model_id:  # Both are strings now
                trading_days.add(td)
        return sorted(list(trading_days))


def update_batch_model_performance_with_indexing():
    """Update the batch_model_performance.py to include index generation"""
    
    # Read the current batch script
    script_path = Path(__file__).parent / "batch_model_performance.py"
    
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        return
    
    # The indexing will be added to the main() function
    index_code = '''
    # Generate performance index for fast lookup
    try:
        from performance_index_generator import PerformanceIndexManager
        
        print("\\nGenerating performance data index...")
        index_manager = PerformanceIndexManager(project_root)
        index_manager.generate_daily_performance_index(start_model_id, end_model_id)
        
    except Exception as e:
        print(f"Warning: Could not generate performance index: {e}")
    '''
    
    print("Index generation code ready to be added to batch_model_performance.py")
    print("Add this code before the final summary in the main() function:")
    print(index_code)


def update_batch_model_regime_performance_with_indexing():
    """Update the batch_model_regime_performance.py to include index generation"""
    
    # Read the current batch script
    script_path = Path(__file__).parent / "batch_model_regime_performance.py"
    
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        return
    
    # The indexing will be added to the main() function
    index_code = '''
    # Generate regime performance index for fast lookup
    try:
        from performance_index_generator import PerformanceIndexManager
        
        print("\\nGenerating regime performance data index...")
        index_manager = PerformanceIndexManager(project_root)
        index_manager.generate_regime_performance_index(start_model_id, end_model_id)
        
    except Exception as e:
        print(f"Warning: Could not generate regime performance index: {e}")
    '''
    
    print("Index generation code ready to be added to batch_model_regime_performance.py")
    print("Add this code before the final summary in the main() function:")
    print(index_code)


if __name__ == "__main__":
    # Example usage
    project_root = "/home/stephen/projects/Testing/TestPy/test-lstm"
    
    # Create index manager
    index_manager = PerformanceIndexManager(project_root)
    
    # Generate indexes for models 1-10 (example)
    index_manager.generate_daily_performance_index(1, 10)
    index_manager.generate_regime_performance_index(1, 10)
    
    # Load indexes into memory
    index_manager.load_daily_index()
    index_manager.load_regime_index()
    
    # Example fast lookup
    daily_data = index_manager.get_daily_performance_fast("00001", 20250110)
    regime_data = index_manager.get_regime_performance_fast("00001", 20250110, 1)
    
    print(f"Daily data found: {daily_data is not None}")
    print(f"Regime data found: {regime_data is not None}")
