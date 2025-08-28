#!/usr/bin/env python3
"""
Model Trading Picker

This script applies optimized weights from weight_optimized.csv to select the best trading model
for each market regime on a specific trading day.

The script:
1. Loads optimized weights for each regime from the CSV file
2. Loads daily performance and daily regime performance data for the trading day
3. Applies weights to find the best model, direction, and threshold for each regime
4. Outputs the results to model_trading_model_pick_yyyymmdd.csv

Usage:
    python model_trading_picker.py <weight_optimized_csv> <trading_day>
    
Examples:
    python model_trading_picker.py ../model_trading/weight_optimized.csv 20250808
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add current directory to path to import model_trading_weighter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_trading.model_trading_weighter import ModelTradingWeighter


class ModelTradingPicker:
    """
    Applies optimized weights to select best trading models for each regime.
    """
    
    def __init__(self, project_root: str = None):
        """
        Initialize the model picker.
        
        Args:
            project_root: Root directory of the project. If None, infers from script location.
        """
        if project_root is None:
            # Infer project root from script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
        
        self.project_root = project_root
        self.daily_performance_dir = os.path.join(project_root, "model_performance", "daily_performance")
        self.daily_regime_performance_dir = os.path.join(project_root, "model_performance", "daily_regime_performance")
        self.output_dir = os.path.join(project_root, "model_trading")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model trading weighter for weight application logic
        self.weighter = ModelTradingWeighter(project_root)
    
    def load_optimized_weights(self, weight_csv_path: str) -> Dict[int, np.ndarray]:
        """
        Load optimized weights from CSV file.
        
        Args:
            weight_csv_path: Path to the weight_optimized.csv file
            
        Returns:
            Dictionary mapping regime ID to weight array
        """
        if not os.path.exists(weight_csv_path):
            raise FileNotFoundError(f"Weight optimized CSV file not found: {weight_csv_path}")
        
        try:
            df = pd.read_csv(weight_csv_path)
            
            # Validate required columns
            if 'regime' not in df.columns:
                raise ValueError("Weight CSV file must have 'regime' column")
            
            weights_dict = {}
            
            for _, row in df.iterrows():
                regime_id = int(row['regime'])
                
                # Extract weight values (all columns except 'regime')
                weight_columns = [col for col in df.columns if col != 'regime']
                weight_values = [float(row[col]) for col in weight_columns]
                
                weights_dict[regime_id] = np.array(weight_values)
                
                print(f"Loaded {len(weight_values)} weights for regime {regime_id}")
            
            print(f"Successfully loaded optimized weights for {len(weights_dict)} regimes")
            return weights_dict
            
        except Exception as e:
            raise Exception(f"Error loading optimized weights: {e}")
    
    def find_best_model_for_regime(self, trading_day: str, regime_id: int, 
                                  weights: np.ndarray) -> Optional[Dict]:
        """
        Find the best model for a specific regime using the optimized weights.
        
        Args:
            trading_day: Trading day (YYYYMMDD)
            regime_id: Market regime ID
            weights: Optimized weight array for this regime
            
        Returns:
            Dictionary with best model info or None if no data found
        """
        try:
            # Use the weighter's method to find best model (single weight array)
            result = self.weighter.get_best_trading_model_batch_vectorized(
                trading_day=trading_day,
                market_regime=regime_id,
                weighting_arrays=[weights],  # Pass as list with single array
                show_metrics=False
            )
            
            if result and len(result) > 0:
                best_result = result[0]  # Only one result since we passed one weight array
                
                return {
                    'regime': regime_id,
                    'model_id': best_result['model_id'],
                    'direction': best_result['direction'],
                    'threshold': best_result['threshold'],
                    'weighted_score': best_result.get('score', 0.0)
                }
            else:
                print(f"  No valid model found for regime {regime_id}")
                return None
                
        except Exception as e:
            print(f"  Error finding best model for regime {regime_id}: {e}")
            return None
    
    def apply_weights_to_trading_day(self, weight_csv_path: str, trading_day: str) -> None:
        """
        Apply optimized weights to find best models for all regimes on a trading day.
        
        Args:
            weight_csv_path: Path to the weight_optimized.csv file
            trading_day: Trading day to apply weights to (YYYYMMDD)
        """
        print(f"Applying optimized weights to trading day {trading_day}")
        print(f"Weight file: {weight_csv_path}")
        
        # Validate trading day format
        if len(trading_day) != 8 or not trading_day.isdigit():
            raise ValueError("Trading day must be in YYYYMMDD format")
        
        # Load optimized weights
        regime_weights = self.load_optimized_weights(weight_csv_path)
        
        # Results storage
        results = []
        
        # Process each regime
        for regime_id in sorted(regime_weights.keys()):
            print(f"\nProcessing regime {regime_id}...")
            
            weights = regime_weights[regime_id]
            
            # Find best model for this regime
            best_model = self.find_best_model_for_regime(trading_day, regime_id, weights)
            
            if best_model:
                results.append(best_model)
                print(f"  Best model for regime {regime_id}: Model {best_model['model_id']}, "
                      f"direction: {best_model['direction']}, threshold: {best_model['threshold']}")
            else:
                # Add a default entry for regimes with no valid models
                results.append({
                    'regime': regime_id,
                    'model_id': 0,
                    'direction': 'up',
                    'threshold': 0.0,
                    'weighted_score': 0.0
                })
                print(f"  No valid model found for regime {regime_id}, using defaults")
        
        # Save results to CSV
        self.save_results(results, trading_day)
    
    def save_results(self, results: List[Dict], trading_day: str) -> None:
        """
        Save model picking results to CSV file.
        
        Args:
            results: List of result dictionaries
            trading_day: Trading day (YYYYMMDD)
        """
        if not results:
            print("No results to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by regime
        df = df.sort_values('regime')
        
        # Save to CSV
        output_filename = f"model_trading_model_pick_{trading_day}.csv"
        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_filename}")
            
            # Print summary
            print(f"\nSummary for trading day {trading_day}:")
            print("="*50)
            for _, row in df.iterrows():
                if row['model_id'] > 0:
                    print(f"Regime {row['regime']:1d}: Model {row['model_id']:3d} | "
                          f"Direction: {row['direction']:>4s} | "
                          f"Threshold: {row['threshold']:6.3f} | "
                          f"Score: {row['weighted_score']:6.3f}")
                else:
                    print(f"Regime {row['regime']:1d}: No valid model found")
            print("="*50)
            
        except Exception as e:
            raise Exception(f"Error saving results: {e}")


def main():
    """
    Main function to parse command line arguments and apply weights.
    """
    if len(sys.argv) != 3:
        print("Usage: python model_trading_picker.py <weight_optimized_csv> <trading_day>")
        print("Examples:")
        print("  python model_trading_picker.py ../model_trading/weight_optimized.csv 20250808")
        print("  python model_trading_picker.py /path/to/weight_optimized.csv 20240615")
        sys.exit(1)
    
    try:
        weight_csv_path = sys.argv[1]
        trading_day = sys.argv[2]
        
        # Validate inputs
        if not os.path.exists(weight_csv_path):
            raise ValueError(f"Weight CSV file not found: {weight_csv_path}")
        
        if len(trading_day) != 8 or not trading_day.isdigit():
            raise ValueError("Trading day must be in YYYYMMDD format")
        
    except ValueError as e:
        print(f"Error: Invalid arguments - {e}")
        sys.exit(1)
    
    print("=" * 80)
    print("MODEL TRADING PICKER - OPTIMIZED WEIGHT APPLICATION")
    print("=" * 80)
    print(f"Weight CSV file:  {weight_csv_path}")
    print(f"Trading Day:      {trading_day}")
    print("=" * 80)
    
    try:
        # Initialize picker
        picker = ModelTradingPicker()
        
        # Apply weights to find best models
        picker.apply_weights_to_trading_day(weight_csv_path, trading_day)
        
        print("\n" + "=" * 80)
        print("MODEL PICKING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
