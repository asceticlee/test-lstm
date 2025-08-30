#!/usr/bin/env python3
"""
Model Trading Picker Details

This script applies optimized weights from weight_optimized.csv to evaluate ALL trading models
for each market regime on a specific trading day, providing detailed results.

The script:
1. Loads optimized weights for each regime from the CSV file
2. Loads daily performance and daily regime performance data for the trading day
3. Applies weights to evaluate ALL model, direction, and threshold combinations for each regime
4. Outputs detailed results sorted by regime, rank, with all coefficients

Usage:
    python model_trading_picker_details.py <weight_optimized_csv> <trading_day>
    
Examples:
    python model_trading_picker_details.py ../model_trading/weight_optimized.csv 20250808
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add current directory to path to import model_trading_weighter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_trading.model_trading_weighter import ModelTradingWeighter


class ModelTradingPickerDetails:
    """
    Applies optimized weights to evaluate ALL trading models for each regime with detailed output.
    """
    
    def __init__(self, project_root: str = None):
        """
        Initialize the detailed model picker.
        
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
    
    def get_all_models_for_regime(self, trading_day: str, regime_id: int, 
                                 weights: np.ndarray) -> List[Dict]:
        """
        Get ALL models for a specific regime with their weighted scores and coefficients.
        
        Args:
            trading_day: Trading day (YYYYMMDD)
            regime_id: Market regime ID
            weights: Optimized weight array for this regime
            
        Returns:
            List of dictionaries with all model info, sorted by score descending
        """
        try:
            # Use the weighter's new vectorized method that returns ALL combinations
            results = self.weighter.get_all_trading_model_scores_vectorized(
                trading_day=trading_day,
                market_regime=regime_id,
                weighting_array=weights
            )
            
            print(f"  Evaluated {len(results)} model combinations for regime {regime_id}")
            return results
            
        except Exception as e:
            print(f"  Error getting models for regime {regime_id}: {e}")
            return []
    
    def apply_weights_to_trading_day_detailed(self, weight_csv_path: str, trading_day: str) -> None:
        """
        Apply optimized weights to evaluate ALL models for all regimes on a trading day.
        
        Args:
            weight_csv_path: Path to the weight_optimized.csv file
            trading_day: Trading day to apply weights to (YYYYMMDD)
        """
        print(f"Applying optimized weights to ALL models for trading day {trading_day}")
        print(f"Weight file: {weight_csv_path}")
        
        # Validate trading day format
        if len(trading_day) != 8 or not trading_day.isdigit():
            raise ValueError("Trading day must be in YYYYMMDD format")
        
        # Load optimized weights
        regime_weights = self.load_optimized_weights(weight_csv_path)
        
        # Results storage
        all_results = []
        
        # Process each regime
        for regime_id in sorted(regime_weights.keys()):
            print(f"\nProcessing regime {regime_id}...")
            
            weights = regime_weights[regime_id]
            
            # Get all models for this regime
            regime_results = self.get_all_models_for_regime(trading_day, regime_id, weights)
            
            if regime_results:
                all_results.extend(regime_results)
                print(f"  Added {len(regime_results)} model combinations for regime {regime_id}")
                print(f"  Best model: Model {regime_results[0]['model_id']}, "
                      f"direction: {regime_results[0]['direction']}, "
                      f"threshold: {regime_results[0]['threshold']}, "
                      f"score: {regime_results[0]['weighted_score']:.6f}")
            else:
                print(f"  No valid models found for regime {regime_id}")
        
        # Save detailed results to CSV
        self.save_detailed_results(all_results, trading_day)
    
    def save_detailed_results(self, results: List[Dict], trading_day: str) -> None:
        """
        Save detailed model evaluation results to CSV file.
        
        Args:
            results: List of result dictionaries with all model details
            trading_day: Trading day (YYYYMMDD)
        """
        if not results:
            print("No results to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by regime, then by rank within regime
        df = df.sort_values(['regime', 'rank_in_regime'])
        
        # Reorder columns for better readability
        base_columns = ['regime', 'rank_in_regime', 'model_id', 'direction', 'threshold', 'weighted_score']
        coeff_columns = [col for col in df.columns if col.startswith('coeff_')]
        coeff_columns.sort()  # Sort coefficient columns numerically
        
        column_order = base_columns + coeff_columns
        df = df[column_order]
        
        # Save to CSV
        output_filename = f"model_trading_model_pick_details_{trading_day}.csv"
        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            df.to_csv(output_path, index=False)
            print(f"\nDetailed results saved to: {output_filename}")
            
            # Print summary
            print(f"\nSummary for trading day {trading_day}:")
            print("="*80)
            
            for regime_id in sorted(df['regime'].unique()):
                regime_df = df[df['regime'] == regime_id]
                if len(regime_df) > 0:
                    best_row = regime_df.iloc[0]  # First row is best (highest score)
                    print(f"Regime {regime_id}: {len(regime_df):4d} combinations | "
                          f"Best: Model {best_row['model_id']:3d} | "
                          f"Direction: {best_row['direction']:>4s} | "
                          f"Threshold: {best_row['threshold']:6.3f} | "
                          f"Score: {best_row['weighted_score']:8.6f}")
                else:
                    print(f"Regime {regime_id}: No valid combinations found")
            
            print("="*80)
            print(f"Total combinations evaluated: {len(df)}")
            print(f"Coefficients per combination: {len(coeff_columns)}")
            
        except Exception as e:
            raise Exception(f"Error saving detailed results: {e}")


def main():
    """
    Main function to parse command line arguments and generate detailed results.
    """
    if len(sys.argv) != 3:
        print("Usage: python model_trading_picker_details.py <weight_optimized_csv> <trading_day>")
        print("Examples:")
        print("  python model_trading_picker_details.py ../model_trading/weight_optimized.csv 20250808")
        print("  python model_trading_picker_details.py /path/to/weight_optimized.csv 20240615")
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
    print("MODEL TRADING PICKER DETAILS - COMPREHENSIVE EVALUATION")
    print("=" * 80)
    print(f"Weight CSV file:  {weight_csv_path}")
    print(f"Trading Day:      {trading_day}")
    print("=" * 80)
    
    try:
        # Initialize detailed picker
        picker = ModelTradingPickerDetails()
        
        # Apply weights to evaluate all models
        picker.apply_weights_to_trading_day_detailed(weight_csv_path, trading_day)
        
        print("\n" + "=" * 80)
        print("DETAILED MODEL EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()