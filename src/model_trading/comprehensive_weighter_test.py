#!/usr/bin/env python3
"""
Comprehensive FastModelTradingWeighter Test Script

This script performs a full test of the FastModelTradingWeighter on all trading days,
all regimes, and all available models with two different weighting arrays.

The script will:
1. Load regime assignments from daily_regime_assignments.csv
2. Load regime characteristics from regime_characteristics.csv  
3. Loop through all unique trading days
4. For each trading day, loop through all available regimes
5. For each regime, test all available models
6. Apply two different weighting arrays and record the selected model and threshold
7. Generate two result CSV files with columns: trading_day, regime, trading_model, threshold

Author: AI Assistant
Date: Created for test-lstm project
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import time
import json
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing

# Add the src directory to the path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent
sys.path.insert(0, str(src_dir))

# Import the FastModelTradingWeighter
from model_trading.fast_model_trading_weighter import FastModelTradingWeighter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_weighter_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveWeighterTest:
    """
    Comprehensive test runner for FastModelTradingWeighter across all trading days, 
    regimes, and models with multiple weighting arrays.
    """
    
    def __init__(self, base_path: str = None):
        """Initialize the comprehensive test."""
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
        self.regime_assignments_path = self.base_path / "market_regime" / "gmm" / "daily" / "daily_regime_assignments.csv"
        self.regime_characteristics_path = self.base_path / "market_regime" / "gmm" / "daily" / "regime_characteristics.csv"
        self.model_performance_path = self.base_path / "model_performance" / "models_alltime_regime_performance.csv"
        
        # Initialize the weighter
        self.weighter = FastModelTradingWeighter(base_path)
        
        # Define two different weighting arrays for testing
        # Weighting Array 1: Accuracy-emphasized (higher weights on accuracy metrics)
        self.weighting_array_1 = np.array([
            0.4,   # 1day_up_acc weight
            0.1,   # 1day_up_pnl weight  
            0.4,   # 1day_down_acc weight
            0.1,   # 1day_down_pnl weight
            0.3,   # 2day_up_acc weight
            0.1,   # 2day_up_pnl weight
            0.3,   # 2day_down_acc weight
            0.1,   # 2day_down_pnl weight
            0.25,  # 3day_up_acc weight
            0.05,  # 3day_up_pnl weight
            0.25,  # 3day_down_acc weight
            0.05,  # 3day_down_pnl weight
            0.2,   # 4day_up_acc weight
            0.05,  # 4day_up_pnl weight
            0.2,   # 4day_down_acc weight
            0.05,  # 4day_down_pnl weight
            0.15,  # 5day_up_acc weight
            0.05,  # 5day_up_pnl weight
            0.15,  # 5day_down_acc weight
            0.05,  # 5day_down_pnl weight
            0.1,   # 10day_up_acc weight
            0.05,  # 10day_up_pnl weight
            0.1,   # 10day_down_acc weight
            0.05,  # 10day_down_pnl weight
            0.08,  # 20day_up_acc weight
            0.02,  # 20day_up_pnl weight
            0.08,  # 20day_down_acc weight
            0.02,  # 20day_down_pnl weight
            0.06,  # 30day_up_acc weight
            0.02,  # 30day_up_pnl weight
            0.06,  # 30day_down_acc weight
            0.02   # 30day_down_pnl weight
        ])
        
        # Weighting Array 2: PnL-emphasized (higher weights on profit/loss metrics)
        self.weighting_array_2 = np.array([
            0.2,   # 1day_up_acc weight
            0.3,   # 1day_up_pnl weight  
            0.2,   # 1day_down_acc weight
            0.3,   # 1day_down_pnl weight
            0.15,  # 2day_up_acc weight
            0.25,  # 2day_up_pnl weight
            0.15,  # 2day_down_acc weight
            0.25,  # 2day_down_pnl weight
            0.1,   # 3day_up_acc weight
            0.2,   # 3day_up_pnl weight
            0.1,   # 3day_down_acc weight
            0.2,   # 3day_down_pnl weight
            0.08,  # 4day_up_acc weight
            0.17,  # 4day_up_pnl weight
            0.08,  # 4day_down_acc weight
            0.17,  # 4day_down_pnl weight
            0.06,  # 5day_up_acc weight
            0.14,  # 5day_up_pnl weight
            0.06,  # 5day_down_acc weight
            0.14,  # 5day_down_pnl weight
            0.04,  # 10day_up_acc weight
            0.11,  # 10day_up_pnl weight
            0.04,  # 10day_down_acc weight
            0.11,  # 10day_down_pnl weight
            0.03,  # 20day_up_acc weight
            0.07,  # 20day_up_pnl weight
            0.03,  # 20day_down_acc weight
            0.07,  # 20day_down_pnl weight
            0.02,  # 30day_up_acc weight
            0.06,  # 30day_up_pnl weight
            0.02,  # 30day_down_acc weight
            0.06   # 30day_down_pnl weight
        ])
        
        # Normalize weights to sum to 1
        self.weighting_array_1 = self.weighting_array_1 / self.weighting_array_1.sum()
        self.weighting_array_2 = self.weighting_array_2 / self.weighting_array_2.sum()
        
        # Storage for results
        self.results_1 = []  # Results for weighting array 1 (accuracy-emphasized)
        self.results_2 = []  # Results for weighting array 2 (PnL-emphasized)
        
        logger.info(f"Comprehensive Weighter Test initialized")
        logger.info(f"Base path: {self.base_path}")
        logger.info(f"Weighting Array 1 (Accuracy-emphasized): sum = {self.weighting_array_1.sum():.6f}")
        logger.info(f"Weighting Array 2 (PnL-emphasized): sum = {self.weighting_array_2.sum():.6f}")
    
    def load_data(self):
        """Load all required data files."""
        logger.info("Loading data files...")
        
        # Load regime assignments
        if not self.regime_assignments_path.exists():
            raise FileNotFoundError(f"Regime assignments file not found: {self.regime_assignments_path}")
        
        self.regime_assignments = pd.read_csv(self.regime_assignments_path)
        logger.info(f"Loaded {len(self.regime_assignments)} regime assignments")
        
        # Load regime characteristics
        if not self.regime_characteristics_path.exists():
            raise FileNotFoundError(f"Regime characteristics file not found: {self.regime_characteristics_path}")
        
        self.regime_characteristics = pd.read_csv(self.regime_characteristics_path)
        logger.info(f"Loaded {len(self.regime_characteristics)} regime characteristics")
        
        # Load model performance data
        if not self.model_performance_path.exists():
            raise FileNotFoundError(f"Model performance file not found: {self.model_performance_path}")
        
        self.model_performance = pd.read_csv(self.model_performance_path)
        logger.info(f"Loaded {len(self.model_performance)} model performance records")
        
        # Get unique values
        self.unique_trading_days = sorted(self.regime_assignments['trading_day'].unique())
        self.unique_regimes = sorted(self.regime_characteristics['Regime'].unique())
        self.unique_models = sorted(self.model_performance['ModelID'].unique())
        
        logger.info(f"Found {len(self.unique_trading_days)} unique trading days")
        logger.info(f"Found {len(self.unique_regimes)} unique regimes: {self.unique_regimes}")
        logger.info(f"Found {len(self.unique_models)} unique models")

    def _test_single_combination(self, args_tuple):
        """
        Worker function to test a single trading day + regime combination.
        This function will be called in parallel.
        """
        trading_day, regime, actual_regime, base_path = args_tuple
        
        try:
            # Each worker needs its own weighter instance to avoid threading issues
            from model_trading.fast_model_trading_weighter import FastModelTradingWeighter
            worker_weighter = FastModelTradingWeighter(base_path)
            
            # Test with both weighting arrays
            results = []
            
            # Test accuracy-emphasized strategy
            try:
                model_id_1, direction_1, threshold_1 = worker_weighter.weight_and_select_model_fast(
                    trading_day=str(trading_day),
                    regime_id=regime,
                    weighting_array=self.weighting_array_1.tolist()
                )
                
                results.append({
                    'trading_day': trading_day,
                    'regime': regime,
                    'actual_regime': actual_regime,
                    'trading_model': model_id_1,
                    'direction': direction_1,
                    'threshold': threshold_1,
                    'weighted_score': 1.0,
                    'weighting_strategy': 'accuracy-emphasized'
                })
            except Exception as e:
                results.append({
                    'trading_day': trading_day,
                    'regime': regime,
                    'actual_regime': actual_regime,
                    'trading_model': 'ERROR',
                    'direction': 'UNKNOWN',
                    'threshold': '0.0',
                    'weighted_score': 0.0,
                    'weighting_strategy': 'accuracy-emphasized'
                })
            
            # Test PnL-emphasized strategy
            try:
                model_id_2, direction_2, threshold_2 = worker_weighter.weight_and_select_model_fast(
                    trading_day=str(trading_day),
                    regime_id=regime,
                    weighting_array=self.weighting_array_2.tolist()
                )
                
                results.append({
                    'trading_day': trading_day,
                    'regime': regime,
                    'actual_regime': actual_regime,
                    'trading_model': model_id_2,
                    'direction': direction_2,
                    'threshold': threshold_2,
                    'weighted_score': 1.0,
                    'weighting_strategy': 'pnl-emphasized'
                })
            except Exception as e:
                results.append({
                    'trading_day': trading_day,
                    'regime': regime,
                    'actual_regime': actual_regime,
                    'trading_model': 'ERROR',
                    'direction': 'UNKNOWN',
                    'threshold': '0.0',
                    'weighted_score': 0.0,
                    'weighting_strategy': 'pnl-emphasized'
                })
            
            return results
            
        except Exception as e:
            # Return error results for both strategies
            return [
                {
                    'trading_day': trading_day,
                    'regime': regime,
                    'actual_regime': actual_regime,
                    'trading_model': 'ERROR',
                    'direction': 'UNKNOWN',
                    'threshold': '0.0',
                    'weighted_score': 0.0,
                    'weighting_strategy': 'accuracy-emphasized'
                },
                {
                    'trading_day': trading_day,
                    'regime': regime,
                    'actual_regime': actual_regime,
                    'trading_model': 'ERROR',
                    'direction': 'UNKNOWN',
                    'threshold': '0.0',
                    'weighted_score': 0.0,
                    'weighting_strategy': 'pnl-emphasized'
                }
            ]

    def run_comprehensive_test_parallel(self, max_workers=None):
        """
        Run comprehensive test using parallel processing for much faster execution.
        """
        logger.info("=== STARTING PARALLEL COMPREHENSIVE WEIGHTER TEST ===")
        
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 cores max
        
        logger.info(f"Using {max_workers} parallel workers")
        
        # Prepare all test combinations
        test_combinations = []
        for trading_day in self.unique_trading_days:
            # Get actual regime for this day
            day_data = self.regime_assignments[self.regime_assignments['trading_day'] == trading_day]
            if day_data.empty:
                continue
            
            actual_regime = day_data.iloc[0]['Regime']
            
            # Test all regimes for this trading day
            for regime in self.unique_regimes:
                test_combinations.append((trading_day, regime, actual_regime, self.base_path))
        
        logger.info(f"Prepared {len(test_combinations)} test combinations")
        
        # Run tests in parallel
        start_time = time.time()
        self.results_1 = []
        self.results_2 = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_combination = {
                executor.submit(self._test_single_combination, combination): combination 
                for combination in test_combinations
            }
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_combination):
                try:
                    results = future.result()
                    # Separate results by strategy
                    for result in results:
                        if result['weighting_strategy'] == 'accuracy-emphasized':
                            self.results_1.append(result)
                        else:
                            self.results_2.append(result)
                    
                    completed += 1
                    if completed % 100 == 0:
                        elapsed = time.time() - start_time
                        progress = completed / len(test_combinations)
                        eta = (elapsed / progress) - elapsed if progress > 0 else 0
                        logger.info(f"Completed {completed}/{len(test_combinations)} combinations "
                                   f"({progress*100:.1f}%) - ETA: {eta/60:.1f} minutes")
                        
                except Exception as e:
                    logger.error(f"Error processing combination: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"Parallel test completed in {total_time/60:.2f} minutes")
        logger.info(f"Generated {len(self.results_1)} results for accuracy-emphasized weighting")
        logger.info(f"Generated {len(self.results_2)} results for PnL-emphasized weighting")

    def run_comprehensive_test(self):
        """Run the comprehensive test across all trading days, regimes, and models."""
        logger.info("Starting comprehensive weighter test...")
        
        start_time = time.time()
        total_iterations = len(self.unique_trading_days) * len(self.unique_regimes)
        current_iteration = 0
        
        for trading_day in self.unique_trading_days:
            # Get the actual regime for this trading day
            day_data = self.regime_assignments[self.regime_assignments['trading_day'] == trading_day]
            if day_data.empty:
                logger.warning(f"No regime data found for trading day {trading_day}")
                continue
            
            actual_regime = day_data.iloc[0]['Regime']
            
            # Test all regimes for this trading day (not just the actual one)
            for regime in self.unique_regimes:
                current_iteration += 1
                elapsed_time = time.time() - start_time
                avg_time_per_iteration = elapsed_time / current_iteration if current_iteration > 0 else 0
                estimated_total_time = avg_time_per_iteration * total_iterations
                remaining_time = estimated_total_time - elapsed_time
                
                logger.info(f"Processing day {trading_day}, regime {regime} "
                           f"({current_iteration}/{total_iterations}) - "
                           f"ETA: {remaining_time/60:.1f} minutes")
                
                try:
                    # Test with weighting array 1 (accuracy-emphasized)
                    best_model_1, best_threshold_1, best_score_1 = self._test_regime_with_weighting(
                        trading_day, regime, self.weighting_array_1, "accuracy-emphasized"
                    )
                    
                    # Test with weighting array 2 (PnL-emphasized)  
                    best_model_2, best_threshold_2, best_score_2 = self._test_regime_with_weighting(
                        trading_day, regime, self.weighting_array_2, "pnl-emphasized"
                    )
                    
                    # Store results
                    self.results_1.append({
                        'trading_day': trading_day,
                        'regime': regime,
                        'actual_regime': actual_regime,
                        'trading_model': best_model_1,
                        'threshold': best_threshold_1,
                        'weighted_score': best_score_1,
                        'weighting_strategy': 'accuracy-emphasized'
                    })
                    
                    self.results_2.append({
                        'trading_day': trading_day,
                        'regime': regime, 
                        'actual_regime': actual_regime,
                        'trading_model': best_model_2,
                        'threshold': best_threshold_2,
                        'weighted_score': best_score_2,
                        'weighting_strategy': 'pnl-emphasized'
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing day {trading_day}, regime {regime}: {str(e)}")
                    # Add placeholder results to maintain data consistency
                    self.results_1.append({
                        'trading_day': trading_day,
                        'regime': regime,
                        'actual_regime': actual_regime,
                        'trading_model': 'ERROR',
                        'threshold': -1,
                        'weighted_score': -999,
                        'weighting_strategy': 'accuracy-emphasized'
                    })
                    
                    self.results_2.append({
                        'trading_day': trading_day,
                        'regime': regime,
                        'actual_regime': actual_regime, 
                        'trading_model': 'ERROR',
                        'threshold': -1,
                        'weighted_score': -999,
                        'weighting_strategy': 'pnl-emphasized'
                    })
        
        total_time = time.time() - start_time
        logger.info(f"Comprehensive test completed in {total_time/60:.2f} minutes")
        logger.info(f"Generated {len(self.results_1)} results for accuracy-emphasized weighting")
        logger.info(f"Generated {len(self.results_2)} results for PnL-emphasized weighting")
    
    def _test_regime_with_weighting(self, trading_day: int, regime: int, 
                                  weighting_array: np.ndarray, strategy_name: str) -> Tuple[str, str, float]:
        """
        Test a specific regime on a trading day with a given weighting array.
        
        Args:
            trading_day: The trading day to test
            regime: The regime to test  
            weighting_array: The weighting array to use
            strategy_name: Name of the weighting strategy for logging
            
        Returns:
            Tuple of (best_model, best_threshold, best_score)
        """
        try:
            # Use the FastModelTradingWeighter to get the best model and parameters
            model_id, direction, threshold = self.weighter.weight_and_select_model_fast(
                trading_day=str(trading_day),
                regime_id=regime,
                weighting_array=weighting_array.tolist()
            )
            
            return model_id, threshold, 1.0  # Return a default score of 1.0 since the method doesn't return scores
            
        except Exception as e:
            self.logger.error(f"Error in _test_regime_with_weighting for day {trading_day}, regime {regime}, strategy {strategy_name}: {e}")
            return 'ERROR', '0.0', 0.0
            
            return best_model, best_threshold, best_score
            
        except Exception as e:
            logger.error(f"Error in _test_regime_with_weighting for day {trading_day}, "
                        f"regime {regime}, strategy {strategy_name}: {str(e)}")
            return 'ERROR', -1.0, -999.0
    
    def save_results(self):
        """Save the test results to CSV files."""
        logger.info("Saving results to CSV files...")
        
        # Convert results to DataFrames
        results_df_1 = pd.DataFrame(self.results_1)
        results_df_2 = pd.DataFrame(self.results_2)
        
        # Define output paths
        output_dir = self.base_path / "test_results"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_file_1 = output_dir / f"weighter_results_accuracy_emphasized_{timestamp}.csv"
        output_file_2 = output_dir / f"weighter_results_pnl_emphasized_{timestamp}.csv"
        
        # Save to CSV
        results_df_1.to_csv(output_file_1, index=False)
        results_df_2.to_csv(output_file_2, index=False)
        
        logger.info(f"Saved accuracy-emphasized results to: {output_file_1}")
        logger.info(f"Saved PnL-emphasized results to: {output_file_2}")
        
        # Generate summary statistics
        self._generate_summary_stats(results_df_1, results_df_2, output_dir, timestamp)
        
        return output_file_1, output_file_2
    
    def _generate_summary_stats(self, results_df_1: pd.DataFrame, results_df_2: pd.DataFrame, 
                               output_dir: Path, timestamp: str):
        """Generate and save summary statistics."""
        logger.info("Generating summary statistics...")
        
        summary_stats = {
            'test_timestamp': timestamp,
            'total_test_combinations': len(results_df_1),
            'unique_trading_days': len(self.unique_trading_days),
            'unique_regimes': len(self.unique_regimes),
            'unique_models': len(self.unique_models),
            
            # Accuracy-emphasized results
            'accuracy_emphasized': {
                'unique_selected_models': results_df_1['trading_model'].nunique(),
                'most_selected_model': results_df_1['trading_model'].mode().iloc[0] if not results_df_1.empty else 'N/A',
                'most_selected_model_count': results_df_1['trading_model'].value_counts().iloc[0] if not results_df_1.empty else 0,
                'average_threshold': results_df_1[results_df_1['threshold'] >= 0]['threshold'].mean(),
                'threshold_distribution': results_df_1[results_df_1['threshold'] >= 0]['threshold'].value_counts().to_dict(),
                'average_weighted_score': results_df_1[results_df_1['weighted_score'] > -999]['weighted_score'].mean(),
                'error_count': len(results_df_1[results_df_1['trading_model'] == 'ERROR'])
            },
            
            # PnL-emphasized results  
            'pnl_emphasized': {
                'unique_selected_models': results_df_2['trading_model'].nunique(),
                'most_selected_model': results_df_2['trading_model'].mode().iloc[0] if not results_df_2.empty else 'N/A',
                'most_selected_model_count': results_df_2['trading_model'].value_counts().iloc[0] if not results_df_2.empty else 0,
                'average_threshold': results_df_2[results_df_2['threshold'] >= 0]['threshold'].mean(),
                'threshold_distribution': results_df_2[results_df_2['threshold'] >= 0]['threshold'].value_counts().to_dict(),
                'average_weighted_score': results_df_2[results_df_2['weighted_score'] > -999]['weighted_score'].mean(),
                'error_count': len(results_df_2[results_df_2['trading_model'] == 'ERROR'])
            }
        }
        
        # Save summary stats
        import json
        summary_file = output_dir / f"weighter_test_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        logger.info(f"Saved summary statistics to: {summary_file}")
        
        # Print key statistics
        logger.info("=== SUMMARY STATISTICS ===")
        logger.info(f"Total test combinations: {summary_stats['total_test_combinations']}")
        logger.info(f"Accuracy-emphasized - Most selected model: {summary_stats['accuracy_emphasized']['most_selected_model']} "
                   f"({summary_stats['accuracy_emphasized']['most_selected_model_count']} times)")
        logger.info(f"PnL-emphasized - Most selected model: {summary_stats['pnl_emphasized']['most_selected_model']} "
                   f"({summary_stats['pnl_emphasized']['most_selected_model_count']} times)")
        logger.info(f"Average threshold (accuracy): {summary_stats['accuracy_emphasized']['average_threshold']:.3f}")
        logger.info(f"Average threshold (PnL): {summary_stats['pnl_emphasized']['average_threshold']:.3f}")
    
    def run_full_test(self, use_parallel=True, max_workers=None):
        """Run the complete comprehensive test."""
        logger.info("=== STARTING COMPREHENSIVE WEIGHTER TEST ===")
        
        try:
            # Load data
            self.load_data()
            
            # Run comprehensive test (parallel by default for speed)
            if use_parallel:
                logger.info("Using PARALLEL processing for maximum speed")
                self.run_comprehensive_test_parallel(max_workers=max_workers)
            else:
                logger.info("Using SEQUENTIAL processing")
                self.run_comprehensive_test()
            
            # Save results
            output_file_1, output_file_2 = self.save_results()
            
            logger.info("=== COMPREHENSIVE TEST COMPLETED SUCCESSFULLY ===")
            logger.info(f"Results saved to:")
            logger.info(f"  Accuracy-emphasized: {output_file_1}")
            logger.info(f"  PnL-emphasized: {output_file_2}")
            
            return output_file_1, output_file_2
            
        except Exception as e:
            logger.error(f"Comprehensive test failed: {str(e)}")
            raise

def main():
    """Main function to run the comprehensive test."""
    try:
        # Initialize and run the test
        test_runner = ComprehensiveWeighterTest()
        output_file_1, output_file_2 = test_runner.run_full_test()
        
        print(f"\n🎉 Comprehensive weighter test completed successfully!")
        print(f"📁 Results saved to:")
        print(f"   📊 Accuracy-emphasized: {output_file_1}")
        print(f"   💰 PnL-emphasized: {output_file_2}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
