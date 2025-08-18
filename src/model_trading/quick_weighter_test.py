#!/usr/bin/env python3
"""
Quick FastModelTradingWeighter Test Script

This script performs a quick test of the FastModelTradingWeighter on a small subset
of trading days and regimes to verify it's working correctly before running the
comprehensive test.

Author: AI Assistant 
Date: Created for test-lstm project
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple
import time

# Add src directory to Python path
current_path = Path(__file__).parent.absolute()
src_path = current_path.parent
sys.path.append(str(src_path))

# Import the FastModelTradingWeighter
from model_trading.fast_model_trading_weighter import FastModelTradingWeighter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickWeighterTest:
    """
    Quick test runner for FastModelTradingWeighter on a small sample of data.
    """
    
    def __init__(self, base_path: str = None):
        """Initialize the quick test."""
        if base_path is None:
            base_path = "/home/stephen/projects/Testing/TestPy/test-lstm"
        
        self.base_path = Path(base_path)
        
        # Initialize the weighter
        self.weighter = FastModelTradingWeighter(base_path)
        
        # Create test weighting strategies
        self.accuracy_weights = self._create_accuracy_weights()
        self.pnl_weights = self._create_pnl_weights()
        
        logger.info("Quick Weighter Test initialized")
        logger.info(f"Base path: {self.base_path}")
        
    def _create_accuracy_weights(self) -> List[float]:
        """Create accuracy-emphasized weighting array."""
        # Higher weights for accuracy metrics
        weights = []
        for i in range(64):  # 8 periods × 2 directions × 2 metrics × 2 threshold scenarios
            if i % 4 < 2:  # Accuracy metrics get higher weight
                weights.append(0.8)
            else:  # PnL metrics get lower weight
                weights.append(0.2)
        
        # Normalize to sum to 1
        weights_sum = sum(weights)
        weights = [w / weights_sum for w in weights]
        
        logger.info(f"Accuracy-emphasized weights created: sum = {sum(weights):.6f}")
        return weights
        
    def _create_pnl_weights(self) -> List[float]:
        """Create PnL-emphasized weighting array."""
        # Higher weights for PnL metrics
        weights = []
        for i in range(64):  # 8 periods × 2 directions × 2 metrics × 2 threshold scenarios
            if i % 4 >= 2:  # PnL metrics get higher weight
                weights.append(0.8)
            else:  # Accuracy metrics get lower weight
                weights.append(0.2)
        
        # Normalize to sum to 1
        weights_sum = sum(weights)
        weights = [w / weights_sum for w in weights]
        
        logger.info(f"PnL-emphasized weights created: sum = {sum(weights):.6f}")
        return weights
    
    def run_quick_test(self) -> bool:
        """Run a quick test on a small sample of trading days and regimes."""
        logger.info("=== STARTING QUICK WEIGHTER TEST ===")
        
        # Test on a small sample
        test_trading_days = ['20250110', '20250113', '20250114', '20250115', '20250116']
        test_regimes = [0, 1, 2]
        
        results = []
        
        start_time = time.time()
        total_tests = len(test_trading_days) * len(test_regimes) * 2  # 2 weighting strategies
        current_test = 0
        
        for trading_day in test_trading_days:
            for regime in test_regimes:
                current_test += 1
                
                logger.info(f"Testing day {trading_day}, regime {regime} ({current_test}/{total_tests})")
                
                # Test accuracy-emphasized strategy
                try:
                    model_id_1, direction_1, threshold_1 = self.weighter.weight_and_select_model_fast(
                        trading_day=trading_day,
                        regime_id=regime,
                        weighting_array=self.accuracy_weights
                    )
                    
                    results.append({
                        'trading_day': trading_day,
                        'regime': regime,
                        'strategy': 'accuracy-emphasized',
                        'model_id': model_id_1,
                        'direction': direction_1,
                        'threshold': threshold_1,
                        'status': 'SUCCESS'
                    })
                    
                    logger.info(f"  Accuracy strategy: Model {model_id_1}, {direction_1}, {threshold_1}")
                    
                except Exception as e:
                    logger.error(f"  Accuracy strategy failed: {e}")
                    results.append({
                        'trading_day': trading_day,
                        'regime': regime,
                        'strategy': 'accuracy-emphasized',
                        'model_id': 'ERROR',
                        'direction': 'ERROR',
                        'threshold': 'ERROR',
                        'status': f'ERROR: {e}'
                    })
                
                # Test PnL-emphasized strategy
                try:
                    model_id_2, direction_2, threshold_2 = self.weighter.weight_and_select_model_fast(
                        trading_day=trading_day,
                        regime_id=regime,
                        weighting_array=self.pnl_weights
                    )
                    
                    results.append({
                        'trading_day': trading_day,
                        'regime': regime,
                        'strategy': 'pnl-emphasized',
                        'model_id': model_id_2,
                        'direction': direction_2,
                        'threshold': threshold_2,
                        'status': 'SUCCESS'
                    })
                    
                    logger.info(f"  PnL strategy: Model {model_id_2}, {direction_2}, {threshold_2}")
                    
                except Exception as e:
                    logger.error(f"  PnL strategy failed: {e}")
                    results.append({
                        'trading_day': trading_day,
                        'regime': regime,
                        'strategy': 'pnl-emphasized',
                        'model_id': 'ERROR',
                        'direction': 'ERROR',
                        'threshold': 'ERROR',
                        'status': f'ERROR: {e}'
                    })
        
        elapsed_time = time.time() - start_time
        
        # Save results
        results_df = pd.DataFrame(results)
        output_path = self.base_path / "test_results" / "quick_weighter_test_results.csv"
        output_path.parent.mkdir(exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        # Print summary
        logger.info("=== QUICK TEST SUMMARY ===")
        logger.info(f"Total tests: {len(results)}")
        success_count = len(results_df[results_df['status'] == 'SUCCESS'])
        error_count = len(results_df[results_df['status'] != 'SUCCESS'])
        logger.info(f"Successful tests: {success_count}")
        logger.info(f"Failed tests: {error_count}")
        logger.info(f"Success rate: {success_count/len(results)*100:.1f}%")
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Average time per test: {elapsed_time/len(results):.3f} seconds")
        logger.info(f"Results saved to: {output_path}")
        
        # Show sample results
        if success_count > 0:
            logger.info("Sample successful results:")
            sample_success = results_df[results_df['status'] == 'SUCCESS'].head(3)
            for _, row in sample_success.iterrows():
                logger.info(f"  {row['trading_day']}, regime {row['regime']}, {row['strategy']}: "
                           f"Model {row['model_id']}, {row['direction']}, {row['threshold']}")
        
        return success_count > 0


def main():
    """Main function to run the quick test."""
    print("🚀 Starting Quick FastModelTradingWeighter Test...")
    print("=" * 60)
    
    try:
        # Run the quick test
        test_runner = QuickWeighterTest()
        success = test_runner.run_quick_test()
        
        if success:
            print("=" * 60)
            print("✅ QUICK TEST COMPLETED SUCCESSFULLY!")
            print("The FastModelTradingWeighter is working correctly.")
            print("You can now run the comprehensive test with confidence.")
        else:
            print("=" * 60)
            print("❌ QUICK TEST FAILED!")
            print("Please check the errors and fix them before running the comprehensive test.")
            return 1
            
    except Exception as e:
        logger.error(f"Quick test failed with exception: {e}")
        print("=" * 60)
        print("❌ QUICK TEST FAILED WITH EXCEPTION!")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
