#!/usr/bin/env python3
"""
Test Script for FastModelTradingWeighter - Execute Comprehensive Test

This script runs the comprehensive test by importing and executing the 
ComprehensiveWeighterTest class from comprehensive_weighter_test.py.

Usage:
    python test_model_trading_weighter.py

Author: AI Assistant  
Date: Created for test-lstm project
"""

import time
import pandas as pd
from comprehensive_weighter_test import ComprehensiveWeighterTest

def main():
    """Main execution function."""
    print("🚀 Starting PARALLEL Comprehensive FastModelTradingWeighter Test...")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Initialize the comprehensive test runner
        test_runner = ComprehensiveWeighterTest()
        
        # Run the full test with parallel processing (much faster!)
        output_file_1, output_file_2 = test_runner.run_full_test(
            use_parallel=True,    # Use parallel processing for speed
            max_workers=None      # Auto-detect optimal worker count
        )
        
        execution_time = time.time() - start_time
        
        # Display results
        print("\n" + "=" * 80)
        print("🎉 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📁 Results saved to:")
        print(f"   📊 Accuracy-emphasized weighting: {output_file_1}")
        print(f"   💰 PnL-emphasized weighting: {output_file_2}")
        print(f"\n⚡ Total execution time: {execution_time/60:.2f} minutes")
        
        # Read and display summary
        df1 = pd.read_csv(output_file_1)
        df2 = pd.read_csv(output_file_2)
        
        print(f"\nTest Summary:")
        print(f"   • Tested {df1['trading_day'].nunique()} trading days")
        print(f"   • Tested {df1['regime'].nunique()} market regimes")
        print(f"   • Evaluated {df1['trading_model'].nunique()} models")
        print(f"   • Generated {len(df1)} result records per weighting strategy")
        
        print(f"\nOutput files contain columns:")
        print(f"   • trading_day: Date of the trading session")
        print(f"   • regime: Market regime (0-4)")
        print(f"   • actual_regime: The actual regime that occurred on that day")
        print(f"   • trading_model: Selected model ID")
        print(f"   • threshold: Selected confidence threshold")
        print(f"   • weighted_score: The weighted performance score")
        print(f"   • weighting_strategy: Strategy used (accuracy-emphasized or pnl-emphasized)")
        
        return 0
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n❌ Test failed after {execution_time/60:.2f} minutes")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())