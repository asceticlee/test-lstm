#!/usr/bin/env python3
"""
Debug script for regime prediction forecaster to test model selection
"""
import sys
import os
sys.path.append('/home/stephen/projects/Testing/TestPy/test-lstm/src')

from regime_prediction_forecaster import RegimePredictionForecaster

def test_model_selection():
    """Test the model selection for different regimes"""
    print("Testing model selection...")
    
    # Create forecaster
    forecaster = RegimePredictionForecaster()
    
    # Load data
    try:
        forecaster.load_data()
        forecaster.load_best_models_data('daily')
        forecaster.prepare_regime_history()
        
        # Test predictions for a specific date
        test_date = 20200106
        print(f"\nTesting model selection for date {test_date}")
        
        # Test each regime
        for regime in [0, 1, 2, 3, 4]:
            print(f"\n--- Regime {regime} ---")
            
            # Test historical selection
            print("Historical selection:")
            best_up_hist, best_down_hist = forecaster.get_best_models(regime, test_date, 'historical')
            if best_up_hist and best_down_hist:
                print(f"  Upside: Model {best_up_hist['model_id']}, Rank {best_up_hist['model_regime_rank']}")
                print(f"  Downside: Model {best_down_hist['model_id']}, Rank {best_down_hist['model_regime_rank']}")
            
            # Test daily selection
            print("Daily selection:")
            best_up_daily, best_down_daily = forecaster.get_best_models(regime, test_date, 'daily')
            if best_up_daily and best_down_daily:
                print(f"  Upside: Model {best_up_daily['model_id']}, Method: {best_up_daily.get('selection_method', 'unknown')}")
                print(f"  Downside: Model {best_down_daily['model_id']}, Method: {best_down_daily.get('selection_method', 'unknown')}")
                if 'reference_date' in best_up_daily:
                    print(f"  Reference date: {best_up_daily['reference_date']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_selection()
