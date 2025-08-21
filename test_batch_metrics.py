#!/usr/bin/env python3
"""
Test the batch vectorized method with show_metrics=True
"""

import sys
import os
sys.path.append('src')
from model_trading.model_trading_weighter import ModelTradingWeighter
import numpy as np

def test_batch_metrics():
    print('Testing batch vectorized with show_metrics...')
    weighter = ModelTradingWeighter()
    dummy_weights = np.ones(76)

    try:
        result = weighter.get_best_trading_model_batch_vectorized(
            trading_day='20200102',
            market_regime=1,
            weighting_arrays=[dummy_weights],
            show_metrics=True
        )
        
        print('SUCCESS: Got result')
        print('Result keys:', list(result[0].keys()))
        print('Has metrics_breakdown:', 'metrics_breakdown' in result[0])
        
        if 'metrics_breakdown' in result[0]:
            metrics_breakdown = result[0]['metrics_breakdown']
            print('Metrics breakdown keys:', list(metrics_breakdown.keys()))
            print('Number of metrics:', len(metrics_breakdown['metrics']))
            print('First 5 column names:')
            for i, m in enumerate(metrics_breakdown['metrics'][:5]):
                print(f"  {i+1}. {m['column_name']}")
            print('Last 5 column names:')
            for i, m in enumerate(metrics_breakdown['metrics'][-5:]):
                print(f"  {len(metrics_breakdown['metrics'])-4+i}. {m['column_name']}")
            return True
        else:
            print('ERROR: No metrics_breakdown found')
            return False
        
    except Exception as e:
        print('ERROR:', e)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_batch_metrics()
    print('Test result:', 'PASS' if success else 'FAIL')
