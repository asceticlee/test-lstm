#!/usr/bin/env python3
"""
GPU Performance Test for Model Trading Weighter

Tests the performance improvements with GPU acceleration using CuPy.
"""

import sys
import os
import numpy as np
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from model_trading.model_trading_weighter import ModelTradingWeighter, get_best_trading_model

def test_gpu_performance():
    """
    Compare performance between different modes: standard, fast (CPU parallel), and GPU.
    """
    print("=" * 80)
    print("🚀 GPU PERFORMANCE TEST - Model Trading Weighter")
    print("=" * 80)
    
    # Test parameters
    trading_day = "20250707"
    market_regime = 1
    
    # Create random weighting array (76 elements)
    np.random.seed(42)  # For reproducible results
    weights = np.random.randn(76)
    
    print(f"Test configuration:")
    print(f"  Trading day: {trading_day}")
    print(f"  Market regime: {market_regime}")
    print(f"  Weighting array: {len(weights)} elements")
    print(f"  Random seed: 42 (for reproducibility)")
    
    weighter = ModelTradingWeighter()
    
    results = {}
    
    # Test 1: Standard (single-threaded) mode
    print(f"\n" + "🐌 STANDARD MODE (single-threaded)")
    print("-" * 50)
    try:
        start_time = time.time()
        result_standard = weighter.get_best_trading_model(trading_day, market_regime, weights)
        end_time = time.time()
        
        standard_time = end_time - start_time
        results['standard'] = {
            'time': standard_time,
            'result': result_standard
        }
        
        print(f"✓ Completed in {standard_time:.2f} seconds")
        print(f"  Best model: {result_standard['model_id']}")
        print(f"  Score: {result_standard['score']:.6f}")
        print(f"  Direction: {result_standard['direction']}")
        print(f"  Threshold: {result_standard['threshold']}")
        
    except Exception as e:
        print(f"✗ Standard mode failed: {e}")
        results['standard'] = None
    
    # Test 2: Fast (CPU parallel) mode
    print(f"\n" + "⚡ FAST MODE (CPU parallel)")
    print("-" * 50)
    try:
        start_time = time.time()
        result_fast = weighter.get_best_trading_model_fast(trading_day, market_regime, weights)
        end_time = time.time()
        
        fast_time = end_time - start_time
        results['fast'] = {
            'time': fast_time,
            'result': result_fast
        }
        
        print(f"✓ Completed in {fast_time:.2f} seconds")
        print(f"  Best model: {result_fast['model_id']}")
        print(f"  Score: {result_fast['score']:.6f}")
        print(f"  Direction: {result_fast['direction']}")
        print(f"  Threshold: {result_fast['threshold']}")
        
        if results['standard']:
            speedup = standard_time / fast_time
            print(f"  🚀 Speedup vs standard: {speedup:.2f}x")
        
    except Exception as e:
        print(f"✗ Fast mode failed: {e}")
        results['fast'] = None
    
    # Test 3: GPU mode
    print(f"\n" + "🔥 GPU MODE (CuPy acceleration)")
    print("-" * 50)
    try:
        start_time = time.time()
        result_gpu = weighter.get_best_trading_model_gpu(trading_day, market_regime, weights)
        end_time = time.time()
        
        gpu_time = end_time - start_time
        results['gpu'] = {
            'time': gpu_time,
            'result': result_gpu
        }
        
        print(f"✓ Completed in {gpu_time:.2f} seconds")
        print(f"  Best model: {result_gpu['model_id']}")
        print(f"  Score: {result_gpu['score']:.6f}")
        print(f"  Direction: {result_gpu['direction']}")
        print(f"  Threshold: {result_gpu['threshold']}")
        
        if results['standard']:
            speedup_vs_standard = standard_time / gpu_time
            print(f"  🚀 Speedup vs standard: {speedup_vs_standard:.2f}x")
        
        if results['fast']:
            speedup_vs_fast = fast_time / gpu_time
            print(f"  🚀 Speedup vs fast: {speedup_vs_fast:.2f}x")
        
    except Exception as e:
        print(f"✗ GPU mode failed: {e}")
        results['gpu'] = None
    
    # Performance Summary
    print(f"\n" + "=" * 80)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 80)
    
    times = []
    modes = []
    
    for mode, data in results.items():
        if data:
            times.append(data['time'])
            modes.append(mode)
            print(f"{mode.upper():>10}: {data['time']:>8.2f} seconds")
    
    if len(times) > 1:
        fastest_idx = np.argmin(times)
        fastest_mode = modes[fastest_idx]
        fastest_time = times[fastest_idx]
        
        print(f"\n🏆 Fastest mode: {fastest_mode.upper()} ({fastest_time:.2f} seconds)")
        
        print(f"\n📈 Speedup factors:")
        for i, (mode, time_taken) in enumerate(zip(modes, times)):
            if i != fastest_idx:
                speedup = time_taken / fastest_time
                print(f"  {mode} → {fastest_mode}: {speedup:.2f}x faster")
    
    # Verify consistency
    print(f"\n" + "🔍 RESULT CONSISTENCY CHECK")
    print("-" * 50)
    
    all_results = [data['result'] for data in results.values() if data]
    
    if len(all_results) > 1:
        # Check if all modes found the same best model
        model_ids = [r['model_id'] for r in all_results]
        scores = [r['score'] for r in all_results]
        directions = [r['direction'] for r in all_results]
        thresholds = [r['threshold'] for r in all_results]
        
        if len(set(model_ids)) == 1:
            print("✓ All modes found the same best model")
        else:
            print("⚠️  Different modes found different best models:")
            for mode, result in zip([k for k in results.keys() if results[k]], all_results):
                print(f"  {mode}: model {result['model_id']}")
        
        print(f"Score range: {min(scores):.6f} to {max(scores):.6f}")
        
    return results

def test_convenience_function():
    """
    Test the convenience function with different modes.
    """
    print(f"\n" + "=" * 80)
    print("🛠️  CONVENIENCE FUNCTION TEST")
    print("=" * 80)
    
    trading_day = "20250707"
    market_regime = 1
    weights = np.random.randn(76)
    
    modes = ['standard', 'fast', 'gpu']
    
    for mode in modes:
        print(f"\n🧪 Testing mode: {mode}")
        try:
            start_time = time.time()
            result = get_best_trading_model(trading_day, market_regime, weights, mode=mode)
            end_time = time.time()
            
            print(f"✓ {mode} mode: {end_time - start_time:.2f}s")
            print(f"  Model: {result['model_id']}, Score: {result['score']:.4f}")
            
        except Exception as e:
            print(f"✗ {mode} mode failed: {e}")

if __name__ == "__main__":
    try:
        # Check if CuPy is available
        try:
            import cupy as cp
            print("✓ CuPy detected - GPU acceleration available")
            try:
                device = cp.cuda.Device()
                print(f"  GPU Device ID: {device.id}")
                mem_info = cp.cuda.runtime.memGetInfo()
                print(f"  GPU Memory: {mem_info[1] / 1e9:.1f} GB total, {mem_info[0] / 1e9:.1f} GB free")
            except Exception as gpu_info_error:
                print(f"  GPU info unavailable: {gpu_info_error}")
        except ImportError:
            print("⚠️  CuPy not available - GPU tests will fall back to CPU")
        
        # Run performance tests
        results = test_gpu_performance()
        
        # Test convenience function
        test_convenience_function()
        
        print(f"\n" + "=" * 80)
        print("🎉 GPU PERFORMANCE TEST COMPLETE")
        print("=" * 80)
        print("The GPU-accelerated model trading weighter is ready for high-speed trading!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
