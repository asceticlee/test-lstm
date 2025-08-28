#!/usr/bin/env python3
"""
Benchmark script to compare parallel vs sequential performance for genetic algorithm optimization.

This script runs the same optimization with and without parallel processing to measure the actual speedup.
"""

import subprocess
import time
import sys
import os

def run_benchmark(from_day, to_day, regime, pop_size, generations=1):
    """
    Run benchmark comparing parallel vs sequential performance.
    
    Args:
        from_day: Start trading day (YYYYMMDD)
        to_day: End trading day (YYYYMMDD)  
        regime: Market regime (0-4)
        pop_size: Population size
        generations: Number of generations (default 1 for quick benchmark)
    """
    
    script_path = os.path.join(os.path.dirname(__file__), "src", "trading_model_regime_weight_optimizer.py")
    
    print("=" * 80)
    print("GENETIC ALGORITHM PARALLELIZATION BENCHMARK")
    print("=" * 80)
    print(f"Trading Days: {from_day} to {to_day}")
    print(f"Market Regime: {regime}")
    print(f"Population Size: {pop_size}")
    print(f"Generations: {generations}")
    print("=" * 80)
    
    # Test 1: Sequential (no parallel processing)
    print("\n🔄 Running SEQUENTIAL test...")
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, script_path,
            from_day, to_day, str(regime), str(pop_size), str(generations), "--no-parallel"
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        sequential_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Sequential completed in {sequential_time:.2f} seconds")
        else:
            print(f"❌ Sequential failed: {result.stderr}")
            return
            
    except subprocess.TimeoutExpired:
        print("❌ Sequential timed out after 1 hour")
        return
    except Exception as e:
        print(f"❌ Sequential error: {e}")
        return
    
    # Test 2: Parallel processing
    print("\n⚡ Running PARALLEL test...")
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, script_path,
            from_day, to_day, str(regime), str(pop_size), str(generations)
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        parallel_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Parallel completed in {parallel_time:.2f} seconds")
        else:
            print(f"❌ Parallel failed: {result.stderr}")
            return
            
    except subprocess.TimeoutExpired:
        print("❌ Parallel timed out after 1 hour")
        return
    except Exception as e:
        print(f"❌ Parallel error: {e}")
        return
    
    # Calculate speedup
    speedup = sequential_time / parallel_time
    efficiency = speedup / os.cpu_count() * 100  # Efficiency as percentage
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Sequential Time:  {sequential_time:.2f} seconds")
    print(f"Parallel Time:    {parallel_time:.2f} seconds")
    print(f"Speedup:          {speedup:.2f}x")
    print(f"Efficiency:       {efficiency:.1f}% (across {os.cpu_count()} CPU cores)")
    
    if speedup > 1.5:
        print("🚀 EXCELLENT: Parallel processing provides significant speedup!")
    elif speedup > 1.1:
        print("✅ GOOD: Parallel processing provides moderate speedup")
    elif speedup > 0.9:
        print("⚠️  NEUTRAL: Parallel processing has minimal impact")
    else:
        print("❌ POOR: Parallel processing is slower than sequential")
        print("   This may indicate overhead issues or insufficient workload size")
    
    print("=" * 80)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if speedup < 1.2:
        print("- Consider increasing population size or trading day range")
        print("- Parallel processing works best with larger workloads")
        print("- For small datasets, sequential processing may be more efficient")
    else:
        print("- Parallel processing is working well for this workload")
        print("- Consider using parallel mode for production runs")
    
    if efficiency < 30:
        print("- Low efficiency suggests memory/I-O bottlenecks")
        print("- Consider reducing max_workers or optimizing data loading")

def main():
    """
    Main function to run benchmark with command line arguments.
    """
    if len(sys.argv) != 6:
        print("Usage: python benchmark_parallel.py <from_day> <to_day> <regime> <pop_size> <generations>")
        print("Examples:")
        print("  python benchmark_parallel.py 20250701 20250703 3 20 1")
        print("  python benchmark_parallel.py 20240601 20240605 2 50 1")
        sys.exit(1)
    
    try:
        from_day = sys.argv[1]
        to_day = sys.argv[2]
        regime = int(sys.argv[3])
        pop_size = int(sys.argv[4])
        generations = int(sys.argv[5])
        
        # Validate inputs
        if len(from_day) != 8 or not from_day.isdigit():
            raise ValueError("from_day must be in YYYYMMDD format")
        if len(to_day) != 8 or not to_day.isdigit():
            raise ValueError("to_day must be in YYYYMMDD format")
        if regime < 0 or regime > 4:
            raise ValueError("regime must be between 0 and 4")
        if pop_size < 4:
            raise ValueError("pop_size must be >= 4")
        if generations < 1:
            raise ValueError("generations must be >= 1")
            
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    run_benchmark(from_day, to_day, regime, pop_size, generations)

if __name__ == "__main__":
    main()
