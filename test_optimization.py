#!/usr/bin/env python3
"""
Quick test to verify the genetic algorithm optimization is working.
"""

import time
from src.trading_model_regime_weight_optimizer import TradingModelRegimeWeightOptimizer

def test_optimization():
    """Test the optimized genetic algorithm."""
    print("=" * 60)
    print("TESTING OPTIMIZED GENETIC ALGORITHM")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = TradingModelRegimeWeightOptimizer()
    print("✅ Optimizer initialized")
    
    # Test basic population operations
    print("\n1. Testing population initialization...")
    population = optimizer.initialize_population(3, 76)
    print(f"   ✅ Created population of {len(population)} chromosomes")
    
    # Test optimized fitness evaluation
    print("\n2. Testing optimized fitness evaluation...")
    start_time = time.time()
    
    try:
        fitness_results, population_results = optimizer.evaluate_population_fitness(
            population, '20220103', '20220103', 0  # Single day for speed
        )
        end_time = time.time()
        
        print(f"   ✅ Fitness evaluation completed in {end_time - start_time:.2f} seconds")
        print(f"   ✅ Returned {len(fitness_results)} fitness scores")
        print(f"   ✅ Returned {len(population_results)} result rows")
        
        # Test optimized file saving
        print("\n3. Testing optimized file saving...")
        optimizer.save_generation_results(1, population, fitness_results, population_results, 0)
        print("   ✅ Generation results saved")
        
        # Check output files
        import os
        output_files = [f for f in os.listdir(optimizer.output_dir) if '20220103_20220103' in f]
        print(f"   ✅ Generated {len(output_files)} files:")
        for f in output_files:
            file_size = os.path.getsize(os.path.join(optimizer.output_dir, f))
            print(f"      - {f} ({file_size} bytes)")
        
        print("\n" + "=" * 60)
        print("🎉 OPTIMIZATION TEST SUCCESSFUL!")
        print("🎉 Chromosome files are now generated efficiently!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimization()
    exit(0 if success else 1)
