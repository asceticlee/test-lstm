#!/usr/bin/env python3
"""
Trading Model Regime Weight Optimizer with Genetic Algorithm

This script optimizes model trading weights for specific market regimes using genetic algorithm by:
1. Initializing population of chromosomes (weight arrays) 
2. Using market regime forecasts to identify regime-specific trading days
3. Evaluating fitness using trading_performance.py composite_score
4. Evolving population through selection, crossover, and mutation
5. Outputting generation-by-generation performance improvements

Each chromosome represents a weight set, each gene is a single weight of the weight set.
After trading evaluation, chromosomes are ranked by composite_score from trading_performance.py.

Usage:
    python trading_model_regime_weight_optimizer.py <from_trading_day> <to_trading_day> <market_regime> <population_size> <generations>
    
Examples:
    python trading_model_regime_weight_optimizer.py 20250701 20250710 3 20 50
    # Optimize for regime 3 from July 1-10, 2025 with population of 20 for 50 generations
"""

import sys
import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Add current directory to path to import model_trading_weighter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_trading.model_trading_weighter import ModelTradingWeighter
from trading_performance import TradingPerformanceAnalyzer


class TradingModelRegimeWeightOptimizer:
    """
    Optimizes trading model weights for specific market regimes using genetic algorithm.
    """
    
    def __init__(self, project_root: str = None):
        """
        Initialize the optimizer.
        
        Args:
            project_root: Root directory of the project. If None, infers from script location.
        """
        if project_root is None:
            # Infer project root from script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
        
        self.project_root = project_root
        self.market_regime_file = os.path.join(project_root, "market_regime", "gmm", "market_regime_forecast.csv")
        self.predictions_dir = os.path.join(project_root, "model_predictions")
        self.output_dir = os.path.join(project_root, "model_trading", "weight_optimizer")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model trading weighter
        self.weighter = ModelTradingWeighter(project_root)
        
        # Initialize trading performance analyzer
        self.performance_analyzer = TradingPerformanceAnalyzer(transaction_fee=0.02)
        
        # Cache for regime forecast data
        self._regime_forecast_cache = None
        
        # Genetic Algorithm Parameters
        self.mutation_rate = 0.1  # 10% chance to mutate each gene
        self.crossover_rate = 0.8  # 80% chance for crossover
        self.elite_size = 2  # Keep top 2 performers
        self.tournament_size = 3  # Tournament selection size
        
        # Dynamic weight column names - will be populated from weighter
        self._weight_column_names = None
        self._first_batch_call = True  # Flag to capture column names on first call
    
    def load_market_regime_forecast(self) -> pd.DataFrame:
        """
        Load market regime forecast data from CSV file.
        
        Returns:
            DataFrame with market regime forecast data
        """
        if self._regime_forecast_cache is not None:
            return self._regime_forecast_cache
        
        if not os.path.exists(self.market_regime_file):
            raise FileNotFoundError(f"Market regime forecast file not found: {self.market_regime_file}")
        
        try:
            # Load the CSV file with proper column names
            df = pd.read_csv(self.market_regime_file)
            
            # Validate required columns exist
            required_columns = ['trading_day', 'ms_of_day', 'predicted_regime']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Market regime forecast file missing required columns: {missing_columns}")
            
            self._regime_forecast_cache = df
            print(f"Loaded {len(df)} market regime forecast records")
            return df
            
        except Exception as e:
            raise Exception(f"Error loading market regime forecast: {e}")
    
    def initialize_population(self, population_size: int, weight_length: int = 76, 
                             weight_range: Tuple[float, float] = (0.0, 1.0)) -> List[np.ndarray]:
        """
        Initialize population of chromosomes (weight arrays) for genetic algorithm.
        
        Args:
            population_size: Number of chromosomes in population
            weight_length: Length of each chromosome (number of genes)
            weight_range: Range for random gene values
            
        Returns:
            List of numpy arrays representing the initial population
        """
        population = []
        min_weight, max_weight = weight_range
        
        for i in range(population_size):
            # Set different random seed for each chromosome
            np.random.seed(42 + i)
            chromosome = np.random.uniform(min_weight, max_weight, weight_length)
            population.append(chromosome)
        
        print(f"Initialized population of {population_size} chromosomes with {weight_length} genes each")
        return population
    
    def evaluate_population_fitness(self, population: List[np.ndarray], from_trading_day: str, 
                                   to_trading_day: str, market_regime: int) -> List[Dict]:
        """
        Evaluate fitness of entire population using optimized daily-outer, chromosome-inner loop.
        
        Args:
            population: List of chromosomes (weight arrays) to evaluate
            from_trading_day: Start trading day
            to_trading_day: End trading day
            market_regime: Target market regime
            
        Returns:
            List of fitness dictionaries for each chromosome
        """
        print(f"Evaluating fitness for {len(population)} chromosomes using optimized daily-outer loop...")
        
        # Get trading days in range
        trading_days = self.get_trading_days_in_range(from_trading_day, to_trading_day)
        
        if len(trading_days) == 0:
            print("No trading days found in range")
            return [self._get_default_fitness() for _ in population]
        
        # Initialize result storage for each chromosome
        population_results = {i: [] for i in range(len(population))}
        
        # OUTER LOOP: Process each trading day (optimized approach)
        for current_day in trading_days:
            print(f"\nProcessing trading day {current_day}")
            
            # Get current day regime rows (read once per day)
            current_day_rows = self.get_regime_rows_for_day(current_day, market_regime)
            
            if len(current_day_rows) == 0:
                print(f"  No regime {market_regime} data for {current_day}, skipping")
                continue
            
            # Get previous trading day for model weighter (read once per day)
            previous_day = self.get_previous_trading_day(current_day)
            if previous_day is None:
                print(f"  No previous trading day found for {current_day}, skipping")
                continue
            
            # Pre-cache model predictions for all potential models (done once per day)
            model_predictions_cache = {}
            
            # INNER LOOP: Process all chromosomes for this trading day using batch operation
            print(f"  Finding best models for all {len(population)} chromosomes using batch operation")
            try:
                # Use batch method to process all chromosomes at once (maximum efficiency!)
                # On first call, use show_metrics=True to capture column names
                use_show_metrics = self._first_batch_call
                
                batch_results = self.weighter.get_best_trading_model_batch_vectorized(
                    trading_day=previous_day,
                    market_regime=market_regime,
                    weighting_arrays=population,
                    show_metrics=use_show_metrics
                )
                
                # Capture column names from the first call with data source prefixes
                if self._first_batch_call and batch_results and len(batch_results) > 0:
                    if 'metrics_breakdown' in batch_results[0]:
                        metrics = batch_results[0]['metrics_breakdown']['metrics']
                        # Create column names with data source prefixes
                        self._weight_column_names = [f"{metric['data_source']}:{metric['column_name']}" for metric in metrics]
                        print(f"  Captured {len(self._weight_column_names)} weight column names from first batch call")
                    else:
                        print("  Warning: No metrics breakdown in first batch call, using fallback names")
                        self._weight_column_names = [f"weight_{i+1:02d}" for i in range(len(population[0]))]
                    
                    self._first_batch_call = False  # Don't use show_metrics for future calls
                
                print(f"  Batch operation complete - got {len(batch_results)} results")
                
                # Process each result from the batch operation
                for batch_result in batch_results:
                    chromosome_index = batch_result['weight_array_index']
                    
                    model_id = batch_result['model_id']
                    direction = batch_result['direction']
                    threshold = batch_result['threshold']
                    
                    print(f"    Chromosome {chromosome_index + 1}: Model {model_id}, direction: {direction}, threshold: {threshold}")
                    
                    # Load model predictions (use cache to avoid repeated disk reads)
                    if model_id not in model_predictions_cache:
                        model_predictions_cache[model_id] = self.load_model_predictions(model_id)
                        print(f"    Cached predictions for model {model_id}")
                    
                    model_predictions = model_predictions_cache[model_id]
                
                    # Get ALL model prediction timestamps for this day (complete dataset)
                    trading_day_int = int(current_day)
                    trading_start = 38100000  # 10:35 AM (first model prediction)
                    trading_end = 43200000    # 12:00 PM (last model prediction)
                    
                    # Get all model timestamps for this trading day
                    model_day_data = model_predictions[model_predictions['TradingDay'] == trading_day_int]
                    all_model_timestamps = []
                    for _, row in model_day_data.iterrows():
                        ms_of_day = row['TradingMsOfDay']
                        if trading_start <= ms_of_day <= trading_end:
                            all_model_timestamps.append(ms_of_day)
                    
                    all_model_timestamps = sorted(all_model_timestamps)
                    
                    # Create result rows for ALL model timestamps (complete dataset for accurate metrics)
                    regime_rows_added = 0
                    predictions_found = 0
                    zeros_added = 0
                    
                    for ms_of_day in all_model_timestamps:
                        # Check if this timestamp is in target regime
                        regime_match = current_day_rows[current_day_rows['ms_of_day'] == ms_of_day]
                        
                        if len(regime_match) > 0:
                            # This timestamp IS in target regime - use actual model data
                            model_match = model_predictions[
                                (model_predictions['TradingDay'] == trading_day_int) & 
                                (model_predictions['TradingMsOfDay'] == ms_of_day)
                            ]
                            actual = model_match.iloc[0]['Actual']
                            predicted = model_match.iloc[0]['Predicted']
                            # Make threshold negative for downside strategies
                            if direction == "down":
                                actual_threshold = -threshold if threshold != 0.0 else -0.0
                            else:
                                actual_threshold = threshold
                            actual_side = direction  # Use direction from weighter ("up" or "down")
                            actual_model_id = model_id
                            predictions_found += 1
                        else:
                            # This timestamp is NOT in target regime - use zeros
                            actual = 0.0
                            predicted = 0.0
                            actual_threshold = 0.0
                            actual_side = "up"  # Default to "up" for zero rows
                            actual_model_id = 0
                            zeros_added += 1
                        
                        result_row = {
                            'TradingDay': trading_day_int,
                            'TradingMsOfDay': ms_of_day,
                            'Actual': actual,
                            'Predicted': predicted,
                            'Threshold': actual_threshold,
                            'Side': actual_side,
                            'ModelID': actual_model_id
                        }
                        population_results[chromosome_index].append(result_row)
                        regime_rows_added += 1
                    
                    print(f"    Chromosome {chromosome_index + 1}: Added {regime_rows_added} rows for {current_day} ({predictions_found} regime matches, {zeros_added} zeros for non-regime)")
                
            except Exception as e:
                print(f"  Error in batch processing for {current_day}: {e}")
                print(f"  Skipping {current_day} - batch processing is required for performance")
                continue
        
        # After processing all trading days, evaluate performance for each chromosome
        fitness_results = []
        for chromosome_index in range(len(population)):
            result_data = population_results[chromosome_index]
            
            if result_data:
                print(f"\nEvaluating performance for chromosome {chromosome_index + 1}")
                
                # Extract data arrays
                trading_days_array = [row['TradingDay'] for row in result_data]
                trading_ms_array = [row['TradingMsOfDay'] for row in result_data]
                actual_array = [row['Actual'] for row in result_data]
                predicted_array = [row['Predicted'] for row in result_data]
                threshold_array = [row['Threshold'] for row in result_data]
                side_array = [row['Side'] for row in result_data]
                
                # Evaluate performance using trading performance analyzer
                try:
                    performance_result = self.performance_analyzer.evaluate_performance(
                        trading_days=trading_days_array,
                        trading_ms=trading_ms_array,
                        actual=actual_array,
                        predicted=predicted_array,
                        thresholds=threshold_array,
                        sides=side_array
                    )
                    
                    if performance_result:
                        fitness_results.append(performance_result)
                        print(f"  Composite Score: {performance_result['composite_score']:.2f}")
                    else:
                        fitness_results.append(self._get_default_fitness())
                    
                except Exception as e:
                    print(f"  Error evaluating performance for chromosome {chromosome_index + 1}: {e}")
                    fitness_results.append(self._get_default_fitness())
            else:
                print(f"\nNo data for chromosome {chromosome_index + 1}")
                fitness_results.append(self._get_default_fitness())
        
        return fitness_results, population_results
    
    def _get_default_fitness(self) -> Dict:
        """Return default fitness values when evaluation fails."""
        return {
            'composite_score': 0.0, 'total_pnl_after_fees': 0.0, 'num_trades': 0, 
            'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0, 
            'profit_factor': 0.0, 'avg_pnl_per_trade': 0.0, 'volatility': 0.0, 'calmar_ratio': 0.0
        }
    
    def tournament_selection(self, population: List[np.ndarray], fitness_scores: List[float], 
                           tournament_size: int = 3) -> np.ndarray:
        """
        Select a parent using tournament selection.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each chromosome
            tournament_size: Number of chromosomes to compete
            
        Returns:
            Selected parent chromosome
        """
        # Randomly select tournament participants
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        
        # Find the best performer in the tournament
        best_idx = max(tournament_indices, key=lambda x: fitness_scores[x])
        
        return population[best_idx].copy()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create two offspring from two parents using single-point crossover.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of two offspring chromosomes
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Single-point crossover
        crossover_point = np.random.randint(1, len(parent1))
        
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        return offspring1, offspring2
    
    def mutate(self, chromosome: np.ndarray, mutation_rate: float = None, 
               weight_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """
        Mutate a chromosome by randomly changing some genes.
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutating each gene
            weight_range: Valid range for gene values
            
        Returns:
            Mutated chromosome
        """
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        
        mutated = chromosome.copy()
        min_weight, max_weight = weight_range
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] = np.random.uniform(min_weight, max_weight)
        
        return mutated
    
    def evolve_population(self, population: List[np.ndarray], fitness_scores: List[float]) -> List[np.ndarray]:
        """
        Evolve the population to create the next generation.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for current population
            
        Returns:
            New population for next generation
        """
        new_population = []
        
        # Elitism: Keep the best performers
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate the rest of the population through crossover and mutation
        while len(new_population) < len(population):
            # Selection
            parent1 = self.tournament_selection(population, fitness_scores, self.tournament_size)
            parent2 = self.tournament_selection(population, fitness_scores, self.tournament_size)
            
            # Crossover
            offspring1, offspring2 = self.crossover(parent1, parent2)
            
            # Mutation
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)
            
            new_population.extend([offspring1, offspring2])
        
        # Trim to exact population size if needed
        return new_population[:len(population)]
    
    def get_trading_days_in_range(self, from_day: str, to_day: str) -> List[str]:
        """
        Get trading days within the specified range from market regime forecast.
        
        Args:
            from_day: Start trading day (YYYYMMDD)
            to_day: End trading day (YYYYMMDD)
            
        Returns:
            List of trading days in chronological order
        """
        # Use cached data if available, otherwise load it
        if self._regime_forecast_cache is not None:
            df = self._regime_forecast_cache
        else:
            df = self.load_market_regime_forecast()
        
        # Convert to integers for comparison
        from_day_int = int(from_day)
        to_day_int = int(to_day)
        
        # Filter trading days in range
        filtered_days = df[
            (df['trading_day'] >= from_day_int) & 
            (df['trading_day'] <= to_day_int)
        ]['trading_day'].unique()
        
        trading_days = sorted([str(day) for day in filtered_days])
        print(f"Found {len(trading_days)} trading days between {from_day} and {to_day}")
        return trading_days
    
    def get_previous_trading_day(self, current_day: str) -> Optional[str]:
        """
        Get the trading day immediately before the current day.
        
        Args:
            current_day: Current trading day (YYYYMMDD)
            
        Returns:
            Previous trading day or None if not found
        """
        # Use cached data if available, otherwise load it
        if self._regime_forecast_cache is not None:
            df = self._regime_forecast_cache
        else:
            df = self.load_market_regime_forecast()
        
        current_day_int = int(current_day)
        
        # Get all trading days before current day
        previous_days = df[df['trading_day'] < current_day_int]['trading_day'].unique()
        
        if len(previous_days) == 0:
            return None
        
        # Return the most recent previous day
        return str(max(previous_days))
    
    def get_regime_rows_for_day(self, trading_day: str, target_regime: int) -> pd.DataFrame:
        """
        Get rows from market regime forecast for specific day and regime.
        
        Args:
            trading_day: Trading day (YYYYMMDD)
            target_regime: Target market regime
            
        Returns:
            DataFrame with matching rows
        """
        # Use cached data if available, otherwise load it
        if self._regime_forecast_cache is not None:
            df = self._regime_forecast_cache
        else:
            df = self.load_market_regime_forecast()
        
        trading_day_int = int(trading_day)
        
        # Filter for specific day and regime
        regime_rows = df[
            (df['trading_day'] == trading_day_int) & 
            (df['predicted_regime'] == target_regime)
        ]
        
        return regime_rows
    
    def load_model_predictions(self, model_id: str) -> pd.DataFrame:
        """
        Load prediction data for a specific model.
        
        Args:
            model_id: Model ID (e.g., "00001")
            
        Returns:
            DataFrame with model predictions
        """
        # Ensure model_id is zero-padded to 5 digits
        model_id_padded = f"{int(model_id):05d}"
        prediction_file = os.path.join(self.predictions_dir, f"model_{model_id_padded}_prediction.csv")
        
        if not os.path.exists(prediction_file):
            raise FileNotFoundError(f"Model prediction file not found: {prediction_file}")
        
        try:
            df = pd.read_csv(prediction_file)
            return df
        except Exception as e:
            raise Exception(f"Error loading model predictions for {model_id}: {e}")
    
    def get_comprehensive_trading_timestamps(self, trading_day: str, model_predictions: pd.DataFrame, 
                                           current_day_regime_rows: pd.DataFrame) -> List[int]:
        """
        Get comprehensive list of trading timestamps by combining model predictions and regime data.
        
        Args:
            trading_day: Trading day (YYYYMMDD)
            model_predictions: DataFrame with model prediction data
            current_day_regime_rows: DataFrame with regime forecast rows
            
        Returns:
            Sorted list of all trading timestamps (within trading hours)
        """
        trading_day_int = int(trading_day)
        
        # Define trading hours based on model prediction availability
        trading_start = 38100000  # 10:35 AM (first model prediction)
        trading_end = 43200000    # 12:00 PM (last model prediction)
        
        # Get timestamps from model predictions for this trading day
        model_timestamps = set()
        model_day_data = model_predictions[model_predictions['TradingDay'] == trading_day_int]
        for _, row in model_day_data.iterrows():
            ms_of_day = row['TradingMsOfDay']
            if trading_start <= ms_of_day <= trading_end:
                model_timestamps.add(ms_of_day)
        
        # Get timestamps from regime data for this trading day
        regime_timestamps = set()
        for _, row in current_day_regime_rows.iterrows():
            ms_of_day = row['ms_of_day']
            if trading_start <= ms_of_day <= trading_end:
                regime_timestamps.add(ms_of_day)
        
        # Combine all timestamps and sort
        all_timestamps = sorted(list(model_timestamps.union(regime_timestamps)))
        
        print(f"    Model timestamps: {len(model_timestamps)}, Regime timestamps: {len(regime_timestamps)}, Combined: {len(all_timestamps)}")
        return all_timestamps
    
    def save_generation_results(self, generation: int, population: List[np.ndarray], 
                               fitness_results: List[Dict], population_results: Dict[int, List[Dict]],
                               from_trading_day: str, to_trading_day: str, market_regime: int) -> None:
        """
        Save generation results to CSV file and best chromosome trading data.
        
        Args:
            generation: Current generation number
            population: Current population
            fitness_results: Fitness evaluation results for each chromosome
            population_results: Trading data results for each chromosome
            from_trading_day: Start trading day
            to_trading_day: End trading day
            market_regime: Market regime
        """
        # Sort by composite_score (fitness) in descending order
        sorted_indices = sorted(range(len(fitness_results)), 
                               key=lambda x: fitness_results[x]['composite_score'], reverse=True)
        
        # Create DataFrame for generation results
        generation_data = []
        for rank, idx in enumerate(sorted_indices, 1):
            result = fitness_results[idx]
            chromosome = population[idx]
            
            row = {
                'rank': rank,
                'chromosome': idx + 1,  # 1-based chromosome numbering
                'composite_score': result['composite_score'],
                'total_pnl_after_fees': result['total_pnl_after_fees'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown'],
                'win_rate': result['win_rate'],
                'num_trades': result['num_trades'],
                'profit_factor': result['profit_factor'],
                'avg_pnl_per_trade': result['avg_pnl_per_trade'],
                'volatility': result['volatility'],
                'calmar_ratio': result['calmar_ratio'],
                'from_trading_day': from_trading_day,
                'to_trading_day': to_trading_day,
                'market_regime': market_regime,
                'generation': generation
            }
            
            # Use cached weight column names (captured from first batch call)
            if self._weight_column_names is not None:
                weight_column_names = self._weight_column_names
            else:
                # Fallback if somehow column names weren't captured
                weight_column_names = [f"weight_{i+1:02d}" for i in range(len(chromosome))]
            
            # Add weight values using the dynamic column names
            for i, weight in enumerate(chromosome):
                if i < len(weight_column_names):
                    row[weight_column_names[i]] = weight
                else:
                    # Fallback for any extra weights (shouldn't happen with proper sizing)
                    row[f'weight_{i+1:02d}'] = weight
            
            generation_data.append(row)
        
        # Save to CSV
        filename = f"{from_trading_day}_{to_trading_day}_regime{market_regime}_generation{generation}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            generation_df = pd.DataFrame(generation_data)
            generation_df.to_csv(filepath, index=False)
            
            print(f"  Saved generation {generation} results to {filename}")
            
            # Print top 3 performers for this generation
            print(f"  Top 3 performers in generation {generation}:")
            for i in range(min(3, len(generation_data))):
                row = generation_data[i]
                print(f"    {i+1}. Chromosome {row['chromosome']:02d}: Score {row['composite_score']:.2f}, "
                      f"P&L ${row['total_pnl_after_fees']:.4f}, Trades: {row['num_trades']}")
            
            # Save best chromosome trading data using already-collected data (efficient!)
            print(f"  Saving best chromosome trading data for generation {generation}...")
            best_idx = sorted_indices[0]  # Best performer is first in sorted list
            best_chromosome = population[best_idx]
            best_chromosome_data = population_results[best_idx]
            
            # Save best chromosome data to CSV file using the same format as before_ga version
            best_filename = f"{from_trading_day}_{to_trading_day}_regime{market_regime}_generation{generation}_best_chromosome_data.csv"
            best_filepath = os.path.join(self.output_dir, best_filename)
            
            with open(best_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # First row: weight values (76 cells)
                writer.writerow(best_chromosome.tolist())
                
                # Second row: column names
                if best_chromosome_data:
                    column_names = list(best_chromosome_data[0].keys())
                    writer.writerow(column_names)
                    
                    # Subsequent rows: result data
                    for row_data in best_chromosome_data:
                        writer.writerow([row_data[col] for col in column_names])
            
            print(f"    Saved best chromosome trading data to {best_filename} ({len(best_chromosome_data)} rows)")
            
        except Exception as e:
            print(f"  Error saving generation {generation} results: {e}")
    
    
    def optimize_weights_with_genetic_algorithm(self, from_trading_day: str, to_trading_day: str, 
                                               market_regime: int, population_size: int, 
                                               generations: int) -> None:
        """
        Main genetic algorithm optimization function.
        
        Args:
            from_trading_day: Start trading day (YYYYMMDD)
            to_trading_day: End trading day (YYYYMMDD)
            market_regime: Target market regime
            population_size: Number of chromosomes in population
            generations: Number of generations to evolve
        """
        # Validate input parameters
        if int(from_trading_day) <= 20200102:
            raise ValueError(f"from_trading_day must be > 20200102, got {from_trading_day}")
        
        if int(from_trading_day) > int(to_trading_day):
            raise ValueError(f"from_trading_day must be <= to_trading_day")
        
        print(f"Starting genetic algorithm optimization for regime {market_regime}")
        print(f"Trading day range: {from_trading_day} to {to_trading_day}")
        print(f"Population size: {population_size}, Generations: {generations}")
        print(f"GA Parameters: mutation_rate={self.mutation_rate}, crossover_rate={self.crossover_rate}, elite_size={self.elite_size}")
        
        # Initialize population
        population = self.initialize_population(population_size)
        
        # Evolution loop
        for generation in range(1, generations + 1):
            print(f"\n" + "="*60)
            print(f"GENERATION {generation}/{generations}")
            print(f"="*60)
            
            # Evaluate fitness for entire population using optimized batch method
            fitness_results, population_results = self.evaluate_population_fitness(
                population, from_trading_day, to_trading_day, market_regime
            )
            
            # Save generation results and best chromosome data efficiently
            self.save_generation_results(generation, population, fitness_results, population_results,
                                       from_trading_day, to_trading_day, market_regime)
            
            # Prepare for next generation (skip for last generation)
            if generation < generations:
                # Extract fitness scores for evolution
                fitness_scores = [result['composite_score'] for result in fitness_results]
                
                # Evolve population
                print(f"  Evolving population for generation {generation + 1}...")
                population = self.evolve_population(population, fitness_scores)
                
                # Print evolution statistics
                best_fitness = max(fitness_scores)
                avg_fitness = np.mean(fitness_scores)
                print(f"  Generation {generation} stats: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}")
        
        print(f"\n" + "="*60)
        print(f"GENETIC ALGORITHM OPTIMIZATION COMPLETE")
        print(f"="*60)
        print(f"Total generations: {generations}")
        print(f"Results saved to: {self.output_dir}")
        print(f"Final generation best performer: {max([r['composite_score'] for r in fitness_results]):.2f}")
        print(f"="*60)
def main():
    """
    Main function to parse command line arguments and run genetic algorithm optimization.
    """
    if len(sys.argv) != 6:
        print("Usage: python trading_model_regime_weight_optimizer.py <from_trading_day> <to_trading_day> <market_regime> <population_size> <generations>")
        print("Examples:")
        print("  python trading_model_regime_weight_optimizer.py 20250701 20250710 3 20 50")
        print("  python trading_model_regime_weight_optimizer.py 20240601 20240630 2 30 100")
        sys.exit(1)
    
    try:
        from_trading_day = sys.argv[1]
        to_trading_day = sys.argv[2]
        market_regime = int(sys.argv[3])
        population_size = int(sys.argv[4])
        generations = int(sys.argv[5])
        
        # Validate trading day format
        if len(from_trading_day) != 8 or not from_trading_day.isdigit():
            raise ValueError("from_trading_day must be in YYYYMMDD format")
        
        if len(to_trading_day) != 8 or not to_trading_day.isdigit():
            raise ValueError("to_trading_day must be in YYYYMMDD format")
        
        # Validate market regime
        if market_regime < 0 or market_regime > 4:
            raise ValueError("market_regime must be between 0 and 4")
        
        # Validate population size
        if population_size < 4:
            raise ValueError("population_size must be >= 4 (minimum for genetic algorithm)")
        
        # Validate generations
        if generations < 1:
            raise ValueError("generations must be >= 1")
        
    except ValueError as e:
        print(f"Error: Invalid arguments - {e}")
        sys.exit(1)
    
    print("=" * 80)
    print("TRADING MODEL REGIME WEIGHT OPTIMIZER - GENETIC ALGORITHM")
    print("=" * 80)
    print(f"From Trading Day: {from_trading_day}")
    print(f"To Trading Day:   {to_trading_day}")
    print(f"Market Regime:    {market_regime}")
    print(f"Population Size:  {population_size}")
    print(f"Generations:      {generations}")
    print(f"Started at:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Initialize optimizer
        optimizer = TradingModelRegimeWeightOptimizer()
        
        # Run genetic algorithm optimization
        optimizer.optimize_weights_with_genetic_algorithm(
            from_trading_day=from_trading_day,
            to_trading_day=to_trading_day,
            market_regime=market_regime,
            population_size=population_size,
            generations=generations
        )
        
        print("=" * 80)
        print("GENETIC ALGORITHM OPTIMIZATION COMPLETED SUCCESSFULLY")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
