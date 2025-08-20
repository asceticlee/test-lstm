#!/usr/bin/env python3
"""
Trading Model Regime Weight Optimizer

This script optimizes model trading weights for specific market regimes by:
1. Generating random weight arrays for model evaluation
2. Using market regime forecasts to identify regime-specific trading days
3. Finding optimal models using the model trading weighter
4. Collecting actual vs predicted performance data
5. Outputting comprehensive performance datasets for analysis

Usage:
    python trading_model_regime_weight_optimizer.py <from_trading_day> <to_trading_day> <market_regime> <candidate_count>
    
Examples:
    python trading_model_regime_weight_optimizer.py 20250701 20250710 3 5
    # Optimize for regime 3 from July 1-10, 2025 with 5 random weight candidates
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
    Optimizes trading model weights for specific market regimes.
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
    
    def generate_random_weights(self, count: int, weight_length: int = 76, 
                               weight_range: Tuple[float, float] = (0.0, 1.0)) -> List[np.ndarray]:
        """
        Generate random weight arrays for model evaluation.
        
        Args:
            count: Number of weight arrays to generate
            weight_length: Length of each weight array (default: 76)
            weight_range: Range for random weights (default: 0.0 to 1.0)
            
        Returns:
            List of numpy arrays containing random weights
        """
        weight_arrays = []
        min_weight, max_weight = weight_range
        
        for i in range(count):
            # Set different random seed for each candidate
            np.random.seed(42 + i)
            weights = np.random.uniform(min_weight, max_weight, weight_length)
            weight_arrays.append(weights)
        
        print(f"Generated {count} random weight arrays of length {weight_length}")
        return weight_arrays
    
    def get_trading_days_in_range(self, from_day: str, to_day: str) -> List[str]:
        """
        Get trading days within the specified range from market regime forecast.
        
        Args:
            from_day: Start trading day (YYYYMMDD)
            to_day: End trading day (YYYYMMDD)
            
        Returns:
            List of trading days in chronological order
        """
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
    
    def optimize_weights_for_regime(self, from_trading_day: str, to_trading_day: str, 
                                   market_regime: int, candidate_count: int) -> None:
        """
        Main optimization function that generates weight candidates and evaluates performance.
        
        Args:
            from_trading_day: Start trading day (YYYYMMDD)
            to_trading_day: End trading day (YYYYMMDD)
            market_regime: Target market regime
            candidate_count: Number of weight candidates to generate
        """
        # Validate input parameters
        if int(from_trading_day) <= 20200102:
            raise ValueError(f"from_trading_day must be > 20200102, got {from_trading_day}")
        
        if int(from_trading_day) > int(to_trading_day):
            raise ValueError(f"from_trading_day must be <= to_trading_day")
        
        print(f"Starting optimization for regime {market_regime}")
        print(f"Trading day range: {from_trading_day} to {to_trading_day}")
        print(f"Generating {candidate_count} weight candidates")
        
        # Generate random weight arrays
        weight_candidates = self.generate_random_weights(candidate_count)
        
        # Get trading days in range
        trading_days = self.get_trading_days_in_range(from_trading_day, to_trading_day)
        
        if len(trading_days) == 0:
            print("No trading days found in the specified range")
            return
        
        # Process each weight candidate
        all_weight_performances = []  # Store performance results for all weight candidates
        
        for set_number, weights in enumerate(weight_candidates, 1):
            print(f"\nProcessing weight candidate {set_number}/{candidate_count}")
            
            # Initialize 2D array for collecting results
            result_data = []
            
            # Process each trading day
            for current_day in trading_days:
                print(f"  Processing trading day {current_day}")
                
                # Get current day regime rows
                current_day_rows = self.get_regime_rows_for_day(current_day, market_regime)
                
                if len(current_day_rows) == 0:
                    print(f"    No regime {market_regime} data for {current_day}, skipping")
                    continue
                
                # Store current day rows for later processing
                current_day_regime_rows = current_day_rows
                
                # Get previous trading day for model weighter
                previous_day = self.get_previous_trading_day(current_day)
                if previous_day is None:
                    print(f"    No previous trading day found for {current_day}, skipping model selection")
                    continue
                
                try:
                    # Find best model using model trading weighter
                    print(f"    Finding best model for previous day {previous_day}, regime {market_regime}")
                    best_model_result = self.weighter.get_best_trading_model(
                        trading_day=previous_day,
                        market_regime=market_regime,
                        weighting_array=weights
                    )
                    
                    model_id = best_model_result['model_id']
                    direction = best_model_result['direction']
                    threshold = best_model_result['threshold']
                    
                    print(f"    Best model: {model_id}, direction: {direction}, threshold: {threshold}")
                    
                    # Load model predictions
                    model_predictions = self.load_model_predictions(model_id)
                    
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
                        regime_match = current_day_regime_rows[
                            current_day_regime_rows['ms_of_day'] == ms_of_day
                        ]
                        
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
                            actual_model_id = int(model_id)
                            actual_side = direction  # Use direction from weighter ("up" or "down")
                            predictions_found += 1
                        else:
                            # This timestamp is NOT in target regime - use zeros
                            actual = 0.0
                            predicted = 0.0
                            actual_threshold = 0.0
                            actual_model_id = 0
                            actual_side = "up"  # Default to "up" for zero rows
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
                        result_data.append(result_row)
                        regime_rows_added += 1
                    
                    print(f"    Added {regime_rows_added} rows for {current_day} ({predictions_found} regime matches, {zeros_added} zeros for non-regime)")
                    
                except Exception as e:
                    print(f"    Error processing {current_day}: {e}")
                    continue
            
            # Evaluate performance using trading performance analyzer
            if result_data:
                print(f"  Evaluating performance for weight candidate {set_number}")
                
                # Extract data arrays (excluding ModelID column)
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
                        # Add weight candidate information
                        performance_result['weight_candidate'] = set_number
                        performance_result['weights'] = weights.tolist()
                        all_weight_performances.append(performance_result)
                        
                        print(f"    Performance metrics collected - Composite Score: {performance_result['composite_score']:.2f}")
                    
                except Exception as e:
                    print(f"    Error evaluating performance for weight candidate {set_number}: {e}")
                
                # Still save the detailed results as before
                self.save_results(from_trading_day, to_trading_day, market_regime, 
                                candidate_count, set_number, weights, result_data)
            else:
                print(f"  No data to save for weight candidate {set_number}")
        
        # After processing all weight candidates, create ranking
        if all_weight_performances:
            self.save_optimized_ranking(from_trading_day, to_trading_day, market_regime, 
                                      candidate_count, all_weight_performances)
        
        print(f"\nOptimization complete! Results saved to {self.output_dir}")
    
    def save_results(self, from_trading_day: str, to_trading_day: str, market_regime: int,
                    candidate_count: int, set_number: int, weights: np.ndarray, 
                    result_data: List[Dict]) -> None:
        """
        Save optimization results to CSV file.
        
        Args:
            from_trading_day: Start trading day
            to_trading_day: End trading day
            market_regime: Market regime
            candidate_count: Total number of candidates
            set_number: Current set number
            weights: Weight array used
            result_data: List of result dictionaries
        """
        # Generate filename
        filename = f"{from_trading_day}_{to_trading_day}_regime{market_regime}_weight{set_number:02d}_{candidate_count}candidate.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # First row: weight values (76 cells)
                writer.writerow(weights.tolist())
                
                # Second row: column names
                if result_data:
                    column_names = list(result_data[0].keys())
                    writer.writerow(column_names)
                    
                    # Subsequent rows: result data
                    for row_data in result_data:
                        writer.writerow([row_data[col] for col in column_names])
                
            print(f"  Saved results to {filename} ({len(result_data)} rows)")
            
        except Exception as e:
            print(f"  Error saving results to {filename}: {e}")
    
    def save_optimized_ranking(self, from_trading_day: str, to_trading_day: str, market_regime: int,
                              candidate_count: int, all_performances: List[Dict]) -> None:
        """
        Save optimized weight ranking sorted by composite score.
        
        Args:
            from_trading_day: Start trading day
            to_trading_day: End trading day
            market_regime: Market regime
            candidate_count: Total number of candidates
            all_performances: List of performance result dictionaries
        """
        print(f"\nGenerating optimized weight ranking...")
        
        # Sort by composite_score in descending order
        sorted_performances = sorted(all_performances, key=lambda x: x['composite_score'], reverse=True)
        
        # Get metric column information by using show_metrics=True approach
        try:
            # Use a sample call to get metrics breakdown with actual column names
            import numpy as np
            np.random.seed(42)
            sample_weights = np.random.uniform(0.0, 1.0, 76)
            
            # Get trading days for sample call
            trading_days = self.get_trading_days_in_range(from_trading_day, to_trading_day)
            if trading_days:
                sample_day = self.get_previous_trading_day(trading_days[0])
                if sample_day:
                    sample_result = self.weighter.get_best_trading_model(
                        trading_day=sample_day,
                        market_regime=market_regime,
                        weighting_array=sample_weights,
                        show_metrics=True
                    )
                    
                    if 'metrics_breakdown' in sample_result and sample_result['metrics_breakdown']:
                        # Extract column names using the correct format from metrics breakdown
                        weight_column_names = []
                        for metric in sample_result['metrics_breakdown']['metrics']:
                            column_name = metric['column_name']
                            data_source = metric['data_source']
                            weight_column_names.append(f"{data_source}:{column_name}")
                        
                        print(f"  Using {len(weight_column_names)} metric column names from model trading weighter")
                        daily_count = sum(1 for name in weight_column_names if name.startswith('daily:'))
                        regime_count = sum(1 for name in weight_column_names if name.startswith('regime:'))
                        print(f"    Daily columns: {daily_count}, Regime columns: {regime_count}")
                    else:
                        raise Exception("No metrics breakdown available from sample call")
                else:
                    raise Exception("No previous trading day found for sample call")
            else:
                raise Exception("No trading days found for sample call")
            
        except Exception as e:
            print(f"  Warning: Could not get metric column names: {e}")
            print(f"  Falling back to generic weight column names")
            # Fallback to generic names if we can't get the actual column names
            weight_column_names = [f"weight_{i+1:02d}" for i in range(76)]
        
        # Create DataFrame for easier manipulation
        ranking_data = []
        for rank, perf in enumerate(sorted_performances, 1):
            row = {
                'rank': rank,
                'weight_candidate': perf['weight_candidate'],
                'composite_score': perf['composite_score'],
                'total_pnl_after_fees': perf['total_pnl_after_fees'],
                'sharpe_ratio': perf['sharpe_ratio'],
                'max_drawdown': perf['max_drawdown'],
                'win_rate': perf['win_rate'],
                'num_trades': perf['num_trades'],
                'profit_factor': perf['profit_factor'],
                'avg_pnl_per_trade': perf['avg_pnl_per_trade'],
                'volatility': perf['volatility'],
                'calmar_ratio': perf['calmar_ratio'],
                'from_trading_day': from_trading_day,
                'to_trading_day': to_trading_day,
                'market_regime': market_regime,
                'candidate_count': candidate_count
            }
            
            # Add weight values using actual metric column names
            weights = perf['weights']
            for i, weight in enumerate(weights):
                if i < len(weight_column_names):
                    row[weight_column_names[i]] = weight
                else:
                    # Fallback for any extra weights
                    row[f'weight_{i+1:02d}'] = weight
            
            ranking_data.append(row)
        
        # Save to CSV
        ranking_filename = "optimized_weight_ranking.csv"
        ranking_filepath = os.path.join(self.output_dir, ranking_filename)
        
        try:
            ranking_df = pd.DataFrame(ranking_data)
            ranking_df.to_csv(ranking_filepath, index=False)
            
            print(f"  Saved optimized ranking to {ranking_filename}")
            print(f"  Top 3 performers by composite score:")
            for i in range(min(3, len(sorted_performances))):
                perf = sorted_performances[i]
                print(f"    {i+1}. Weight {perf['weight_candidate']:02d}: Score {perf['composite_score']:.2f}, "
                      f"P&L ${perf['total_pnl_after_fees']:.4f}, Sharpe {perf['sharpe_ratio']:.4f}")
            
        except Exception as e:
            print(f"  Error saving ranking to {ranking_filename}: {e}")


def main():
    """
    Main function to parse command line arguments and run optimization.
    """
    if len(sys.argv) != 5:
        print("Usage: python trading_model_regime_weight_optimizer.py <from_trading_day> <to_trading_day> <market_regime> <candidate_count>")
        print("Examples:")
        print("  python trading_model_regime_weight_optimizer.py 20250701 20250710 3 5")
        print("  python trading_model_regime_weight_optimizer.py 20240601 20240630 2 10")
        sys.exit(1)
    
    try:
        from_trading_day = sys.argv[1]
        to_trading_day = sys.argv[2]
        market_regime = int(sys.argv[3])
        candidate_count = int(sys.argv[4])
        
        # Validate trading day format
        if len(from_trading_day) != 8 or not from_trading_day.isdigit():
            raise ValueError("from_trading_day must be in YYYYMMDD format")
        
        if len(to_trading_day) != 8 or not to_trading_day.isdigit():
            raise ValueError("to_trading_day must be in YYYYMMDD format")
        
        # Validate market regime
        if market_regime < 0 or market_regime > 4:
            raise ValueError("market_regime must be between 0 and 4")
        
        # Validate candidate count
        if candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")
        
    except ValueError as e:
        print(f"Error: Invalid arguments - {e}")
        sys.exit(1)
    
    print("=" * 70)
    print("TRADING MODEL REGIME WEIGHT OPTIMIZER")
    print("=" * 70)
    print(f"From Trading Day: {from_trading_day}")
    print(f"To Trading Day:   {to_trading_day}")
    print(f"Market Regime:    {market_regime}")
    print(f"Candidate Count:  {candidate_count}")
    print(f"Started at:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Initialize optimizer
        optimizer = TradingModelRegimeWeightOptimizer()
        
        # Run optimization
        optimizer.optimize_weights_for_regime(
            from_trading_day=from_trading_day,
            to_trading_day=to_trading_day,
            market_regime=market_regime,
            candidate_count=candidate_count
        )
        
        print("=" * 70)
        print("OPTIMIZATION COMPLETED SUCCESSFULLY")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
