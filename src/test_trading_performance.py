#!/usr/bin/env python3
"""
Test Trading Performance Script

This script tests the trading_performance.py module using pre-calculated model predictions
from the model_predictions directory. It loads prediction data from model_xxxxx_prediction.csv
files and evaluates trading performance ac    print(f"\n📂 All files saved to: {output_dir}")
    
    return results_dfnt thresholds.

Usage:
    python test_trading_performance.py [model_id]
    
Examples:
    python test_trading_performance.py          # Test model 00001 by default
    python test_trading_performance.py 377     # Test model 00377
"""

import sys
import os
import pandas as pd
import numpy as np
from trading_performance import TradingPerformanceAnalyzer

def find_available_models(prediction_dir):
    """
    Find all available model prediction files
    
    Args:
        prediction_dir: Directory containing model prediction files
        
    Returns:
        list: List of available model IDs
    """
    available_models = []
    
    if not os.path.exists(prediction_dir):
        return available_models
    
    for filename in os.listdir(prediction_dir):
        if filename.startswith('model_') and filename.endswith('_prediction.csv'):
            # Extract model ID from filename
            model_part = filename.replace('model_', '').replace('_prediction.csv', '')
            try:
                model_id = int(model_part)
                available_models.append(model_id)
            except ValueError:
                continue
    
    return sorted(available_models)

def get_model_training_period(model_id, models_dir):
    """
    Get the training period for a specific model from model_log.csv
    
    Args:
        model_id: Model ID (integer)
        models_dir: Directory containing model_log.csv
        
    Returns:
        tuple: (train_from, train_to) or (None, None) if not found
    """
    model_log_path = os.path.join(models_dir, 'model_log.csv')
    
    if not os.path.exists(model_log_path):
        print(f"Warning: model_log.csv not found at {model_log_path}")
        return None, None
    
    try:
        df = pd.read_csv(model_log_path)
        # The model_id in CSV is stored as integer, not zero-padded string
        model_row = df[df['model_id'] == model_id]
        
        if len(model_row) == 0:
            print(f"Warning: Model {model_id} not found in model_log.csv")
            return None, None
        
        train_from = model_row.iloc[0]['train_from']
        train_to = model_row.iloc[0]['train_to']
        
        print(f"Found training period for model {model_id:05d}: {train_from} to {train_to}")
        return int(train_from), int(train_to)
        
    except Exception as e:
        print(f"Warning: Error reading model_log.csv: {e}")
        return None, None

def load_model_predictions(model_id, prediction_dir):
    """
    Load prediction data for a specific model
    
    Args:
        model_id: Model ID (integer)
        prediction_dir: Directory containing prediction files
        
    Returns:
        pd.DataFrame: Prediction data with columns [TradingDay, TradingMsOfDay, Actual, Predicted]
    """
    model_file = os.path.join(prediction_dir, f'model_{model_id:05d}_prediction.csv')
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Prediction file not found: {model_file}")
    
    try:
        df = pd.read_csv(model_file)
        
        # Validate required columns
        required_cols = ['TradingDay', 'TradingMsOfDay', 'Actual', 'Predicted']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"Loaded {len(df):,} prediction records for model {model_id:05d}")
        return df
        
    except Exception as e:
        raise Exception(f"Error loading prediction file {model_file}: {e}")

def test_model_performance(model_id, prediction_dir, models_dir, output_dir, transaction_fee=0.02, generate_input_files=False):
    """
    Test trading performance for a specific model
    
    Args:
        model_id: Model ID to test
        prediction_dir: Directory containing prediction files
        models_dir: Directory containing model files and model_log.csv
        output_dir: Directory to save trading performance results
        transaction_fee: Transaction fee per trade in dollars (e.g., 0.02 = $0.02)
        generate_input_files: Whether to generate detailed input data files for each threshold
        
    Returns:
        pd.DataFrame: Performance results
    """
    print(f"\n" + "="*80)
    print(f"TESTING MODEL {model_id:05d} TRADING PERFORMANCE")
    print(f"="*80)
    
    # Load prediction data
    try:
        pred_df = load_model_predictions(model_id, prediction_dir)
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    
    # Get training period to exclude from testing
    train_from, train_to = get_model_training_period(model_id, models_dir)
    
    # Filter out training data if training period is found
    original_count = len(pred_df)
    if train_from is not None and train_to is not None:
        pred_df = pred_df[~((pred_df['TradingDay'] >= train_from) & (pred_df['TradingDay'] <= train_to))].copy()
        excluded_count = original_count - len(pred_df)
        print(f"Excluded training period ({train_from} to {train_to}): {excluded_count:,} records")
    else:
        print("Warning: Training period not found, using all prediction data")
    
    # Display data overview
    print(f"\nData Overview:")
    print(f"  Total records: {original_count:,}")
    print(f"  Records after excluding training: {len(pred_df):,}")
    print(f"  Date range: {pred_df['TradingDay'].min()} to {pred_df['TradingDay'].max()}")
    print(f"  Actual range: {pred_df['Actual'].min():.4f} to {pred_df['Actual'].max():.4f}")
    print(f"  Predicted range: {pred_df['Predicted'].min():.4f} to {pred_df['Predicted'].max():.4f}")
    
    # Calculate correlations
    correlation = pred_df['Actual'].corr(pred_df['Predicted'])
    print(f"  Actual vs Predicted correlation: {correlation:.4f}")
    
    # Initialize performance analyzer
    analyzer = TradingPerformanceAnalyzer(transaction_fee=transaction_fee)
    
    # Define threshold range from -0.8 to 0.8 (step 0.1, include 0.0, skip 0.05)
    all_thresholds = [round(x, 2) for x in np.arange(-0.8, 0.81, 0.1) if abs(round(x, 2) - 0.05) > 1e-8]
    
    print(f"\nTesting {len(all_thresholds)} thresholds: {all_thresholds}")
    
    # Log parameters being passed to trading_performance.py
    print(f"\nParameters passed to TradingPerformanceAnalyzer:")
    print(f"  transaction_fee: ${transaction_fee:.4f}")
    print(f"  thresholds to test: {all_thresholds}")
    print(f"  data_points: {len(pred_df):,}")
    print(f"  excluded_training_period: {train_from} to {train_to}" if train_from else "  excluded_training_period: None")
    
    # Collect all results
    all_results = []
    
    # Create input files directory if needed
    input_files_dir = os.path.join(output_dir, 'trading_performance_input')
    if generate_input_files:
        os.makedirs(input_files_dir, exist_ok=True)
    
    # Test each threshold individually
    for threshold in all_thresholds:
        print(f"\n" + "="*60)
        print(f"Testing threshold: {threshold}")
        print(f"="*60)

        threshold_array = np.full(len(pred_df), threshold)

        # For threshold == 0.0, test both sides explicitly
        if abs(threshold) < 1e-8:
            for side in ['up', 'down']:
                print(f"Testing threshold: {threshold} with side: {side}")
                
                # Generate input data file if requested
                input_data_file = None
                if generate_input_files:
                    # Create input data DataFrame with side column
                    input_df = pd.DataFrame({
                        'TradingDay': pred_df['TradingDay'].values,
                        'TradingMsOfDay': pred_df['TradingMsOfDay'].values,
                        'Actual': pred_df['Actual'].values,
                        'Predicted': pred_df['Predicted'].values,
                        'Threshold': threshold_array,
                        'Side': side
                    })
                    
                    # Save input data file with explicit threshold value
                    if side == 'up':
                        threshold_str = "0p00"
                        display_threshold = 0.0
                    else:
                        threshold_str = "neg0p00"  
                        display_threshold = -0.0
                    
                    input_filename = f"model_{model_id:05d}_threshold_{threshold_str}_{side}_input_data.csv"
                    input_data_file = os.path.join(input_files_dir, input_filename)
                    input_df.to_csv(input_data_file, index=False)
                    print(f"    📄 Generated input file: {os.path.basename(input_data_file)}")
                
                result = analyzer.evaluate_performance(
                    pred_df['TradingDay'].values,
                    pred_df['TradingMsOfDay'].values,
                    pred_df['Actual'].values,
                    pred_df['Predicted'].values,
                    threshold_array,
                    sides=np.full(len(pred_df), side)
                )
                if result:
                    # Override threshold to show proper +0.0 vs -0.0
                    if side == 'up':
                        result['threshold'] = 0.0
                    else:
                        result['threshold'] = -0.0
                    
                    # Add input data file path to result if generated
                    if input_data_file:
                        result['input_data_file'] = input_data_file
                    
                    # Print individual summary
                    analyzer.print_performance_summary(result)
                    all_results.append(result)
        else:
            # For non-zero thresholds, determine side based on threshold sign
            if threshold > 0:
                side = 'up'
            else:
                side = 'down'
            
            print(f"Testing threshold: {threshold} with side: {side}")
            
            # Generate input data file if requested
            input_data_file = None
            if generate_input_files:
                # Create input data DataFrame with side column
                input_df = pd.DataFrame({
                    'TradingDay': pred_df['TradingDay'].values,
                    'TradingMsOfDay': pred_df['TradingMsOfDay'].values,
                    'Actual': pred_df['Actual'].values,
                    'Predicted': pred_df['Predicted'].values,
                    'Threshold': threshold_array,
                    'Side': side
                })
                
                # Save input data file with side included in filename
                threshold_str = f"{threshold:.2f}".replace('-', 'neg').replace('.', 'p')
                input_filename = f"model_{model_id:05d}_threshold_{threshold_str}_{side}_input_data.csv"
                input_data_file = os.path.join(input_files_dir, input_filename)
                input_df.to_csv(input_data_file, index=False)
                print(f"    📄 Generated input file: {os.path.basename(input_data_file)}")
            
            result = analyzer.evaluate_performance(
                pred_df['TradingDay'].values,
                pred_df['TradingMsOfDay'].values,
                pred_df['Actual'].values,
                pred_df['Predicted'].values,
                threshold_array,
                sides=np.full(len(pred_df), side)
            )
            if result:
                # Add input data file path to result if generated
                if input_data_file:
                    result['input_data_file'] = input_data_file
                
                # Print individual summary
                analyzer.print_performance_summary(result)
                all_results.append(result)
    
    if len(all_results) == 0:
        print("No results generated for any threshold")
        return None
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Additional detailed analysis
    print(f"\n" + "="*80)
    print(f"DETAILED ANALYSIS")
    print(f"="*80)
    
    # Separate upside and downside strategies
    upside_results = results_df[results_df['trade_type'] == 'upside']
    downside_results = results_df[results_df['trade_type'] == 'downside']
    
    if len(upside_results) > 0:
        print(f"\nUpside Trading Strategies ({len(upside_results)} tested):")
        print("-" * 60)
        
        profitable_upside = upside_results[upside_results['total_pnl_after_fees'] > 0]
        print(f"  Profitable strategies: {len(profitable_upside)} / {len(upside_results)}")
        
        if len(profitable_upside) > 0:
            best_upside = profitable_upside.loc[profitable_upside['total_pnl_after_fees'].idxmax()]
            print(f"  Best upside threshold: {best_upside['threshold']:.3f}")
            print(f"  Best upside P&L: {best_upside['total_pnl_after_fees']:.4f}")
            print(f"  Best upside Sharpe: {best_upside['sharpe_ratio']:.4f}")
            print(f"  Best upside Win Rate: {best_upside['win_rate']:.1%}")
    
    if len(downside_results) > 0:
        print(f"\nDownside Trading Strategies ({len(downside_results)} tested):")
        print("-" * 60)
        
        profitable_downside = downside_results[downside_results['total_pnl_after_fees'] > 0]
        print(f"  Profitable strategies: {len(profitable_downside)} / {len(downside_results)}")
        
        if len(profitable_downside) > 0:
            best_downside = profitable_downside.loc[profitable_downside['total_pnl_after_fees'].idxmax()]
            print(f"  Best downside threshold: {best_downside['threshold']:.3f}")
            print(f"  Best downside P&L: {best_downside['total_pnl_after_fees']:.4f}")
            print(f"  Best downside Sharpe: {best_downside['sharpe_ratio']:.4f}")
            print(f"  Best downside Win Rate: {best_downside['win_rate']:.1%}")
    
    # Show top performers in each category
    print(f"\nTop 5 Performers by Category:")
    print("-" * 80)
    
    categories = [
        ('total_pnl_after_fees', 'Total P&L', 'highest'),
        ('sharpe_ratio', 'Sharpe Ratio', 'highest'),
        ('win_rate', 'Win Rate', 'highest'),
        ('profit_factor', 'Profit Factor', 'highest')
    ]
    
    for metric, name, order in categories:
        print(f"\n{name} (Top 5):")
        top_5 = results_df.nlargest(5, metric) if order == 'highest' else results_df.nsmallest(5, metric)
        
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"  {i}. Threshold: {row['threshold']:7.3f} | "
                  f"Type: {row['trade_type']:8s} | "
                  f"{name}: {row[metric]:8.4f} | "
                  f"Trades: {row['num_trades']:5.0f} | "
                  f"Win%: {row['win_rate']:6.1%}")
    
    # Export results to CSV in the specified output directory
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f'model_{model_id:05d}_trading_performance.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ Results exported to: {results_file}")
    
    # Log summary of all input data files created
    print(f"\n📁 Input Data Files Created:")
    print("-" * 60)
    for result in all_results:
        if 'input_data_file' in result:
            threshold = result['threshold']
            num_trades = result['num_trades']
            input_file = result['input_data_file']
            print(f"  Threshold {threshold:7.3f}: {num_trades:5,} trades → {os.path.basename(input_file)}")
    
    print(f"\n📂 All files saved to: {output_dir}")
    
    return results_df

def main():
    # Parse command line arguments
    if len(sys.argv) > 3:
        print("Usage: python test_trading_performance.py [model_id] [--generate-input-files]")
        print("Examples:")
        print("  python test_trading_performance.py                    # Test model 00001 by default")
        print("  python test_trading_performance.py 377               # Test model 00377")
        print("  python test_trading_performance.py 377 --generate-input-files  # Test model 00377 and generate input files")
        sys.exit(1)
    
    # Determine model ID and options
    model_id = 1  # Default to model 1
    generate_input_files = False
    
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--generate-input-files':
            generate_input_files = True
        else:
            try:
                model_id = int(arg)
            except ValueError:
                print(f"ERROR: Invalid argument '{arg}'. Model ID must be an integer.")
                sys.exit(1)
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    prediction_dir = os.path.join(project_root, 'model_predictions')
    models_dir = os.path.join(project_root, 'models')
    output_dir = os.path.join(project_root, 'model_trading', 'model_trading_performance')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if prediction directory exists
    if not os.path.exists(prediction_dir):
        print(f"ERROR: Prediction directory not found: {prediction_dir}")
        print("Please run batch_model_prediction.py first to generate prediction files")
        sys.exit(1)
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print(f"ERROR: Models directory not found: {models_dir}")
        print("Please ensure the models directory exists with model_log.csv")
        sys.exit(1)
    
    # Find available models
    available_models = find_available_models(prediction_dir)
    
    if not available_models:
        print(f"ERROR: No model prediction files found in {prediction_dir}")
        print("Please run batch_model_prediction.py first to generate prediction files")
        sys.exit(1)
    
    print(f"Available models: {available_models[:10]}{'...' if len(available_models) > 10 else ''}")
    print(f"Total available models: {len(available_models)}")
    print(f"Output directory: {output_dir}")
    
    # Check if requested model is available
    if model_id not in available_models:
        print(f"ERROR: Model {model_id:05d} prediction file not found")
        print(f"Available models: {available_models}")
        sys.exit(1)
    
    # Test the model
    try:
        results = test_model_performance(model_id, prediction_dir, models_dir, output_dir, 
                                       transaction_fee=0.02, generate_input_files=generate_input_files)
        
        if results is not None:
            print(f"\n🎉 Trading performance test completed successfully!")
            print(f"Model {model_id:05d} tested with {len(results)} threshold configurations")
            print(f"Results saved to: {output_dir}")
        else:
            print(f"❌ Trading performance test failed")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
