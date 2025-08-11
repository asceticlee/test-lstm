#!/usr/bin/env python3
"""
Model Regime Performance Tester

This script tests each model's performance on different market regimes and generates ranking files.
Supports both daily and sequence-based regime clustering methods.

Pipeline:
1. Load model metadata from models/model_log_avg.csv
2. Load regime assignments from:
   - Daily clustering: market_regime/gmm/daily/daily_regime_assignments.csv
   - Sequence clustering: market_regime/gmm/sequence/sequence_regime_assignments.csv
3. For each model, test performance on each regime
4. Generate model_regime_test_results_{method}.csv with detailed performance metrics
5. Generate model_regime_ranking_{method}.csv with regime-based rankings

Usage:
    python model_regime_tester.py --clustering_method daily
    python model_regime_tester.py --clustering_method sequence
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import json
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import test functions
current_dir = Path(__file__).parent
project_root = current_dir / ".." / ".."
sys.path.append(str(project_root / "src"))

class ModelRegimeTester:
    """Test models on different market regimes and generate rankings"""
    
    def __init__(self, max_models=None, clustering_method='daily'):
        self.project_root = Path(__file__).parent / ".." / ".."
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data"
        self.market_regime_dir = self.project_root / "market_regime"
        self.output_dir = self.project_root / "model_regime"
        self.max_models = max_models  # Limit number of models for testing
        self.clustering_method = clustering_method  # 'daily' or 'sequence'
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.model_log = None
        self.regime_assignments = None
        self.trading_data = None
        self.regime_characteristics = None
        
        # Results storage
        self.test_results = []
        
    def load_data(self):
        """Load all required data files"""
        print(f"Loading data files for {self.clustering_method} clustering method...")
        
        # Load model log
        model_log_file = self.models_dir / "model_log_avg.csv"
        if not model_log_file.exists():
            raise FileNotFoundError(f"Model log not found: {model_log_file}")
        self.model_log = pd.read_csv(model_log_file)
        
        # Limit models if specified
        if self.max_models:
            self.model_log = self.model_log.head(self.max_models)
            print(f"Limited to first {len(self.model_log)} models for testing")
        
        print(f"Loaded {len(self.model_log)} models from model log")
        
        # Load regime assignments based on clustering method
        if self.clustering_method == 'daily':
            regime_file = self.market_regime_dir / "gmm" / "daily" / "daily_regime_assignments.csv"
            regime_char_file = self.market_regime_dir / "gmm" / "daily" / "regime_characteristics.csv"
        elif self.clustering_method == 'sequence':
            regime_file = self.market_regime_dir / "gmm" / "sequence" / "sequence_regime_assignments.csv"
            regime_char_file = self.market_regime_dir / "gmm" / "sequence" / "regime_characteristics.csv"
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}. Use 'daily' or 'sequence'")
        
        if not regime_file.exists():
            raise FileNotFoundError(f"Regime assignments not found: {regime_file}")
        self.regime_assignments = pd.read_csv(regime_file)
        print(f"Loaded regime assignments for {len(self.regime_assignments)} entries from {self.clustering_method} clustering")
        
        # Load trading data
        trading_file = self.data_dir / "trainingData.csv"
        if not trading_file.exists():
            raise FileNotFoundError(f"Trading data not found: {trading_file}")
        self.trading_data = pd.read_csv(trading_file)
        print(f"Loaded trading data with {len(self.trading_data)} rows")
        
        # Load regime characteristics to get number of regimes
        if not regime_char_file.exists():
            raise FileNotFoundError(f"Regime characteristics not found: {regime_char_file}")
        self.regime_characteristics = pd.read_csv(regime_char_file)
        print(f"Found {len(self.regime_characteristics)} regimes using {self.clustering_method} clustering")
        
    def create_sequences_for_inference(self, df, seq_length, feature_cols, target_col):
        """Create sequences for inference - targeting exact single label, not average"""
        df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
        features = df[feature_cols].values
        target_values = df[target_col].values  # Single target column, not multiple
        days = df['TradingDay'].values
        ms = df['TradingMsOfDay'].values
        
        X = []
        y = []
        
        i = 0
        while i <= len(df) - seq_length:
            # Check if the sequence is consecutive (same day, 60000 ms increments)
            is_consecutive = True
            current_day = days[i]
            current_ms = ms[i]
            
            for j in range(1, seq_length):
                if days[i + j] != current_day or ms[i + j] != current_ms + j * 60000:
                    is_consecutive = False
                    break
            
            if is_consecutive:
                seq = features[i:i + seq_length]
                # Use the exact target label at prediction timestep (last timestep)
                target_value = target_values[i + seq_length - 1]  # Exact single label value
                X.append(seq)
                y.append(target_value)
                i += 1  # Sliding window, step by 1
            else:
                # Skip to the next potential start (after the gap)
                i += 1
        
        return np.array(X), np.array(y)
    
    def create_sequences_like_training(self, df, seq_length, feature_cols, target_cols):
        """Create sequences exactly like the training script"""
        df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
        features = df[feature_cols].values
        target_values = df[target_cols].values
        days = df['TradingDay'].values
        ms = df['TradingMsOfDay'].values
        
        X = []
        y = []
        
        i = 0
        while i <= len(df) - seq_length:
            # Check if the sequence is consecutive (same day, 60000 ms increments)
            is_consecutive = True
            current_day = days[i]
            current_ms = ms[i]
            
            for j in range(1, seq_length):
                if days[i + j] != current_day or ms[i + j] != current_ms + j * 60000:
                    is_consecutive = False
                    break
            
            if is_consecutive:
                seq = features[i:i + seq_length]
                # Calculate average of target labels at the prediction timestep (last timestep)
                individual_labels = target_values[i + seq_length - 1]  # Individual label values at prediction timestep
                label_avg = np.mean(individual_labels)  # Average of the specific labels we want to predict
                X.append(seq)
                y.append(label_avg)
                i += 1  # Sliding window, step by 1
            else:
                # Skip to the next potential start (after the gap)
                i += 1
        
        return np.array(X), np.array(y)
    
    def create_sequences(self, df, seq_length, feature_cols, target_col):
        """Create sequences for LSTM testing, respecting data continuity"""
        df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
        features = df[feature_cols].values
        targets = df[target_col].values
        days = df['TradingDay'].values
        ms = df['TradingMsOfDay'].values
        
        X = []
        y = []
        
        for i in range(seq_length, len(df)):
            # Check if we have a continuous sequence
            current_day = days[i]
            current_ms = ms[i]
            
            # Get the sequence window
            seq_days = days[i-seq_length:i]
            seq_ms = ms[i-seq_length:i]
            
            # Check for day continuity (allow same day or consecutive days)
            day_diff = np.diff(seq_days)
            valid_day_sequence = np.all((day_diff == 0) | (day_diff == 1))
            
            # Check for time continuity within days
            valid_time_sequence = True
            for j in range(len(seq_days)-1):
                if seq_days[j] == seq_days[j+1]:
                    # Same day - check time continuity
                    if seq_ms[j+1] != seq_ms[j] + 300000:  # 5 minutes = 300,000 ms
                        valid_time_sequence = False
                        break
                else:
                    # Different day - last time of previous day should be close to end, first time should be close to start
                    if seq_ms[j] < 43200000 - 300000 or seq_ms[j+1] > 38100000 + 300000:  # Allow some tolerance
                        valid_time_sequence = False
                        break
            
            if valid_day_sequence and valid_time_sequence:
                X.append(features[i-seq_length:i])
                y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def get_thresholded_direction_accuracies(self, actual, predicted):
        """Calculate thresholded accuracies for all thresholds"""
        actual_up = (actual > 0)
        actual_down = (actual <= 0)
        
        results = {}
        thresholds = np.arange(0, 0.81, 0.1)
        
        for t in thresholds:
            # Upside predictions (positive threshold)
            pred_up_thr = (predicted > t)
            n_pred_up_thr = np.sum(pred_up_thr)
            tu_thr = np.sum(pred_up_thr & actual_up)
            up_acc_thr = tu_thr / n_pred_up_thr if n_pred_up_thr > 0 else 0.0
            
            # Downside predictions (negative threshold)
            pred_down_thr = (predicted < -t)
            n_pred_down_thr = np.sum(pred_down_thr)
            td_thr = np.sum(pred_down_thr & actual_down)
            down_acc_thr = td_thr / n_pred_down_thr if n_pred_down_thr > 0 else 0.0
            
            results[f'upside_{t:.1f}'] = up_acc_thr
            results[f'downside_{t:.1f}'] = down_acc_thr
            
        return results
    
    def test_model_on_regime(self, model_id, regime):
        """Test a single model on a specific regime"""
        try:
            print(f"Testing model {model_id} on regime {regime}...")
            
            # Get model info
            model_info = self.model_log[self.model_log['model_id'] == model_id].iloc[0]
            seq_length = model_info['seq_length']
            
            # Load model files
            model_file = self.models_dir / f"lstm_stock_model_avg_{model_id:05d}.keras"
            scaler_file = self.models_dir / f"scaler_params_avg_{model_id:05d}.json"
            
            if not model_file.exists():
                print(f"Warning: Model file not found: {model_file}")
                return None
                
            if not scaler_file.exists():
                print(f"Warning: Scaler file not found: {scaler_file}")
                return None
            
            # Load model
            model = tf.keras.models.load_model(model_file)
            
            # Load scaler parameters
            with open(scaler_file, 'r') as f:
                scaler_params = json.load(f)
            
            # Get regime days - handle different formats for daily vs sequence clustering
            if self.clustering_method == 'daily':
                # Daily clustering: one regime per trading day
                regime_days = self.regime_assignments[self.regime_assignments['Regime'] == regime]['trading_day'].unique()
            elif self.clustering_method == 'sequence':
                # Sequence clustering: multiple regime assignments per day, need to get unique trading days
                regime_days = self.regime_assignments[self.regime_assignments['Regime'] == regime]['trading_day'].unique()
            
            if len(regime_days) == 0:
                print(f"Warning: No days found for regime {regime}")
                return None
            
            # Filter trading data for the specific regime dates
            if self.clustering_method == 'daily':
                regime_dates = self.regime_assignments[self.regime_assignments['Regime'] == regime]['trading_day'].values
            else:  # sequence
                regime_dates = self.regime_assignments[self.regime_assignments['Regime'] == regime]['trading_day'].values
            
            # Convert regime dates to match trading data format (as float)
            # Trading data has dates as float64 (e.g., 20200102.0)
            regime_dates_formatted = [float(int(date)) for date in regime_dates]
            
            # CRITICAL: Filter out training period to avoid data leakage
            train_start = model_info['train_from']
            train_end = model_info['train_to']
            
            # Convert training dates to proper format for filtering
            if isinstance(train_start, str):
                train_start = float(train_start)
            if isinstance(train_end, str):
                train_end = float(train_end)
            
            # Remove regime dates that fall within the training period
            regime_dates_test_only = [date for date in regime_dates_formatted 
                                    if not (train_start <= date <= train_end)]
            
            print(f"Model {model_id} training period: {train_start} to {train_end}")
            if self.clustering_method == 'daily':
                print(f"Regime {regime}: {len(regime_dates_formatted)} total days, {len(regime_dates_test_only)} test days (excluding training period)")
            else:  # sequence
                print(f"Regime {regime}: {len(regime_dates_formatted)} total time windows, {len(regime_dates_test_only)} test days (excluding training period)")
            
            if len(regime_dates_test_only) == 0:
                print(f"Warning: No test data for regime {regime} outside training period for model {model_id}")
                return None
            
            regime_data = self.trading_data[self.trading_data['TradingDay'].isin(regime_dates_test_only)]
            
            if len(regime_data) == 0:
                print(f"Warning: No trading data found for regime {regime} test period")
                return None
                
            # Define feature columns exactly as used in training (columns 5-48, 0-based index)
            # Get all columns and select the feature range
            all_columns = regime_data.columns.tolist()
            if len(all_columns) < 49:
                print(f"Warning: Expected at least 49 columns, but found {len(all_columns)}")
                return None
            
            # Use the same feature columns as training script (columns 5-48)
            available_feature_cols = all_columns[5:49]
            num_features = len(available_feature_cols)
            
            # Calculate target label range (same as training script for understanding)
            label_number = model_info['label_number']
            half_window = label_number // 2
            start_label = label_number - half_window
            end_label = label_number + half_window
            label_range = list(range(start_label, end_label + 1))
            
            # For inference, we want to predict the EXACT target label, not the average
            target_col = f'Label_{label_number}'
            
            # Verify target column exists
            if target_col not in regime_data.columns:
                print(f"Warning: Target column {target_col} not found for model {model_id}")
                return None
            
            print(f"Model {model_id} was trained on average of labels {label_range}, testing on exact {target_col}")
            
            # Create sequences using the same method as training (for features)
            # But target the exact label instead of the average
            X_test, y_test = self.create_sequences_for_inference(regime_data, seq_length, available_feature_cols, target_col)
            
            if len(X_test) == 0:
                print(f"Warning: No valid sequences created for model {model_id} on regime {regime}")
                return None
            
            # Reconstruct scaler from saved params
            if 'Min' in scaler_params and 'Max' in scaler_params:
                # MinMaxScaler
                scaler = MinMaxScaler()
                scaler.min_ = np.array(scaler_params['Min'])
                scaler.scale_ = np.array(scaler_params['Max']) - np.array(scaler_params['Min'])
                scaler.data_min_ = np.array(scaler_params['Min'])
                scaler.data_max_ = np.array(scaler_params['Max'])
                scaler.data_range_ = scaler.scale_
            elif 'Mean' in scaler_params and 'Variance' in scaler_params:
                # StandardScaler
                scaler = StandardScaler()
                scaler.mean_ = np.array(scaler_params['Mean'])
                scaler.var_ = np.array(scaler_params['Variance'])
                scaler.scale_ = np.sqrt(scaler.var_)
            else:
                raise ValueError(f"Unknown scaler parameters: {list(scaler_params.keys())}")
            
            # Scale features
            X_test_reshaped = X_test.reshape(-1, num_features)
            X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled, verbose=0).flatten()
            
            # Calculate performance metrics
            mae = np.mean(np.abs(y_test - y_pred))
            accuracies = self.get_thresholded_direction_accuracies(y_test, y_pred)
            
            # Prepare result
            result = {
                'model_id': model_id,
                'regime': regime,
                'training_from': model_info['train_from'],
                'training_to': model_info['train_to'],
                'test_days_available': len(regime_dates_test_only),
                'test_samples': len(y_test),
                'mae': mae
            }
            
            # Add threshold accuracies
            result.update(accuracies)
            
            return result
            
        except Exception as e:
            print(f"Error testing model {model_id} on regime {regime}: {e}")
            return None
    
    def test_filtering_logic(self):
        """Quick test to verify training period filtering logic is working"""
        print("\n" + "="*50)
        print("TESTING TRAINING PERIOD FILTERING LOGIC")
        print("="*50)
        
        if self.model_log is None or self.regime_assignments is None:
            print("Error: Data not loaded. Call load_data() first.")
            return
        
        # Test with first model and first regime
        test_model_id = self.model_log.iloc[0]['model_id']
        test_regime = self.regime_characteristics.iloc[0]['Regime']
        
        model_info = self.model_log[self.model_log['model_id'] == test_model_id].iloc[0]
        train_start = model_info['train_from']
        train_end = model_info['train_to']
        
        print(f"Testing with Model {test_model_id}")
        print(f"Model training period: {train_start} to {train_end}")
        print(f"Testing on Regime {test_regime}")
        
        # Get regime dates
        regime_dates = self.regime_assignments[self.regime_assignments['Regime'] == test_regime]['trading_day'].values
        regime_dates_formatted = [float(int(date)) for date in regime_dates]
        
        print(f"Total regime days: {len(regime_dates_formatted)}")
        
        # Apply filtering
        if isinstance(train_start, str):
            train_start = float(train_start)
        if isinstance(train_end, str):
            train_end = float(train_end)
            
        regime_dates_test_only = [date for date in regime_dates_formatted 
                                if not (train_start <= date <= train_end)]
        
        print(f"Test days (after filtering training period): {len(regime_dates_test_only)}")
        
        # Show some example dates
        training_dates = [date for date in regime_dates_formatted 
                         if train_start <= date <= train_end]
        
        print(f"Training dates removed: {len(training_dates)}")
        if training_dates:
            print(f"  Example training dates: {training_dates[:5]}...")
        if regime_dates_test_only:
            print(f"  Example test dates: {regime_dates_test_only[:5]}...")
        
        print("✅ Filtering logic verification completed")
        
        return len(regime_dates_test_only) > 0

    def run_all_tests(self):
        """Test all models on all regimes"""
        print("Starting comprehensive model-regime testing...")
        
        models = self.model_log['model_id'].tolist()
        regimes = self.regime_characteristics['Regime'].tolist()
        
        total_tests = len(models) * len(regimes)
        completed_tests = 0
        
        for model_id in models:
            for regime in regimes:
                result = self.test_model_on_regime(model_id, regime)
                if result is not None:
                    self.test_results.append(result)
                
                completed_tests += 1
                if completed_tests % 10 == 0:
                    print(f"Progress: {completed_tests}/{total_tests} tests completed ({completed_tests/total_tests*100:.1f}%)")
        
        print(f"Completed {len(self.test_results)} successful tests out of {total_tests} attempted")
    
    def calculate_rankings(self):
        """Calculate model rankings for each regime based on performance metrics"""
        if not self.test_results:
            raise ValueError("No test results available. Run tests first.")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.test_results)
        
        # Calculate rankings for each regime and direction
        ranking_results = []
        
        regimes = results_df['regime'].unique()
        
        for regime in regimes:
            regime_results = results_df[results_df['regime'] == regime].copy()
            
            if len(regime_results) == 0:
                continue
            
            # Calculate average upside and downside accuracies across all thresholds
            upside_cols = [col for col in regime_results.columns if col.startswith('upside_')]
            downside_cols = [col for col in regime_results.columns if col.startswith('downside_')]
            
            regime_results['avg_upside_acc'] = regime_results[upside_cols].mean(axis=1)
            regime_results['avg_downside_acc'] = regime_results[downside_cols].mean(axis=1)
            
            # Rank models by average accuracy (higher is better)
            regime_results['upside_rank'] = regime_results['avg_upside_acc'].rank(ascending=False, method='min')
            regime_results['downside_rank'] = regime_results['avg_downside_acc'].rank(ascending=False, method='min')
            
            # Add ranking columns to original results
            for idx, row in regime_results.iterrows():
                # Find the corresponding result in test_results
                for i, result in enumerate(self.test_results):
                    if (result['model_id'] == row['model_id'] and 
                        result['regime'] == row['regime']):
                        self.test_results[i]['model_regime_upside_rank'] = row['upside_rank']
                        self.test_results[i]['model_regime_downside_rank'] = row['downside_rank']
                        break
    
    def save_test_results(self):
        """Save detailed test results to CSV"""
        if not self.test_results:
            raise ValueError("No test results to save")
        
        results_df = pd.DataFrame(self.test_results)
        
        # Reorder columns to match expected format
        base_cols = ['model_id', 'regime', 'training_from', 'training_to', 
                    'model_regime_upside_rank', 'model_regime_downside_rank', 
                    'test_days_available', 'test_samples', 'mae']
        
        # Group threshold columns by direction instead of alternating
        upside_cols = []
        downside_cols = []
        for t in np.arange(0, 0.81, 0.1):
            upside_cols.append(f'upside_{t:.1f}')
            downside_cols.append(f'downside_{t:.1f}')
        
        threshold_cols = upside_cols + downside_cols
        
        # Order columns
        ordered_cols = base_cols + threshold_cols
        available_cols = [col for col in ordered_cols if col in results_df.columns]
        
        results_df = results_df[available_cols]
        
        # Save to file
        output_file = self.output_dir / f"model_regime_test_results_{self.clustering_method}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Saved detailed test results to: {output_file}")
    
    def save_ranking_summary(self):
        """Save model ranking summary to CSV"""
        if not self.test_results:
            raise ValueError("No test results to save")
        
        results_df = pd.DataFrame(self.test_results)
        
        # Create ranking summary
        ranking_data = []
        
        models = results_df['model_id'].unique()
        regimes = sorted(results_df['regime'].unique())
        
        for model_id in models:
            model_results = results_df[results_df['model_id'] == model_id]
            model_info = self.model_log[self.model_log['model_id'] == model_id].iloc[0]
            
            ranking_row = {
                'model_id': model_id,
                'training_from': model_info['train_from'],
                'training_to': model_info['train_to']
            }
            
            # Initialize best regime tracking
            best_upside_regime = None
            best_upside_rank = float('inf')
            best_downside_regime = None
            best_downside_rank = float('inf')
            
            # Add regime rankings - group by direction instead of alternating
            upside_rankings = {}
            downside_rankings = {}
            
            for regime in regimes:
                regime_result = model_results[model_results['regime'] == regime]
                
                if len(regime_result) > 0:
                    upside_rank = regime_result.iloc[0]['model_regime_upside_rank']
                    downside_rank = regime_result.iloc[0]['model_regime_downside_rank']
                    
                    upside_rankings[f'regime_{regime}_up'] = int(upside_rank)
                    downside_rankings[f'regime_{regime}_down'] = int(downside_rank)
                    
                    # Track best regimes
                    if upside_rank < best_upside_rank:
                        best_upside_rank = upside_rank
                        best_upside_regime = regime
                    
                    if downside_rank < best_downside_rank:
                        best_downside_rank = downside_rank
                        best_downside_regime = regime
                else:
                    upside_rankings[f'regime_{regime}_up'] = None
                    downside_rankings[f'regime_{regime}_down'] = None
            
            # Add upside rankings first, then downside rankings
            ranking_row.update(upside_rankings)
            ranking_row.update(downside_rankings)
            
            # Add best regime columns
            ranking_row['model_upside_regime'] = best_upside_regime
            ranking_row['model_downside_regime'] = best_downside_regime
            
            ranking_data.append(ranking_row)
        
        # Create DataFrame and save
        ranking_df = pd.DataFrame(ranking_data)
        
        output_file = self.output_dir / f"model_regime_ranking_{self.clustering_method}.csv"
        ranking_df.to_csv(output_file, index=False)
        print(f"Saved ranking summary to: {output_file}")
        
        # Print summary statistics
        print("\n=== RANKING SUMMARY ===")
        print(f"Total models tested: {len(ranking_df)}")
        print(f"Total regimes: {len(regimes)}")
        
        # Show best performing models per regime
        results_df_with_ranks = pd.DataFrame(self.test_results)
        
        print("\nBest models per regime (upside):")
        for regime in regimes:
            regime_results = results_df_with_ranks[results_df_with_ranks['regime'] == regime]
            if len(regime_results) > 0:
                best_model = regime_results.loc[regime_results['model_regime_upside_rank'].idxmin()]
                print(f"  Regime {regime}: Model {best_model['model_id']} (rank {best_model['model_regime_upside_rank']:.0f})")
        
        print("\nBest models per regime (downside):")
        for regime in regimes:
            regime_results = results_df_with_ranks[results_df_with_ranks['regime'] == regime]
            if len(regime_results) > 0:
                best_model = regime_results.loc[regime_results['model_regime_downside_rank'].idxmin()]
                print(f"  Regime {regime}: Model {best_model['model_id']} (rank {best_model['model_regime_downside_rank']:.0f})")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model performance on market regimes')
    parser.add_argument('--max_models', type=int, default=None, 
                       help='Maximum number of models to test (for quick testing)')
    parser.add_argument('--test_filtering', action='store_true',
                       help='Only test the filtering logic without running full tests')
    parser.add_argument('--clustering_method', choices=['daily', 'sequence'], default='daily',
                       help='Clustering method to use: daily (GMM daily clustering) or sequence (GMM sequence clustering)')
    args = parser.parse_args()
    
    print("="*60)
    print(f"MODEL REGIME PERFORMANCE TESTER ({args.clustering_method.upper()} CLUSTERING)")
    print("="*60)
    
    # Create tester instance
    tester = ModelRegimeTester(max_models=args.max_models, clustering_method=args.clustering_method)
    
    try:
        # Load data
        tester.load_data()
        
        if args.test_filtering:
            # Only test filtering logic
            tester.test_filtering_logic()
            return
        
        # Run all tests
        tester.run_all_tests()
        
        # Calculate rankings
        tester.calculate_rankings()
        
        # Save results
        tester.save_test_results()
        tester.save_ranking_summary()
        
        print("\n" + "="*60)
        print(f"TESTING COMPLETED SUCCESSFULLY ({args.clustering_method.upper()} CLUSTERING)")
        print("="*60)
        print(f"Results saved to: {tester.output_dir}")
        print(f"Clustering method: {args.clustering_method}")
        print(f"Output files:")
        print(f"  - model_regime_test_results_{args.clustering_method}.csv")
        print(f"  - model_regime_ranking_{args.clustering_method}.csv")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
