#!/usr/bin/env python3
"""
Comprehensive Model Testing Across All Regimes

This script tests all models from model_log.csv against all regimes,
evaluating performance with threshold-based accuracy metrics.

Usage:
    python test_all_models_regimes.py [--start_model 1] [--end_model 425]
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import json
import csv
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")

# Get project paths
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
data_dir = project_root / 'data'
models_dir = project_root / 'models'
paradigm_dir = project_root / 'regime_analysis'
results_dir = project_root / 'test_results'

# Ensure directories exist
results_dir.mkdir(exist_ok=True)

class ModelTester:
    """Test models across all paradigms"""
    
    def __init__(self):
        self.regime_assignments = None
        self.trading_data = None
        self.paradigm_data_cache = {}
        self.load_regime_assignments()
        
    def load_regime_assignments(self):
        """Load regime assignments"""
        print("Loading regime assignments...")
        paradigm_file = paradigm_dir / 'regime_assignments.csv'
        if not paradigm_file.exists():
            raise FileNotFoundError(f"Regime assignments file not found: {paradigm_file}")
        
        self.regime_assignments = pd.read_csv(paradigm_file)
        print(f"Loaded regime assignments: {len(self.regime_assignments):,} rows")
        
        # Get available paradigms
        self.available_paradigms = sorted(self.regime_assignments['Regime'].unique())
        print(f"Available paradigms: {self.available_paradigms}")
        
    def load_trading_data(self):
        """Load full trading data"""
        if self.trading_data is None:
            print("Loading trading data...")
            data_file = data_dir / 'trainingData.csv'
            if not data_file.exists():
                raise FileNotFoundError(f"Training data file not found: {data_file}")
            
            self.trading_data = pd.read_csv(data_file)
            print(f"Loaded trading data: {len(self.trading_data):,} rows")
            
    def get_paradigm_data(self, paradigm_number):
        """Get data for a specific regime (cached)"""
        if paradigm_number not in self.paradigm_data_cache:
            print(f"Caching data for regime {paradigm_number}...")
            
            # Get trading days for this regime
            paradigm_data = self.regime_assignments[
                self.regime_assignments['Regime'] == paradigm_number
            ]
            paradigm_trading_days = set(paradigm_data['TradingDay'].unique())
            
            # Filter trading data
            filtered_data = self.trading_data[
                self.trading_data['TradingDay'].isin(paradigm_trading_days)
            ].reset_index(drop=True)
            
            self.paradigm_data_cache[paradigm_number] = filtered_data
            print(f"Cached {len(filtered_data):,} rows for regime {paradigm_number}")
            
        return self.paradigm_data_cache[paradigm_number]
    
    def exclude_training_period(self, data, train_from, train_to):
        """Exclude training period from test data"""
        # Convert date strings to integers for comparison
        train_from_int = int(train_from)
        train_to_int = int(train_to)
        
        # Filter out training period
        filtered_data = data[
            (data['TradingDay'] < train_from_int) | 
            (data['TradingDay'] > train_to_int)
        ].reset_index(drop=True)
        
        return filtered_data
    
    def create_sequences(self, df, seq_length, feature_cols, target_col):
        """Create sequences for testing"""
        if len(df) == 0:
            return np.array([]), np.array([])
            
        df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
        features = df[feature_cols].values
        targets = df[target_col].values
        days = df['TradingDay'].values
        ms = df['TradingMsOfDay'].values
        
        X = []
        y = []
        
        i = 0
        while i <= len(df) - seq_length:
            # Check if the sequence is consecutive
            is_consecutive = True
            current_day = days[i]
            current_ms = ms[i]
            
            for j in range(1, seq_length):
                if days[i + j] != current_day or ms[i + j] != current_ms + j * 60000:
                    is_consecutive = False
                    break
            
            if is_consecutive:
                seq = features[i:i + seq_length]
                label = targets[i + seq_length - 1]
                X.append(seq)
                y.append(label)
                i += 1
            else:
                i += 1
        
        return np.array(X), np.array(y)
    
    def get_thresholded_accuracies(self, actual, predicted):
        """Calculate thresholded accuracies for all thresholds"""
        actual_up = (actual > 0)
        actual_down = (actual <= 0)
        
        results = {}
        thresholds = np.arange(0, 0.9, 0.1)
        
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
    
    def test_model_on_paradigm(self, model_info, paradigm_number):
        """Test a single model on a single regime"""
        model_id = model_info['model_id']
        seq_length = int(model_info['seq_length'])
        label_number = int(model_info['label_number'])
        train_from = model_info['train_from']
        train_to = model_info['train_to']
        
        try:
            # Load model - using the correct filename format
            model_path = models_dir / f'lstm_stock_model_{model_id}.keras'
            if not model_path.exists():
                print(f"Model file not found: {model_path}")
                return None
                
            model = keras.models.load_model(model_path)
            
            # Load scaler - using the correct filename format
            scaler_path = models_dir / f'scaler_params_{model_id}.json'
            if not scaler_path.exists():
                print(f"Scaler file not found: {scaler_path}")
                return None
                
            with open(scaler_path, 'r') as f:
                scaler_params = json.load(f)
            
            scaler = MinMaxScaler()
            scaler.data_min_ = np.array(scaler_params['Min'])
            scaler.data_max_ = np.array(scaler_params['Max'])
            scaler.scale_ = 1.0 / (scaler.data_max_ - scaler.data_min_)
            scaler.min_ = -scaler.data_min_ * scaler.scale_
            
            # Get regime data (excluding training period)
            paradigm_data = self.get_paradigm_data(paradigm_number)
            test_data = self.exclude_training_period(paradigm_data, train_from, train_to)
            
            if len(test_data) == 0:
                print(f"No test data available for regime {paradigm_number} after excluding training period")
                return None
            
            # Define features and target
            feature_cols = test_data.columns[5:49]  # Features: columns 6 to 49
            target_col = f'Label_{label_number}'
            num_features = len(feature_cols)
            
            # Create sequences
            X_test, y_test = self.create_sequences(test_data, seq_length, feature_cols, target_col)
            
            if len(X_test) == 0:
                print(f"No sequences available for testing regime {paradigm_number}")
                return None
            
            # Scale features
            X_test_reshaped = X_test.reshape(-1, num_features)
            X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
            
            # Make predictions
            predictions = model.predict(X_test_scaled, verbose=0).flatten()
            
            # Calculate accuracies
            accuracies = self.get_thresholded_accuracies(y_test, predictions)
            
            # Calculate MAE for comparison
            mae = np.mean(np.abs(y_test - predictions))
            
            result = {
                'model_id': model_id,
                'regime': paradigm_number,
                'test_samples': len(y_test),
                'mae': mae,
                **accuracies
            }
            
            return result
            
        except Exception as e:
            print(f"Error testing model {model_id} on regime {paradigm_number}: {str(e)}")
            return None
    
    def find_best_paradigm(self, model_results):
        """Find the best regime for a model based on average accuracy"""
        if not model_results:
            return None, 0.0, None
            
        best_paradigm = None
        best_score = -1
        best_side = None
        
        for regime, result in model_results.items():
            if result is None:
                continue
                
            # Calculate average accuracy for upside and downside separately
            upside_accs = [result[f'upside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
            downside_accs = [result[f'downside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
            
            upside_avg = np.mean(upside_accs)
            downside_avg = np.mean(downside_accs)
            
            # Check if upside is better
            if upside_avg > best_score:
                best_score = upside_avg
                best_paradigm = regime
                best_side = 'up'
            
            # Check if downside is better
            if downside_avg > best_score:
                best_score = downside_avg
                best_paradigm = regime
                best_side = 'down'
                
        return best_paradigm, best_score, best_side
    
    def test_all_models(self, start_model=1, end_model=425):
        """Test all models across all paradigms"""
        print(f"Testing models {start_model} to {end_model} across all paradigms...")
        
        # Load trading data
        self.load_trading_data()
        
        # Load model log
        model_log_path = models_dir / 'model_log.csv'
        if not model_log_path.exists():
            raise FileNotFoundError(f"Model log file not found: {model_log_path}")
        
        # Read model log with proper data types
        model_log = pd.read_csv(model_log_path, dtype={'model_id': str})
        
        # Filter models in range
        available_models = []
        for _, row in model_log.iterrows():
            try:
                model_id_str = str(row['model_id']).strip()
                # Handle the case where pandas reads as float
                if '.' in model_id_str:
                    model_num = int(float(model_id_str))
                    model_id_str = f"{model_num:05d}"
                else:
                    model_num = int(model_id_str)
                
                if start_model <= model_num <= end_model:
                    available_models.append((model_num, model_id_str))
            except (ValueError, TypeError):
                continue
        
        available_models = sorted(available_models)
        print(f"Found {len(available_models)} models to test: {[m[0] for m in available_models[:10]]}...")
        
        if len(available_models) == 0:
            print("No models found in the specified range. Check your model_log.csv file.")
            print("Available model IDs:")
            for _, row in model_log.head(10).iterrows():
                print(f"  {row['model_id']}")
            return []
        
        # Prepare results
        all_results = []
        
        # Test each model
        for i, (model_num, model_id_str) in enumerate(available_models):
            print(f"\nTesting model {model_num} ({i+1}/{len(available_models)})...")
            
            # Get model info
            model_info = model_log[model_log['model_id'] == model_id_str].iloc[0]
            
            # Test on all paradigms
            model_results = {}
            for regime in self.available_paradigms:
                print(f"  Testing on regime {regime}...")
                result = self.test_model_on_paradigm(model_info, regime)
                model_results[regime] = result
            
            # Save results for each regime
            for regime, result in model_results.items():
                if result is not None:
                    # Add model metadata
                    result.update({
                        'training_from': model_info['train_from'],
                        'training_to': model_info['train_to']
                    })
                    all_results.append(result)
            
            print(f"  Completed testing model {model_num} on all regimes")
        
        # Save results
        self.save_results(all_results, start_model, end_model)
        
        return all_results
    
    def save_results(self, results, start_model, end_model):
        """Save test results to CSV"""
        if not results:
            print("No results to save")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Calculate ranking within each regime
        print("Calculating regime-specific rankings...")
        
        # For each regime, calculate separate upside and downside rankings
        for regime in sorted(df['regime'].unique()):
            regime_indices = df[df['regime'] == regime].index
            regime_data = df.loc[regime_indices].copy()
            
            # Calculate upside and downside scores separately
            upside_scores = []
            downside_scores = []
            for idx in regime_indices:
                row = df.loc[idx]
                upside_accs = [row[f'upside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
                downside_accs = [row[f'downside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
                upside_scores.append(np.mean(upside_accs))
                downside_scores.append(np.mean(downside_accs))
            
            # Rank models separately for upside and downside (1 = best, higher numbers = worse)
            upside_ranks = pd.Series(upside_scores).rank(method='dense', ascending=False).astype(int)
            downside_ranks = pd.Series(downside_scores).rank(method='dense', ascending=False).astype(int)
            
            # Update the dataframe with separate rankings
            df.loc[regime_indices, 'model_regime_upside_rank'] = upside_ranks.values
            df.loc[regime_indices, 'model_regime_downside_rank'] = downside_ranks.values
        
        # Remove temporary columns
        if 'is_best_regime' in df.columns:
            df = df.drop('is_best_regime', axis=1)
        if 'best_paradigm' in df.columns:
            df = df.drop('best_paradigm', axis=1)
        if 'best_regime' in df.columns:
            df = df.drop('best_regime', axis=1)
        if 'best_side' in df.columns:
            df = df.drop('best_side', axis=1)
        
        # Reorder columns - remove best_regime and best_side, split model_regime_rank
        base_cols = ['model_id', 'regime', 'training_from', 'training_to', 
                    'model_regime_upside_rank', 'model_regime_downside_rank', 'test_samples', 'mae']
        
        # Add threshold columns - all upside first, then all downside
        upside_cols = [f'upside_{t:.1f}' for t in np.arange(0, 0.9, 0.1)]
        downside_cols = [f'downside_{t:.1f}' for t in np.arange(0, 0.9, 0.1)]
        threshold_cols = upside_cols + downside_cols
        
        ordered_cols = base_cols + threshold_cols
        df = df[ordered_cols]
        
        # Save main results
        output_file = results_dir / f'model_regime_test_results_{start_model}_{end_model}.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved detailed results to {output_file}")
        
        # Generate daily best model tracking
        self.generate_daily_best_models(df, start_model, end_model)
        
        # Generate weekly best model tracking
        self.generate_weekly_best_models(df, start_model, end_model)
        
        # Create new ranking-based summary
        print("Creating ranking-based summary...")
        
        # Get unique models and regimes
        unique_models = sorted(df['model_id'].unique())
        unique_regimes = sorted(df['regime'].unique())
        
        # Initialize summary data with model info
        summary_data = []
        
        for model_id in unique_models:
            model_data = df[df['model_id'] == model_id]
            if len(model_data) == 0:
                continue
                
            # Get training period info (should be same for all regimes for this model)
            training_from = model_data['training_from'].iloc[0]
            training_to = model_data['training_to'].iloc[0]
            
            # Initialize row with basic info
            row = {
                'model_id': model_id,
                'training_from': training_from,
                'training_to': training_to
            }
            
            # Add regime rankings - upside first, then downside
            for regime in unique_regimes:
                regime_model_data = model_data[model_data['regime'] == regime]
                
                if len(regime_model_data) > 0:
                    # Calculate upside and downside scores for this model in this regime
                    model_row = regime_model_data.iloc[0]
                    
                    upside_accs = [model_row[f'upside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
                    downside_accs = [model_row[f'downside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
                    
                    upside_score = np.mean(upside_accs)
                    downside_score = np.mean(downside_accs)
                    
                    # Get all models' scores for this regime to calculate ranks
                    regime_all_data = df[df['regime'] == regime]
                    
                    # Calculate upside ranks
                    all_upside_scores = []
                    model_ids_list = list(regime_all_data['model_id'])
                    
                    for _, row_data in regime_all_data.iterrows():
                        all_upside_accs = [row_data[f'upside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
                        all_upside_scores.append(np.mean(all_upside_accs))
                    
                    upside_rank = pd.Series(all_upside_scores).rank(method='dense', ascending=False)
                    model_upside_rank = upside_rank.iloc[model_ids_list.index(model_id)]
                    
                    # Calculate downside ranks
                    all_downside_scores = []
                    for _, row_data in regime_all_data.iterrows():
                        all_downside_accs = [row_data[f'downside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
                        all_downside_scores.append(np.mean(all_downside_accs))
                    
                    downside_rank = pd.Series(all_downside_scores).rank(method='dense', ascending=False)
                    model_downside_rank = downside_rank.iloc[model_ids_list.index(model_id)]
                    
                    row[f'regime_{regime}_up'] = int(model_upside_rank)
                    row[f'regime_{regime}_down'] = int(model_downside_rank)
                else:
                    # No data for this regime
                    row[f'regime_{regime}_up'] = np.nan
                    row[f'regime_{regime}_down'] = np.nan
            
            summary_data.append(row)
        
        # Create summary DataFrame
        best_regime_summary = pd.DataFrame(summary_data)
        
        # Save summary
        summary_file = results_dir / f'best_regime_summary_{start_model}_{end_model}.csv'
        best_regime_summary.to_csv(summary_file, index=False)
        print(f"Saved ranking-based summary to {summary_file}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"Total test combinations: {len(df)}")
        print(f"Unique models tested: {df['model_id'].nunique()}")
        print(f"Summary includes rankings for {len(unique_regimes)} regimes: {unique_regimes}")
        print(f"Ranking columns: upside and downside rankings for each regime")
    
    def generate_daily_best_models(self, df, start_model, end_model):
        """Generate daily best model tracking using competitive regime-based selection"""
        print("Generating daily best model tracking with competitive regime-based selection...")
        
        # Load the enhanced best regime summary with regime base columns
        best_regime_file = results_dir / 'best_regime_summary_1_425.csv'
        if not best_regime_file.exists():
            print(f"ERROR: Best regime summary file not found: {best_regime_file}")
            print("Please run the full model analysis first to generate this file.")
            return None
            
        regime_base_df = pd.read_csv(best_regime_file)
        print(f"Loaded regime base summary with {len(regime_base_df)} models")
        
        # Get all unique trading days from regime assignments
        unique_trading_days = sorted(self.regime_assignments['TradingDay'].unique())
        unique_regimes = sorted(df['regime'].unique())
        
        # Organize models by their regime bases
        print("Organizing models by regime bases...")
        upside_regime_models = {}  # regime -> list of model_ids
        downside_regime_models = {}  # regime -> list of model_ids
        
        for _, row in regime_base_df.iterrows():
            model_id = f"{int(row['model_id']):05d}"  # Convert back to 00001 format
            
            # Group by upside regime base
            upside_regime = row['model_regime_upside']
            if upside_regime not in upside_regime_models:
                upside_regime_models[upside_regime] = []
            upside_regime_models[upside_regime].append(model_id)
            
            # Group by downside regime base
            downside_regime = row['model_regime_downside'] 
            if downside_regime not in downside_regime_models:
                downside_regime_models[downside_regime] = []
            downside_regime_models[downside_regime].append(model_id)
        
        # Print regime base distributions
        for regime in sorted(upside_regime_models.keys()):
            count = len(upside_regime_models[regime])
            print(f"  Upside regime {regime}: {count} models")
            
        for regime in sorted(downside_regime_models.keys()):
            count = len(downside_regime_models[regime])
            print(f"  Downside regime {regime}: {count} models")
        
        # Filter models to the requested range
        valid_models = set(f"{i:05d}" for i in range(start_model, end_model + 1))
        
        # Filter regime model lists to only include valid models
        for regime in upside_regime_models:
            upside_regime_models[regime] = [m for m in upside_regime_models[regime] if m in valid_models]
            
        for regime in downside_regime_models:
            downside_regime_models[regime] = [m for m in downside_regime_models[regime] if m in valid_models]
        
        print(f"Filtered to models in range {start_model}-{end_model}")
        
        daily_results = []
        
        for trading_day in unique_trading_days:
            # Get the regime for this trading day
            day_regime_data = self.regime_assignments[
                self.regime_assignments['TradingDay'] == trading_day
            ]
            if len(day_regime_data) == 0:
                continue
                
            actual_regime = day_regime_data['Regime'].iloc[0]
            
            # For each test regime, find the best models through competition
            for test_regime in unique_regimes:
                
                # Get candidate models for upside in this regime
                upside_candidates = upside_regime_models.get(test_regime, [])
                # Filter out models trained on this day
                upside_candidates = [
                    model_id for model_id in upside_candidates
                    if not self._model_trained_on_day(model_id, trading_day, regime_base_df)
                ]
                
                # Get candidate models for downside in this regime  
                downside_candidates = downside_regime_models.get(test_regime, [])
                # Filter out models trained on this day
                downside_candidates = [
                    model_id for model_id in downside_candidates
                    if not self._model_trained_on_day(model_id, trading_day, regime_base_df)
                ]
                
                # Simulate competition: find best performing model for upside
                best_upside_model = None
                best_upside_score = None
                if upside_candidates:
                    # Use performance scores from the test regime for competition
                    upside_results = []
                    for model_id in upside_candidates:
                        # Find this model's performance in the test regime
                        model_data = df[
                            (df['model_id'] == int(model_id)) & 
                            (df['regime'] == test_regime)
                        ]
                        if len(model_data) > 0:
                            row = model_data.iloc[0]
                            upside_accs = [row[f'upside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
                            avg_upside_score = np.mean(upside_accs)
                            upside_results.append((model_id, avg_upside_score))
                    
                    # Select best upside model (highest score wins the competition)
                    if upside_results:
                        upside_results.sort(key=lambda x: x[1], reverse=True)
                        best_upside_model, best_upside_score = upside_results[0]
                
                # Simulate competition: find best performing model for downside
                best_downside_model = None  
                best_downside_score = None
                if downside_candidates:
                    # Use performance scores from the test regime for competition
                    downside_results = []
                    for model_id in downside_candidates:
                        # Find this model's performance in the test regime
                        model_data = df[
                            (df['model_id'] == int(model_id)) & 
                            (df['regime'] == test_regime)
                        ]
                        if len(model_data) > 0:
                            row = model_data.iloc[0]
                            downside_accs = [row[f'downside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
                            avg_downside_score = np.mean(downside_accs)
                            downside_results.append((model_id, avg_downside_score))
                    
                    # Select best downside model (highest score wins the competition)
                    if downside_results:
                        downside_results.sort(key=lambda x: x[1], reverse=True)
                        best_downside_model, best_downside_score = downside_results[0]
                
                # Record the results
                daily_results.append({
                    'trading_day': trading_day,
                    'actual_regime': actual_regime,
                    'test_regime': test_regime,
                    'best_upside_model_id': best_upside_model,
                    'best_upside_score': best_upside_score,
                    'best_upside_rank': 1 if best_upside_model else None,
                    'best_downside_model_id': best_downside_model,
                    'best_downside_score': best_downside_score,
                    'best_downside_rank': 1 if best_downside_model else None,
                    'competing_upside_models': len(upside_candidates),
                    'competing_downside_models': len(downside_candidates)
                })
        
        # Convert to DataFrame and save
        daily_df = pd.DataFrame(daily_results)
        
        # Sort by trading day and regime
        daily_df = daily_df.sort_values(['trading_day', 'test_regime']).reset_index(drop=True)
        
        # Save the daily best models file
        daily_output_file = results_dir / f'daily_best_models_{start_model}_{end_model}.csv'
        daily_df.to_csv(daily_output_file, index=False)
        print(f"Saved daily best models to {daily_output_file}")
        
        # Print some statistics
        print(f"Daily best models summary:")
        print(f"  Total trading days: {len(unique_trading_days):,}")
        print(f"  Total day-regime combinations: {len(daily_df):,}")
        print(f"  Average competing upside models per day-regime: {daily_df['competing_upside_models'].mean():.1f}")
        print(f"  Average competing downside models per day-regime: {daily_df['competing_downside_models'].mean():.1f}")
        
        # Show distribution of competing models
        upside_counts = daily_df['competing_upside_models'].value_counts().sort_index()
        downside_counts = daily_df['competing_downside_models'].value_counts().sort_index()
        print(f"  Competing upside models distribution:")
        for count, freq in upside_counts.head(10).items():
            print(f"    {count} models: {freq:,} combinations")
        print(f"  Competing downside models distribution:")
        for count, freq in downside_counts.head(10).items():
            print(f"    {count} models: {freq:,} combinations")
        
        # Show some examples of selected models
        sample_df = daily_df.sample(min(10, len(daily_df)))
        print(f"\nSample daily competitive selections:")
        for _, row in sample_df.iterrows():
            print(f"  Day {row['trading_day']}, Regime {row['test_regime']}: "
                  f"Upside={row['best_upside_model_id']} (vs {row['competing_upside_models']} competitors), "
                  f"Downside={row['best_downside_model_id']} (vs {row['competing_downside_models']} competitors)")
        
        return daily_df
    
    def _model_trained_on_day(self, model_id, trading_day, regime_base_df):
        """Check if a model was trained on the given trading day"""
        model_row = regime_base_df[regime_base_df['model_id'] == int(model_id)]
        if len(model_row) == 0:
            return True  # Conservative: exclude if model not found
            
        training_from = int(model_row.iloc[0]['training_from'])
        training_to = int(model_row.iloc[0]['training_to'])
        
        return training_from <= trading_day <= training_to
    
    def generate_weekly_best_models(self, df, start_model, end_model):
        """Generate weekly best model tracking using competitive regime-based selection"""
        print("Generating weekly best model tracking with competitive regime-based selection...")
        
        # Load the enhanced best regime summary with regime base columns
        best_regime_file = results_dir / 'best_regime_summary_1_425.csv'
        if not best_regime_file.exists():
            print(f"ERROR: Best regime summary file not found: {best_regime_file}")
            print("Please run the full model analysis first to generate this file.")
            return None
            
        regime_base_df = pd.read_csv(best_regime_file)
        print(f"Loaded regime base summary with {len(regime_base_df)} models")
        
        # Get all unique trading days from regime assignments
        unique_trading_days = sorted(self.regime_assignments['TradingDay'].unique())
        unique_regimes = sorted(df['regime'].unique())
        
        # Organize models by their regime bases (same as daily)
        print("Organizing models by regime bases for weekly competition...")
        upside_regime_models = {}  # regime -> list of model_ids
        downside_regime_models = {}  # regime -> list of model_ids
        
        for _, row in regime_base_df.iterrows():
            model_id = f"{int(row['model_id']):05d}"  # Convert back to 00001 format
            
            # Group by upside regime base
            upside_regime = row['model_regime_upside']
            if upside_regime not in upside_regime_models:
                upside_regime_models[upside_regime] = []
            upside_regime_models[upside_regime].append(model_id)
            
            # Group by downside regime base
            downside_regime = row['model_regime_downside'] 
            if downside_regime not in downside_regime_models:
                downside_regime_models[downside_regime] = []
            downside_regime_models[downside_regime].append(model_id)
        
        # Filter models to the requested range
        valid_models = set(f"{i:05d}" for i in range(start_model, end_model + 1))
        
        # Filter regime model lists to only include valid models
        for regime in upside_regime_models:
            upside_regime_models[regime] = [m for m in upside_regime_models[regime] if m in valid_models]
            
        for regime in downside_regime_models:
            downside_regime_models[regime] = [m for m in downside_regime_models[regime] if m in valid_models]
        
        # Group trading days into weeks
        trading_days_df = pd.DataFrame({'trading_day': unique_trading_days})
        trading_days_df['date'] = pd.to_datetime(trading_days_df['trading_day'], format='%Y%m%d')
        trading_days_df['week_start'] = trading_days_df['date'].dt.to_period('W').dt.start_time
        trading_days_df['week_end'] = trading_days_df['date'].dt.to_period('W').dt.end_time
        
        # Group by week
        weekly_groups = trading_days_df.groupby(['week_start', 'week_end'])
        
        weekly_results = []
        
        for (week_start, week_end), week_data in weekly_groups:
            week_trading_days = week_data['trading_day'].tolist()
            
            # Get regime distribution for this week
            week_regime_data = self.regime_assignments[
                self.regime_assignments['TradingDay'].isin(week_trading_days)
            ]
            
            if len(week_regime_data) == 0:
                continue
            
            # Get the most common regime for this week
            regime_counts = week_regime_data['Regime'].value_counts()
            most_common_regime = regime_counts.index[0]
            
            # Calculate regime distribution percentages
            regime_distribution = {}
            total_records = len(week_regime_data)
            for regime in unique_regimes:
                count = regime_counts.get(regime, 0)
                regime_distribution[f'regime_{regime}_pct'] = count / total_records * 100
            
            # For each test regime, run competitive selection
            for test_regime in unique_regimes:
                
                # Get candidate models for upside in this regime
                upside_candidates = upside_regime_models.get(test_regime, [])
                # Filter out models trained during this week
                upside_candidates = [
                    model_id for model_id in upside_candidates
                    if not self._model_trained_during_week(model_id, week_trading_days, regime_base_df)
                ]
                
                # Get candidate models for downside in this regime  
                downside_candidates = downside_regime_models.get(test_regime, [])
                # Filter out models trained during this week
                downside_candidates = [
                    model_id for model_id in downside_candidates
                    if not self._model_trained_during_week(model_id, week_trading_days, regime_base_df)
                ]
                
                # Simulate weekly competition: find best performing model for upside
                best_upside_model = None
                best_upside_score = None
                if upside_candidates:
                    # Use performance scores from the test regime for competition
                    upside_results = []
                    for model_id in upside_candidates:
                        # Find this model's performance in the test regime
                        model_data = df[
                            (df['model_id'] == int(model_id)) & 
                            (df['regime'] == test_regime)
                        ]
                        if len(model_data) > 0:
                            row = model_data.iloc[0]
                            upside_accs = [row[f'upside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
                            avg_upside_score = np.mean(upside_accs)
                            upside_results.append((model_id, avg_upside_score))
                    
                    # Select best upside model (highest score wins the weekly competition)
                    if upside_results:
                        upside_results.sort(key=lambda x: x[1], reverse=True)
                        best_upside_model, best_upside_score = upside_results[0]
                
                # Simulate weekly competition: find best performing model for downside
                best_downside_model = None  
                best_downside_score = None
                if downside_candidates:
                    # Use performance scores from the test regime for competition
                    downside_results = []
                    for model_id in downside_candidates:
                        # Find this model's performance in the test regime
                        model_data = df[
                            (df['model_id'] == int(model_id)) & 
                            (df['regime'] == test_regime)
                        ]
                        if len(model_data) > 0:
                            row = model_data.iloc[0]
                            downside_accs = [row[f'downside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
                            avg_downside_score = np.mean(downside_accs)
                            downside_results.append((model_id, avg_downside_score))
                    
                    # Select best downside model (highest score wins the weekly competition)
                    if downside_results:
                        downside_results.sort(key=lambda x: x[1], reverse=True)
                        best_downside_model, best_downside_score = downside_results[0]
                
                # Record the weekly results
                result = {
                    'week_start': week_start.strftime('%Y%m%d'),
                    'week_end': week_end.strftime('%Y%m%d'),
                    'week_trading_days': len(week_trading_days),
                    'most_common_regime': most_common_regime,
                    'test_regime': test_regime,
                    'best_upside_model_id': best_upside_model,
                    'best_upside_score': best_upside_score,
                    'best_upside_rank': 1 if best_upside_model else None,
                    'best_downside_model_id': best_downside_model,
                    'best_downside_score': best_downside_score,
                    'best_downside_rank': 1 if best_downside_model else None,
                    'competing_upside_models': len(upside_candidates),
                    'competing_downside_models': len(downside_candidates)
                }
                # Add regime distribution
                result.update(regime_distribution)
                weekly_results.append(result)
        
        # Convert to DataFrame and save
        weekly_df = pd.DataFrame(weekly_results)
        
        # Sort by week start and regime
        weekly_df = weekly_df.sort_values(['week_start', 'test_regime']).reset_index(drop=True)
        
        # Save the weekly best models file
        weekly_output_file = results_dir / f'weekly_best_models_{start_model}_{end_model}.csv'
        weekly_df.to_csv(weekly_output_file, index=False)
        print(f"Saved weekly best models to {weekly_output_file}")
        
        # Print some statistics
        print(f"Weekly best models summary:")
        print(f"  Total weeks: {len(weekly_df['week_start'].unique()):,}")
        print(f"  Total week-regime combinations: {len(weekly_df):,}")
        print(f"  Average competing upside models per week-regime: {weekly_df['competing_upside_models'].mean():.1f}")
        print(f"  Average competing downside models per week-regime: {weekly_df['competing_downside_models'].mean():.1f}")
        print(f"  Average trading days per week: {weekly_df['week_trading_days'].mean():.1f}")
        
        # Show distribution of competing models
        upside_counts = weekly_df['competing_upside_models'].value_counts().sort_index()
        downside_counts = weekly_df['competing_downside_models'].value_counts().sort_index()
        print(f"  Competing upside models distribution:")
        for count, freq in upside_counts.head(10).items():
            print(f"    {count} models: {freq:,} combinations")
        print(f"  Competing downside models distribution:")
        for count, freq in downside_counts.head(10).items():
            print(f"    {count} models: {freq:,} combinations")
        
        # Show some examples of selected models
        sample_df = weekly_df.sample(min(10, len(weekly_df)))
        print(f"\nSample weekly competitive selections:")
        for _, row in sample_df.iterrows():
            print(f"  Week {row['week_start']}-{row['week_end']}, Regime {row['test_regime']}: "
                  f"Upside={row['best_upside_model_id']} (vs {row['competing_upside_models']} competitors), "
                  f"Downside={row['best_downside_model_id']} (vs {row['competing_downside_models']} competitors)")
        
        return weekly_df
    
    def _model_trained_during_week(self, model_id, week_trading_days, regime_base_df):
        """Check if a model was trained during any day in the given week"""
        model_row = regime_base_df[regime_base_df['model_id'] == int(model_id)]
        if len(model_row) == 0:
            return True  # Conservative: exclude if model not found
            
        training_from = int(model_row.iloc[0]['training_from'])
        training_to = int(model_row.iloc[0]['training_to'])
        
        # Check if any day in the week falls within training period
        return any(
            training_from <= trading_day <= training_to 
            for trading_day in week_trading_days
        )

def main():
    parser = argparse.ArgumentParser(description='Test all models across paradigms')
    parser.add_argument('--start_model', type=int, default=1, help='Starting model number (default: 1)')
    parser.add_argument('--end_model', type=int, default=425, help='Ending model number (default: 425)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_model < 1 or args.end_model < args.start_model:
        raise ValueError("Invalid model range")
    
    # Create tester and run
    tester = ModelTester()
    results = tester.test_all_models(args.start_model, args.end_model)
    
    print(f"\nTesting completed successfully!")

if __name__ == "__main__":
    main()
