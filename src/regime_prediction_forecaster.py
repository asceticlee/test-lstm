#!/usr/bin/env python3
"""
Regime Prediction and LSTM Model Selection Forecaster

This script:
1. Analyzes market data to predict future regimes using HMM or naive models
2. Selects best upside and downside models based on regime predictions
3. Makes minute-by-minute regression forecasts using selected models
4. Prevents using models trained on the prediction period

Usage:
    python regime_prediction_forecaster.py [--period daily|weekly] [--method hmm|naive] [--start_date YYYYMMDD] [--end_date YYYYMMDD]
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
import argparse
from datetime import datetime, timedelta
import warnings
from hmmlearn import hmm
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

# Get project paths
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
data_dir = project_root / 'data'
models_dir = project_root / 'models'
regime_dir = project_root / 'regime_analysis'
results_dir = project_root / 'test_results'
output_dir = project_root / 'forecasts'

# Ensure directories exist
output_dir.mkdir(exist_ok=True)

class RegimePredictionForecaster:
    """
    Predicts market regimes and uses best models for forecasting
    """
    
    def __init__(self):
        # Data storage
        self.trading_data = None
        self.regime_assignments = None
        self.model_results = None
        self.best_regime_summary = None
        self.model_log = None
        
        # Regime prediction models
        self.hmm_model = None
        self.regime_history = None
        
        print("Regime Prediction Forecaster initialized")
        
    def load_data(self):
        """Load all required data files"""
        print("Loading data files...")
        
        # Load trading data
        data_file = data_dir / 'trainingData.csv'
        if not data_file.exists():
            raise FileNotFoundError(f"Trading data file not found: {data_file}")
        self.trading_data = pd.read_csv(data_file)
        print(f"Loaded trading data: {len(self.trading_data):,} rows")
        
        # Load regime assignments
        regime_file = regime_dir / 'regime_assignments.csv'
        if not regime_file.exists():
            raise FileNotFoundError(f"Regime assignments file not found: {regime_file}")
        self.regime_assignments = pd.read_csv(regime_file)
        print(f"Loaded regime assignments: {len(self.regime_assignments):,} rows")
        
        # Load model test results
        model_results_file = results_dir / 'model_regime_test_results_1_425.csv'
        if not model_results_file.exists():
            raise FileNotFoundError(f"Model test results file not found: {model_results_file}")
        self.model_results = pd.read_csv(model_results_file)
        print(f"Loaded model test results: {len(self.model_results):,} rows")
        
        # Load best regime summary
        best_summary_file = results_dir / 'best_regime_summary_1_425.csv'
        if not best_summary_file.exists():
            raise FileNotFoundError(f"Best regime summary file not found: {best_summary_file}")
        self.best_regime_summary = pd.read_csv(best_summary_file)
        print(f"Loaded best regime summary: {len(self.best_regime_summary):,} rows")
        
        # Load model log
        model_log_file = models_dir / 'model_log.csv'
        if not model_log_file.exists():
            raise FileNotFoundError(f"Model log file not found: {model_log_file}")
        self.model_log = pd.read_csv(model_log_file, dtype={'model_id': str})
        print(f"Loaded model log: {len(self.model_log):,} rows")
        
    def prepare_regime_history(self):
        """Prepare historical regime sequence for prediction models"""
        print("Preparing regime history...")
        
        # Get unique weeks with their regimes
        weekly_regimes = self.regime_assignments.groupby('TradingDay')['Regime'].first().reset_index()
        weekly_regimes = weekly_regimes.sort_values('TradingDay')
        
        # Create regime sequence
        self.regime_history = weekly_regimes['Regime'].values
        self.trading_days = weekly_regimes['TradingDay'].values
        
        print(f"Prepared regime history: {len(self.regime_history)} periods")
        print(f"Regime distribution: {Counter(self.regime_history)}")
        
    def train_hmm_model(self, n_components=None):
        """Train HMM model for regime prediction"""
        print("Training HMM model for regime prediction...")
        
        if n_components is None:
            n_components = len(np.unique(self.regime_history))
        
        # Use a simpler approach: Multinomial HMM for discrete regime sequences
        try:
            from hmmlearn import hmm
            
            # Convert regime sequence to proper format
            regime_sequence = self.regime_history.reshape(-1, 1).astype(float)
            
            # Train Multinomial HMM instead of Gaussian HMM for discrete regimes
            self.hmm_model = hmm.MultinomialHMM(
                n_components=n_components,
                n_iter=100,
                random_state=42
            )
            
            # For MultinomialHMM, we need to encode regimes as integers starting from 0
            unique_regimes = sorted(np.unique(self.regime_history))
            regime_mapping = {regime: i for i, regime in enumerate(unique_regimes)}
            encoded_regimes = np.array([regime_mapping[regime] for regime in self.regime_history])
            
            # Fit the model
            self.hmm_model.fit(encoded_regimes.reshape(-1, 1))
            
            # Store mapping for later use
            self.regime_mapping = regime_mapping
            self.reverse_mapping = {v: k for k, v in regime_mapping.items()}
            
            # Evaluate model fit
            log_likelihood = self.hmm_model.score(encoded_regimes.reshape(-1, 1))
            print(f"HMM model trained with log-likelihood: {log_likelihood:.3f}")
            
        except Exception as e:
            print(f"HMM training failed: {str(e)}")
            print("Falling back to naive prediction method")
            self.hmm_model = None
        
    def predict_regime_hmm(self, current_date, lookback_periods=10):
        """Predict next regime using HMM"""
        if self.hmm_model is None:
            return self.predict_regime_naive(current_date)
            
        # Find current position in regime history
        current_idx = np.where(self.trading_days <= int(current_date))[0]
        
        if len(current_idx) == 0:
            # If before historical data, use naive approach
            return self.predict_regime_naive(current_date)
        
        current_idx = current_idx[-1]
        
        # Get recent regime history
        start_idx = max(0, current_idx - lookback_periods + 1)
        recent_regimes = self.regime_history[start_idx:current_idx + 1]
        
        if len(recent_regimes) == 0:
            return self.predict_regime_naive(current_date)
        
        # Encode regimes using the mapping
        try:
            encoded_regimes = np.array([self.regime_mapping[regime] for regime in recent_regimes])
            
            # Predict next regime
            predicted_encoded = self.hmm_model.predict(encoded_regimes.reshape(-1, 1))[-1]
            predicted_regime = self.reverse_mapping[predicted_encoded]
            
            return int(predicted_regime)
        except Exception as e:
            print(f"HMM prediction failed: {str(e)}, falling back to naive method")
            # Fallback to naive method
            return self.predict_regime_naive(current_date)
            
    def predict_regime_naive(self, current_date, lookback_periods=5):
        """Predict next regime using naive approach (most common recent regime)"""
        # Find current position in regime history
        current_idx = np.where(self.trading_days <= int(current_date))[0]
        
        if len(current_idx) == 0:
            # If before historical data, return most common regime overall
            return int(Counter(self.regime_history).most_common(1)[0][0])
        
        current_idx = current_idx[-1]
        
        # Get recent regime history
        start_idx = max(0, current_idx - lookback_periods + 1)
        recent_regimes = self.regime_history[start_idx:current_idx + 1]
        
        if len(recent_regimes) == 0:
            return int(Counter(self.regime_history).most_common(1)[0][0])
        
        # Return most common recent regime
        return int(Counter(recent_regimes).most_common(1)[0][0])
    
    def get_best_models_for_regime(self, regime, prediction_date):
        """Get best upside and downside models for a given regime, excluding models trained on prediction date"""
        prediction_date_int = int(prediction_date)
        
        # Filter models that don't overlap with prediction date
        valid_models = []
        for _, model_row in self.model_log.iterrows():
            train_from = int(model_row['train_from'])
            train_to = int(model_row['train_to'])
            
            # Exclude models if prediction date falls within training period
            if not (train_from <= prediction_date_int <= train_to):
                # Convert model_id to match format in model_results (integer)
                model_id_str = str(model_row['model_id']).strip()
                if '.' in model_id_str:
                    model_id_int = int(float(model_id_str))
                else:
                    model_id_int = int(model_id_str)
                valid_models.append(model_id_int)
        
        if len(valid_models) == 0:
            print(f"Warning: No valid models found for regime {regime} on date {prediction_date}")
            return None, None
        
        # Filter model results for this regime and valid models
        regime_models = self.model_results[
            (self.model_results['regime'] == regime) & 
            (self.model_results['model_id'].isin(valid_models))
        ]
        
        if len(regime_models) == 0:
            print(f"Warning: No model results found for regime {regime}")
            return None, None
        
        # Calculate upside and downside scores
        upside_scores = []
        downside_scores = []
        
        for _, row in regime_models.iterrows():
            # Calculate average upside accuracy
            upside_accs = [row[f'upside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
            upside_score = np.mean(upside_accs)
            upside_scores.append((upside_score, row))
            
            # Calculate average downside accuracy
            downside_accs = [row[f'downside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
            downside_score = np.mean(downside_accs)
            downside_scores.append((downside_score, row))
        
        # Get best models
        best_upside = max(upside_scores, key=lambda x: x[0])[1] if upside_scores else None
        best_downside = max(downside_scores, key=lambda x: x[0])[1] if downside_scores else None
        
        return best_upside, best_downside
    
    def load_model_and_scaler(self, model_id):
        """Load LSTM model and its scaler"""
        try:
            # Convert model_id to properly formatted string
            if isinstance(model_id, (int, np.integer)):
                model_id_str = f"{model_id:05d}"
            else:
                model_id_str = str(model_id).strip()
                if '.' in model_id_str:
                    model_num = int(float(model_id_str))
                    model_id_str = f"{model_num:05d}"
                elif model_id_str.isdigit() and len(model_id_str) < 5:
                    model_id_str = f"{int(model_id_str):05d}"
            
            # Load model
            model_path = models_dir / f'lstm_stock_model_{model_id_str}.keras'
            if not model_path.exists():
                print(f"Model file not found: {model_path}")
                return None, None
            
            model = keras.models.load_model(model_path)
            
            # Load scaler
            scaler_path = models_dir / f'scaler_params_{model_id_str}.json'
            if not scaler_path.exists():
                print(f"Scaler file not found: {scaler_path}")
                return None, None
            
            with open(scaler_path, 'r') as f:
                scaler_params = json.load(f)
            
            scaler = MinMaxScaler()
            scaler.data_min_ = np.array(scaler_params['Min'])
            scaler.data_max_ = np.array(scaler_params['Max'])
            scaler.scale_ = 1.0 / (scaler.data_max_ - scaler.data_min_)
            scaler.min_ = -scaler.data_min_ * scaler.scale_
            
            return model, scaler
            
        except Exception as e:
            print(f"Error loading model {model_id}: {str(e)}")
            return None, None
    
    def create_sequences(self, df, seq_length, feature_cols, target_col):
        """Create sequences for prediction"""
        if len(df) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
        features = df[feature_cols].values
        targets = df[target_col].values
        days = df['TradingDay'].values
        ms = df['TradingMsOfDay'].values
        
        X = []
        y = []
        days_out = []
        ms_out = []
        
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
                days_out.append(days[i + seq_length - 1])
                ms_out.append(ms[i + seq_length - 1])
                i += 1
            else:
                i += 1
        
        return np.array(X), np.array(y), np.array(days_out), np.array(ms_out)
    
    def make_predictions(self, model, scaler, data, seq_length, feature_cols, target_col):
        """Make predictions using the model"""
        X, y_actual, days, ms = self.create_sequences(data, seq_length, feature_cols, target_col)
        
        if len(X) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Scale features
        num_features = len(feature_cols)
        X_reshaped = X.reshape(-1, num_features)
        X_scaled = scaler.transform(X_reshaped).reshape(X.shape)
        
        # Make predictions
        predictions = model.predict(X_scaled, verbose=0).flatten()
        
        return predictions, y_actual, days, ms
    
    def run_forecasting(self, start_date, end_date, period='daily', method='hmm'):
        """Run the complete forecasting process"""
        print(f"Running forecasting from {start_date} to {end_date}")
        print(f"Period: {period}, Method: {method}")
        
        # Load all required data
        self.load_data()
        self.prepare_regime_history()
        
        # Train regime prediction model if using HMM
        if method == 'hmm':
            self.train_hmm_model()
        
        # Generate date range
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        if period == 'daily':
            date_range = pd.date_range(start_dt, end_dt, freq='D')
        else:  # weekly
            date_range = pd.date_range(start_dt, end_dt, freq='W-MON')
        
        # Results storage
        all_results = []
        
        for current_date in date_range:
            current_date_str = current_date.strftime('%Y%m%d')
            current_date_int = int(current_date_str)
            
            print(f"\nProcessing date: {current_date_str}")
            
            # Skip if no trading data for this date
            day_data = self.trading_data[self.trading_data['TradingDay'] == current_date_int]
            if len(day_data) == 0:
                print(f"No trading data for {current_date_str}, skipping...")
                continue
            
            # Predict regime for this period
            if method == 'hmm':
                predicted_regime = self.predict_regime_hmm(current_date_str)
            else:
                predicted_regime = self.predict_regime_naive(current_date_str)
            
            print(f"Predicted regime: {predicted_regime}")
            
            # Get best models for this regime
            best_upside, best_downside = self.get_best_models_for_regime(predicted_regime, current_date_str)
            
            if best_upside is None or best_downside is None:
                print(f"No valid models found for regime {predicted_regime} on {current_date_str}")
                continue
            
            print(f"Selected upside model: {best_upside['model_id']} (rank: {best_upside['model_regime_rank']})")
            print(f"Selected downside model: {best_downside['model_id']} (rank: {best_downside['model_regime_rank']})")
            
            # Load models and make predictions
            upside_model, upside_scaler = self.load_model_and_scaler(best_upside['model_id'])
            downside_model, downside_scaler = self.load_model_and_scaler(best_downside['model_id'])
            
            if upside_model is None or downside_model is None:
                print(f"Failed to load models for {current_date_str}")
                continue
            
            # Get model parameters
            upside_model_info = self.model_log[self.model_log['model_id'] == f"{best_upside['model_id']:05d}"].iloc[0]
            downside_model_info = self.model_log[self.model_log['model_id'] == f"{best_downside['model_id']:05d}"].iloc[0]
            
            upside_seq_length = int(upside_model_info['seq_length'])
            downside_seq_length = int(downside_model_info['seq_length'])
            upside_label_number = int(upside_model_info['label_number'])
            downside_label_number = int(downside_model_info['label_number'])
            
            # Define feature columns
            feature_cols = day_data.columns[5:49]  # Features: columns 6 to 49
            upside_target_col = f'Label_{upside_label_number}'
            downside_target_col = f'Label_{downside_label_number}'
            
            # Make predictions
            upside_predictions, upside_actual, days, ms = self.make_predictions(
                upside_model, upside_scaler, day_data, upside_seq_length, feature_cols, upside_target_col
            )
            
            downside_predictions, downside_actual, _, _ = self.make_predictions(
                downside_model, downside_scaler, day_data, downside_seq_length, feature_cols, downside_target_col
            )
            
            # Align predictions if sequence lengths are different
            min_length = min(len(upside_predictions), len(downside_predictions))
            if min_length == 0:
                print(f"No valid predictions for {current_date_str}")
                continue
            
            # Truncate to common length
            upside_predictions = upside_predictions[:min_length]
            downside_predictions = downside_predictions[:min_length]
            upside_actual = upside_actual[:min_length]
            days = days[:min_length]
            ms = ms[:min_length]
            
            # Store results
            for i in range(min_length):
                result = {
                    'trading_day': int(days[i]),
                    'ms_of_day': int(ms[i]),
                    'actual': float(upside_actual[i]),
                    'upside_predict': float(upside_predictions[i]),
                    'upside_model_id': best_upside['model_id'],
                    'upside_regime': int(predicted_regime),
                    'upside_rank_of_model': int(best_upside['model_regime_rank']),
                    'downside_predict': float(downside_predictions[i]),
                    'downside_model_id': best_downside['model_id'],
                    'downside_regime': int(predicted_regime),
                    'downside_rank_of_model': int(best_downside['model_regime_rank'])
                }
                all_results.append(result)
            
            print(f"Generated {min_length} predictions for {current_date_str}")
        
        # Save results
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Reorder columns
            column_order = [
                'trading_day', 'ms_of_day', 'actual', 
                'upside_predict', 'upside_model_id', 'upside_regime', 'upside_rank_of_model',
                'downside_predict', 'downside_model_id', 'downside_regime', 'downside_rank_of_model'
            ]
            results_df = results_df[column_order]
            
            output_file = output_dir / f'regime_forecast_{method}_{period}_{start_date}_{end_date}.csv'
            results_df.to_csv(output_file, index=False)
            print(f"\nSaved {len(all_results)} predictions to {output_file}")
            
            # Print summary statistics
            print(f"\nSummary:")
            print(f"Total predictions: {len(all_results)}")
            print(f"Unique trading days: {results_df['trading_day'].nunique()}")
            print(f"Regime distribution: {dict(results_df['upside_regime'].value_counts().sort_index())}")
            
            return results_df
        else:
            print("No predictions generated")
            return None

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Regime Prediction and LSTM Forecasting')
    parser.add_argument('--period', type=str, choices=['daily', 'weekly'], default='daily',
                       help='Prediction period (default: daily)')
    parser.add_argument('--method', type=str, choices=['hmm', 'naive'], default='hmm',
                       help='Regime prediction method (default: hmm)')
    parser.add_argument('--start_date', type=str, required=True,
                       help='Start date for forecasting (YYYYMMDD)')
    parser.add_argument('--end_date', type=str, required=True,
                       help='End date for forecasting (YYYYMMDD)')
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.start_date, '%Y%m%d')
        datetime.strptime(args.end_date, '%Y%m%d')
    except ValueError:
        raise ValueError("Dates must be in YYYYMMDD format")
    
    # Create forecaster and run
    forecaster = RegimePredictionForecaster()
    results = forecaster.run_forecasting(
        start_date=args.start_date,
        end_date=args.end_date,
        period=args.period,
        method=args.method
    )
    
    if results is not None:
        print(f"\nForecasting completed successfully!")
    else:
        print(f"\nForecasting completed with no results.")

if __name__ == "__main__":
    main()
