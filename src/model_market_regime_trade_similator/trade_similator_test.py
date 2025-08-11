#!/usr/bin/env python3
"""
Market Regime Trading Simulator

This script simulates trading using market regime-based model selection.
It combines:
1. Market regime detection from sequence_regime_assignments.csv
2. Model performance rankings from model_regime_test_results_sequence.csv  
3. Actual LSTM model predictions using trained models
4. Training period filtering to avoid data leakage

The simulator makes predictions at each sequence window end time, selecting
the best-performing model for the detected market regime while ensuring
the model wasn't trained on the prediction period.

Usage:
    python trade_similator_test.py [--start_date YYYYMMDD] [--end_date YYYYMMDD] [--max_days N]
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir / ".." / ".."
sys.path.append(str(project_root / "src"))

class MarketRegimeTradingSimulator:
    """
    Trading simulator using market regime-based model selection
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent / ".." / ".."
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.regime_dir = self.project_root / "market_regime" / "gmm" / "sequence"
        self.results_dir = self.project_root / "model_regime"
        self.output_dir = self.project_root / "model_market_regime_trade_similator"
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.trading_data = None
        self.sequence_regimes = None
        self.model_rankings = None
        self.model_log = None
        self.loaded_models = {}  # Cache for loaded models
        self.loaded_scalers = {}  # Cache for loaded scalers
        
        print(f"Trading Simulator initialized")
        print(f"Project root: {self.project_root}")
        print(f"Output directory: {self.output_dir}")
    
    def load_data(self):
        """Load all required data files"""
        print("Loading data files...")
        
        # Load trading data
        trading_file = self.data_dir / "trainingData.csv"
        if not trading_file.exists():
            raise FileNotFoundError(f"Trading data not found: {trading_file}")
        self.trading_data = pd.read_csv(trading_file)
        print(f"Loaded trading data: {len(self.trading_data)} rows")
        
        # Load sequence regime assignments
        regime_file = self.regime_dir / "sequence_regime_assignments.csv"
        if not regime_file.exists():
            raise FileNotFoundError(f"Sequence regime assignments not found: {regime_file}")
        self.sequence_regimes = pd.read_csv(regime_file)
        print(f"Loaded sequence regimes: {len(self.sequence_regimes)} assignments")
        
        # Load model rankings
        rankings_file = self.results_dir / "model_regime_test_results_sequence.csv"
        if not rankings_file.exists():
            raise FileNotFoundError(f"Model rankings not found: {rankings_file}")
        self.model_rankings = pd.read_csv(rankings_file)
        print(f"Loaded model rankings: {len(self.model_rankings)} entries")
        
        # Load model log
        model_log_file = self.models_dir / "model_log_avg.csv"
        if not model_log_file.exists():
            raise FileNotFoundError(f"Model log not found: {model_log_file}")
        self.model_log = pd.read_csv(model_log_file)
        print(f"Loaded model log: {len(self.model_log)} models")
        
        # Sort data for efficient processing
        self.trading_data = self.trading_data.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
        self.sequence_regimes = self.sequence_regimes.sort_values(['trading_day', 'window_end_ms']).reset_index(drop=True)
        
        print("Data loading completed")
    
    def get_best_model_for_regime(self, regime, direction, trading_day):
        """
        Get the best-performing model for a regime and direction that wasn't trained on the trading_day
        
        Args:
            regime: Market regime number
            direction: 'upside' or 'downside'
            trading_day: Trading day to avoid (YYYYMMDD format)
            
        Returns:
            model_id or None if no suitable model found
        """
        # Filter rankings for this regime
        regime_rankings = self.model_rankings[self.model_rankings['regime'] == regime].copy()
        
        if len(regime_rankings) == 0:
            return None
        
        # Sort by rank for the specified direction
        rank_col = f'model_regime_{direction}_rank'
        if rank_col not in regime_rankings.columns:
            return None
        
        regime_rankings = regime_rankings.sort_values(rank_col)
        
        # Find the best model that wasn't trained on this trading day
        trading_day_float = float(trading_day)
        
        for _, row in regime_rankings.iterrows():
            model_id = row['model_id']
            train_from = float(row['training_from'])
            train_to = float(row['training_to'])
            
            # Check if trading day is outside training period
            if not (train_from <= trading_day_float <= train_to):
                return int(model_id)
        
        return None
    
    def load_model_and_scaler(self, model_id):
        """Load model and scaler, caching for efficiency"""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id], self.loaded_scalers[model_id]
        
        # Format model ID
        model_id_str = f"{model_id:05d}"
        
        # Load model
        model_file = self.models_dir / f"lstm_stock_model_avg_{model_id_str}.keras"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        model = tf.keras.models.load_model(model_file)
        
        # Load scaler parameters
        scaler_file = self.models_dir / f"scaler_params_avg_{model_id_str}.json"
        if not scaler_file.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
        
        with open(scaler_file, 'r') as f:
            scaler_params = json.load(f)
        
        # Reconstruct scaler
        if 'Min' in scaler_params and 'Max' in scaler_params:
            scaler = MinMaxScaler()
            scaler.min_ = np.array(scaler_params['Min'])
            scaler.scale_ = np.array(scaler_params['Max']) - np.array(scaler_params['Min'])
            scaler.data_min_ = np.array(scaler_params['Min'])
            scaler.data_max_ = np.array(scaler_params['Max'])
            scaler.data_range_ = scaler.scale_
        elif 'Mean' in scaler_params and 'Variance' in scaler_params:
            scaler = StandardScaler()
            scaler.mean_ = np.array(scaler_params['Mean'])
            scaler.var_ = np.array(scaler_params['Variance'])
            scaler.scale_ = np.sqrt(scaler.var_)
        else:
            raise ValueError(f"Unknown scaler parameters: {list(scaler_params.keys())}")
        
        # Cache for future use
        self.loaded_models[model_id] = model
        self.loaded_scalers[model_id] = scaler
        
        return model, scaler
    
    def get_model_label_info(self, model_id):
        """Get label information for a specific model"""
        model_info = self.model_log[self.model_log['model_id'] == model_id]
        if len(model_info) == 0:
            raise ValueError(f"Model {model_id} not found in model log")
        
        model_info = model_info.iloc[0]
        label_number = model_info['label_number']
        seq_length = model_info['seq_length']
        
        return label_number, seq_length
    
    def create_sequences_for_prediction(self, data, seq_length, feature_cols, target_col, end_idx):
        """
        Create a single sequence for prediction ending at the specified index
        
        Args:
            data: DataFrame with trading data
            seq_length: Length of sequence needed
            feature_cols: Feature column names
            target_col: Target column name
            end_idx: Index where sequence should end (prediction point)
            
        Returns:
            X (features), y (actual), or None if sequence cannot be created
        """
        if end_idx < seq_length:
            return None, None
        
        # Get the sequence window
        start_idx = end_idx - seq_length
        seq_data = data.iloc[start_idx:end_idx + 1]  # Include end_idx for target
        
        # Check data continuity (same logic as training)
        days = seq_data['TradingDay'].values
        ms = seq_data['TradingMsOfDay'].values
        
        # Verify we have enough data
        if len(seq_data) < seq_length + 1:
            return None, None
        
        # Check time continuity (1-minute intervals = 60000 ms)
        for i in range(len(ms) - 1):
            expected_next_ms = ms[i] + 60000
            
            # Handle day boundary - allow end of day to start of next day
            if ms[i+1] < ms[i]:  # Next day
                # Allow transition from last trading time to first trading time of next day
                if days[i+1] != days[i] + 1:  # Must be consecutive trading days
                    return None, None
                # Don't check ms continuity across days
            else:  # Same day
                if ms[i+1] != expected_next_ms or days[i+1] != days[i]:
                    return None, None
        
        # Extract features and target
        features = seq_data.iloc[:seq_length][feature_cols].values  # First seq_length rows
        actual = seq_data.iloc[seq_length][target_col]  # Target at end_idx
        
        return features.reshape(1, seq_length, -1), actual
    
    def run_simulation(self, start_date=None, end_date=None, max_days=None):
        """
        Run the trading simulation
        
        Args:
            start_date: Start date (YYYYMMDD format), if None use first available
            end_date: End date (YYYYMMDD format), if None use last available  
            max_days: Maximum number of days to simulate (for testing)
        """
        print("Starting trading simulation...")
        
        # Filter sequence regimes by date range
        sequence_regimes = self.sequence_regimes.copy()
        if start_date:
            sequence_regimes = sequence_regimes[sequence_regimes['trading_day'] >= int(start_date)]
        if end_date:
            sequence_regimes = sequence_regimes[sequence_regimes['trading_day'] <= int(end_date)]
        
        # Limit days for testing
        if max_days:
            unique_days = sequence_regimes['trading_day'].unique()[:max_days]
            sequence_regimes = sequence_regimes[sequence_regimes['trading_day'].isin(unique_days)]
        
        print(f"Simulating on {len(sequence_regimes)} sequence windows")
        print(f"Date range: {sequence_regimes['trading_day'].min()} to {sequence_regimes['trading_day'].max()}")
        
        # Feature columns (same as training - columns 5-48)
        feature_cols = self.trading_data.columns[5:49].tolist()
        
        # Results storage
        results = []
        processed_count = 0
        
        # Process each sequence window
        for idx, seq_row in sequence_regimes.iterrows():
            trading_day = seq_row['trading_day']
            window_end_ms = seq_row['window_end_ms'] 
            regime = seq_row['Regime']
            
            try:
                # Find the corresponding data point in trading data
                trading_data_point = self.trading_data[
                    (self.trading_data['TradingDay'] == trading_day) & 
                    (self.trading_data['TradingMsOfDay'] == window_end_ms)
                ]
                
                if len(trading_data_point) == 0:
                    continue
                
                data_idx = trading_data_point.index[0]
                
                # Get best models for this regime and trading day
                upside_model_id = self.get_best_model_for_regime(regime, 'upside', trading_day)
                downside_model_id = self.get_best_model_for_regime(regime, 'downside', trading_day)
                
                upside_prediction = np.nan
                downside_prediction = np.nan
                actual_value = np.nan
                
                # Make upside prediction
                if upside_model_id is not None:
                    try:
                        label_number, seq_length = self.get_model_label_info(upside_model_id)
                        target_col = f'Label_{label_number}'
                        
                        if target_col in self.trading_data.columns:
                            model, scaler = self.load_model_and_scaler(upside_model_id)
                            
                            X, actual = self.create_sequences_for_prediction(
                                self.trading_data, seq_length, feature_cols, target_col, data_idx
                            )
                            
                            if X is not None:
                                # Scale features
                                X_reshaped = X.reshape(-1, len(feature_cols))
                                X_scaled = scaler.transform(X_reshaped).reshape(X.shape)
                                
                                # Predict
                                upside_prediction = model.predict(X_scaled, verbose=0)[0, 0]
                                if np.isnan(actual_value):  # Only set once
                                    actual_value = actual
                    except Exception as e:
                        print(f"Error in upside prediction for day {trading_day}, ms {window_end_ms}: {e}")
                
                # Make downside prediction
                if downside_model_id is not None:
                    try:
                        label_number, seq_length = self.get_model_label_info(downside_model_id)
                        target_col = f'Label_{label_number}'
                        
                        if target_col in self.trading_data.columns:
                            model, scaler = self.load_model_and_scaler(downside_model_id)
                            
                            X, actual = self.create_sequences_for_prediction(
                                self.trading_data, seq_length, feature_cols, target_col, data_idx
                            )
                            
                            if X is not None:
                                # Scale features
                                X_reshaped = X.reshape(-1, len(feature_cols))
                                X_scaled = scaler.transform(X_reshaped).reshape(X.shape)
                                
                                # Predict
                                downside_prediction = model.predict(X_scaled, verbose=0)[0, 0]
                                if np.isnan(actual_value):  # Only set once
                                    actual_value = actual
                    except Exception as e:
                        print(f"Error in downside prediction for day {trading_day}, ms {window_end_ms}: {e}")
                
                # Store result
                results.append({
                    'trading_day': trading_day,
                    'trading_ms_of_day': window_end_ms,
                    'market_regime': regime,
                    'upside_model': upside_model_id,
                    'downside_model': downside_model_id,
                    'actual_value': actual_value,
                    'upside_predict_value': upside_prediction,
                    'downside_predict_value': downside_prediction
                })
                
                processed_count += 1
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} sequence windows...")
                    
            except Exception as e:
                print(f"Error processing sequence window {idx}: {e}")
                continue
        
        print(f"Simulation completed. Processed {len(results)} predictions.")
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # Add summary statistics
            print(f"\nSimulation Summary:")
            print(f"Total predictions: {len(results_df)}")
            print(f"Unique trading days: {results_df['trading_day'].nunique()}")
            print(f"Unique regimes: {results_df['market_regime'].nunique()}")
            print(f"Upside predictions with models: {results_df['upside_model'].notna().sum()}")
            print(f"Downside predictions with models: {results_df['downside_model'].notna().sum()}")
            
            # Save results
            start_str = str(start_date) if start_date else "start"
            end_str = str(end_date) if end_date else "end"
            output_file = self.output_dir / f"trading_simulation_{start_str}_{end_str}.csv"
            
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            return results_df
        else:
            print("No valid predictions generated.")
            return None

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Market Regime Trading Simulator')
    parser.add_argument('--start_date', type=str, default=None,
                       help='Start date (YYYYMMDD format)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date (YYYYMMDD format)')
    parser.add_argument('--max_days', type=int, default=None,
                       help='Maximum number of days to simulate (for testing)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MARKET REGIME TRADING SIMULATOR")
    print("="*60)
    
    # Create simulator
    simulator = MarketRegimeTradingSimulator()
    
    try:
        # Load data
        simulator.load_data()
        
        # Run simulation
        results = simulator.run_simulation(
            start_date=args.start_date,
            end_date=args.end_date,
            max_days=args.max_days
        )
        
        if results is not None:
            print("\n" + "="*60)
            print("SIMULATION COMPLETED SUCCESSFULLY")
            print("="*60)
        else:
            print("Simulation failed to generate results")
            
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
