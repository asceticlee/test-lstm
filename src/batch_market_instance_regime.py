#!/usr/bin/env python3
"""
Batch Market Instance Regime Analysis

This script processes history_spot_quote.csv minute by minute to generate
regime forecasts using the gmm_regime_instance_clustering.py. It creates
a market regime forecast for every minute of trading data.

Output: test-lstm/market_regime/gmm/market_regime_forecast.csv

The script supports append mode for incremental processing and includes
progress tracking and error handling.

Usage:
    python batch_market_instance_regime.py [--start_date YYYYMMDD] [--end_date YYYYMMDD]
    
Examples:
    python batch_market_instance_regime.py                    # Process all data
    python batch_market_instance_regime.py --start_date 20240101  # From specific date
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import argparse
import warnings
from datetime import datetime
import time
warnings.filterwarnings('ignore')

# Add the src directory to the path for imports  
script_dir = Path(__file__).parent
src_dir = script_dir
sys.path.insert(0, str(src_dir))

from market_regime.gmm_regime_instance_clustering import GMMRegimeInstanceClassifier

class BatchMarketRegimeForecast:
    """
    Batch processing for minute-by-minute market regime forecasting
    """
    
    def __init__(self, quote_data_path='data/history_spot_quote.csv', 
                 output_path='market_regime/gmm/market_regime_forecast.csv'):
        """
        Initialize the batch processor
        
        Args:
            quote_data_path: Path to history_spot_quote.csv (relative to project root)
            output_path: Path to output CSV file (relative to project root)
        """
        # Get project root (two levels up from src/)
        self.project_root = script_dir.parent
        
        # Resolve file paths
        self.quote_data_path = self.project_root / quote_data_path
        self.output_path = self.project_root / output_path
        
        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize regime classifier
        self.classifier = GMMRegimeInstanceClassifier()
        
        # Data storage
        self.quote_data = None
        self.existing_forecasts = set()
        
        print(f"Batch Market Regime Forecast initialized")
        print(f"Quote data: {self.quote_data_path}")
        print(f"Output file: {self.output_path}")
        print(f"Trading period: {self.classifier.ms_to_time(self.classifier.trading_start_ms)} to {self.classifier.ms_to_time(self.classifier.trading_end_ms)}")
    
    def load_quote_data(self, start_date=None, end_date=None):
        """Load quote data with optional date filtering"""
        print("Loading historical quote data...")
        
        if not self.quote_data_path.exists():
            raise FileNotFoundError(f"Quote data file not found: {self.quote_data_path}")
        
        # Load data
        self.quote_data = pd.read_csv(self.quote_data_path)
        print(f"Loaded {len(self.quote_data):,} quote records")
        
        # Apply date filtering if specified
        if start_date is not None or end_date is not None:
            original_count = len(self.quote_data)
            
            if start_date is not None:
                self.quote_data = self.quote_data[self.quote_data['trading_day'] >= start_date]
            
            if end_date is not None:
                self.quote_data = self.quote_data[self.quote_data['trading_day'] <= end_date]
            
            filtered_count = len(self.quote_data)
            print(f"Filtered to {filtered_count:,} records ({original_count - filtered_count:,} excluded)")
        
        # Sort data
        self.quote_data = self.quote_data.sort_values(['trading_day', 'ms_of_day']).reset_index(drop=True)
        
        # Get unique trading days
        unique_days = sorted(self.quote_data['trading_day'].unique())
        print(f"Date range: {unique_days[0]} to {unique_days[-1]} ({len(unique_days)} trading days)")
        
        return self.quote_data
    
    def load_existing_forecasts(self):
        """Load existing forecast data to avoid reprocessing"""
        if self.output_path.exists():
            try:
                existing_df = pd.read_csv(self.output_path)
                
                # Create set of (trading_day, ms_of_day) tuples for fast lookup
                self.existing_forecasts = set(
                    zip(existing_df['trading_day'], existing_df['ms_of_day'])
                )
                
                print(f"Found {len(self.existing_forecasts):,} existing forecast records")
                
                if len(existing_df) > 0:
                    last_day = existing_df['trading_day'].max()
                    last_time = existing_df[existing_df['trading_day'] == last_day]['ms_of_day'].max()
                    print(f"Last processed: {last_day} at {self.classifier.ms_to_time(last_time)}")
                
            except Exception as e:
                print(f"Warning: Could not load existing forecasts: {e}")
                self.existing_forecasts = set()
        else:
            print("No existing forecast file found - will create new file")
            self.existing_forecasts = set()
    
    def calculate_overnight_gap(self, current_day, previous_day_data):
        """Calculate overnight gap for current day"""
        if previous_day_data is None or len(previous_day_data) == 0:
            return 0  # Default for first day
        
        # Get last price of previous day
        prev_close = previous_day_data['mid'].iloc[-1]
        
        # Get first price of current day
        current_day_data = self.quote_data[
            (self.quote_data['trading_day'] == current_day) &
            (self.quote_data['ms_of_day'] >= self.classifier.trading_start_ms)
        ]
        
        if len(current_day_data) == 0:
            return 0
        
        current_open = current_day_data['mid'].iloc[0]
        
        # Calculate absolute gap
        overnight_gap = current_open - prev_close
        
        return overnight_gap
    
    def get_cumulative_data_for_minute(self, trading_day, target_ms, overnight_gap):
        """
        Get cumulative quote data from start of trading day up to target minute
        
        Args:
            trading_day: Trading day
            target_ms: Target time in milliseconds
            overnight_gap: Overnight gap for this day
            
        Returns:
            dict: Data needed for regime classification or None if insufficient
        """
        # Get data from start of trading period up to target time
        day_data = self.quote_data[
            (self.quote_data['trading_day'] == trading_day) &
            (self.quote_data['ms_of_day'] >= self.classifier.trading_start_ms) &
            (self.quote_data['ms_of_day'] <= target_ms)
        ].copy()
        
        if len(day_data) < 5:  # Need minimum data for feature extraction
            return None
        
        # Sort by time and convert to list of dicts
        day_data = day_data.sort_values('ms_of_day')
        quote_data_list = day_data[['ms_of_day', 'bid', 'ask', 'mid']].to_dict('records')
        
        return {
            'overnight_gap': overnight_gap,
            'quote_data': quote_data_list,
            'data_points': len(quote_data_list)
        }
    
    def process_trading_day(self, trading_day, previous_day_data):
        """Process all minutes for a single trading day"""
        print(f"  Processing {trading_day}...")
        
        # Calculate overnight gap
        overnight_gap = self.calculate_overnight_gap(trading_day, previous_day_data)
        
        # Get all unique timestamps for this day within trading hours
        day_quotes = self.quote_data[
            (self.quote_data['trading_day'] == trading_day) &
            (self.quote_data['ms_of_day'] >= self.classifier.trading_start_ms) &
            (self.quote_data['ms_of_day'] <= self.classifier.trading_end_ms)
        ]
        
        if len(day_quotes) == 0:
            print(f"    No trading data found for {trading_day}")
            return []
        
        unique_times = sorted(day_quotes['ms_of_day'].unique())
        print(f"    Processing {len(unique_times)} time points (gap: {overnight_gap:.3f})")
        
        # Process each minute
        forecasts = []
        processed_count = 0
        skipped_count = 0
        
        for ms_of_day in unique_times:
            try:
                # Check if already processed (for append mode)
                if (trading_day, ms_of_day) in self.existing_forecasts:
                    skipped_count += 1
                    continue
                
                # Get cumulative data up to this minute
                minute_data = self.get_cumulative_data_for_minute(trading_day, ms_of_day, overnight_gap)
                
                if minute_data is None:
                    continue  # Skip if insufficient data
                
                # Classify regime
                regime = self.classifier.classify_regime(
                    minute_data['overnight_gap'], 
                    minute_data['quote_data']
                )
                
                # Get regime probabilities
                regime_with_probs, probabilities = self.classifier.classify_regime_with_probabilities(
                    minute_data['overnight_gap'],
                    minute_data['quote_data']
                )
                
                # Create forecast record
                forecast = {
                    'trading_day': trading_day,
                    'ms_of_day': ms_of_day,
                    'time_str': self.classifier.ms_to_time(ms_of_day),
                    'overnight_gap': overnight_gap,
                    'data_points_used': minute_data['data_points'],
                    'predicted_regime': regime,
                    'regime_prob_0': probabilities[0],
                    'regime_prob_1': probabilities[1], 
                    'regime_prob_2': probabilities[2],
                    'regime_prob_3': probabilities[3],
                    'regime_prob_4': probabilities[4],
                    'max_probability': np.max(probabilities),
                    'confidence': np.max(probabilities) - np.sort(probabilities)[-2],  # Margin over 2nd place
                }
                
                forecasts.append(forecast)
                processed_count += 1
                
            except Exception as e:
                print(f"    Error processing {trading_day} at {self.classifier.ms_to_time(ms_of_day)}: {e}")
                continue
        
        print(f"    Generated {processed_count} new forecasts, skipped {skipped_count} existing")
        return forecasts
    
    def save_forecasts(self, forecasts, append_mode=True):
        """Save forecasts to CSV file"""
        if len(forecasts) == 0:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(forecasts)
        
        # Define column order
        columns = [
            'trading_day', 'ms_of_day', 'time_str', 'overnight_gap', 'data_points_used',
            'predicted_regime', 'regime_prob_0', 'regime_prob_1', 'regime_prob_2', 
            'regime_prob_3', 'regime_prob_4', 'max_probability', 'confidence'
        ]
        
        df = df[columns]
        
        # Save to file
        if append_mode and self.output_path.exists():
            # Append to existing file
            df.to_csv(self.output_path, mode='a', header=False, index=False)
        else:
            # Create new file
            df.to_csv(self.output_path, index=False)
        
        print(f"    Saved {len(df)} forecasts to {self.output_path}")
    
    def run_batch_processing(self, start_date=None, end_date=None, save_frequency=100):
        """Run the complete batch processing pipeline"""
        print("Starting batch market regime forecasting...")
        start_time = time.time()
        
        # Load data
        self.load_quote_data(start_date, end_date)
        self.load_existing_forecasts()
        
        # Get unique trading days to process
        all_trading_days = sorted(self.quote_data['trading_day'].unique())
        
        # Filter days based on existing forecasts if in append mode
        if len(self.existing_forecasts) > 0:
            # Find days that might need processing
            last_processed_day = max(day for day, _ in self.existing_forecasts)
            
            # Process from the last day (in case it's incomplete) and any new days
            days_to_process = [day for day in all_trading_days if day >= last_processed_day]
        else:
            days_to_process = all_trading_days
        
        print(f"Processing {len(days_to_process)} trading days...")
        
        # Process each trading day
        all_forecasts = []
        previous_day_data = None
        
        for i, trading_day in enumerate(days_to_process):
            try:
                # Get previous day data for overnight gap calculation
                if i > 0:
                    prev_day = days_to_process[i-1]
                    previous_day_data = self.quote_data[
                        (self.quote_data['trading_day'] == prev_day) &
                        (self.quote_data['ms_of_day'] <= self.classifier.trading_end_ms)
                    ]
                elif trading_day != all_trading_days[0]:
                    # Find actual previous trading day
                    prev_day_idx = all_trading_days.index(trading_day) - 1
                    if prev_day_idx >= 0:
                        prev_day = all_trading_days[prev_day_idx]
                        previous_day_data = self.quote_data[
                            (self.quote_data['trading_day'] == prev_day) &
                            (self.quote_data['ms_of_day'] <= self.classifier.trading_end_ms)
                        ]
                
                # Process this trading day
                day_forecasts = self.process_trading_day(trading_day, previous_day_data)
                
                # Add to collection
                all_forecasts.extend(day_forecasts)
                
                # Save periodically
                if len(all_forecasts) >= save_frequency:
                    self.save_forecasts(all_forecasts, append_mode=True)
                    all_forecasts = []  # Clear buffer
                
                # Progress update
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    progress = (i + 1) / len(days_to_process) * 100
                    print(f"  Progress: {progress:.1f}% ({i+1}/{len(days_to_process)} days), Elapsed: {elapsed:.1f}s")
                
            except Exception as e:
                print(f"  ERROR processing {trading_day}: {e}")
                continue
        
        # Save remaining forecasts
        if len(all_forecasts) > 0:
            self.save_forecasts(all_forecasts, append_mode=True)
        
        # Final summary
        elapsed_time = time.time() - start_time
        print(f"\n" + "="*60)
        print(f"BATCH PROCESSING COMPLETED")
        print(f"="*60)
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Processed {len(days_to_process)} trading days")
        print(f"Output file: {self.output_path}")
        
        # Show final file stats
        if self.output_path.exists():
            final_df = pd.read_csv(self.output_path)
            print(f"Total forecast records: {len(final_df):,}")
            
            if len(final_df) > 0:
                date_range = f"{final_df['trading_day'].min()} to {final_df['trading_day'].max()}"
                print(f"Date range: {date_range}")
                
                # Regime distribution
                regime_counts = final_df['predicted_regime'].value_counts().sort_index()
                print("Regime distribution:")
                for regime, count in regime_counts.items():
                    pct = count / len(final_df) * 100
                    print(f"  Regime {regime}: {count:,} ({pct:.1f}%)")

def parse_date(date_str):
    """Parse date string in YYYYMMDD format"""
    try:
        return int(date_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYYMMDD format.")

def main():
    parser = argparse.ArgumentParser(description='Batch Market Instance Regime Forecasting')
    parser.add_argument('--start_date', type=parse_date, default=None,
                       help='Start date in YYYYMMDD format (e.g., 20240101)')
    parser.add_argument('--end_date', type=parse_date, default=None,
                       help='End date in YYYYMMDD format (e.g., 20241231)')
    parser.add_argument('--save_frequency', type=int, default=100,
                       help='Save forecasts every N records (default: 100)')
    parser.add_argument('--quote_data', default='data/history_spot_quote.csv',
                       help='Path to quote data file (relative to project root)')
    parser.add_argument('--output', default='market_regime/gmm/market_regime_forecast.csv',
                       help='Output file path (relative to project root)')
    
    args = parser.parse_args()
    
    try:
        # Initialize batch processor
        processor = BatchMarketRegimeForecast(
            quote_data_path=args.quote_data,
            output_path=args.output
        )
        
        # Run batch processing
        processor.run_batch_processing(
            start_date=args.start_date,
            end_date=args.end_date,
            save_frequency=args.save_frequency
        )
        
        print("\n✅ Batch processing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
