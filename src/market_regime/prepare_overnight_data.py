#!/usr/bin/env python3
"""
Data Preparation for Overnight Gap Analysis

This script enhances the history_spot_quote.csv data with previous day close prices
to enable proper overnight gap calculation.

Usage:
    python prepare_overnight_data.py [--input_file] [--output_file]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def prepare_overnight_data(input_file, output_file=None):
    """
    Prepare data with previous day close prices for overnight gap analysis
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
    """
    print(f"Loading data from: {input_file}")
    
    # Load the data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows of data")
    
    # Ensure data is sorted by trading_day and ms_of_day
    df = df.sort_values(['trading_day', 'ms_of_day']).reset_index(drop=True)
    
    # Convert trading_day to datetime for easier manipulation
    df['date'] = pd.to_datetime(df['trading_day'])
    
    # Find the last price of each trading day (closest to market close)
    # Assuming market close is around 16:00 (57600000 ms)
    market_close_ms = 57600000  # 4:00 PM
    
    # For each day, find the last available price
    daily_closes = []
    
    for trading_day in df['trading_day'].unique():
        day_data = df[df['trading_day'] == trading_day]
        
        # Get the last price of the day
        last_price = day_data['mid'].iloc[-1] if len(day_data) > 0 else np.nan
        
        daily_closes.append({
            'trading_day': trading_day,
            'close_price': last_price
        })
    
    # Convert to DataFrame
    closes_df = pd.DataFrame(daily_closes)
    closes_df['date'] = pd.to_datetime(closes_df['trading_day'])
    
    # Sort by date
    closes_df = closes_df.sort_values('date').reset_index(drop=True)
    
    # Create previous day close mapping
    closes_df['prev_date'] = closes_df['date'].shift(-1)  # Next day's previous date is today
    closes_df['prev_close'] = closes_df['close_price'].shift(1)  # Previous day's close
    
    # Create mapping dictionary
    prev_close_map = dict(zip(closes_df['trading_day'], closes_df['prev_close']))
    
    # Add previous close to main dataframe
    df['prev_close'] = df['trading_day'].map(prev_close_map)
    
    # Fill missing previous close values
    df['prev_close'] = df['prev_close'].fillna(method='ffill')
    
    # Calculate overnight gap for verification
    # Only for first data point of each day
    df['is_first_of_day'] = df.groupby('trading_day')['ms_of_day'].transform('min') == df['ms_of_day']
    
    # Calculate overnight gap
    df['overnight_gap'] = np.where(
        df['is_first_of_day'] & df['prev_close'].notna() & (df['prev_close'] != 0),
        (df['mid'] - df['prev_close']) / df['prev_close'],
        np.nan
    )
    
    # Remove helper columns
    df = df.drop(['date', 'is_first_of_day'], axis=1)
    
    print(f"Added previous close prices for overnight gap calculation")
    print(f"Sample overnight gaps:")
    
    # Show some overnight gap statistics
    overnight_gaps = df['overnight_gap'].dropna()
    if len(overnight_gaps) > 0:
        print(f"  Number of overnight gaps: {len(overnight_gaps)}")
        print(f"  Mean overnight gap: {overnight_gaps.mean():.4f} ({overnight_gaps.mean()*100:.2f}%)")
        print(f"  Std overnight gap: {overnight_gaps.std():.4f} ({overnight_gaps.std()*100:.2f}%)")
        print(f"  Min overnight gap: {overnight_gaps.min():.4f} ({overnight_gaps.min()*100:.2f}%)")
        print(f"  Max overnight gap: {overnight_gaps.max():.4f} ({overnight_gaps.max()*100:.2f}%)")
        
        # Count significant gaps
        large_gaps_up = (overnight_gaps > 0.01).sum()
        large_gaps_down = (overnight_gaps < -0.01).sum()
        print(f"  Large gaps up (>1%): {large_gaps_up}")
        print(f"  Large gaps down (<-1%): {large_gaps_down}")
    
    # Save the enhanced data
    if output_file is None:
        output_file = input_file.replace('.csv', '_with_overnight.csv')
    
    df.to_csv(output_file, index=False)
    print(f"Saved enhanced data to: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Prepare data for overnight gap analysis')
    parser.add_argument('--input_file', default='data/history_spot_quote.csv',
                       help='Input CSV file path')
    parser.add_argument('--output_file', default=None,
                       help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent.absolute()
    input_path = script_dir / ".." / ".." / args.input_file
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    output_path = args.output_file
    if output_path is not None:
        output_path = script_dir / ".." / ".." / output_path
    
    try:
        enhanced_file = prepare_overnight_data(str(input_path), str(output_path) if output_path else None)
        print(f"\nData preparation completed successfully!")
        print(f"Enhanced file: {enhanced_file}")
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
