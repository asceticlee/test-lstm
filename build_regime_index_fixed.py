#!/usr/bin/env python3
"""
Script to build complete regime performance index for all models.
"""

import os
import pandas as pd
import glob
from tqdm import tqdm

def build_complete_regime_performance_index():
    """Build complete regime performance index for all models - grouped by trading day"""
    
    # Define paths
    performance_dir = "model_performance/model_regime_performance"
    index_file = "model_performance/regime_performance_index.csv"
    
    print(f"Building regime performance index from {performance_dir}")
    print(f"Output index file: {index_file}")
    
    # Get all performance files
    pattern = f"{performance_dir}/model_*_regime_performance.csv"
    files = glob.glob(pattern)
    files.sort()
    
    print(f"Found {len(files)} performance files")
    
    if len(files) == 0:
        print("No performance files found!")
        return
    
    # Build index entries
    index_entries = []
    
    for file_path in tqdm(files, desc="Processing files"):
        # Extract model ID from filename
        filename = os.path.basename(file_path)
        # filename format: model_XXXXX_regime_performance.csv
        model_id = filename.split('_')[1]
        
        try:
            # Read file to get trading day information
            df = pd.read_csv(file_path, usecols=['TradingDay'])
            
            # Group consecutive rows by trading day for batch loading
            current_trading_day = None
            start_row = None
            
            for row_idx, trading_day in enumerate(df['TradingDay'], start=1):
                if trading_day != current_trading_day:
                    # Save previous group if exists
                    if current_trading_day is not None:
                        index_entries.append({
                            'model_id': model_id,
                            'trading_day': int(current_trading_day),
                            'file_path': file_path,
                            'start_row': start_row,
                            'end_row': row_idx - 1  # End row of previous group
                        })
                    
                    # Start new group
                    current_trading_day = trading_day
                    start_row = row_idx
            
            # Save final group
            if current_trading_day is not None:
                index_entries.append({
                    'model_id': model_id,
                    'trading_day': int(current_trading_day),
                    'file_path': file_path,
                    'start_row': start_row,
                    'end_row': len(df)  # End row is last row
                })
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if len(index_entries) == 0:
        print("No index entries created!")
        return
    
    # Create index DataFrame
    index_df = pd.DataFrame(index_entries)
    
    print(f"Created index with {len(index_df)} entries")
    print(f"Models: {index_df['model_id'].nunique()}")
    print(f"Trading days: {index_df['trading_day'].nunique()}")
    
    # Save index
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    index_df.to_csv(index_file, index=False)
    
    print(f"Index saved to {index_file}")
    
    # Show sample
    print("\nSample index entries:")
    print(index_df.head(10))
    print("\nTrading day distribution:")
    day_counts = index_df['trading_day'].value_counts().head(10)
    print(day_counts)
    
    return index_df

if __name__ == "__main__":
    print("Building complete regime performance index...")
    result = build_complete_regime_performance_index()
    print("Done!")
