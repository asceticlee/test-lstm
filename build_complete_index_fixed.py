#!/usr/bin/env python3
"""
Fixed script to build complete daily performance index for all models.
"""

import os
import pandas as pd
import glob
from tqdm import tqdm

def build_complete_daily_performance_index():
    """Build complete daily performance index for all models"""
    
    # Define paths
    performance_dir = "model_performance/model_daily_performance"
    index_file = "model_performance/daily_performance_index.csv"
    
    print(f"Building daily performance index from {performance_dir}")
    print(f"Output index file: {index_file}")
    
    # Get all performance files
    pattern = f"{performance_dir}/model_*_daily_performance.csv"
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
        # filename format: model_XXXXX_daily_performance.csv
        model_id = filename.split('_')[1]
        
        try:
            # Read just the TradingDay column to get all trading days
            df = pd.read_csv(file_path, usecols=['TradingDay'])
            
            # Add index entries for each row
            for row_idx, trading_day in enumerate(df['TradingDay']):
                index_entries.append({
                    'model_id': model_id,
                    'trading_day': trading_day,
                    'file_path': file_path,
                    'row_number': row_idx + 1  # 1-based indexing (header is row 0)
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
    print("\nModel distribution:")
    model_counts = index_df['model_id'].value_counts().head(10)
    print(model_counts)
    
    return index_df

if __name__ == "__main__":
    print("Building complete daily performance index...")
    result = build_complete_daily_performance_index()
    print("Done!")
