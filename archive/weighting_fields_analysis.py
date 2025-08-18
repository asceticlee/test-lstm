#!/usr/bin/env python3
"""
Weighting Fields Analysis Script
===============================
This script extracts and documents all fields and weights used in the 
FastModelTradingWeighter calculations for manual verification.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append('/home/stephen/projects/Testing/TestPy/test-lstm/src')

def analyze_weighting_fields():
    """Extract and document all weighting fields and their values."""
    
    print("=" * 80)
    print("WEIGHTING FIELDS ANALYSIS")
    print("=" * 80)
    
    # Import the FastModelTradingWeighter
    try:
        from model_trading.fast_model_trading_weighter import FastModelTradingWeighter
        print("✅ Successfully imported FastModelTradingWeighter")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Initialize the weighter
    try:
        weighter = FastModelTradingWeighter()
        print("✅ Successfully initialized FastModelTradingWeighter")
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return
    
    # Analysis sections
    print("\n" + "="*80)
    print("1. WEIGHTING ARRAYS ANALYSIS")
    print("="*80)
    
    # Define the actual weighting arrays used in the system
    # These are the same arrays used in detailed_calculation_demo.py
    accuracy_weights = [2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5] * 100  # 1000 weights total
    pnl_weights = [0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0] * 100  # 1000 weights total
    
    # Truncate to match actual performance data length (872 fields)
    accuracy_weights = accuracy_weights[:872]
    pnl_weights = pnl_weights[:872]
    
    print(f"\n📊 ACCURACY-FOCUSED WEIGHTING ARRAY:")
    print(f"    Length: {len(accuracy_weights)}")
    print(f"    Values: {accuracy_weights}")
    print(f"    Sum: {np.sum(accuracy_weights)}")
    print(f"    Non-zero count: {np.count_nonzero(accuracy_weights)}")
    
    print(f"\n💰 PNL-FOCUSED WEIGHTING ARRAY:")
    print(f"    Length: {len(pnl_weights)}")
    print(f"    Values: {pnl_weights}")
    print(f"    Sum: {np.sum(pnl_weights)}")
    print(f"    Non-zero count: {np.count_nonzero(pnl_weights)}")
    
    print("\n" + "="*80)
    print("2. PERFORMANCE METRICS FIELDS ANALYSIS")
    print("="*80)
    
    # Load a sample performance file to see the actual fields
    sample_model = "00001"
    sample_date = "20200108"
    
    try:
        # Try to load sample performance data
        sample_perf_data = weighter._get_performance_data_fast(
            [sample_model], sample_date, 0, accuracy_weights
        )
        
        if sample_perf_data and sample_model in sample_perf_data:
            sample_df = sample_perf_data[sample_model]
            
            print(f"\n📈 SAMPLE PERFORMANCE DATA (Model {sample_model}, Date {sample_date}):")
            print(f"    Total fields: {len(sample_df.columns)}")
            print(f"    Data shape: {sample_df.shape}")
            
            # Group fields by category
            field_categories = {
                'accuracy': [],
                'numerator': [],
                'denominator': [],
                'pnl': []
            }
            
            for col in sample_df.columns:
                if 'acc_' in col.lower():
                    field_categories['accuracy'].append(col)
                elif 'num_' in col.lower():
                    field_categories['numerator'].append(col)
                elif 'den_' in col.lower():
                    field_categories['denominator'].append(col)
                elif 'pnl_' in col.lower():
                    field_categories['pnl'].append(col)
            
            print(f"\n🎯 ACCURACY FIELDS ({len(field_categories['accuracy'])}):")
            for i, field in enumerate(field_categories['accuracy'][:10]):  # Show first 10
                weight_idx = i if i < len(accuracy_weights) else len(accuracy_weights) - 1
                acc_weight = accuracy_weights[weight_idx] if weight_idx < len(accuracy_weights) else 0
                pnl_weight = pnl_weights[weight_idx] if weight_idx < len(pnl_weights) else 0
                print(f"    [{i:2d}] {field:40} | Acc_Weight: {acc_weight:6.3f} | PnL_Weight: {pnl_weight:6.3f}")
            
            if len(field_categories['accuracy']) > 10:
                print(f"    ... and {len(field_categories['accuracy']) - 10} more accuracy fields")
            
            print(f"\n🔢 NUMERATOR FIELDS ({len(field_categories['numerator'])}):")
            for i, field in enumerate(field_categories['numerator'][:10]):  # Show first 10
                base_idx = len(field_categories['accuracy'])
                weight_idx = base_idx + i if base_idx + i < len(accuracy_weights) else len(accuracy_weights) - 1
                acc_weight = accuracy_weights[weight_idx] if weight_idx < len(accuracy_weights) else 0
                pnl_weight = pnl_weights[weight_idx] if weight_idx < len(pnl_weights) else 0
                print(f"    [{weight_idx:2d}] {field:40} | Acc_Weight: {acc_weight:6.3f} | PnL_Weight: {pnl_weight:6.3f}")
            
            if len(field_categories['numerator']) > 10:
                print(f"    ... and {len(field_categories['numerator']) - 10} more numerator fields")
            
            print(f"\n🔢 DENOMINATOR FIELDS ({len(field_categories['denominator'])}):")
            for i, field in enumerate(field_categories['denominator'][:10]):  # Show first 10
                base_idx = len(field_categories['accuracy']) + len(field_categories['numerator'])
                weight_idx = base_idx + i if base_idx + i < len(accuracy_weights) else len(accuracy_weights) - 1
                acc_weight = accuracy_weights[weight_idx] if weight_idx < len(accuracy_weights) else 0
                pnl_weight = pnl_weights[weight_idx] if weight_idx < len(pnl_weights) else 0
                print(f"    [{weight_idx:2d}] {field:40} | Acc_Weight: {acc_weight:6.3f} | PnL_Weight: {pnl_weight:6.3f}")
            
            if len(field_categories['denominator']) > 10:
                print(f"    ... and {len(field_categories['denominator']) - 10} more denominator fields")
            
            print(f"\n💰 PNL FIELDS ({len(field_categories['pnl'])}):")
            for i, field in enumerate(field_categories['pnl'][:10]):  # Show first 10
                base_idx = len(field_categories['accuracy']) + len(field_categories['numerator']) + len(field_categories['denominator'])
                weight_idx = base_idx + i if base_idx + i < len(accuracy_weights) else len(accuracy_weights) - 1
                acc_weight = accuracy_weights[weight_idx] if weight_idx < len(accuracy_weights) else 0
                pnl_weight = pnl_weights[weight_idx] if weight_idx < len(pnl_weights) else 0
                print(f"    [{weight_idx:2d}] {field:40} | Acc_Weight: {acc_weight:6.3f} | PnL_Weight: {pnl_weight:6.3f}")
            
            if len(field_categories['pnl']) > 10:
                print(f"    ... and {len(field_categories['pnl']) - 10} more PnL fields")
            
    except Exception as e:
        print(f"❌ Error loading sample performance data: {e}")
    
    print("\n" + "="*80)
    print("3. FIELD-TO-WEIGHT MAPPING ANALYSIS")
    print("="*80)
    
    # Create comprehensive mapping
    try:
        # Try to get complete field list from performance index
        index_manager_path = "/home/stephen/projects/Testing/TestPy/test-lstm/src/model_trading"
        sys.path.append(index_manager_path)
        
        from performance_index_manager import PerformanceIndexManager
        index_manager = PerformanceIndexManager()
        
        # Get field names from index
        if hasattr(index_manager, 'get_field_names'):
            field_names = index_manager.get_field_names()
            print(f"\n📋 COMPLETE FIELD-TO-WEIGHT MAPPING:")
            print(f"    Total performance fields: {len(field_names)}")
            
            for i, field_name in enumerate(field_names):
                if i < len(accuracy_weights) and i < len(pnl_weights):
                    acc_weight = accuracy_weights[i]
                    pnl_weight = pnl_weights[i]
                    importance = "HIGH" if (acc_weight > 0.1 or pnl_weight > 0.1) else "LOW" if (acc_weight > 0 or pnl_weight > 0) else "ZERO"
                    print(f"    [{i:3d}] {field_name:50} | Acc: {acc_weight:7.4f} | PnL: {pnl_weight:7.4f} | {importance}")
                else:
                    print(f"    [{i:3d}] {field_name:50} | Acc: N/A      | PnL: N/A      | NO_WEIGHT")
        
    except Exception as e:
        print(f"❌ Error getting complete field mapping: {e}")
    
    print("\n" + "="*80)
    print("4. REGIME ADJUSTMENT ANALYSIS")
    print("="*80)
    
    print("\n🐂 REGIME 0 (BULL MARKET) ADJUSTMENTS:")
    print("    - Favors upward predictions")
    print("    - PnL metrics get regime-specific weighting")
    print("    - Direction bias: UP")
    
    print("\n🐻 REGIME 1 (BEAR MARKET) ADJUSTMENTS:")
    print("    - Favors downward predictions") 
    print("    - PnL metrics get regime-specific weighting")
    print("    - Direction bias: DOWN")
    
    print("\n" + "="*80)
    print("5. CALCULATION FORMULA")
    print("="*80)
    
    print("\n📐 MATHEMATICAL FORMULA:")
    print("    For each model i:")
    print("    weighted_score_i = Σ(metric_i,j × weight_j) for j in [0, 871]")
    print("    ")
    print("    Where:")
    print("    - metric_i,j = performance value for model i, field j")
    print("    - weight_j = accuracy_weights[j] OR pnl_weights[j] (strategy dependent)")
    print("    - j ranges from 0 to 871 (872 total fields)")
    print("    ")
    print("    Selection:")
    print("    optimal_model = argmax(weighted_score_i) for all valid models")
    
    print("\n" + "="*80)
    print("6. WEIGHT DISTRIBUTION SUMMARY")
    print("="*80)
    
    # Analyze weight distribution
    acc_nonzero = accuracy_weights[accuracy_weights != 0]
    pnl_nonzero = pnl_weights[pnl_weights != 0]
    
    print(f"\n📊 ACCURACY WEIGHTS DISTRIBUTION:")
    print(f"    Total weights: {len(accuracy_weights)}")
    print(f"    Non-zero weights: {len(acc_nonzero)}")
    print(f"    Zero weights: {len(accuracy_weights) - len(acc_nonzero)}")
    print(f"    Min non-zero: {np.min(acc_nonzero) if len(acc_nonzero) > 0 else 'N/A'}")
    print(f"    Max weight: {np.max(accuracy_weights)}")
    print(f"    Mean weight: {np.mean(accuracy_weights):.6f}")
    
    print(f"\n💰 PNL WEIGHTS DISTRIBUTION:")
    print(f"    Total weights: {len(pnl_weights)}")
    print(f"    Non-zero weights: {len(pnl_nonzero)}")
    print(f"    Zero weights: {len(pnl_weights) - len(pnl_nonzero)}")
    print(f"    Min non-zero: {np.min(pnl_nonzero) if len(pnl_nonzero) > 0 else 'N/A'}")
    print(f"    Max weight: {np.max(pnl_weights)}")
    print(f"    Mean weight: {np.mean(pnl_weights):.6f}")
    
    print("\n" + "="*80)
    print("✅ WEIGHTING FIELDS ANALYSIS COMPLETE")
    print("="*80)
    
    return {
        'accuracy_weights': accuracy_weights,
        'pnl_weights': pnl_weights,
        'field_categories': field_categories if 'field_categories' in locals() else None
    }

def save_detailed_weights_to_file():
    """Save detailed weight mapping to a CSV file for manual inspection."""
    
    print("\n" + "="*80)
    print("SAVING DETAILED WEIGHTS TO FILE")
    print("="*80)
    
    try:
        from model_trading.fast_model_trading_weighter import FastModelTradingWeighter
        weighter = FastModelTradingWeighter()
        
        # Define the actual weighting arrays used in the system
        # These are the same arrays used in detailed_calculation_demo.py
        accuracy_weights = [2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5] * 100  # 1000 weights total
        pnl_weights = [0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0] * 100  # 1000 weights total
        
        # Truncate to match actual performance data length (872 fields)
        accuracy_weights = accuracy_weights[:872]
        pnl_weights = pnl_weights[:872]
        
        # Try to get sample data to extract field names
        sample_perf_data = weighter._get_performance_data_fast(
            ["00001"], "20200108", 0, accuracy_weights
        )
        
        if sample_perf_data and "00001" in sample_perf_data:
            field_names = list(sample_perf_data["00001"].columns)
        else:
            # Fallback: create generic field names
            field_names = [f"field_{i:03d}" for i in range(len(accuracy_weights))]
        
        # Create detailed mapping
        detailed_mapping = []
        for i in range(max(len(accuracy_weights), len(pnl_weights), len(field_names))):
            field_name = field_names[i] if i < len(field_names) else f"field_{i:03d}"
            acc_weight = accuracy_weights[i] if i < len(accuracy_weights) else 0.0
            pnl_weight = pnl_weights[i] if i < len(pnl_weights) else 0.0
            
            # Categorize field
            category = "unknown"
            if 'acc_' in field_name.lower():
                category = "accuracy"
            elif 'num_' in field_name.lower():
                category = "numerator"
            elif 'den_' in field_name.lower():
                category = "denominator"
            elif 'pnl_' in field_name.lower():
                category = "pnl"
            
            # Determine importance
            importance = "zero"
            if acc_weight > 0.1 or pnl_weight > 0.1:
                importance = "high"
            elif acc_weight > 0.01 or pnl_weight > 0.01:
                importance = "medium"
            elif acc_weight > 0 or pnl_weight > 0:
                importance = "low"
            
            detailed_mapping.append({
                'index': i,
                'field_name': field_name,
                'category': category,
                'accuracy_weight': acc_weight,
                'pnl_weight': pnl_weight,
                'importance': importance,
                'weight_difference': abs(acc_weight - pnl_weight),
                'max_weight': max(acc_weight, pnl_weight)
            })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(detailed_mapping)
        
        # Save main file
        output_file = "weighting_fields_detailed.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Saved detailed weights to: {output_file}")
        
        # Save high-importance fields only
        high_importance = df[df['importance'].isin(['high', 'medium'])]
        high_output_file = "weighting_fields_important.csv"
        high_importance.to_csv(high_output_file, index=False)
        print(f"✅ Saved important weights to: {high_output_file}")
        
        # Save summary statistics
        summary_file = "weighting_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("WEIGHTING FIELDS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total fields: {len(df)}\n")
            f.write(f"Fields with accuracy weight > 0: {len(df[df['accuracy_weight'] > 0])}\n")
            f.write(f"Fields with PnL weight > 0: {len(df[df['pnl_weight'] > 0])}\n")
            f.write(f"High importance fields: {len(df[df['importance'] == 'high'])}\n")
            f.write(f"Medium importance fields: {len(df[df['importance'] == 'medium'])}\n")
            
            f.write(f"\nAccuracy weights sum: {df['accuracy_weight'].sum():.6f}\n")
            f.write(f"PnL weights sum: {df['pnl_weight'].sum():.6f}\n")
            
            f.write("\nField categories:\n")
            category_counts = df['category'].value_counts()
            for category, count in category_counts.items():
                f.write(f"  {category}: {count}\n")
            
            f.write("\nTop 20 highest weighted fields (accuracy):\n")
            top_acc = df.nlargest(20, 'accuracy_weight')[['field_name', 'accuracy_weight', 'category']]
            for _, row in top_acc.iterrows():
                f.write(f"  {row['field_name']}: {row['accuracy_weight']:.6f} ({row['category']})\n")
            
            f.write("\nTop 20 highest weighted fields (PnL):\n")
            top_pnl = df.nlargest(20, 'pnl_weight')[['field_name', 'pnl_weight', 'category']]
            for _, row in top_pnl.iterrows():
                f.write(f"  {row['field_name']}: {row['pnl_weight']:.6f} ({row['category']})\n")
        
        print(f"✅ Saved summary to: {summary_file}")
        
        # Print quick overview
        print(f"\n📊 QUICK OVERVIEW:")
        print(f"    Total fields analyzed: {len(df)}")
        print(f"    High importance fields: {len(df[df['importance'] == 'high'])}")
        print(f"    Medium importance fields: {len(df[df['importance'] == 'medium'])}")
        print(f"    Accuracy weights sum: {df['accuracy_weight'].sum():.6f}")
        print(f"    PnL weights sum: {df['pnl_weight'].sum():.6f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error saving weights to file: {e}")
        return None

if __name__ == "__main__":
    # Run analysis
    results = analyze_weighting_fields()
    
    # Save to files
    detailed_df = save_detailed_weights_to_file()
    
    print(f"\n🎯 FILES CREATED FOR MANUAL VERIFICATION:")
    print(f"    1. weighting_fields_detailed.csv - Complete field-to-weight mapping")
    print(f"    2. weighting_fields_important.csv - High/medium importance fields only")
    print(f"    3. weighting_summary.txt - Human-readable summary")
    print(f"\n📝 Review these files to verify the weighting logic implementation.")
