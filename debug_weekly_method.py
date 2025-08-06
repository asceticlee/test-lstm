#!/usr/bin/env python3
"""
Enhanced debug to trace exactly what happens in the weekly method
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Patch the ModelTester class to add debug prints for the problematic week
import sys
sys.path.append('src')
from test_all_models_regimes import ModelTester

# Create a custom subclass with debug prints
class DebugModelTester(ModelTester):
    def generate_weekly_best_models(self, df, start_model, end_model):
        """Enhanced version with debug prints for the problematic case"""
        print("🔍 DEBUG: Generating weekly best model tracking with debug...")
        
        # Load the enhanced best regime summary with regime base columns
        results_dir = Path('test_results')
        best_regime_file = results_dir / 'best_regime_summary_1_425.csv'
        regime_base_df = pd.read_csv(best_regime_file)
        
        # Get all unique trading days from regime assignments
        unique_trading_days = sorted(self.regime_assignments['TradingDay'].unique())
        unique_regimes = sorted(df['regime'].unique())
        
        # Build regime models (abbreviated for debug)
        valid_models = set(f"{i:05d}" for i in range(start_model, end_model + 1))
        upside_regime_models = {}
        downside_regime_models = {}
        
        for _, row in regime_base_df.iterrows():
            model_id = f"{int(row['model_id']):05d}"
            if model_id not in valid_models:
                continue
            
            upside_regime = row['model_regime_upside']
            if upside_regime not in upside_regime_models:
                upside_regime_models[upside_regime] = []
            upside_regime_models[upside_regime].append(model_id)
            
            downside_regime = row['model_regime_downside'] 
            if downside_regime not in downside_regime_models:
                downside_regime_models[downside_regime] = []
            downside_regime_models[downside_regime].append(model_id)
        
        # Group trading days into weeks
        trading_days_df = pd.DataFrame({'trading_day': unique_trading_days})
        trading_days_df['date'] = pd.to_datetime(trading_days_df['trading_day'], format='%Y%m%d')
        trading_days_df['week_start'] = trading_days_df['date'].dt.to_period('W').dt.start_time
        trading_days_df['week_end'] = trading_days_df['date'].dt.to_period('W').dt.end_time
        
        weekly_groups = trading_days_df.groupby(['week_start', 'week_end'])
        weekly_results = []
        
        target_week_start = pd.to_datetime('2025-01-20')
        target_week_end = pd.to_datetime('2025-01-26')
        
        for (week_start, week_end), week_data in weekly_groups:
            week_trading_days = week_data['trading_day'].tolist()
            
            # Check if this is our target problematic week
            is_target_week = (week_start <= target_week_start <= week_end) or (20250120 in week_trading_days)
            
            # Get regime distribution for this week
            week_regime_data = self.regime_assignments[
                self.regime_assignments['TradingDay'].isin(week_trading_days)
            ]
            
            if len(week_regime_data) == 0:
                continue
            
            regime_counts = week_regime_data['Regime'].value_counts()
            most_common_regime = regime_counts.index[0]
            
            regime_distribution = {}
            total_records = len(week_regime_data)
            for regime in unique_regimes:
                count = regime_counts.get(regime, 0)
                regime_distribution[f'regime_{regime}_pct'] = count / total_records * 100
            
            # For each test regime, run competitive selection
            for test_regime in unique_regimes:
                
                if is_target_week and test_regime == 0:
                    print(f"\\n🎯 TARGET WEEK FOUND: {week_start.strftime('%Y%m%d')}-{week_end.strftime('%Y%m%d')}, regime {test_regime}")
                    print(f"Week trading days: {week_trading_days}")
                
                # Get candidate models for upside in this regime
                upside_candidates = upside_regime_models.get(test_regime, [])
                
                if is_target_week and test_regime == 0:
                    print(f"Initial upside candidates: {upside_candidates}")
                
                # Filter out models trained during this week
                upside_candidates = [
                    model_id for model_id in upside_candidates
                    if not self._model_trained_during_week(model_id, week_trading_days, regime_base_df)
                ]
                
                if is_target_week and test_regime == 0:
                    print(f"After filtering upside candidates: {upside_candidates}")
                    # Check specific model 00053
                    should_exclude_53 = self._model_trained_during_week('00053', week_trading_days, regime_base_df)
                    print(f"Model 00053 should be excluded: {should_exclude_53}")
                
                # Get candidate models for downside in this regime  
                downside_candidates = downside_regime_models.get(test_regime, [])
                downside_candidates = [
                    model_id for model_id in downside_candidates
                    if not self._model_trained_during_week(model_id, week_trading_days, regime_base_df)
                ]
                
                if is_target_week and test_regime == 0:
                    print(f"After filtering downside candidates: {downside_candidates}")
                
                # Simulate weekly competition: find best performing model for upside
                best_upside_model = None
                best_upside_score = None
                if upside_candidates:
                    upside_results = []
                    if is_target_week and test_regime == 0:
                        print("Running upside competition...")
                        
                    for model_id in upside_candidates:
                        model_data = df[
                            (df['model_id'] == int(model_id)) & 
                            (df['regime'] == test_regime)
                        ]
                        if is_target_week and test_regime == 0:
                            print(f"  Model {model_id}: found {len(model_data)} performance records")
                            
                        if len(model_data) > 0:
                            row = model_data.iloc[0]
                            upside_accs = [row[f'upside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
                            avg_upside_score = np.mean(upside_accs)
                            upside_results.append((model_id, avg_upside_score))
                            
                            if is_target_week and test_regime == 0:
                                print(f"    Average score: {avg_upside_score:.4f}")
                    
                    if upside_results:
                        upside_results.sort(key=lambda x: x[1], reverse=True)
                        best_upside_model, best_upside_score = upside_results[0]
                        
                        if is_target_week and test_regime == 0:
                            print(f"Upside winner: {best_upside_model} with score {best_upside_score:.4f}")
                            print(f"All upside results: {upside_results}")
                
                # Simulate weekly competition: find best performing model for downside
                best_downside_model = None  
                best_downside_score = None
                if downside_candidates:
                    downside_results = []
                    for model_id in downside_candidates:
                        model_data = df[
                            (df['model_id'] == int(model_id)) & 
                            (df['regime'] == test_regime)
                        ]
                        if len(model_data) > 0:
                            row = model_data.iloc[0]
                            downside_accs = [row[f'downside_{t:.1f}'] for t in np.arange(0, 0.9, 0.1)]
                            avg_downside_score = np.mean(downside_accs)
                            downside_results.append((model_id, avg_downside_score))
                    
                    if downside_results:
                        downside_results.sort(key=lambda x: x[1], reverse=True)
                        best_downside_model, best_downside_score = downside_results[0]
                        
                        if is_target_week and test_regime == 0:
                            print(f"Downside winner: {best_downside_model}")
                
                if is_target_week and test_regime == 0:
                    print(f"Final selections - Upside: {best_upside_model}, Downside: {best_downside_model}")
                
                # Record the weekly results
                result = {
                    'week_start': int(week_start.strftime('%Y%m%d')),
                    'week_end': int(week_end.strftime('%Y%m%d')),
                    'week_trading_days': len(week_trading_days),
                    'most_common_regime': most_common_regime,
                    'test_regime': test_regime,
                    'best_upside_model_id': int(best_upside_model) if best_upside_model else None,
                    'best_upside_score': best_upside_score,
                    'best_upside_rank': 1 if best_upside_model else None,
                    'best_downside_model_id': int(best_downside_model) if best_downside_model else None,
                    'best_downside_score': best_downside_score,
                    'best_downside_rank': 1 if best_downside_model else None,
                    'competing_upside_models': len(upside_candidates),
                    'competing_downside_models': len(downside_candidates)
                }
                result.update(regime_distribution)
                weekly_results.append(result)
                
                if is_target_week and test_regime == 0:
                    print(f"Saved to results: upside_model_id={result['best_upside_model_id']}, downside_model_id={result['best_downside_model_id']}")
        
        # Convert to DataFrame and save
        weekly_df = pd.DataFrame(weekly_results)
        weekly_df = weekly_df.sort_values(['week_start', 'test_regime']).reset_index(drop=True)
        
        weekly_output_file = results_dir / f'weekly_best_models_{start_model}_{end_model}.csv'
        weekly_df.to_csv(weekly_output_file, index=False)
        print(f"\\nSaved debug weekly best models to {weekly_output_file}")
        
        return weekly_df

# Run the debug version
if __name__ == "__main__":
    debug_tester = DebugModelTester()
    df = pd.read_csv('test_results/model_regime_test_results_1_425.csv')
    result = debug_tester.generate_weekly_best_models(df, 40, 70)
