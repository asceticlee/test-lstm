#!/usr/bin/env python3
"""
Overnight Gap Impact Analysis for Market Regime Clustering

This script tests how including overnight gap features affects:
1. Market regime clustering quality
2. Early regime prediction accuracy (50% accuracy milestone timing)
3. Feature importance and regime characteristics

It compares four scenarios:
- Standard clustering (no overnight features)
- Overnight gap only
- Overnight indicators only  
- Full overnight analysis (gap + indicators)

Usage:
    python overnight_gap_analysis.py [--test_intervals N]
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import argparse
import warnings
import subprocess
import json
warnings.filterwarnings('ignore')

# Add the src directory to the path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent
sys.path.insert(0, str(src_dir))

class OvernightGapAnalyzer:
    """
    Analyze the impact of overnight gap features on regime clustering and prediction
    """
    
    def __init__(self, data_path='data/history_spot_quote.csv',
                 base_output_dir='../../market_regime/overnight_analysis',
                 trading_start='09:30', trading_end='12:00'):
        """
        Initialize the overnight gap analyzer
        """
        script_dir = Path(__file__).parent.absolute()
        
        self.data_path = data_path
        self.base_output_dir = script_dir / base_output_dir
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trading_start = trading_start
        self.trading_end = trading_end
        
        # Test scenarios
        self.scenarios = {
            'standard': {
                'name': 'Standard (No Overnight)',
                'include_overnight_gap': False,
                'overnight_indicators': False,
                'output_suffix': 'standard'
            },
            'gap_only': {
                'name': 'Overnight Gap Only',
                'include_overnight_gap': True,
                'overnight_indicators': False,
                'output_suffix': 'gap_only'
            },
            'indicators_only': {
                'name': 'Overnight Indicators Only',
                'include_overnight_gap': False,
                'overnight_indicators': True,
                'output_suffix': 'indicators_only'
            },
            'full_overnight': {
                'name': 'Full Overnight Analysis',
                'include_overnight_gap': True,
                'overnight_indicators': True,
                'output_suffix': 'full_overnight'
            }
        }
        
        self.results = {}
        
        print(f"Overnight Gap Analyzer initialized")
        print(f"Testing period: {trading_start} to {trading_end}")
        print(f"Scenarios to test: {len(self.scenarios)}")
        print(f"Output directory: {self.base_output_dir}")
    
    def run_clustering_scenario(self, scenario_key):
        """Run GMM clustering for a specific scenario"""
        scenario = self.scenarios[scenario_key]
        print(f"\n{'='*60}")
        print(f"RUNNING SCENARIO: {scenario['name']}")
        print(f"{'='*60}")
        
        # Create output directory for this scenario
        output_dir = self.base_output_dir / scenario['output_suffix']
        output_dir.mkdir(exist_ok=True)
        
        # Build command for GMM clustering
        cmd = [
            sys.executable,
            'gmm_regime_clustering.py',
            '--data_path', self.data_path,
            '--trading_start', self.trading_start,
            '--trading_end', self.trading_end,
            '--output_dir', str(output_dir)
        ]
        
        # Add overnight parameters
        if scenario['include_overnight_gap']:
            cmd.append('--include_overnight_gap')
        
        if scenario['overnight_indicators']:
            cmd.append('--overnight_indicators')
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run the clustering
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
            
            if result.returncode == 0:
                print(f"✅ Clustering completed successfully for {scenario['name']}")
                
                # Load and store results
                clustering_info_file = output_dir / 'clustering_info.json'
                regime_assignments_file = output_dir / 'daily_regime_assignments.csv'
                
                if clustering_info_file.exists() and regime_assignments_file.exists():
                    with open(clustering_info_file, 'r') as f:
                        clustering_info = json.load(f)
                    
                    regime_assignments = pd.read_csv(regime_assignments_file)
                    
                    self.results[scenario_key] = {
                        'scenario': scenario,
                        'clustering_info': clustering_info,
                        'regime_assignments': regime_assignments,
                        'output_dir': output_dir,
                        'success': True
                    }
                else:
                    print(f"❌ Output files not found for {scenario['name']}")
                    self.results[scenario_key] = {'success': False, 'error': 'Output files not found'}
            else:
                print(f"❌ Clustering failed for {scenario['name']}")
                print(f"Error: {result.stderr}")
                self.results[scenario_key] = {'success': False, 'error': result.stderr}
                
        except Exception as e:
            print(f"❌ Exception running {scenario['name']}: {e}")
            self.results[scenario_key] = {'success': False, 'error': str(e)}
    
    def run_prediction_accuracy_test(self, scenario_key, test_intervals=15):
        """Run progressive prediction accuracy test for a scenario"""
        if scenario_key not in self.results or not self.results[scenario_key]['success']:
            print(f"Skipping prediction test for {scenario_key} - clustering failed")
            return None
        
        scenario = self.scenarios[scenario_key]
        output_dir = self.results[scenario_key]['output_dir']
        
        print(f"\n--- Running Prediction Accuracy Test for {scenario['name']} ---")
        
        # Create prediction test output directory
        pred_output_dir = output_dir / 'prediction_accuracy'
        pred_output_dir.mkdir(exist_ok=True)
        
        # Build command for prediction accuracy test
        cmd = [
            sys.executable,
            'progressive_regime_prediction_test.py',
            '--data_path', self.data_path,
            '--regime_assignments', str(output_dir / 'daily_regime_assignments.csv'),
            '--models_path', str(output_dir),
            '--output_dir', str(pred_output_dir),
            '--test_intervals', str(test_intervals)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
            
            if result.returncode == 0:
                print(f"✅ Prediction test completed for {scenario['name']}")
                
                # Load prediction results
                pred_results_file = pred_output_dir / 'progressive_regime_prediction_results.csv'
                if pred_results_file.exists():
                    pred_results = pd.read_csv(pred_results_file)
                    self.results[scenario_key]['prediction_results'] = pred_results
                    
                    # Find 50% accuracy milestone
                    milestone_50 = pred_results[pred_results['overall_accuracy'] >= 0.5]
                    if len(milestone_50) > 0:
                        first_50 = milestone_50.iloc[0]
                        self.results[scenario_key]['accuracy_50_time'] = first_50['end_time']
                        self.results[scenario_key]['accuracy_50_minutes'] = first_50['period_minutes']
                        print(f"  50% accuracy achieved at: {first_50['end_time']} ({first_50['period_minutes']:.0f} minutes)")
                    else:
                        self.results[scenario_key]['accuracy_50_time'] = 'Never'
                        self.results[scenario_key]['accuracy_50_minutes'] = 999
                        print(f"  50% accuracy: Never achieved")
                else:
                    print(f"❌ Prediction results file not found")
            else:
                print(f"❌ Prediction test failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Exception in prediction test: {e}")
    
    def compare_scenarios(self):
        """Compare results across all scenarios"""
        print(f"\n{'='*80}")
        print("OVERNIGHT GAP IMPACT ANALYSIS - COMPARISON RESULTS")
        print(f"{'='*80}")
        
        successful_scenarios = {k: v for k, v in self.results.items() if v.get('success', False)}
        
        if len(successful_scenarios) == 0:
            print("❌ No successful scenarios to compare")
            return
        
        print(f"\n📊 REGIME CLUSTERING COMPARISON:")
        print("-" * 60)
        print(f"{'Scenario':<25} {'Regimes':<10} {'Features':<10} {'50% Accuracy Time':<20}")
        print("-" * 60)
        
        for scenario_key, results in successful_scenarios.items():
            scenario_name = self.scenarios[scenario_key]['name']
            n_regimes = results['clustering_info']['n_regimes']
            n_features = len(results['clustering_info']['feature_names'])
            accuracy_time = results.get('accuracy_50_time', 'Not tested')
            
            print(f"{scenario_name:<25} {n_regimes:<10} {n_features:<10} {accuracy_time:<20}")
        
        # Feature analysis
        print(f"\n🔍 FEATURE ANALYSIS:")
        print("-" * 60)
        
        # Compare feature counts and types
        for scenario_key, results in successful_scenarios.items():
            scenario_name = self.scenarios[scenario_key]['name']
            feature_names = results['clustering_info']['feature_names']
            
            # Count different types of features
            standard_features = [f for f in feature_names if not any(keyword in f.lower() for keyword in ['overnight', 'gap', 'opening', 'vwap'])]
            overnight_features = [f for f in feature_names if any(keyword in f.lower() for keyword in ['overnight', 'gap', 'opening', 'vwap'])]
            
            print(f"\n{scenario_name}:")
            print(f"  Total features: {len(feature_names)}")
            print(f"  Standard features: {len(standard_features)}")
            print(f"  Overnight features: {len(overnight_features)}")
            
            if len(overnight_features) > 0:
                print(f"  Overnight feature list: {overnight_features}")
        
        # Regime distribution comparison
        print(f"\n📈 REGIME DISTRIBUTION COMPARISON:")
        print("-" * 60)
        
        # Compare regime distributions
        for scenario_key, results in successful_scenarios.items():
            scenario_name = self.scenarios[scenario_key]['name']
            regime_assignments = results['regime_assignments']
            regime_counts = regime_assignments['Regime'].value_counts().sort_index()
            
            print(f"\n{scenario_name}:")
            for regime, count in regime_counts.items():
                percentage = count / len(regime_assignments) * 100
                print(f"  Regime {regime}: {count:4d} days ({percentage:5.1f}%)")
        
        # Prediction accuracy timeline comparison
        if any('prediction_results' in results for results in successful_scenarios.values()):
            print(f"\n⏰ PREDICTION ACCURACY TIMELINE COMPARISON:")
            print("-" * 60)
            
            # Find common time points for comparison
            all_pred_results = {}
            for scenario_key, results in successful_scenarios.items():
                if 'prediction_results' in results:
                    all_pred_results[scenario_key] = results['prediction_results']
            
            if all_pred_results:
                # Create comparison table
                print(f"{'Time':<10}", end='')
                for scenario_key in all_pred_results.keys():
                    scenario_name = self.scenarios[scenario_key]['name'][:12]  # Truncate for display
                    print(f"{scenario_name:<15}", end='')
                print()
                
                print("-" * (10 + 15 * len(all_pred_results)))
                
                # Get common time points (use first scenario as reference)
                reference_scenario = list(all_pred_results.keys())[0]
                reference_times = all_pred_results[reference_scenario]['end_time'].values
                
                for time_point in reference_times:
                    print(f"{time_point:<10}", end='')
                    
                    for scenario_key, pred_results in all_pred_results.items():
                        # Find accuracy for this time point
                        time_match = pred_results[pred_results['end_time'] == time_point]
                        if len(time_match) > 0:
                            accuracy = time_match['overall_accuracy'].iloc[0]
                            print(f"{accuracy:<15.3f}", end='')
                        else:
                            print(f"{'N/A':<15}", end='')
                    print()
        
        # Summary insights
        print(f"\n💡 KEY INSIGHTS:")
        print("-" * 60)
        
        # Find best performing scenario for early prediction
        best_early_scenario = None
        best_early_time = 999
        
        for scenario_key, results in successful_scenarios.items():
            if 'accuracy_50_minutes' in results:
                if results['accuracy_50_minutes'] < best_early_time:
                    best_early_time = results['accuracy_50_minutes']
                    best_early_scenario = scenario_key
        
        if best_early_scenario:
            scenario_name = self.scenarios[best_early_scenario]['name']
            time_str = self.results[best_early_scenario]['accuracy_50_time']
            print(f"🏆 Best early prediction: {scenario_name}")
            print(f"   Achieves 50% accuracy at: {time_str} ({best_early_time:.0f} minutes)")
        
        # Compare feature complexity
        feature_counts = {k: len(v['clustering_info']['feature_names']) for k, v in successful_scenarios.items()}
        min_features = min(feature_counts.values())
        max_features = max(feature_counts.values())
        
        print(f"\n📊 Feature complexity range: {min_features} to {max_features} features")
        
        # Identify scenarios with similar regime counts
        regime_counts = {k: v['clustering_info']['n_regimes'] for k, v in successful_scenarios.items()}
        most_common_regimes = max(set(regime_counts.values()), key=list(regime_counts.values()).count)
        
        print(f"🎯 Most common regime count: {most_common_regimes} regimes")
        scenarios_with_common_regimes = [k for k, v in regime_counts.items() if v == most_common_regimes]
        print(f"   Scenarios with {most_common_regimes} regimes: {[self.scenarios[k]['name'] for k in scenarios_with_common_regimes]}")
    
    def save_comparison_results(self):
        """Save detailed comparison results"""
        print("\nSaving overnight gap analysis results...")
        
        # Create summary report
        summary_file = self.base_output_dir / 'overnight_gap_analysis_summary.json'
        
        summary_data = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'trading_period': f"{self.trading_start} to {self.trading_end}",
            'scenarios_tested': len(self.scenarios),
            'successful_scenarios': len([r for r in self.results.values() if r.get('success', False)]),
            'scenario_results': {}
        }
        
        for scenario_key, results in self.results.items():
            if results.get('success', False):
                scenario_summary = {
                    'scenario_name': self.scenarios[scenario_key]['name'],
                    'include_overnight_gap': self.scenarios[scenario_key]['include_overnight_gap'],
                    'overnight_indicators': self.scenarios[scenario_key]['overnight_indicators'],
                    'n_regimes': results['clustering_info']['n_regimes'],
                    'n_features': len(results['clustering_info']['feature_names']),
                    'total_days': results['clustering_info']['total_days'],
                    'accuracy_50_time': results.get('accuracy_50_time', 'Not tested'),
                    'accuracy_50_minutes': results.get('accuracy_50_minutes', 999)
                }
                
                # Add regime distribution
                regime_assignments = results['regime_assignments']
                regime_dist = regime_assignments['Regime'].value_counts().sort_index().to_dict()
                scenario_summary['regime_distribution'] = regime_dist
                
                summary_data['scenario_results'][scenario_key] = scenario_summary
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Saved summary to: {summary_file}")
        
        # Create CSV comparison
        comparison_rows = []
        for scenario_key, results in self.results.items():
            if results.get('success', False):
                row = {
                    'scenario': self.scenarios[scenario_key]['name'],
                    'overnight_gap': self.scenarios[scenario_key]['include_overnight_gap'],
                    'overnight_indicators': self.scenarios[scenario_key]['overnight_indicators'],
                    'n_regimes': results['clustering_info']['n_regimes'],
                    'n_features': len(results['clustering_info']['feature_names']),
                    'total_days': results['clustering_info']['total_days'],
                    'accuracy_50_time': results.get('accuracy_50_time', 'Not tested'),
                    'accuracy_50_minutes': results.get('accuracy_50_minutes', 999)
                }
                comparison_rows.append(row)
        
        if comparison_rows:
            comparison_df = pd.DataFrame(comparison_rows)
            comparison_file = self.base_output_dir / 'scenario_comparison.csv'
            comparison_df.to_csv(comparison_file, index=False)
            print(f"Saved comparison table to: {comparison_file}")
    
    def run_full_analysis(self, test_intervals=15):
        """Run the complete overnight gap analysis"""
        print(f"Starting comprehensive overnight gap analysis...")
        print(f"Testing {len(self.scenarios)} scenarios with prediction accuracy tests")
        
        # Step 1: Run clustering for all scenarios
        for scenario_key in self.scenarios.keys():
            self.run_clustering_scenario(scenario_key)
        
        # Step 2: Run prediction accuracy tests
        for scenario_key in self.scenarios.keys():
            if self.results.get(scenario_key, {}).get('success', False):
                self.run_prediction_accuracy_test(scenario_key, test_intervals)
        
        # Step 3: Compare results
        self.compare_scenarios()
        
        # Step 4: Save results
        self.save_comparison_results()
        
        print(f"\n{'='*80}")
        print("OVERNIGHT GAP ANALYSIS COMPLETED")
        print(f"{'='*80}")
        print(f"Results saved to: {self.base_output_dir}")
        
        return self.results

def main():
    parser = argparse.ArgumentParser(description='Overnight Gap Impact Analysis')
    parser.add_argument('--data_path', default='data/history_spot_quote.csv',
                       help='Path to trading data CSV file')
    parser.add_argument('--trading_start', default='09:30',
                       help='Trading start time (HH:MM format)')
    parser.add_argument('--trading_end', default='12:00',
                       help='Trading end time (HH:MM format)')
    parser.add_argument('--test_intervals', type=int, default=15,
                       help='Test intervals for prediction accuracy (minutes)')
    parser.add_argument('--output_dir', default='../../market_regime/overnight_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyzer = OvernightGapAnalyzer(
        data_path=args.data_path,
        base_output_dir=args.output_dir,
        trading_start=args.trading_start,
        trading_end=args.trading_end
    )
    
    results = analyzer.run_full_analysis(args.test_intervals)
    
    return 0

if __name__ == "__main__":
    exit(main())
