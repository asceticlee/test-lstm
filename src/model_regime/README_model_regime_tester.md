# Model Regime Tester - Updated Guide

## Overview
The `model_regime_tester.py` script has been updated to support both daily and sequence-based regime clustering methods from the GMM clustering pipeline.

## Directory Structure
The script now reads regime assignments from the updated directory structure:

### Daily Clustering
- **Input**: `market_regime/gmm/daily/daily_regime_assignments.csv`
- **Characteristics**: `market_regime/gmm/daily/regime_characteristics.csv`
- **Output**: 
  - `model_regime/model_regime_test_results_daily.csv`
  - `model_regime/model_regime_ranking_daily.csv`

### Sequence Clustering  
- **Input**: `market_regime/gmm/sequence/sequence_regime_assignments.csv`
- **Characteristics**: `market_regime/gmm/sequence/regime_characteristics.csv`
- **Output**:
  - `model_regime/model_regime_test_results_sequence.csv`
  - `model_regime/model_regime_ranking_sequence.csv`

## Usage

### Daily Clustering (Traditional)
```bash
# Test all models using daily regime clustering
python model_regime_tester.py --clustering_method daily

# Test first 5 models only
python model_regime_tester.py --clustering_method daily --max_models 5

# Test filtering logic only
python model_regime_tester.py --clustering_method daily --test_filtering
```

### Sequence Clustering (Minute-by-minute)
```bash
# Test all models using sequence regime clustering
python model_regime_tester.py --clustering_method sequence

# Test first 3 models only  
python model_regime_tester.py --clustering_method sequence --max_models 3

# Test filtering logic only
python model_regime_tester.py --clustering_method sequence --test_filtering
```

## Key Differences

### Daily Clustering
- **Granularity**: One regime assignment per trading day
- **Time Period**: 10:35 AM to 12:00 PM daily analysis
- **Regimes**: Typically 5-6 market regimes
- **Sample Size**: ~1,400 regime assignments (one per trading day)

### Sequence Clustering
- **Granularity**: Multiple regime assignments per trading day
- **Time Period**: Sliding 30-minute windows from 10:05-11:30 (reference times)
- **Analysis Windows**: 10:06-10:36, 10:07-10:37, ..., 11:31-12:00
- **Regimes**: Typically 6-7 market regimes
- **Sample Size**: ~120,000+ regime assignments (85+ per trading day)

## Output Files

Both clustering methods generate the same structure but with different suffixes:

### Test Results (`model_regime_test_results_{method}.csv`)
Contains detailed performance metrics for each model-regime combination:
- Model ID, regime, training period
- Test samples and MAE
- Threshold-based accuracy metrics (upside/downside for 0.0-0.8 thresholds)
- Regime-specific rankings

### Ranking Summary (`model_regime_ranking_{method}.csv`)
Contains model rankings across all regimes:
- Model training periods
- Rankings for each regime (upside/downside separately)
- Best regime identification for each model

## Performance Considerations

### Daily Clustering
- **Speed**: Fast execution (~1-2 minutes for full test suite)
- **Memory**: Low memory usage
- **Best for**: Traditional daily regime analysis

### Sequence Clustering  
- **Speed**: Moderate execution (~5-10 minutes for full test suite)
- **Memory**: Higher memory usage due to larger datasets
- **Best for**: Intraday regime analysis and minute-by-minute regime changes

## Examples

The script maintains backward compatibility. Legacy output files without clustering method suffixes will continue to work but are not updated by the new versions.

Current output files:
```
model_regime/
├── model_regime_test_results_daily.csv      # Daily clustering results
├── model_regime_test_results_sequence.csv   # Sequence clustering results
├── model_regime_ranking_daily.csv           # Daily clustering rankings
├── model_regime_ranking_sequence.csv        # Sequence clustering rankings
├── model_regime_test_results.csv            # Legacy (not updated)
└── model_regime_ranking.csv                 # Legacy (not updated)
```
