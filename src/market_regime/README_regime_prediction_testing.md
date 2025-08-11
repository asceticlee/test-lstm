# Market Regime Prediction Accuracy Testing

This directory contains scripts to test how accurately market regimes can be predicted as trading data accumulates throughout the day.

## Problem Statement

When using market regime-based trading strategies, you need to know the current market regime to select the appropriate model. However, the regime is determined using the full trading day's data (10:35 AM to 12:00 PM). This creates a practical challenge: **How early in the trading day can we accurately predict the final regime?**

## Scripts Overview

### 1. `regime_prediction_accuracy_test.py` (Main Test Script)

**Purpose**: Tests regime prediction accuracy as the trading day progresses from 10:35 AM to 12:00 PM.

**Key Features**:
- Uses partial day data to predict regimes at multiple time points
- Compares predictions against ground truth (full day regimes)
- Tracks accuracy progression over time
- Handles feature alignment with pre-trained GMM models

**Usage**:
```bash
python regime_prediction_accuracy_test.py --test_intervals 20 --min_duration 5
```

**Parameters**:
- `--test_intervals`: Number of time points to test (default: 20)
- `--min_duration`: Minimum minutes from start before testing (default: 5)
- `--data_path`: Path to trading data (default: data/history_spot_quote.csv)
- `--reference_model_path`: Path to trained GMM models (default: ../../market_regime/gmm/daily)

### 2. `regime_prediction_analysis_summary.py` (Results Interpreter)

**Purpose**: Provides practical interpretation of accuracy test results.

**Key Outputs**:
- Accuracy progression milestones
- Trading decision thresholds
- Practical trading recommendations
- Regime-specific performance analysis

**Usage**:
```bash
python regime_prediction_analysis_summary.py
```

### 3. `advanced_regime_analysis.py` (Deep Analysis)

**Purpose**: Advanced analysis including confidence thresholds, optimal trading windows, and risk assessment.

**Key Features**:
- Confidence threshold analysis
- Accuracy improvement rate calculation
- Critical decision points identification
- Trading strategy recommendations
- Risk assessment for different time periods

**Usage**:
```bash
python advanced_regime_analysis.py
```

## Key Findings

Based on testing with 1,407 trading days:

### Accuracy Progression
- **10:45 AM (11 min)**: 18.8% accuracy - Too early for reliable predictions
- **11:16 AM (41 min)**: 45.7% accuracy - Moderate confidence emerges
- **11:38 AM (63 min)**: 64.3% accuracy - **OPTIMAL TRADING WINDOW**
- **11:46 AM (72 min)**: 74.6% accuracy - High confidence
- **11:51 AM (76 min)**: 81.3% accuracy - Very high confidence
- **12:00 PM (85 min)**: 100.0% accuracy - Perfect (by definition)

### Trading Recommendations

#### Optimal Strategy
- **Start monitoring**: 10:35 AM (collect data)
- **Begin trading**: 11:38 AM (64% accuracy, 22 minutes left)
- **Strategy type**: Quick execution with moderate position sizes
- **Risk level**: Medium (36% chance of wrong regime)

#### Alternative Thresholds
- **Conservative (70% accuracy)**: Wait until 11:46 AM (13 minutes left)
- **Aggressive (50% accuracy)**: Start at 11:25 AM (35 minutes left)
- **High conviction (80% accuracy)**: Wait until 11:51 AM (9 minutes left)

### Regime-Specific Performance

Different regimes show varying predictability:
- **Regimes 1, 2, 3, 4**: Generally predictable by mid-day
- **Regime 0**: Harder to predict early, improves significantly late
- **Regime 5**: Consistently difficult to predict (rare regime, only 0.7% of days)

## Output Files

The scripts generate several output files in `../../market_regime/prediction_accuracy/`:

1. **`regime_prediction_accuracy_results.csv`**: Detailed accuracy results for each test time point
2. **`accuracy_test_summary.json`**: Summary statistics and key metrics
3. **`accuracy_over_time.png`**: Visualization of accuracy progression

## Practical Implementation

### For Live Trading:
1. **Data Collection Phase** (10:35-11:38): Collect and analyze incoming market data
2. **Decision Phase** (11:38): Begin regime-based model selection with 64% confidence
3. **Execution Phase** (11:38-12:00): Execute trades with 22-minute window
4. **Risk Management**: Use position sizing appropriate for 36% error risk

### For Strategy Development:
- Use these results to optimize regime-based model selection timing
- Consider ensemble approaches combining early and late predictions
- Develop fallback strategies for low-confidence periods

## Dependencies

- pandas, numpy (data handling)
- scikit-learn (ML models)
- matplotlib (visualization)
- Pre-trained GMM models from `gmm_regime_clustering.py`
- Statistical features from `statistical_features.py`

## Notes

- All tests use the same feature extraction pipeline as the original GMM training
- Feature alignment ensures compatibility with pre-trained models
- Results are based on historical data and may vary with market conditions
- The 100% accuracy at 12:00 PM is by definition (using full data to predict itself)
