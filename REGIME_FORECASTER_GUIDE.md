# Regime Prediction Forecaster - Usage Guide

## Overview

The `regime_prediction_forecaster.py` script implements a comprehensive forecasting system that:

1. **Predicts market regimes** using HMM (Hidden Markov Model) or naive statistical methods
2. **Selects optimal models** for each regime (separate upside and downside models)
3. **Prevents data leakage** by excluding models trained on the prediction period
4. **Generates minute-by-minute predictions** using the selected models

## Output Format

The script generates CSV files with the following columns:

- `trading_day`: Trading day (YYYYMMDD format)
- `ms_of_day`: Millisecond of the trading day
- `actual`: Actual market movement
- `upside_predict`: Prediction from the best upside model
- `upside_model_id`: ID of the upside model used
- `upside_regime`: Predicted regime for upside
- `upside_rank_of_model`: Rank of the upside model within the regime
- `downside_predict`: Prediction from the best downside model
- `downside_model_id`: ID of the downside model used
- `downside_regime`: Predicted regime for downside
- `downside_rank_of_model`: Rank of the downside model within the regime

## Usage Examples

### Basic Usage

```bash
# Daily forecasting with naive regime prediction
python regime_prediction_forecaster.py --start_date 20200102 --end_date 20200110 --period daily --method naive

# Weekly forecasting with HMM regime prediction
python regime_prediction_forecaster.py --start_date 20200106 --end_date 20200120 --period weekly --method hmm
```

### Command Line Arguments

- `--start_date`: Start date for forecasting (required, format: YYYYMMDD)
- `--end_date`: End date for forecasting (required, format: YYYYMMDD)
- `--period`: Prediction period - `daily` or `weekly` (default: daily)
- `--method`: Regime prediction method - `hmm` or `naive` (default: hmm)

### Examples with Different Configurations

```bash
# Single day forecast with HMM
python regime_prediction_forecaster.py --start_date 20200102 --end_date 20200102 --period daily --method hmm

# Multi-day forecast with naive method
python regime_prediction_forecaster.py --start_date 20200106 --end_date 20200115 --period daily --method naive

# Weekly intervals with HMM
python regime_prediction_forecaster.py --start_date 20200106 --end_date 20200220 --period weekly --method hmm
```

## How It Works

### 1. Data Loading
- Loads trading data from `data/trainingData.csv`
- Loads regime assignments from `regime_analysis/regime_assignments.csv`
- Loads model performance data from `test_results/model_regime_test_results_1_425.csv`
- Loads best regime summary from `test_results/best_regime_summary_1_425.csv`
- Loads model metadata from `models/model_log.csv`

### 2. Regime Prediction

**Naive Method:**
- Uses the most common regime from the last 5 periods
- Fast and simple, good baseline

**HMM Method:**
- Trains a Hidden Markov Model on historical regime sequences
- Uses recent regime history to predict the next regime
- More sophisticated but requires more computation

### 3. Model Selection

For each predicted regime and date:
- Identifies all models that weren't trained on the prediction date (prevents data leakage)
- Finds models that were tested on the predicted regime
- Selects the best upside model (highest average upside accuracy)
- Selects the best downside model (highest average downside accuracy)
- Uses model rankings from the test results

### 4. Prediction Generation

- Loads the selected LSTM models and their scalers
- Creates sequences from the minute-by-minute data
- Generates predictions for each valid sequence
- Combines upside and downside predictions in the output

## Output Files

Files are saved to the `forecasts/` directory with the naming convention:
```
regime_forecast_{method}_{period}_{start_date}_{end_date}.csv
```

Examples:
- `regime_forecast_hmm_daily_20200102_20200110.csv`
- `regime_forecast_naive_weekly_20200106_20200220.csv`

## Performance Considerations

- **HMM training**: Takes a few seconds, done once per run
- **Model loading**: Models are loaded fresh for each date (memory efficient)
- **GPU usage**: Automatically uses GPU if available for faster predictions
- **Memory usage**: Processes one day at a time to manage memory

## Error Handling

The script includes comprehensive error handling:
- Falls back to naive method if HMM training fails
- Skips dates with no trading data
- Handles missing models gracefully
- Provides informative error messages

## Dependencies

- pandas, numpy: Data manipulation
- tensorflow/keras: LSTM model loading and prediction
- sklearn: Data preprocessing (scalers)
- hmmlearn: Hidden Markov Model implementation (for HMM method)

## Example Output

```csv
trading_day,ms_of_day,actual,upside_predict,upside_model_id,upside_regime,upside_rank_of_model,downside_predict,downside_model_id,downside_regime,downside_rank_of_model
20200102,38100000,-0.08,0.236,42,3,8,-0.220,67,3,7
20200102,38160000,0.02,0.271,42,3,8,-0.178,67,3,7
20200102,38220000,-0.06,0.142,42,3,8,-0.221,67,3,7
```

This output shows that:
- On 2020-01-02 at 38100000ms, the actual movement was -0.08
- The upside model (ID 42) predicted +0.236, ranking 8th in regime 3
- The downside model (ID 67) predicted -0.220, ranking 7th in regime 3
- Both models were selected based on predicted regime 3

## Integration with Trading Systems

The output can be easily integrated into trading systems by:
1. Loading the CSV files
2. Using appropriate predictions based on market direction expectations
3. Applying the model rankings for confidence weighting
4. Combining with other signals for final trading decisions
