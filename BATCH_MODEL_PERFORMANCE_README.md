# Batch Model Performance Script Documentation

## Overview

The `batch_model_performance.py` script generates detailed performance data for LSTM models by running each model on the entire `tradingData.csv` dataset. The output includes actual vs predicted values for every trading day and time point.

## Features

- **Batch Processing**: Process multiple models in a single run
- **Incremental Updates**: Automatically skips existing data, only adds new records
- **Robust Error Handling**: Continues processing even if individual models fail
- **Progress Tracking**: Shows detailed progress and timing information
- **Memory Efficient**: Processes one model at a time to manage memory usage

## Output Format

Each model generates a CSV file named `model_xxxxx_performance.csv` in the `test-lstm/model_performance/` directory with the following columns:

```csv
TradingDay,TradingMsOfDay,Actual,Predicted
20200102,38100000,-0.08,-0.3757208585739136
20200102,38160000,0.02,-0.24234908819198608
...
```

- **TradingDay**: Date in YYYYMMDD format
- **TradingMsOfDay**: Time of day in milliseconds from midnight
- **Actual**: Actual target value from the dataset
- **Predicted**: Model's predicted value

## Usage

### Basic Usage
```bash
cd /home/stephen/projects/Testing/TestPy/test-lstm/src
source /home/stephen/projects/Testing/TestPy/test-lstm/venv-test-lstm/bin/activate
python batch_model_performance.py <start_model_id> <end_model_id>
```

### Examples
```bash
# Process a single model
python batch_model_performance.py 377 377

# Process a range of models
python batch_model_performance.py 1 10

# Process models 375-380
python batch_model_performance.py 375 380
```

## Required Files

The script expects the following files to exist:

- **Model files**: `test-lstm/models/lstm_stock_model_xxxxx.keras`
- **Scaler files**: `test-lstm/models/scaler_params_xxxxx.json`
- **Training data**: `test-lstm/data/trainingData.csv`
- **Model log**: `test-lstm/models/model_log.csv` (for label_number lookup)

## Incremental Processing

The script intelligently handles incremental updates:

1. **First run**: Creates new CSV file with all available data
2. **Subsequent runs**: Only adds missing records (new trading days/times)
3. **Existing data**: Automatically skipped to avoid duplication

This allows you to:
- Add new training data and re-run to get predictions for new dates
- Resume interrupted batch processing
- Update specific models without reprocessing everything

## Status Checking

Use the companion script to check processing status:

```bash
# Check all processed models
python check_model_performance_status.py

# Check specific range
python check_model_performance_status.py 1 100
```

## Performance Information

**Processing Speed**: Approximately 7-10 seconds per model (depends on data size and GPU)

**File Sizes**: Each performance file is approximately 5MB for ~120,000 records

**Memory Usage**: Processes one model at a time to manage memory efficiently

## Example Output

```
Loading trading data...
Loaded 162,127 rows of trading data

Processing models 00375 to 00377
Output directory: /home/stephen/projects/Testing/TestPy/test-lstm/model_performance
Started at: 2025-08-13 16:24:58
============================================================
Processing model 00375...
  Found 0 existing records
  Using 44 features and target Label_10
  Created 120722 sequences
  Running predictions...
  Skipped 0 existing records
  Adding 120722 new records
  Successfully saved to model_00375_performance.csv

Processing model 00376...
  Found 0 existing records
  Using 44 features and target Label_10
  Created 120722 sequences
  Running predictions...
  Skipped 0 existing records
  Adding 120722 new records
  Successfully saved to model_00376_performance.csv

Processing model 00377...
  Found 120722 existing records
  Using 44 features and target Label_10
  Created 120722 sequences
  Running predictions...
  Skipped 120722 existing records
  Adding 0 new records
  No new data to add for model 00377

============================================================
Batch processing completed at: 2025-08-13 16:25:16
Successfully processed: 3 models
Failed to process: 0 models
Total models attempted: 3
```

## Error Handling

The script handles common errors gracefully:

- **Missing model files**: Logs error and continues with next model
- **Missing scaler files**: Logs error and continues with next model
- **Corrupted data**: Logs error and continues with next model
- **Invalid model IDs**: Validates input parameters before processing

## Tips

1. **Virtual Environment**: Always activate the virtual environment first
2. **GPU Memory**: If you encounter GPU memory issues, process smaller batches
3. **Disk Space**: Each model generates ~5MB files, plan accordingly
4. **Backup**: Consider backing up the model_performance directory for large runs
5. **Monitoring**: Use the status checker to monitor progress of long-running batches
