import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
def get_thresholded_direction_accuracies(actual, predicted):
    actual_up = (actual > 0)
    actual_down = (actual <= 0)
    results = {}
    for t in np.arange(0, 0.81, 0.1):
        pred_up_thr = (predicted > t)
        pred_down_thr = (predicted < -t)
        n_pred_up_thr = np.sum(pred_up_thr)
        n_pred_down_thr = np.sum(pred_down_thr)
        tu_thr = np.sum(pred_up_thr & actual_up)
        td_thr = np.sum(pred_down_thr & actual_down)
        up_acc_thr = tu_thr / n_pred_up_thr if n_pred_up_thr > 0 else 0.0
        down_acc_thr = td_thr / n_pred_down_thr if n_pred_down_thr > 0 else 0.0
        results[f'test_up_acc_thr_{t:.1f}'] = up_acc_thr
        results[f'test_down_acc_thr_{t:.1f}'] = down_acc_thr
    return results

def test_model(model_id, test_from, test_to, accuracy_only=False):
    """
    Test a model on a specific date range and return results.
    
    Args:
        model_id: Model ID string
        test_from: Start date (yyyymmdd string)
        test_to: End date (yyyymmdd string)
        accuracy_only: If True, return only accuracy metrics
    
    Returns:
        If accuracy_only=True: list of [model_id, test_from, test_to, ...accuracy_values]
        If accuracy_only=False: dict with detailed results
    """
    try:
        # Paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_file = os.path.join(project_root, 'data', 'trainingData.csv')
        model_dir = os.path.join(project_root, 'models')
        model_path = os.path.join(model_dir, f'lstm_stock_model_{model_id}.keras')
        scaler_path = os.path.join(model_dir, f'scaler_params_{model_id}.json')

        # Load model and scaler params
        model = keras.models.load_model(model_path)
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)

        # Get label number from model log
        import csv
        model_log_path = os.path.join(project_root, 'models', 'model_log.csv')
        label_number = 10  # Default fallback
        
        if os.path.exists(model_log_path):
            with open(model_log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('model_id') == model_id:
                        label_number = int(row.get('label_number', 10))
                        break

        # Load data
        df = pd.read_csv(data_file)
        test_df = df[(df['TradingDay'] >= int(test_from)) & (df['TradingDay'] <= int(test_to))].reset_index(drop=True)

        # Feature columns (same as training)
        feature_cols = test_df.columns[5:49]
        target_col = f'Label_{label_number}'
        num_features = len(feature_cols)
        seq_length = 30  # Must match training

        # Sequence creation (same as in training script)
        def create_sequences(df, seq_length):
            df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
            features = df[feature_cols].values
            targets = df[target_col].values
            days = df['TradingDay'].values
            ms = df['TradingMsOfDay'].values
            X, y = [], []
            i = 0
            while i <= len(df) - seq_length:
                is_consecutive = True
                current_day = days[i]
                current_ms = ms[i]
                for j in range(1, seq_length):
                    if days[i + j] != current_day or ms[i + j] != current_ms + j * 60000:
                        is_consecutive = False
                        break
                if is_consecutive:
                    seq = features[i:i + seq_length]
                    label = targets[i + seq_length - 1]
                    X.append(seq)
                    y.append(label)
                    i += 1
                else:
                    i += 1
            return np.array(X), np.array(y)

        X_test, y_test = create_sequences(test_df, seq_length)

        # Correctly reconstruct MinMaxScaler from saved params
        scaler = MinMaxScaler()
        scaler.data_min_ = np.array(scaler_params['Min'])
        scaler.data_max_ = np.array(scaler_params['Max'])
        scaler.data_range_ = scaler.data_max_ - scaler.data_min_
        scaler.feature_range = (0, 1)
        scaler.scale_ = (scaler.feature_range[1] - scaler.feature_range[0]) / scaler.data_range_
        scaler.min_ = scaler.feature_range[0] - scaler.data_min_ * scaler.scale_

        X_test_reshaped = X_test.reshape(-1, num_features)
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

        # Predict
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()

        if accuracy_only:
            accs = get_thresholded_direction_accuracies(y_test, y_pred)
            # Return up accuracies first, then down accuracies to match header order
            up_values = [f'{accs[f"test_up_acc_thr_{t:.1f}"]:.6f}' for t in np.arange(0, 0.81, 0.1)]
            down_values = [f'{accs[f"test_down_acc_thr_{t:.1f}"]:.6f}' for t in np.arange(0, 0.81, 0.1)]
            row = [model_id, test_from, test_to] + up_values + down_values
            return row
        else:
            # Return detailed results for non-accuracy-only mode
            # ... (implement if needed)
            return {"message": "Detailed results not implemented in function mode"}
    
    except Exception as e:
        # Return error row for accuracy_only mode
        if accuracy_only:
            return [model_id, test_from, test_to] + [''] * 18  # 18 accuracy columns
        else:
            return {"error": str(e)}

if __name__ == '__main__':
    import argparse
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('model_id', type=str, help='Model ID')
        parser.add_argument('testFrom', type=str, help='Test from date (yyyymmdd)')
        parser.add_argument('testTo', type=str, help='Test to date (yyyymmdd)')
        parser.add_argument('--accuracy-only', action='store_true', help='Print only thresholded up/down accuracy as CSV row')
        args = parser.parse_args()

        model_id = args.model_id
        testFrom = args.testFrom
        testTo = args.testTo

        # Paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_file = os.path.join(project_root, 'data', 'trainingData.csv')
        model_dir = os.path.join(project_root, 'models')
        model_path = os.path.join(model_dir, f'lstm_stock_model_{model_id}.keras')
        scaler_path = os.path.join(model_dir, f'scaler_params_{model_id}.json')

        # Load model and scaler params
        model = keras.models.load_model(model_path)
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)

        # Get label number from model log
        import csv
        model_log_path = os.path.join(project_root, 'models', 'model_log.csv')
        label_number = 10  # Default fallback
        
        if os.path.exists(model_log_path):
            with open(model_log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('model_id') == model_id:
                        label_number = int(row.get('label_number', 10))
                        break

        # Load data
        df = pd.read_csv(data_file)
        test_df = df[(df['TradingDay'] >= int(testFrom)) & (df['TradingDay'] <= int(testTo))].reset_index(drop=True)

        # Feature columns (same as training)
        feature_cols = test_df.columns[5:49]
        target_col = f'Label_{label_number}'
        num_features = len(feature_cols)
        seq_length = 30  # Must match training

        # Sequence creation (same as in training script)
        def create_sequences(df, seq_length):
            df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
            features = df[feature_cols].values
            targets = df[target_col].values
            days = df['TradingDay'].values
            ms = df['TradingMsOfDay'].values
            X, y = [], []
            i = 0
            while i <= len(df) - seq_length:
                is_consecutive = True
                current_day = days[i]
                current_ms = ms[i]
                for j in range(1, seq_length):
                    if days[i + j] != current_day or ms[i + j] != current_ms + j * 60000:
                        is_consecutive = False
                        break
                if is_consecutive:
                    seq = features[i:i + seq_length]
                    label = targets[i + seq_length - 1]
                    X.append(seq)
                    y.append(label)
                    i += 1
                else:
                    i += 1
            return np.array(X), np.array(y)

        X_test, y_test = create_sequences(test_df, seq_length)

        # Correctly reconstruct MinMaxScaler from saved params
        scaler = MinMaxScaler()
        scaler.data_min_ = np.array(scaler_params['Min'])
        scaler.data_max_ = np.array(scaler_params['Max'])
        scaler.data_range_ = scaler.data_max_ - scaler.data_min_
        scaler.feature_range = (0, 1)
        scaler.scale_ = (scaler.feature_range[1] - scaler.feature_range[0]) / scaler.data_range_
        scaler.min_ = scaler.feature_range[0] - scaler.data_min_ * scaler.scale_

        X_test_reshaped = X_test.reshape(-1, num_features)
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

        # Predict
        y_pred = model.predict(X_test_scaled).flatten()

        # Recover TradingDay and TradingMsOfDay for each sequence (as in training script)
        def get_seq_indices(df, seq_length):
            days = df['TradingDay'].values
            ms = df['TradingMsOfDay'].values
            seq_idx = []
            for i in range(len(df) - seq_length + 1):
                is_consecutive = True
                current_day = days[i]
                current_ms = ms[i]
                for j in range(1, seq_length):
                    if days[i + j] != current_day or ms[i + j] != current_ms + j * 60000:
                        is_consecutive = False
                        break
                if is_consecutive:
                    seq_idx.append(i + seq_length - 1)
            return np.array(seq_idx)

        test_seq_idx = get_seq_indices(test_df, seq_length)
        if len(test_seq_idx) > len(y_test):
            test_seq_idx = test_seq_idx[:len(y_test)]

        if args.accuracy_only:
            accs = get_thresholded_direction_accuracies(y_test, y_pred)
            row = [model_id, testFrom, testTo] + [f'{accs[k]:.6f}' for k in sorted(accs.keys())]
            print(','.join(row))
            sys.exit(0)

        print(f"Test samples: {len(y_test)}")

        # Build DataFrame to match the test predictions CSV from training script
        results_df = pd.DataFrame({
            'TradingDay': test_df['TradingDay'].values[test_seq_idx],
            'TradingMsOfDay': test_df['TradingMsOfDay'].values[test_seq_idx],
            'Actual': y_test,
            'Predicted': y_pred
        })

        print("TradingDay,TradingMsOfDay,Actual,Predicted")
        for row in results_df.itertuples(index=False):
            print(f"{row.TradingDay},{row.TradingMsOfDay},{row.Actual:.6f},{row.Predicted:.6f}")

        # Optionally, compare with test predictions CSV from training script (if exists)
        test_pred_csv = os.path.join(project_root, f"test_predictions_regression_{model_id}_{testFrom}_{testTo}.csv")
        if os.path.exists(test_pred_csv):
            ref_df = pd.read_csv(test_pred_csv)
            # Compare shape and values
            if len(ref_df) != len(results_df):
                print(f"WARNING: Row count mismatch with {test_pred_csv} (expected {len(ref_df)}, got {len(results_df)})")
            else:
                mismatches = (ref_df[['TradingDay','TradingMsOfDay','Actual']].values != results_df[['TradingDay','TradingMsOfDay','Actual']].values).any()
                if mismatches:
                    print(f"WARNING: Data mismatch with {test_pred_csv}")
                else:
                    print(f"Test results match {test_pred_csv} (except for possible floating point differences in Predicted column)")

        # Optionally, print overall MAE/MSE
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Test MAE: {mae:.6f}")
        print(f"Test MSE: {mse:.6f}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

# (Removed stray function definitions and statements at the end of the file)