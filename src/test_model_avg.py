import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
import tensorflow as tf

# Define asymmetric MSE loss function (needed for loading models that use it)
def asymmetric_mse(y_true, y_pred):
    error = y_true - y_pred
    return tf.reduce_mean(tf.where(error > 0, error**2, 1.5 * error**2))

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

def test_model_avg(model_id, test_from, test_to, accuracy_only=False):
    """
    Test an averaging model on a specific date range and return results.
    
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
        model_path = os.path.join(model_dir, f'lstm_stock_model_avg_{model_id}.keras')
        scaler_path = os.path.join(model_dir, f'scaler_params_avg_{model_id}.json')

        # Load model and scaler params
        model = keras.models.load_model(model_path)
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)

        # Get label number and label range from model log
        import csv
        model_log_path = os.path.join(project_root, 'models', 'model_log_avg.csv')
        label_number = 5  # Default fallback
        label_range = [3, 4, 5, 6, 7]  # Default fallback
        
        if os.path.exists(model_log_path):
            with open(model_log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('model_id') == model_id:
                        label_number = int(row.get('label_number', 5))
                        # Parse label_range from the format "start-end"
                        label_range_str = row.get('label_range', '3-7')
                        start_label, end_label = map(int, label_range_str.split('-'))
                        label_range = list(range(start_label, end_label + 1))
                        break

        # Load data
        df = pd.read_csv(data_file)
        test_df = df[(df['TradingDay'] >= int(test_from)) & (df['TradingDay'] <= int(test_to))].reset_index(drop=True)

        # Feature columns (same as training)
        feature_cols = test_df.columns[5:49]
        target_cols = [f'Label_{i}' for i in label_range]
        num_features = len(feature_cols)
        seq_length = 15  # Must match training

        # Verify that all required label columns exist
        missing_cols = [col for col in target_cols if col not in test_df.columns]
        if missing_cols:
            raise ValueError(f"Missing label columns in data: {missing_cols}")

        # Sequence creation (same as in averaging training script)
        def create_sequences(df, seq_length):
            df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
            features = df[feature_cols].values
            target_values = df[target_cols].values
            days = df['TradingDay'].values
            ms = df['TradingMsOfDay'].values
            
            X = []
            y = []
            y_individual = []
            
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
                    # Get individual labels at the prediction timestep (last timestep)
                    individual_labels = target_values[i + seq_length - 1]
                    # Use the center label (labelNumber) as the actual value
                    center_label_value = individual_labels[label_range.index(label_number)]
                    # Calculate average of target labels for comparison
                    label_avg = np.mean(individual_labels)
                    X.append(seq)
                    y.append(center_label_value)  # Use center label, not average
                    y_individual.append(individual_labels)
                    i += 1
                else:
                    i += 1
            
            return np.array(X), np.array(y), np.array(y_individual)

        X_test, y_test, y_test_individual = create_sequences(test_df, seq_length)

        # Correctly reconstruct scaler from saved params
        if 'Min' in scaler_params and 'Max' in scaler_params:
            # MinMaxScaler
            scaler = MinMaxScaler()
            scaler.data_min_ = np.array(scaler_params['Min'])
            scaler.data_max_ = np.array(scaler_params['Max'])
            scaler.data_range_ = scaler.data_max_ - scaler.data_min_
            scaler.feature_range = (0, 1)
            scaler.scale_ = (scaler.feature_range[1] - scaler.feature_range[0]) / scaler.data_range_
            scaler.min_ = scaler.feature_range[0] - scaler.data_min_ * scaler.scale_
        elif 'Mean' in scaler_params and 'Variance' in scaler_params:
            # StandardScaler
            scaler = StandardScaler()
            scaler.mean_ = np.array(scaler_params['Mean'])
            scaler.var_ = np.array(scaler_params['Variance'])
            scaler.scale_ = np.sqrt(scaler.var_)
        else:
            raise ValueError(f"Unknown scaler parameters: {list(scaler_params.keys())}")

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
        model_path = os.path.join(model_dir, f'lstm_stock_model_avg_{model_id}.keras')
        scaler_path = os.path.join(model_dir, f'scaler_params_avg_{model_id}.json')

        # Load model and scaler params
        model = keras.models.load_model(model_path)
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)

        # Get label number and label range from model log
        import csv
        model_log_path = os.path.join(project_root, 'models', 'model_log_avg.csv')
        label_number = 5  # Default fallback
        label_range = [3, 4, 5, 6, 7]  # Default fallback
        
        if os.path.exists(model_log_path):
            with open(model_log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('model_id') == model_id:
                        label_number = int(row.get('label_number', 5))
                        # Parse label_range from the format "start-end"
                        label_range_str = row.get('label_range', '3-7')
                        start_label, end_label = map(int, label_range_str.split('-'))
                        label_range = list(range(start_label, end_label + 1))
                        break

        print(f"Testing model {model_id} for labels {label_range} (centered at label {label_number})")

        # Load data
        df = pd.read_csv(data_file)
        test_df = df[(df['TradingDay'] >= int(testFrom)) & (df['TradingDay'] <= int(testTo))].reset_index(drop=True)

        # Feature columns (same as training)
        feature_cols = test_df.columns[5:49]
        target_cols = [f'Label_{i}' for i in label_range]
        num_features = len(feature_cols)
        seq_length = 15  # Must match training

        # Verify that all required label columns exist
        missing_cols = [col for col in target_cols if col not in test_df.columns]
        if missing_cols:
            raise ValueError(f"Missing label columns in data: {missing_cols}")

        # Sequence creation (same as in averaging training script)
        def create_sequences(df, seq_length):
            df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
            features = df[feature_cols].values
            target_values = df[target_cols].values
            days = df['TradingDay'].values
            ms = df['TradingMsOfDay'].values
            
            X = []
            y = []
            y_individual = []
            
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
                    # Get individual labels at the prediction timestep (last timestep)
                    individual_labels = target_values[i + seq_length - 1]
                    # Use the center label (labelNumber) as the actual value
                    center_label_value = individual_labels[label_range.index(label_number)]
                    # Calculate average of target labels for comparison
                    label_avg = np.mean(individual_labels)
                    X.append(seq)
                    y.append(center_label_value)  # Use center label, not average
                    y_individual.append(individual_labels)
                    i += 1
                else:
                    i += 1
            
            return np.array(X), np.array(y), np.array(y_individual)

        X_test, y_test, y_test_individual = create_sequences(test_df, seq_length)

        # Correctly reconstruct scaler from saved params
        if 'Min' in scaler_params and 'Max' in scaler_params:
            # MinMaxScaler
            scaler = MinMaxScaler()
            scaler.data_min_ = np.array(scaler_params['Min'])
            scaler.data_max_ = np.array(scaler_params['Max'])
            scaler.data_range_ = scaler.data_max_ - scaler.data_min_
            scaler.feature_range = (0, 1)
            scaler.scale_ = (scaler.feature_range[1] - scaler.feature_range[0]) / scaler.data_range_
            scaler.min_ = scaler.feature_range[0] - scaler.data_min_ * scaler.scale_
        elif 'Mean' in scaler_params and 'Variance' in scaler_params:
            # StandardScaler
            scaler = StandardScaler()
            scaler.mean_ = np.array(scaler_params['Mean'])
            scaler.var_ = np.array(scaler_params['Variance'])
            scaler.scale_ = np.sqrt(scaler.var_)
        else:
            raise ValueError(f"Unknown scaler parameters: {list(scaler_params.keys())}")

        X_test_reshaped = X_test.reshape(-1, num_features)
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

        # Predict
        y_pred = model.predict(X_test_scaled).flatten()

        # Helper to get sequence indices and actual average values
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

        def get_actual_avg_values(df, seq_idx):
            """Get the actual average values for the given sequence indices"""
            actual_avg = []
            for idx in seq_idx:
                if idx < len(df):
                    row_values = df.iloc[idx][target_cols].values
                    avg_value = np.mean(row_values)
                    actual_avg.append(avg_value)
                else:
                    actual_avg.append(np.nan)
            return np.array(actual_avg)

        test_seq_idx = get_seq_indices(test_df, seq_length)
        if len(test_seq_idx) > len(y_test):
            test_seq_idx = test_seq_idx[:len(y_test)]

        # Get actual average values for test predictions
        test_actual_avg = get_actual_avg_values(test_df, test_seq_idx)

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
            'ActualAvg': test_actual_avg,
            'PredictedAvg': y_pred
        })

        print("TradingDay,TradingMsOfDay,Actual,ActualAvg,PredictedAvg")
        for row in results_df.itertuples(index=False):
            print(f"{row.TradingDay},{row.TradingMsOfDay},{row.Actual:.6f},{row.ActualAvg:.6f},{row.PredictedAvg:.6f}")

        # Direction accuracy calculation using ActualAvg and PredictedAvg
        def print_direction_accuracy(actual, predicted, label):
            actual_up = (actual > 0)
            actual_down = (actual <= 0)
            pred_up = (predicted > 0)
            pred_down = (predicted <= 0)

            n_pred_up = np.sum(pred_up)
            n_pred_down = np.sum(pred_down)
            n_actual_up = np.sum(actual_up)
            n_actual_down = np.sum(actual_down)

            # True ups: predicted up and actually up
            tu = np.sum(pred_up & actual_up)
            # True downs: predicted down and actually down
            td = np.sum(pred_down & actual_down)

            up_acc = tu / n_pred_up if n_pred_up > 0 else 0.0
            down_acc = td / n_pred_down if n_pred_down > 0 else 0.0

            print(f"\n{label} set direction prediction:")
            print(f"  Actual up count: {n_actual_up}, down count: {n_actual_down}")
            print(f"  Predicted up count: {n_pred_up}, down count: {n_pred_down}")
            print(f"  Up prediction accuracy: {up_acc:.2%} ({tu}/{n_pred_up})")
            print(f"  Down prediction accuracy: {down_acc:.2%} ({td}/{n_pred_down})")

            # Thresholded accuracy
            print(f"\n{label} set thresholded direction prediction:")
            for t in np.arange(0, 0.81, 0.1):
                pred_up_thr = (predicted > t)
                pred_down_thr = (predicted < -t)
                n_pred_up_thr = np.sum(pred_up_thr)
                n_pred_down_thr = np.sum(pred_down_thr)
                tu_thr = np.sum(pred_up_thr & actual_up)
                td_thr = np.sum(pred_down_thr & actual_down)
                up_acc_thr = tu_thr / n_pred_up_thr if n_pred_up_thr > 0 else 0.0
                down_acc_thr = td_thr / n_pred_down_thr if n_pred_down_thr > 0 else 0.0
                print(f"  Threshold: {t:.1f} | Up acc: {up_acc_thr:.2%} ({tu_thr}/{n_pred_up_thr}), Down acc: {down_acc_thr:.2%} ({td_thr}/{n_pred_down_thr})")

        # Print direction accuracy using ActualAvg and PredictedAvg
        print_direction_accuracy(results_df['ActualAvg'].values, results_df['PredictedAvg'].values, 'Test')

        # Optionally, compare with test predictions CSV from training script (if exists)
        test_pred_csv = os.path.join(project_root, f"test_predictions_regression_avg_{model_id}_{testFrom}_{testTo}.csv")
        if os.path.exists(test_pred_csv):
            ref_df = pd.read_csv(test_pred_csv)
            # Compare shape and values
            if len(ref_df) != len(results_df):
                print(f"WARNING: Row count mismatch with {test_pred_csv} (expected {len(ref_df)}, got {len(results_df)})")
            else:
                mismatches = (ref_df[['TradingDay','TradingMsOfDay','Actual','ActualAvg']].values != results_df[['TradingDay','TradingMsOfDay','Actual','ActualAvg']].values).any()
                if mismatches:
                    print(f"WARNING: Data mismatch with {test_pred_csv}")
                else:
                    print(f"Test results match {test_pred_csv} (except for possible floating point differences in PredictedAvg column)")

        # Optionally, print overall MAE/MSE using ActualAvg
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(results_df['ActualAvg'].values, results_df['PredictedAvg'].values)
        mse = mean_squared_error(results_df['ActualAvg'].values, results_df['PredictedAvg'].values)
        print(f"Test MAE (ActualAvg vs PredictedAvg): {mae:.6f}")
        print(f"Test MSE (ActualAvg vs PredictedAvg): {mse:.6f}")
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
