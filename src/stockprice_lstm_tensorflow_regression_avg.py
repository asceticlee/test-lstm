import pandas as pd
import numpy as np
import os
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras  # This loads tf.keras as 'keras'
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.layers import GRU
import tf2onnx
import onnx

import sys

from keras.callbacks import ReduceLROnPlateau

print(tf.__version__)  # Must show ≥2.5.0

script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets src/ dir
project_root = os.path.dirname(script_dir)  # Gets test-lstm/ root
data_dir = os.path.join(project_root, 'data/')  # Full path to data/
data_file = os.path.join(data_dir, 'trainingData.csv')
export_dir = os.path.join(project_root, 'models')
os.makedirs(export_dir, exist_ok=True)

# Read date parameters and label number from command line
if len(sys.argv) != 8:
    print("Usage: python stockprice_lstm_tensorflow_regression_avg.py trainFrom trainTo validationFrom validationTo testFrom testTo labelNumber")
    print("Example: python stockprice_lstm_tensorflow_regression_avg.py 20250501 20250531 20250601 20250607 20250701 20250707 9")
    print("Note: labelNumber must be odd (5, 7, 9, 11, 13, etc.)")
    sys.exit(1)

trainFrom, trainTo, validationFrom, validationTo, testFrom, testTo, labelNumber = sys.argv[1:8]
labelNumber = int(labelNumber)

# Validate that labelNumber is odd
if labelNumber % 2 == 0:
    print(f"Error: labelNumber must be odd, but got {labelNumber}")
    print("Valid examples: 5, 7, 9, 11, 13, etc.")
    sys.exit(1)

# Calculate the range of labels for averaging
half_window = labelNumber // 2
start_label = labelNumber - half_window
end_label = labelNumber + half_window
label_range = list(range(start_label, end_label + 1))

print(f"Predicting average of labels {label_range} (centered at label {labelNumber})")

# Validate that all required labels exist (assuming labels go up to at least 30)
if start_label < 1 or end_label > 30:
    print(f"Error: Label range {start_label} to {end_label} is outside valid range (1-30)")
    sys.exit(1)

# Parameters
seq_length = 30
learning_rate = 0.0009
epochs = 150
batch_size = 32

# Load data

# Load full data
full_df = pd.read_csv(data_file)

# Verify that all required label columns exist
required_label_cols = [f'Label_{i}' for i in label_range]
missing_cols = [col for col in required_label_cols if col not in full_df.columns]
if missing_cols:
    print(f"Error: Missing label columns in data: {missing_cols}")
    sys.exit(1)

# Filter for train, validation, and test sets by TradingDay (assume column 0 is TradingDay)
train_df = full_df[(full_df.iloc[:,0] >= int(trainFrom)) & (full_df.iloc[:,0] <= int(trainTo))].reset_index(drop=True)
validation_df = full_df[(full_df.iloc[:,0] >= int(validationFrom)) & (full_df.iloc[:,0] <= int(validationTo))].reset_index(drop=True)
test_df = full_df[(full_df.iloc[:,0] >= int(testFrom)) & (full_df.iloc[:,0] <= int(testTo))].reset_index(drop=True)

# Define feature and target columns
# Features: columns 6 to 49 (0-based index 5 to 48)
feature_cols = train_df.columns[5:49]
target_cols = required_label_cols
num_features = len(feature_cols)

# Function to create sequences, respecting continuity
def create_sequences(df, seq_length):
    df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
    features = df[feature_cols].values
    # Get all target label values
    target_values = df[target_cols].values
    days = df['TradingDay'].values
    ms = df['TradingMsOfDay'].values
    
    X = []
    y = []
    y_individual = []  # Store individual label values for analysis
    
    i = 0
    while i <= len(df) - seq_length:
        # Check if the sequence is consecutive (same day, 60000 ms increments)
        is_consecutive = True
        current_day = days[i]
        current_ms = ms[i]
        
        for j in range(1, seq_length):
            if days[i + j] != current_day or ms[i + j] != current_ms + j * 60000:
                is_consecutive = False
                break
        
        if is_consecutive:
            seq = features[i:i + seq_length]
            # Calculate average of target labels at the prediction timestep (last timestep)
            individual_labels = target_values[i + seq_length - 1]  # Individual label values at prediction timestep
            label_avg = np.mean(individual_labels)  # Average of the specific labels we want to predict
            X.append(seq)
            y.append(label_avg)
            y_individual.append(individual_labels)
            i += 1  # Sliding window, step by 1
        else:
            # Skip to the next potential start (after the gap)
            i += 1
    
    return np.array(X), np.array(y), np.array(y_individual)


# Create sequences for train, validation, and test
X_train, y_train, y_train_individual = create_sequences(train_df, seq_length)
X_validation, y_validation, y_validation_individual = create_sequences(validation_df, seq_length)
X_test, y_test, y_test_individual = create_sequences(test_df, seq_length)

# Store original indices before balancing
train_original_indices = np.arange(len(y_train))

# Balance positive and negative samples in training set
pos_idx = np.where(y_train > 0)[0]
neg_idx = np.where(y_train <= 0)[0]
min_count = min(len(pos_idx), len(neg_idx))
np.random.seed(42)
pos_sample = np.random.choice(pos_idx, min_count, replace=False)
neg_sample = np.random.choice(neg_idx, min_count, replace=False)
balanced_idx = np.concatenate([pos_sample, neg_sample])
np.random.shuffle(balanced_idx)

# Apply balancing to all training data
X_train = X_train[balanced_idx]
y_train = y_train[balanced_idx]
y_train_individual = y_train_individual[balanced_idx]
train_balanced_indices = train_original_indices[balanced_idx]  # Store which original indices were selected

print(f"Balanced train class counts: positive={np.sum(y_train > 0)}, negative={np.sum(y_train <= 0)}")

# Print the size of x_train, x_validation, x_test
print(f'Train sequences shape: {X_train.shape}, Train labels shape: {y_train.shape}')
print(f'Validation sequences shape: {X_validation.shape}, Validation labels shape: {y_validation.shape}')
print(f'Test sequences shape: {X_test.shape}, Test labels shape: {y_test.shape}')


# Scale features (fit on train, apply to all sets)
scaler = MinMaxScaler()
# scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, num_features)
X_validation_reshaped = X_validation.reshape(-1, num_features)
X_test_reshaped = X_test.reshape(-1, num_features)

scaler.fit(X_train_reshaped)
X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_validation_scaled = scaler.transform(X_validation_reshaped).reshape(X_validation.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

# Add after scaler.fit(X_train_reshaped)


# --- Model ID and Logging ---
import json
import csv
model_log_path = os.path.join(export_dir, 'model_log_avg.csv')

# Determine next model_id
if os.path.exists(model_log_path):
    with open(model_log_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        if len(rows) > 1:
            last_id = int(rows[-1][0])
            model_id = last_id + 1
        else:
            model_id = 1
else:
    model_id = 1
model_id_str = f"{model_id:05d}"

# Save scaler params with model_id
# scaler_params = {
#     "Mean": scaler.mean_.tolist(),
#     "Variance": scaler.var_.tolist()
# }
scaler_params = {
    "Min": scaler.data_min_.tolist(),
    "Max": scaler.data_max_.tolist()
}
scaler_params_path = os.path.join(export_dir, f'scaler_params_avg_{model_id_str}.json')
with open(scaler_params_path, 'w') as f:
    json.dump(scaler_params, f)

# Compile with asymmetric MSE loss
def asymmetric_mse(y_true, y_pred):
    error = y_true - y_pred
    return tf.reduce_mean(tf.where(error > 0, error**2, 1.5 * error**2))  # 1.5x penalty for downs (tune)

model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, num_features)),
    keras.layers.LSTM(50),
    keras.layers.Dense(1) # Output layer for regression
])

# Compile
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
# model.compile(optimizer=Adam(learning_rate=learning_rate), loss=asymmetric_mse, metrics=['mae', asymmetric_mse])

# Train
early_stop = EarlyStopping(monitor='val_mae', patience=100, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=120, min_lr=1e-6, verbose=1)
history = model.fit(
    X_train_scaled, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_validation_scaled, y_validation),
    callbacks=[
        # early_stop, 
        reduce_lr
    ],
    verbose=1
)
# Get final train/val metrics from history
train_loss = history.history['loss'][-1] if 'loss' in history.history else None
train_mae = history.history['mae'][-1] if 'mae' in history.history else None
val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
val_mae = history.history['val_mae'][-1] if 'val_mae' in history.history else None

# Evaluate
validation_loss, validation_mae = model.evaluate(X_validation_scaled, y_validation, verbose=1)
print(f'Validation Loss (MSE): {validation_loss}')
print(f'Validation MAE: {validation_mae}')


# Save model with model_id in filename
model.save(os.path.join(export_dir, f'lstm_stock_model_avg_{model_id_str}.keras'))

# Helper to recover TradingDay and TradingMsOfDay for each sequence
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

# Helper to get actual average values from dataframe
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

# Train predictions
train_seq_idx = get_seq_indices(train_df, seq_length)

# Use the balanced indices to get the corresponding sequence indices
train_seq_idx_balanced = train_seq_idx[train_balanced_indices]

# Get actual average values for train predictions using the correct indices
train_actual_avg = get_actual_avg_values(train_df, train_seq_idx_balanced)

train_results_df = pd.DataFrame({
    'TradingDay': train_df['TradingDay'].values[train_seq_idx_balanced],
    'TradingMsOfDay': train_df['TradingMsOfDay'].values[train_seq_idx_balanced],
    'Actual': y_train,
    'ActualAvg': train_actual_avg,
    'PredictedAvg': model.predict(X_train_scaled).flatten()
})

# train_output_file = os.path.join(project_root, f'train_predictions_regression_avg_{model_id_str}_{trainFrom}_{trainTo}.csv')
# train_results_df.to_csv(train_output_file, index=False)
# print(f"Saved all {len(y_train)} train predictions and actuals to '{train_output_file}'")

# Validation predictions
validation_seq_idx = get_seq_indices(validation_df, seq_length)

# Get actual average values for validation predictions
validation_actual_avg = get_actual_avg_values(validation_df, validation_seq_idx)

validation_results_df = pd.DataFrame({
    'TradingDay': validation_df['TradingDay'].values[validation_seq_idx],
    'TradingMsOfDay': validation_df['TradingMsOfDay'].values[validation_seq_idx],
    'Actual': y_validation,
    'ActualAvg': validation_actual_avg,
    'PredictedAvg': model.predict(X_validation_scaled).flatten()
})

# validation_output_file = os.path.join(project_root, f'validation_predictions_regression_avg_{model_id_str}_{validationFrom}_{validationTo}.csv')
# validation_results_df.to_csv(validation_output_file, index=False)
# print(f"Saved all {len(y_validation)} validation predictions and actuals to '{validation_output_file}'")


# Test predictions
test_seq_idx = get_seq_indices(test_df, seq_length)

# Get actual average values for test predictions
test_actual_avg = get_actual_avg_values(test_df, test_seq_idx)

test_results_df = pd.DataFrame({
    'TradingDay': test_df['TradingDay'].values[test_seq_idx],
    'TradingMsOfDay': test_df['TradingMsOfDay'].values[test_seq_idx],
    'Actual': y_test,
    'ActualAvg': test_actual_avg,
    'PredictedAvg': model.predict(X_test_scaled).flatten()
})

# test_output_file = os.path.join(project_root, f'test_predictions_regression_avg_{model_id_str}_{testFrom}_{testTo}.csv')
# test_results_df.to_csv(test_output_file, index=False)
# print(f"Saved all {len(y_test)} test predictions and actuals to '{test_output_file}'")


# Direction accuracy calculation and printout (using "up" and "down" instead of "positive" and "negative")

# Direction accuracy calculation and printout (with thresholding)

# Direction accuracy calculation and printout (with thresholding)
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
        results[f'up_acc_thr_{t:.1f}'] = up_acc_thr
        results[f'down_acc_thr_{t:.1f}'] = down_acc_thr
    return results

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

    print(f"{label} set direction prediction:")
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

# Compute and print, and store for logging
# Use the ActualAvg and PredictedAvg for accuracy calculations
train_thr_accs = get_thresholded_direction_accuracies(train_results_df['ActualAvg'].values, train_results_df['PredictedAvg'].values)
val_thr_accs = get_thresholded_direction_accuracies(validation_results_df['ActualAvg'].values, validation_results_df['PredictedAvg'].values)
test_thr_accs = get_thresholded_direction_accuracies(test_results_df['ActualAvg'].values, test_results_df['PredictedAvg'].values)

print_direction_accuracy(train_results_df['ActualAvg'].values, train_results_df['PredictedAvg'].values, 'Train')
print_direction_accuracy(validation_results_df['ActualAvg'].values, validation_results_df['PredictedAvg'].values, 'Validation')
print_direction_accuracy(test_results_df['ActualAvg'].values, test_results_df['PredictedAvg'].values, 'Test')

# --- Enhanced model_log.csv logging with thresholded direction accuracy ---
base_header = ['model_id', 'train_from', 'train_to', 'val_from', 'val_to', 'test_from', 'test_to', 'seq_length', 'label_number', 'label_range', 'learning_rate', 'epochs', 'batch_size', 'train_loss', 'train_mae', 'val_loss', 'val_mae']
base_row = [model_id_str, trainFrom, trainTo, validationFrom, validationTo, testFrom, testTo, seq_length, labelNumber, f"{start_label}-{end_label}", learning_rate, epochs, batch_size, train_loss, train_mae, val_loss, val_mae]

# Collect all thresholds (ensure consistent order)
thresholds = [round(t, 1) for t in np.arange(0, 0.81, 0.1)]
split_names = ['train', 'val', 'test']
split_accs = [train_thr_accs, val_thr_accs, test_thr_accs]

# Build dynamic header and row
header = base_header.copy()
row = base_row.copy()
for split, accs in zip(split_names, split_accs):
    for t in thresholds:
        header.append(f'{split}_up_acc_thr_{t:.1f}')
        header.append(f'{split}_down_acc_thr_{t:.1f}')
        row.append(accs.get(f'up_acc_thr_{t:.1f}', 0.0))
        row.append(accs.get(f'down_acc_thr_{t:.1f}', 0.0))

# Write or append to CSV
write_header = not os.path.exists(model_log_path)
if not write_header:
    # Check if header matches; if not, rewrite header (optional, for robustness)
    with open(model_log_path, 'r') as f:
        reader = csv.reader(f)
        existing_header = next(reader, None)
        if existing_header != header:
            write_header = True

mode = 'w' if write_header else 'a'
with open(model_log_path, mode, newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
    writer.writerow(row)

print(f"\nModel {model_id_str} completed successfully!")
print(f"Predicted average of labels {label_range} (centered at label {labelNumber})")
print(f"Results saved with ActualAvg and PredictedAvg columns")
