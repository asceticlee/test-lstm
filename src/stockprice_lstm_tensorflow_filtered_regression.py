import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras  # This loads tf.keras as 'keras'
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.layers import GRU
import tf2onnx
import onnx

print(tf.__version__)  # Must show ≥2.5.0
# Define paths and parameters
script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets src/ dir
project_root = os.path.dirname(script_dir)  # Gets test-lstm/ root
data_dir = os.path.join(project_root, 'data/')  # Full path to data/
train_file = os.path.join(data_dir, 'training_data_spy_with_iv_20240101_20250430.csv')
test_file = os.path.join(data_dir, 'testing_data_spy_with_iv_20250501_20250715.csv')
export_dir = os.path.join(project_root, 'models')
os.makedirs(export_dir, exist_ok=True)
# Parameters
seq_length = 20
learning_rate = 0.001
epochs = 1000
batch_size = 32

# Load data
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Define feature and target columns
# Features: columns 6 to 65 (0-based index 5 to 65)
feature_cols = train_df.columns[5:66]
target_col = 'Label_8'
num_features = len(feature_cols)

# Function to create sequences, respecting continuity
def create_sequences(df, seq_length):
    df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
    features = df[feature_cols].values
    targets = df[target_col].values
    days = df['TradingDay'].values
    ms = df['TradingMsOfDay'].values
    
    X = []
    y = []
    
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
            label = targets[i + seq_length - 1]  # Label_8 of the last timestep
            X.append(seq)
            y.append(label)
            i += 1  # Sliding window, step by 1
        else:
            # Skip to the next potential start (after the gap)
            i += 1
    
    return np.array(X), np.array(y)

# Create sequences for train and test

X_train, y_train = create_sequences(train_df, seq_length)
X_test, y_test = create_sequences(test_df, seq_length)

# Balance positive and negative samples in training set
pos_idx = np.where(y_train > 0)[0]
neg_idx = np.where(y_train <= 0)[0]
min_count = min(len(pos_idx), len(neg_idx))
np.random.seed(42)
pos_sample = np.random.choice(pos_idx, min_count, replace=False)
neg_sample = np.random.choice(neg_idx, min_count, replace=False)
balanced_idx = np.concatenate([pos_sample, neg_sample])
np.random.shuffle(balanced_idx)
X_train = X_train[balanced_idx]
y_train = y_train[balanced_idx]
print(f"Balanced train class counts: positive={np.sum(y_train > 0)}, negative={np.sum(y_train <= 0)}")

# # Balancing the samples up down size
# up_idx = np.where(y_train > 0)[0]
# down_idx = np.where(y_train <= 0)[0]

# # Find the smaller class size
# min_count = min(len(up_idx), len(down_idx))

# # Randomly sample from each class
# np.random.seed(42)  # for reproducibility
# up_sample = np.random.choice(up_idx, min_count, replace=False)
# down_sample = np.random.choice(down_idx, min_count, replace=False)

# # Combine and shuffle
# balanced_idx = np.concatenate([up_sample, down_sample])
# np.random.shuffle(balanced_idx)

# # Subset the data
# X_train_bal = X_train[balanced_idx]
# y_train_bal = y_train[balanced_idx]



# Print the size of x_train and y_train
print(f'Train sequences shape: {X_train.shape}, Train labels shape: {y_train.shape}')
print(f'Test sequences shape: {X_test.shape}, Test labels shape: {y_test.shape}')

# Scale features (fit on train, apply to both)
scaler = MinMaxScaler()
X_train_reshaped = X_train.reshape(-1, num_features)
X_test_reshaped = X_test.reshape(-1, num_features)

scaler.fit(X_train_reshaped)
X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

# Print the scaled first test sequence for debugging
# print("\nFirst test sequence (raw):")
# print(X_test[0])
# print("\nFirst test sequence (scaled):")
# print(X_test_scaled[0])

# Add after scaler.fit(X_train_reshaped)
import json
scaler_params = {
    "Min": scaler.data_min_.tolist(),
    "Max": scaler.data_max_.tolist()
}
scaler_params_path = os.path.join(export_dir, 'scaler_params.json')
with open(scaler_params_path, 'w') as f:
    json.dump(scaler_params, f)

# Compile with asymmetric MSE loss
def asymmetric_mse(y_true, y_pred):
    error = y_true - y_pred
    return tf.reduce_mean(tf.where(error > 0, error**2, 1.5 * error**2))  # 1.5x penalty for downs (tune)

# Build LSTM model (assuming regression for price movement)
# model = keras.Sequential([
#     keras.layers.GRU(50, return_sequences=True, input_shape=(seq_length, num_features)),
#     keras.layers.GRU(50),
#     keras.layers.Dense(1)
# ])
model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, num_features)),
    keras.layers.LSTM(50),
    keras.layers.Dense(1) # Output layer for regression
])
# model = keras.Sequential([
#     keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, num_features), kernel_regularizer=l2(0.0001)),
#     keras.layers.Dropout(0.1),
#     keras.layers.LSTM(50, kernel_regularizer=l2(0.0001)),
#     keras.layers.Dropout(0.1),
#     keras.layers.Dense(1)
# ])
# model = keras.Sequential([
#     keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, num_features)),
#     keras.layers.Dropout(0.2),  # Add dropout after first LSTM
#     keras.layers.LSTM(50),
#     keras.layers.Dropout(0.2),  # Add dropout after second LSTM
#     keras.layers.Dense(1)  # Output layer for regression
# ])
# model = keras.Sequential([
#     keras.layers.LSTM(32, input_shape=(seq_length, num_features)),  # Single layer, fewer units
#     keras.layers.Dropout(0.2),  # Keep dropout for regularization
#     keras.layers.Dense(1)  # Output layer
# ])
# Build LSTM model with bidirectional processing
# model = keras.Sequential([
#     keras.layers.Bidirectional(keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, num_features))),
#     keras.layers.Dropout(0.2),
#     keras.layers.Bidirectional(keras.layers.LSTM(50)),
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(1)
# ])
# Build LSTM model with BatchNorm
# model = keras.Sequential([
#     keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, num_features)),
#     keras.layers.BatchNormalization(),  # Add after first LSTM
#     keras.layers.Dropout(0.2),
#     keras.layers.LSTM(50),
#     keras.layers.BatchNormalization(),  # Add after second
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(1)
# ])

# Compile
# model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
model.compile(optimizer=Adam(learning_rate=learning_rate), loss=asymmetric_mse, metrics=['mae'])

# Train
early_stop = EarlyStopping(monitor='val_mae', patience=300, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test_scaled, y_test), 
                    callbacks=[early_stop],
                    verbose=1)

# Evaluate
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f'Test Loss (MSE): {test_loss}')
print(f'Test MAE: {test_mae}')

# # Analyze feature importance with SHAP (after training)
# import shap

# explainer = shap.DeepExplainer(model, X_train_scaled[:100])  # Use a sample of train data for speed (increase if needed)
# shap_values = explainer.shap_values(X_test_scaled[:100])  # Explain on a sample of test data
# shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=feature_cols.tolist(), show=True)  # Generates a summary plot; saved as image if desired

# Save as SavedModel for ML.NET
model.export(os.path.join(export_dir, 'saved_model'))
print("Model exported as SavedModel directory: saved_model")

# Optional: Save model
model.save(os.path.join(export_dir, 'lstm_stock_model.keras'))

# Optional: Save model in HDF5 format for C# loading
model.save(os.path.join(export_dir, 'lstm_stock_model.h5'))
print("Model saved as HDF5: lstm_stock_model.h5")

# Optional: Save model in onnx format for C# loading
if isinstance(model, keras.Sequential):
    inputs = keras.Input(shape=(seq_length, num_features), name='input')
    outputs = model(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

onnx_model_path = os.path.join(export_dir, 'lstm_stock_model.onnx')
input_signature = [tf.TensorSpec((None, seq_length, num_features), tf.float32, name='input')]
tf2onnx.convert.from_keras(model, input_signature=input_signature, output_path=onnx_model_path)
print(f"Model saved in ONNX format: {onnx_model_path}")

# To make predictions on train data
train_predictions = model.predict(X_train_scaled)

train_results_df = pd.DataFrame({
    'Actual': y_train,
    'Predicted': train_predictions.flatten()  # Flatten to match y_train's shape
})

# Save to CSV (in the project root; adjust path if needed)
train_output_file = os.path.join(project_root, 'train_predictions_regression.csv')
train_results_df.to_csv(train_output_file, index=False)

print(f"Saved all {len(y_train)} train predictions and actuals to '{train_output_file}'")

# To make predictions on test data
predictions = model.predict(X_test_scaled)

# Create DataFrame with all actuals and predictions
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions.flatten()  # Flatten to match y_test's shape
})

# Save to CSV (in the project root; adjust path if needed)
output_file = os.path.join(project_root, 'test_predictions_regression.csv')
results_df.to_csv(output_file, index=False)

print(f"Saved all {len(y_test)} test predictions and actuals to '{output_file}'")
