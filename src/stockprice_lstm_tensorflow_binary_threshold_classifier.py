import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras  # This loads tf.keras as 'keras'
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# Define paths and parameters
script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets src/ dir
project_root = os.path.dirname(script_dir)  # Gets test-lstm/ root
data_dir = os.path.join(project_root, 'data/')  # Full path to data/
train_file = os.path.join(data_dir, 'training_data_spy_20250101_20250624.csv')
test_file = os.path.join(data_dir, 'testing_data_spy_20250625_20250710.csv')
seq_length = 20
learning_rate = 0.001
epochs = 1000
batch_size = 32
threshold = 0.1  # Threshold for binary categorization (up if > threshold, down if < -threshold; ignore in-between)

# Load data
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Define feature and target columns
# Features: columns 6 to 49 (0-based index 5 to 48)
feature_cols = train_df.columns[5:49]
target_col = 'Label_8'  # Raw continuous label for categorization
num_features = len(feature_cols)

# Function to create sequences, respecting continuity, and categorize into binary labels (only up/down, skip idle)
def create_sequences(df, seq_length, threshold):
    df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
    features = df[feature_cols].values
    raw_targets = df[target_col].values  # Raw continuous values
    days = df['TradingDay'].values
    ms = df['TradingMsOfDay'].values
    
    X = []
    y = []  # Binary labels: 0=down, 1=up (skip if idle)
    
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
            raw_label = raw_targets[i + seq_length - 1]  # Raw value of last timestep
            
            # Binary categorization (skip idle)
            if raw_label > threshold:
                label = 1  # Up (long)
                X.append(seq)
                y.append(label)
            elif raw_label < -threshold:
                label = 0  # Down (short)
                X.append(seq)
                y.append(label)
            # Else skip (idle)
            
            i += 1  # Sliding window, step by 1
        else:
            # Skip to the next potential start (after the gap)
            i += 1
    
    return np.array(X), np.array(y)

# Create sequences for train and test
X_train, y_train = create_sequences(train_df, seq_length, threshold)
X_test, y_test = create_sequences(test_df, seq_length, threshold)

# Print the size of x_train and y_train
print(f'Train sequences shape: {X_train.shape}, Train labels shape: {y_train.shape}')
print(f'Test sequences shape: {X_test.shape}, Test labels shape: {y_test.shape}')

# Compute class weights for binary imbalance (if needed)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# Scale features (fit on train, apply to both)
scaler = MinMaxScaler()
X_train_reshaped = X_train.reshape(-1, num_features)
X_test_reshaped = X_test.reshape(-1, num_features)

scaler.fit(X_train_reshaped)
X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

# Build LSTM model for binary classification
model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, num_features)),
    keras.layers.LSTM(50),
    keras.layers.Dense(1, activation='sigmoid')  # Binary output: probability of up (1)
])

# Compile for binary classification
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=1001, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test_scaled, y_test), verbose=1,
                    callbacks=[early_stop],
                    class_weight=class_weight_dict)  # Optional; remove if imbalance is low

# Evaluate
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

# Save model
model.save('lstm_binary_model.keras')

# Predictions on train data
train_predictions = model.predict(X_train_scaled)  # Shape: (n_samples, 1) probabilities for up
train_classes = (train_predictions > 0.5).astype(int).flatten()  # 0=down, 1=up
train_probs = train_predictions.flatten()  # Probability of up

train_results_df = pd.DataFrame({
    'Actual_Class': y_train,
    'Predicted_Class': train_classes,
    'Prob_Up': train_probs
})

train_output_file = os.path.join(project_root, 'train_predictions_binary_threshold.csv')
train_results_df.to_csv(train_output_file, index=False)
print(f"Saved train predictions to '{train_output_file}'")

# Predictions on test data
test_predictions = model.predict(X_test_scaled)
test_classes = (test_predictions > 0.5).astype(int).flatten()
test_probs = test_predictions.flatten()

test_results_df = pd.DataFrame({
    'Actual_Class': y_test,
    'Predicted_Class': test_classes,
    'Prob_Up': test_probs
})

output_file = os.path.join(project_root, 'test_predictions_binary_threshold.csv')
test_results_df.to_csv(output_file, index=False)
print(f"Saved test predictions to '{output_file}'")