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
train_file = os.path.join(data_dir, 'training_data_spy_with_iv_20240101_20250430.csv')
test_file = os.path.join(data_dir, 'testing_data_spy_with_iv_20250501_20250715.csv')
export_dir = os.path.join(project_root, 'models')
os.makedirs(export_dir, exist_ok=True)
seq_length = 20
learning_rate = 0.001
epochs = 1000
batch_size = 32
significant_threshold = 0.3  # Threshold for binary labeling (adjustable)

# Load data
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Define feature and target columns
# Features: columns 6 to 65 (0-based index 5 to 65)
feature_cols = train_df.columns[5:66]
target_col = 'Label_8'
num_features = len(feature_cols)

# Function to create sequences, respecting continuity, with binary labels (1=significant move, 0=idle)
def create_sequences(df, seq_length, threshold):
    df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
    features = df[feature_cols].values
    raw_targets = df[target_col].values  # Raw continuous values
    days = df['TradingDay'].values
    ms = df['TradingMsOfDay'].values
    
    X = []
    y = []  # Binary: 1 if |raw_label| > threshold (significant), 0 otherwise (idle)
    
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
            raw_label = raw_targets[i + seq_length - 1]  # Raw value of last timestep
            
            # Binary label
            label = 1 if abs(raw_label) > threshold else 0
            
            X.append(seq)
            y.append(label)
            i += 1
        else:
            i += 1
    
    return np.array(X), np.array(y)

# Create sequences for train and test
X_train, y_train = create_sequences(train_df, seq_length, significant_threshold)
X_test, y_test = create_sequences(test_df, seq_length, significant_threshold)

# Print the size and class counts
print(f'Train sequences shape: {X_train.shape}, Train labels shape: {y_train.shape}')
print(f'Test sequences shape: {X_test.shape}, Test labels shape: {y_test.shape}')
print("Train class counts (0=idle, 1=significant):", np.bincount(y_train))
print("Test class counts (0=idle, 1=significant):", np.bincount(y_test))

# Resample minority class using RandomOverSampler
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
# Flatten each sequence for resampling
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_train_res, y_train_res = ros.fit_resample(X_train_flat, y_train)
X_train = X_train_res.reshape(X_train_res.shape[0], seq_length, num_features)
y_train = y_train_res
print("Resampled train class counts (0=idle, 1=significant):", np.bincount(y_train))

# Class weights (optional, can be omitted after resampling)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights after resampling:", class_weight_dict)

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
    keras.layers.Dense(1, activation='sigmoid')  # Prob of significant move (1)
])

# Compile
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test_scaled, y_test), verbose=1,
                    callbacks=[early_stop],
                    class_weight=class_weight_dict)

# Evaluate
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

# Save model
model.save(os.path.join(export_dir,'binary_significant_model.keras'))

# Predictions on train
train_predictions = model.predict(X_train_scaled).flatten()  # Probs of significant
train_classes = (train_predictions > 0.5).astype(int)  # 0=idle, 1=significant

train_results_df = pd.DataFrame({
    'Actual_Class': y_train,
    'Predicted_Class': train_classes,
    'Prob_Significant': train_predictions
})
train_results_df.to_csv(os.path.join(project_root, 'train_binary_significant.csv'), index=False)

# Predictions on test
test_predictions = model.predict(X_test_scaled).flatten()
test_classes = (test_predictions > 0.5).astype(int)

test_results_df = pd.DataFrame({
    'Actual_Class': y_test,
    'Predicted_Class': test_classes,
    'Prob_Significant': test_predictions
})
test_results_df.to_csv(os.path.join(project_root, 'test_binary_significant.csv'), index=False)