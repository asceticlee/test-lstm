import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras  # This loads tf.keras as 'keras'
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score

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
threshold = 0.2  # For labeling (tune for balance)
confidence_threshold = 0.7  # Starting point for high-confidence trades; will tune dynamically

# Load data
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Define feature and target columns
feature_cols = train_df.columns[5:49]
target_col = 'Label_8'  # Raw continuous label for categorization
num_features = len(feature_cols)

# Function to create sequences, respecting continuity, and categorize labels
def create_sequences(df, seq_length, threshold):
    df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
    features = df[feature_cols].values
    raw_targets = df[target_col].values  # Raw continuous values
    days = df['TradingDay'].values
    ms = df['TradingMsOfDay'].values
    
    X = []
    y = []  # Class labels: 0=idle, 1=long, 2=short
    
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
            
            # Categorize into 3 classes
            if raw_label > threshold:
                label = 1  # Long (up)
            elif raw_label < -threshold:
                label = 2  # Short (down)
            else:
                label = 0  # Idle
            
            X.append(seq)
            y.append(label)
            i += 1
        else:
            i += 1
    
    return np.array(X), np.array(y)

# Create sequences for train and test
X_train, y_train = create_sequences(train_df, seq_length, threshold)
X_test, y_test = create_sequences(test_df, seq_length, threshold)

# Resample train data to balance (optional but recommended)
ros = RandomOverSampler(random_state=42)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_train_res, y_train_res = ros.fit_resample(X_train_reshaped, y_train)
X_train_res = X_train_res.reshape(X_train_res.shape[0], seq_length, num_features)
print("Resampled train class counts:", np.bincount(y_train_res))

# Compute class weights on resampled data
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_res), y=y_train_res)
class_weight_dict = dict(enumerate(class_weights))
# Manual boost for up/down
class_weight_dict[1] *= 1.2  # Long
class_weight_dict[2] *= 1.2  # Short
if 0 in class_weight_dict:
    class_weight_dict[0] *= 0.8  # Decrease for idle (class 0) to de-emphasize
print("Class weights:", class_weight_dict)

# Print sizes
print(f'Train sequences shape: {X_train_res.shape}, Train labels shape: {y_train_res.shape}')
print(f'Test sequences shape: {X_test.shape}, Test labels shape: {y_test.shape}')

# Scale features (fit on resampled train)
scaler = MinMaxScaler()
X_train_reshaped = X_train_res.reshape(-1, num_features)
X_test_reshaped = X_test.reshape(-1, num_features)

scaler.fit(X_train_reshaped)
X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train_res.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

# Custom focal loss
def sparse_categorical_focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)  # Cast to int
        y_true = tf.one_hot(y_true, depth=y_pred.shape[-1])
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1. - keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow(1 - y_pred, gamma)
        focal_loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    return loss

# Build LSTM model for 3-class classification
model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, num_features)),
    keras.layers.LSTM(50),
    keras.layers.Dense(3, activation='softmax')  # 3 classes
])

# Compile with focal loss
model.compile(optimizer=Adam(learning_rate=learning_rate), loss=sparse_categorical_focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train_res, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test_scaled, y_test), verbose=1,
                    callbacks=[early_stop],
                    class_weight=class_weight_dict)

# Evaluate
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

# Save model
model.save('lstm_3class_model.keras')

# Temperature scaling for calibration
class CalibratedModel(keras.Model):
    def __init__(self, base_model, temperature=1.5):  # Tune temperature >1 for less confidence
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature
        
    def call(self, inputs):
        logits = self.base_model(inputs)
        return tf.nn.softmax(logits / self.temperature)

calibrated_model = CalibratedModel(model, temperature=1.5)  # Higher temp softens probs

# Predictions on train (use calibrated)
train_predictions = calibrated_model.predict(X_train_scaled)
train_classes = np.argmax(train_predictions, axis=1)
train_probs = np.max(train_predictions, axis=1)

# Filter traded (non-idle predicted, confidence > threshold)
traded_mask = (train_classes != 0) & (train_probs > confidence_threshold)
traded_actual = y_train_res[traded_mask]
traded_pred = train_classes[traded_mask]
traded_acc = accuracy_score(traded_actual, traded_pred) if len(traded_actual) > 0 else 0

# Split up/down accuracy
up_mask = (traded_pred == 1)
down_mask = (traded_pred == 2)
up_acc = accuracy_score(traded_actual[up_mask], traded_pred[up_mask]) if len(up_mask[up_mask]) > 0 else 0
down_acc = accuracy_score(traded_actual[down_mask], traded_pred[down_mask]) if len(down_mask[down_mask]) > 0 else 0

print(f'Train traded samples: {len(traded_actual)} ({len(traded_actual)/len(y_train_res)*100:.2f}%)')
print(f'Train directional accuracy on traded (total): {traded_acc * 100:.2f}%')
print(f'Train directional accuracy on traded (up): {up_acc * 100:.2f}%')
print(f'Train directional accuracy on traded (down): {down_acc * 100:.2f}%')

train_results_df = pd.DataFrame({
    'Actual_Class': y_train_res,
    'Predicted_Class': train_classes,
    'Confidence_Prob': train_probs,
    'Prob_Idle': train_predictions[:, 0],
    'Prob_Long': train_predictions[:, 1],
    'Prob_Short': train_predictions[:, 2]
})
train_output_file = os.path.join(project_root, 'train_predictions_3class.csv')
train_results_df.to_csv(train_output_file, index=False)

# Predictions on test (use calibrated)
test_predictions = calibrated_model.predict(X_test_scaled)
test_classes = np.argmax(test_predictions, axis=1)
test_probs = np.max(test_predictions, axis=1)

# Tune threshold for ~7% trade rate on test
desired_trade_rate = 0.07  # 7%
start_threshold = 0.5  # Start low, increase to reduce trades
step = 0.05
current_threshold = start_threshold

while current_threshold <= 0.95:
    traded_mask = (test_classes != 0) & (test_probs > current_threshold)
    trade_rate = len(traded_mask[traded_mask]) / len(test_probs)
    if trade_rate <= desired_trade_rate:
        break
    current_threshold += step

traded_actual = y_test[traded_mask]
traded_pred = test_classes[traded_mask]
traded_acc = accuracy_score(traded_actual, traded_pred) if len(traded_actual) > 0 else 0

# Split up/down accuracy for test
up_mask = (traded_pred == 1)
down_mask = (traded_pred == 2)
up_acc = accuracy_score(traded_actual[up_mask], traded_pred[up_mask]) if len(up_mask[up_mask]) > 0 else 0
down_acc = accuracy_score(traded_actual[down_mask], traded_pred[down_mask]) if len(down_mask[down_mask]) > 0 else 0

print(f'Tuned threshold: {current_threshold:.2f}')
print(f'Test traded samples: {len(traded_actual)} ({trade_rate*100:.2f}%)')
print(f'Test directional accuracy on traded (total): {traded_acc * 100:.2f}%')
print(f'Test directional accuracy on traded (up): {up_acc * 100:.2f}%')
print(f'Test directional accuracy on traded (down): {down_acc * 100:.2f}%')

test_results_df = pd.DataFrame({
    'Actual_Class': y_test,
    'Predicted_Class': test_classes,
    'Confidence_Prob': test_probs,
    'Prob_Idle': test_predictions[:, 0],
    'Prob_Long': test_predictions[:, 1],
    'Prob_Short': test_predictions[:, 2]
})
output_file = os.path.join(project_root, 'test_predictions_3class.csv')
test_results_df.to_csv(output_file, index=False)