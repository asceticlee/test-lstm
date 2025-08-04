import os
import json
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets src/ dir
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Gets test-lstm/ root

# Model 1 paths
MODEL1_PATH = os.path.join(os.path.join(PROJECT_ROOT, 'models'), 'lstm_stock_model_00201.keras')
SCALER1_PATH = os.path.join(os.path.join(PROJECT_ROOT, 'models'), 'scaler_params_00201.json')

# Model 2 paths (adjust these to your second model)
MODEL2_PATH = os.path.join(os.path.join(PROJECT_ROOT, 'models'), 'lstm_stock_model_00205.keras')
SCALER2_PATH = os.path.join(os.path.join(PROJECT_ROOT, 'models'), 'scaler_params_00205.json')

SEQ_LENGTH = 30  # adjust if needed
NUM_FEATURES = 44  # adjust if needed

# Load models
model1 = tf.keras.models.load_model(MODEL1_PATH)
model2 = tf.keras.models.load_model(MODEL2_PATH)

# Load scaler params for model 1
with open(SCALER1_PATH, 'r') as f:
    scaler1_params = json.load(f)

scaler1 = MinMaxScaler()
scaler1.data_min_ = np.array(scaler1_params['Min'])
scaler1.data_max_ = np.array(scaler1_params['Max'])
scaler1.data_range_ = scaler1.data_max_ - scaler1.data_min_
scaler1.scale_ = np.where(scaler1.data_range_ != 0, 1.0 / scaler1.data_range_, 0.0)
scaler1.min_ = -scaler1.data_min_ * scaler1.scale_

# Load scaler params for model 2
with open(SCALER2_PATH, 'r') as f:
    scaler2_params = json.load(f)

scaler2 = MinMaxScaler()
scaler2.data_min_ = np.array(scaler2_params['Min'])
scaler2.data_max_ = np.array(scaler2_params['Max'])
scaler2.data_range_ = scaler2.data_max_ - scaler2.data_min_
scaler2.scale_ = np.where(scaler2.data_range_ != 0, 1.0 / scaler2.data_range_, 0.0)
scaler2.min_ = -scaler2.data_min_ * scaler2.scale_

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Expecting: {"sequence": [[...], [...], ...]} shape (SEQ_LENGTH, NUM_FEATURES)
    sequence = data.get('sequence')
    if not sequence or len(sequence) != SEQ_LENGTH or len(sequence[0]) != NUM_FEATURES:
        return jsonify({'error': f'Input must be a {SEQ_LENGTH}x{NUM_FEATURES} array.'}), 400
    
    X = np.array(sequence).reshape(1, SEQ_LENGTH, NUM_FEATURES)
    
    # Scale features for model 1
    X_reshaped = X.reshape(-1, NUM_FEATURES)
    X_scaled1 = scaler1.transform(X_reshaped).reshape(1, SEQ_LENGTH, NUM_FEATURES)
    
    # Scale features for model 2
    X_scaled2 = scaler2.transform(X_reshaped).reshape(1, SEQ_LENGTH, NUM_FEATURES)

    # Print the scaled sequences for debugging (optional)
    # print("\nReceived sequence (raw):")
    # print(np.array(sequence))
    # print("\nScaled sequence for model 1:")
    # print(X_scaled1[0])
    # print("\nScaled sequence for model 2:")
    # print(X_scaled2[0])

    # Predict with both models
    pred1 = model1.predict(X_scaled1, verbose=0)
    pred2 = model2.predict(X_scaled2, verbose=0)
    
    return jsonify({
        'prediction_model1': float(pred1.flatten()[0]),
        'prediction_model2': float(pred2.flatten()[0])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8300)
