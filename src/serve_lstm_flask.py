import os
import json
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets src/ dir
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Gets test-lstm/ root
MODEL_PATH = os.path.join(os.path.join(PROJECT_ROOT, 'models_prod'), 'lstm_stock_model_30_050_20250608_20250705_6136_4676.keras')
SCALER_PATH = os.path.join(os.path.join(PROJECT_ROOT, 'models_prod'), 'scaler_params_30_050_20250608_20250705_6136_4676.json')
SEQ_LENGTH = 30  # adjust if needed
NUM_FEATURES = 44  # adjust if needed

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load scaler params
with open(SCALER_PATH, 'r') as f:
    scaler_params = json.load(f)

scaler = MinMaxScaler()
scaler.data_min_ = np.array(scaler_params['Min'])
scaler.data_max_ = np.array(scaler_params['Max'])
scaler.data_range_ = scaler.data_max_ - scaler.data_min_
# Avoid division by zero for constant features
scaler.scale_ = np.where(scaler.data_range_ != 0, 1.0 / scaler.data_range_, 0.0)
scaler.min_ = -scaler.data_min_ * scaler.scale_

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Expecting: {"sequence": [[...], [...], ...]} shape (SEQ_LENGTH, NUM_FEATURES)
    sequence = data.get('sequence')
    if not sequence or len(sequence) != SEQ_LENGTH or len(sequence[0]) != NUM_FEATURES:
        return jsonify({'error': f'Input must be a {SEQ_LENGTH}x{NUM_FEATURES} array.'}), 400
    X = np.array(sequence).reshape(1, SEQ_LENGTH, NUM_FEATURES)
    # Scale features
    X_reshaped = X.reshape(-1, NUM_FEATURES)
    X_scaled = scaler.transform(X_reshaped).reshape(1, SEQ_LENGTH, NUM_FEATURES)


    # Print the scaled sequence for debugging
    # print("\nReceived sequence (raw):")
    # print(np.array(sequence))
    # print("\nScaled sequence (Flask):")
    # print(X_scaled[0])

    # Predict
    pred = model.predict(X_scaled)
    return jsonify({'prediction': float(pred.flatten()[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8300)
