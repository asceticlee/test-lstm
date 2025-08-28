#!/usr/bin/env python3
"""
Serve Regime-Based LSTM Flask API

This Flask API provides regime-aware LSTM predictions by:
1. Accepting overnight_gap, quote_data, and sequence parameters
2. Using GMMRegimeInstanceClassifier to predict current market regime
3. Loading the appropriate model based on model_trading_model_pick.csv
4. Making LSTM predictions with the selected model
5. Returning regime, model info, and prediction

The model selection file (model_trading_model_pick.csv) should be updated daily
before market hours using the model_trading_picker.py script.

Usage:
    python serve_regime_lstm_flask.py
    
API Endpoint:
    POST /predict_regime_lstm
    {
        "overnight_gap": 0.05,
        "quote_data": [{"ms_of_day": 38100000, "bid": 1.2345, "ask": 1.2347, "mid": 1.2346}, ...],
        "sequence": [[...], [...], ...]  // shape (30, 44)
    }
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import sys
from typing import Dict, List, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from market_regime.gmm_regime_instance_clustering import GMMRegimeInstanceClassifier

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets src/ dir
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Gets test-lstm/ root
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PICK_FILE = os.path.join(PROJECT_ROOT, 'model_trading', 'model_trading_model_pick.csv')

# Model configuration
SEQ_LENGTH = 30
NUM_FEATURES = 44

# Global variables for caching
regime_classifier = None
loaded_models = {}  # Cache for loaded models and scalers
model_selections = {}  # Cache for model selection data

app = Flask(__name__)


def initialize_regime_classifier():
    """Initialize the GMM regime classifier"""
    global regime_classifier
    if regime_classifier is None:
        print("Initializing GMM Regime Classifier...")
        regime_classifier = GMMRegimeInstanceClassifier()
        print("GMM Regime Classifier initialized successfully")


def load_model_selections():
    """Load model selection data from CSV file"""
    global model_selections
    
    if not os.path.exists(MODEL_PICK_FILE):
        raise FileNotFoundError(f"Model selection file not found: {MODEL_PICK_FILE}")
    
    try:
        df = pd.read_csv(MODEL_PICK_FILE)
        
        # Validate required columns
        required_columns = ['regime', 'model_id', 'direction', 'threshold']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Model selection file missing columns: {missing_columns}")
        
        # Convert to dictionary for fast lookup
        model_selections = {}
        for _, row in df.iterrows():
            regime_id = int(row['regime'])
            model_selections[regime_id] = {
                'model_id': int(row['model_id']),
                'direction': str(row['direction']),
                'threshold': float(row['threshold'])
            }
        
        print(f"Loaded model selections for {len(model_selections)} regimes")
        return model_selections
        
    except Exception as e:
        raise Exception(f"Error loading model selections: {e}")


def load_model_and_scaler(model_id: int) -> Dict:
    """
    Load LSTM model and scaler for given model ID
    
    Args:
        model_id: Model ID (e.g., 43, 421)
        
    Returns:
        Dictionary with 'model' and 'scaler' keys
    """
    global loaded_models
    
    # Check if already loaded
    if model_id in loaded_models:
        return loaded_models[model_id]
    
    # Construct file paths
    model_path = os.path.join(MODELS_DIR, f'lstm_stock_model_{model_id:05d}.keras')
    scaler_path = os.path.join(MODELS_DIR, f'scaler_params_{model_id:05d}.json')
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    try:
        # Load model
        print(f"Loading model {model_id:05d}...")
        model = tf.keras.models.load_model(model_path)
        
        # Load scaler parameters
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)
        
        # Reconstruct scaler
        scaler = MinMaxScaler()
        scaler.data_min_ = np.array(scaler_params['Min'])
        scaler.data_max_ = np.array(scaler_params['Max'])
        scaler.data_range_ = scaler.data_max_ - scaler.data_min_
        scaler.scale_ = np.where(scaler.data_range_ != 0, 1.0 / scaler.data_range_, 0.0)
        scaler.min_ = -scaler.data_min_ * scaler.scale_
        
        # Cache the loaded model and scaler
        loaded_models[model_id] = {
            'model': model,
            'scaler': scaler
        }
        
        print(f"Successfully loaded model {model_id:05d}")
        return loaded_models[model_id]
        
    except Exception as e:
        raise Exception(f"Error loading model {model_id:05d}: {e}")


def predict_market_regime(overnight_gap: float, quote_data: List[Dict]) -> Dict:
    """
    Predict market regime using the classifier
    
    Args:
        overnight_gap: Overnight gap value
        quote_data: List of quote data dictionaries
        
    Returns:
        Dictionary with regime prediction and probabilities
    """
    global regime_classifier
    
    if regime_classifier is None:
        raise RuntimeError("Regime classifier not initialized")
    
    try:
        # Classify regime
        predicted_regime = regime_classifier.classify_regime(overnight_gap, quote_data)
        
        # Get probabilities
        regime_with_probs, probabilities = regime_classifier.classify_regime_with_probabilities(
            overnight_gap, quote_data
        )
        
        return {
            'predicted_regime': int(predicted_regime),
            'regime_probabilities': {
                '0': float(probabilities[0]),
                '1': float(probabilities[1]),
                '2': float(probabilities[2]),
                '3': float(probabilities[3]),
                '4': float(probabilities[4])
            },
            'max_probability': float(max(probabilities)),
            'confidence': float(max(probabilities))
        }
        
    except Exception as e:
        raise Exception(f"Error predicting regime: {e}")


def make_lstm_prediction(sequence: List[List[float]], model_id: int) -> float:
    """
    Make LSTM prediction using the specified model
    
    Args:
        sequence: Input sequence (SEQ_LENGTH x NUM_FEATURES)
        model_id: Model ID to use for prediction
        
    Returns:
        Prediction value
    """
    # Load model and scaler
    model_data = load_model_and_scaler(model_id)
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Validate sequence shape
    if len(sequence) != SEQ_LENGTH or len(sequence[0]) != NUM_FEATURES:
        raise ValueError(f"Sequence must be {SEQ_LENGTH}x{NUM_FEATURES}, got {len(sequence)}x{len(sequence[0])}")
    
    # Prepare input data
    X = np.array(sequence).reshape(1, SEQ_LENGTH, NUM_FEATURES)
    
    # Scale features
    X_reshaped = X.reshape(-1, NUM_FEATURES)
    X_scaled = scaler.transform(X_reshaped).reshape(1, SEQ_LENGTH, NUM_FEATURES)
    
    # Make prediction
    prediction = model.predict(X_scaled, verbose=0)
    
    return float(prediction.flatten()[0])


@app.route('/predict_regime_lstm', methods=['POST'])
def predict_regime_lstm():
    """
    Main API endpoint for regime-aware LSTM predictions
    """
    try:
        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required parameters
        overnight_gap = data.get('overnight_gap')
        quote_data = data.get('quote_data')
        sequence = data.get('sequence')
        
        if overnight_gap is None:
            return jsonify({'error': 'overnight_gap parameter is required'}), 400
        if not quote_data:
            return jsonify({'error': 'quote_data parameter is required'}), 400
        if not sequence:
            return jsonify({'error': 'sequence parameter is required'}), 400
        
        # Validate quote_data format
        if not isinstance(quote_data, list) or len(quote_data) == 0:
            return jsonify({'error': 'quote_data must be a non-empty list'}), 400
        
        # Validate sequence format
        if not isinstance(sequence, list) or len(sequence) != SEQ_LENGTH:
            return jsonify({'error': f'sequence must be a list of length {SEQ_LENGTH}'}), 400
        if len(sequence[0]) != NUM_FEATURES:
            return jsonify({'error': f'Each sequence element must have {NUM_FEATURES} features'}), 400
        
        # Step 1: Predict market regime
        print(f"Predicting regime with overnight_gap={overnight_gap}, {len(quote_data)} quote points")
        regime_prediction = predict_market_regime(overnight_gap, quote_data)
        predicted_regime = regime_prediction['predicted_regime']
        
        print(f"Predicted regime: {predicted_regime}")
        
        # Step 2: Get model selection for this regime
        if predicted_regime not in model_selections:
            return jsonify({
                'error': f'No model selection found for regime {predicted_regime}',
                'regime_prediction': regime_prediction
            }), 400
        
        model_selection = model_selections[predicted_regime]
        model_id = model_selection['model_id']
        direction = model_selection['direction']
        threshold = model_selection['threshold']
        
        print(f"Using model {model_id:05d} (direction: {direction}, threshold: {threshold})")
        
        # Step 3: Make LSTM prediction
        if model_id == 0:
            # Handle case where no valid model was found for this regime
            lstm_prediction = 0.0
            print("No valid model found for this regime, using default prediction of 0.0")
        else:
            lstm_prediction = make_lstm_prediction(sequence, model_id)
            print(f"LSTM prediction: {lstm_prediction}")
        
        # Step 4: Return comprehensive response
        response = {
            'success': True,
            'regime_prediction': {
                'predicted_regime': predicted_regime,
                'regime_probabilities': regime_prediction['regime_probabilities'],
                'confidence': regime_prediction['confidence']
            },
            'model_selection': {
                'model_id': model_id,
                'direction': direction,
                'threshold': threshold
            },
            'lstm_prediction': lstm_prediction,
            'metadata': {
                'overnight_gap': overnight_gap,
                'quote_data_points': len(quote_data),
                'sequence_shape': [len(sequence), len(sequence[0])]
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in predict_regime_lstm: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if all components are loaded
        status = {
            'regime_classifier': regime_classifier is not None,
            'model_selections_loaded': len(model_selections) > 0,
            'cached_models': len(loaded_models),
            'model_pick_file_exists': os.path.exists(MODEL_PICK_FILE)
        }
        
        return jsonify({
            'status': 'healthy',
            'components': status
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/reload_model_selections', methods=['POST'])
def reload_model_selections():
    """Reload model selections from CSV file"""
    try:
        load_model_selections()
        return jsonify({
            'success': True,
            'message': f'Reloaded model selections for {len(model_selections)} regimes'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def initialize_server():
    """Initialize all server components"""
    print("=" * 60)
    print("REGIME-BASED LSTM FLASK SERVER INITIALIZATION")
    print("=" * 60)
    
    try:
        # Initialize regime classifier
        initialize_regime_classifier()
        
        # Load model selections
        load_model_selections()
        
        print("=" * 60)
        print("SERVER INITIALIZATION COMPLETED SUCCESSFULLY")
        print(f"Model selections loaded for regimes: {sorted(model_selections.keys())}")
        print(f"Model pick file: {MODEL_PICK_FILE}")
        print("=" * 60)
        
    except Exception as e:
        print(f"ERROR: Server initialization failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # Initialize server components
    initialize_server()
    
    # Start Flask server
    print("Starting Flask server on http://0.0.0.0:8301")
    app.run(host='0.0.0.0', port=8301, debug=False)
