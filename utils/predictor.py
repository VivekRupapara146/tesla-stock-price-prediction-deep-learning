"""
predictor.py
------------
Handles GRU model loading and inference for Tesla stock price prediction.

Responsibilities:
    - Load the saved GRU model from disk (.keras format).
    - Validate model input shape before inference.
    - Run prediction and return the final USD price as a float.
    - Expose model metadata for the UI dashboard.

Design note:
    load_model() and load_scaler() are kept separate from predict()
    intentionally. In app.py, both will be wrapped with
    @st.cache_resource so they load ONCE per session — not on
    every prediction call.

Public API:
    load_model(model_path)                    -> tf.keras.Model
    predict(model, model_input, scaler)       -> float  (USD price)
    get_model_metadata()                      -> dict
"""

import numpy as np
import os

# Suppress TensorFlow INFO/WARNING logs — only show errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from utils.preprocessing import inverse_transform_prediction


# ── Constants ────────────────────────────────────────────────────────────────

EXPECTED_INPUT_SHAPE = (None, 60, 5)   # (batch, timesteps, features)
EXPECTED_OUTPUT_UNITS = 1              # Single Close price prediction

# Model performance metrics — source: training notebook
MODEL_METADATA = {
    "model_name": "GRU Neural Network",
    "model_file": "tesla_gru_model.keras",
    "input_window": 60,
    "features": ["Open", "High", "Low", "Close", "Volume"],
    "target": "Next Day Closing Price",
    "metrics": {
        "MAE":  "8.81 USD",
        "RMSE": "15.22 USD",
        "MAPE": "2.77%",
        "R2":   "0.9603",
    },
}


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str) -> tf.keras.Model:
    """
    Load the pre-trained GRU model from a .keras file.

    Designed to be called once and cached via @st.cache_resource in app.py.
    Validates that the loaded model has the correct input/output shape
    to catch wrong model files early.

    Args:
        model_path (str): Path to tesla_gru_model.keras.

    Returns:
        tf.keras.Model: Compiled and ready-to-predict GRU model.

    Raises:
        FileNotFoundError: If the .keras file is not found.
        ValueError: If the model input/output shape is unexpected.
        RuntimeError: If TensorFlow fails to load the model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: '{model_path}'\n"
            "Place tesla_gru_model.keras inside the model/ directory."
        )

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(
            f"TensorFlow failed to load the model from '{model_path}'.\n"
            f"Ensure the file is a valid Keras model. Error: {e}"
        )

    # ── Validate input shape ──
    # model.input_shape = (None, 60, 5) — batch dim is None (flexible)
    actual_input_shape = model.input_shape
    if actual_input_shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"Model input shape mismatch.\n"
            f"Expected: {EXPECTED_INPUT_SHAPE}\n"
            f"Got:      {actual_input_shape}\n"
            "Ensure you are using the correct tesla_gru_model.keras file."
        )

    # ── Validate output shape ──
    # Output should be (None, 1) — single predicted Close price
    actual_output_shape = model.output_shape
    if actual_output_shape[-1] != EXPECTED_OUTPUT_UNITS:
        raise ValueError(
            f"Model output units mismatch.\n"
            f"Expected: {EXPECTED_OUTPUT_UNITS} output unit(s)\n"
            f"Got:      {actual_output_shape[-1]}\n"
            "Ensure you are using the correct tesla_gru_model.keras file."
        )

    return model


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(model: tf.keras.Model, model_input: np.ndarray, scaler) -> float:
    """
    Run inference on preprocessed input and return the predicted USD price.

    Pipeline:
        model_input: np.ndarray (1, 60, 5)
            -> model.predict()                -> scaled output (1, 1)
            -> extract scalar float           -> e.g. 0.6823
            -> inverse_transform_prediction() -> e.g. 280.45

    Args:
        model (tf.keras.Model): Loaded GRU model from load_model().
        model_input (np.ndarray): Preprocessed array of shape (1, 60, 5).
        scaler: Fitted MinMaxScaler from load_scaler().

    Returns:
        float: Predicted next-day Tesla closing price in USD.

    Raises:
        ValueError: If model_input has wrong shape.
        RuntimeError: If model inference fails.
    """
    # ── Validate input shape ──
    if model_input.shape != (1, 60, 5):
        raise ValueError(
            f"model_input has wrong shape: {model_input.shape}. "
            "Expected (1, 60, 5). Pass the output of preprocess_window() directly."
        )

    # ── Run inference ──
    # verbose=0 suppresses the TF progress bar in Streamlit logs
    try:
        raw_output = model.predict(model_input, verbose=0)
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {e}")

    # raw_output shape: (1, 1) — extract the single scalar value
    scaled_prediction = float(raw_output[0][0])

    # ── Inverse transform to USD ──
    predicted_price = inverse_transform_prediction(scaled_prediction, scaler)

    return round(predicted_price, 2)


# ── Metadata ──────────────────────────────────────────────────────────────────

def get_model_metadata() -> dict:
    """
    Return model configuration and performance metrics for the UI dashboard.

    Returns:
        dict: Model name, input specs, and evaluation metrics.
    """
    return MODEL_METADATA
