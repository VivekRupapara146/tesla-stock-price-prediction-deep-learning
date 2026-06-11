"""
preprocessing.py
----------------
Handles all data transformation between raw OHLCV data and
the exact input format expected by the trained GRU model.

Responsibilities:
    - Load the saved MinMaxScaler (fitted on training data only).
    - Scale a (60, 5) OHLCV window using the loaded scaler.
    - Reshape scaled data to GRU input shape: (1, 60, 5).
    - Inverse-transform the model's scalar output back to USD price.

CRITICAL constraints:
    - NEVER fit a new scaler — always load tesla_scaler.pkl.
    - NEVER change feature order: [Open, High, Low, Close, Volume].
    - NEVER use fewer or more than 60 rows.
    - Inverse transform requires a dummy (1, 5) array — see below.

Public API:
    load_scaler(scaler_path)                    -> MinMaxScaler
    preprocess_window(df, scaler)               -> np.ndarray  shape (1,60,5)
    inverse_transform_prediction(val, scaler)   -> float  (USD price)
"""

import numpy as np
import pandas as pd
import joblib
import os


# ── Constants ────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
CLOSE_COL_INDEX = 3       # Position of 'Close' in FEATURE_COLUMNS
N_FEATURES = 5
LOOKBACK_WINDOW = 60      # Must match training configuration


# ── Scaler Loading ────────────────────────────────────────────────────────────

def load_scaler(scaler_path: str):
    """
    Load the pre-fitted MinMaxScaler from disk.

    This scaler was fitted ONLY on the training data split.
    It must be reused as-is during deployment — never refit.

    Args:
        scaler_path (str): Absolute or relative path to tesla_scaler.pkl.

    Returns:
        sklearn.preprocessing.MinMaxScaler: Loaded scaler object.

    Raises:
        FileNotFoundError: If the .pkl file does not exist at the path.
        ValueError: If the loaded object is not a valid sklearn scaler.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler file not found at: '{scaler_path}'\n"
            "Place tesla_scaler.pkl inside the model/ directory."
        )

    scaler = joblib.load(scaler_path)

    # Sanity check — verify it has transform/inverse_transform methods
    if not (hasattr(scaler, "transform") and hasattr(scaler, "inverse_transform")):
        raise ValueError(
            f"Loaded object from '{scaler_path}' is not a valid scaler. "
            "Expected a fitted sklearn MinMaxScaler."
        )

    # Verify it was fitted on 5 features (OHLCV)
    if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != N_FEATURES:
        raise ValueError(
            f"Scaler was fitted on {scaler.n_features_in_} features, "
            f"but expected {N_FEATURES} (OHLCV). "
            "Ensure you are using the correct scaler file."
        )

    return scaler


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_window(df: pd.DataFrame, scaler) -> np.ndarray:
    """
    Transform a (60, 5) OHLCV DataFrame into the GRU model's input format.

    Pipeline:
        DataFrame (60, 5)
            → enforce column order [Open, High, Low, Close, Volume]
            → convert to NumPy float32
            → MinMaxScaler.transform()    → scaled (60, 5)
            → np.reshape(1, 60, 5)        → model-ready (1, 60, 5)

    Args:
        df (pd.DataFrame): Exactly 60 rows with OHLCV columns.
        scaler: Fitted MinMaxScaler loaded via load_scaler().

    Returns:
        np.ndarray: Shape (1, 60, 5), dtype float32.

    Raises:
        ValueError: If df does not have exactly 60 rows or 5 columns.
    """
    # ── Validate input shape ──
    if len(df) != LOOKBACK_WINDOW:
        raise ValueError(
            f"Expected {LOOKBACK_WINDOW} rows, got {len(df)}. "
            "Pass the output of get_latest_window() directly."
        )

    if not all(col in df.columns for col in FEATURE_COLUMNS):
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        raise ValueError(f"Missing columns before preprocessing: {missing}")

    # ── Enforce feature order ──
    # This is the single most important step — wrong order = wrong prediction
    data = df[FEATURE_COLUMNS].values.astype(np.float32)   # shape: (60, 5)

    # ── Scale using the pre-fitted scaler ──
    # scaler.transform() uses the min/max learned from training data
    scaled = scaler.transform(data)    # shape: (60, 5), values in [0, 1]

    # ── Reshape to GRU input format ──
    # GRU expects: (batch_size, timesteps, features) = (1, 60, 5)
    model_input = scaled.reshape(1, LOOKBACK_WINDOW, N_FEATURES)

    return model_input


# ── Inverse Transform ─────────────────────────────────────────────────────────

def inverse_transform_prediction(scaled_value: float, scaler) -> float:
    """
    Convert the GRU model's scaled output back to a USD price.

    WHY A DUMMY ARRAY?
    ------------------
    The scaler was fitted on all 5 features simultaneously.
    Its inverse_transform() therefore expects shape (n, 5).

    The model outputs a single scaled Close value (scalar float).
    We cannot pass this directly to inverse_transform.

    Solution — dummy array:
        Build a (1, 5) zero array.
        Place scaled_value at index 3 (Close column position).
        Call scaler.inverse_transform() on the full array.
        Extract index [0][3] — the recovered Close price.

    This is mathematically equivalent to:
        actual_close = scaled_close * (max_close - min_close) + min_close
    but uses the scaler's own stored statistics correctly.

    Args:
        scaled_value (float): Raw scalar output from the GRU model.
        scaler: The same fitted MinMaxScaler used in preprocess_window().

    Returns:
        float: Predicted Tesla closing price in USD.
    """
    # Build dummy array: shape (1, 5), all zeros
    dummy = np.zeros((1, N_FEATURES), dtype=np.float32)

    # Place the model's prediction at the Close column position
    dummy[0][CLOSE_COL_INDEX] = scaled_value

    # Inverse transform — recovers all 5 features, we only need Close
    recovered = scaler.inverse_transform(dummy)

    # Extract the Close value
    predicted_price = float(recovered[0][CLOSE_COL_INDEX])

    return predicted_price

