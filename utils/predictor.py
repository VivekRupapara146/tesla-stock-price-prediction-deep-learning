"""
predictor.py
------------
Handles GRU model loading and inference for Tesla stock price prediction.

Public API:
    load_model(model_path)                          -> tf.keras.Model
    predict(model, model_input, scaler)             -> float
    predict_multi_step(model, window_df, scaler, n) -> list[dict]
    get_model_metadata()                            -> dict
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from utils.preprocessing import preprocess_window, inverse_transform_prediction
from utils.config import (
    LOOKBACK_WINDOW, FEATURE_COLUMNS,
    VOLUME_ROLLING_WINDOW, RANGE_ROLLING_WINDOW,
    TREND_WINDOW, TREND_BLEND_DECAY, MIN_MODEL_WEIGHT,
)


# ── Constants ────────────────────────────────────────────────────────────────

EXPECTED_INPUT_SHAPE  = (None, 60, 5)
EXPECTED_OUTPUT_UNITS = 1

MODEL_METADATA = {
    "model_name":   "GRU Neural Network",
    "model_file":   "tesla_gru_model.keras",
    "input_window": 60,
    "features":     FEATURE_COLUMNS,
    "target":       "Next Day Closing Price",
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

    Validates input shape (None, 60, 5) and output units (1).

    Raises:
        FileNotFoundError: If model file is absent.
        ValueError: If shapes don't match expectations.
        RuntimeError: If TensorFlow fails to load.
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
            f"TensorFlow failed to load model from '{model_path}': {e}"
        )

    if model.input_shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"Model input shape mismatch.\n"
            f"Expected: {EXPECTED_INPUT_SHAPE}\nGot: {model.input_shape}"
        )

    if model.output_shape[-1] != EXPECTED_OUTPUT_UNITS:
        raise ValueError(
            f"Model output units mismatch.\n"
            f"Expected: {EXPECTED_OUTPUT_UNITS}\nGot: {model.output_shape[-1]}"
        )

    return model


# ── Single-Step Prediction ────────────────────────────────────────────────────

def predict(model: tf.keras.Model, model_input: np.ndarray, scaler) -> float:
    """
    Run inference on a preprocessed (1, 60, 5) array.

    Returns:
        float: Predicted next-day Tesla closing price in USD (2dp).
    """
    if model_input.shape != (1, 60, 5):
        raise ValueError(
            f"model_input shape {model_input.shape} — expected (1, 60, 5)."
        )

    try:
        raw_output = model.predict(model_input, verbose=0)
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {e}")

    scaled_pred = float(raw_output[0][0])
    return round(inverse_transform_prediction(scaled_pred, scaler), 2)


# ── Multi-Step Recursive Forecasting ─────────────────────────────────────────

def predict_multi_step(
    model:     tf.keras.Model,
    window_df: pd.DataFrame,
    scaler,
    n_days:    int,
) -> list[dict]:
    """
    Recursively forecast up to n_days ahead using the single-step GRU.

    ALGORITHM (recursive / auto-regressive forecasting):
    -------------------------------------------------------
    Each predicted Close is fed back as part of a synthetic OHLCV row
    to build the next 60-day window:

        Step 1:  window = last 60 real rows
                 predict → Close_1
                 build synthetic_row_1 = [Open, High, Low, Close_1, Volume]

        Step 2:  window = rows [1:60] + synthetic_row_1
                 predict → Close_2
                 build synthetic_row_2 ...

        ...repeat for n_days steps.

    SYNTHETIC ROW CONSTRUCTION:
    -------------------------------------------------------
    Since the model only predicts Close, the other 4 features are
    estimated from historical statistics of the real window:

        Open   = previous day's Close        (gap-open assumption)
        High   = pred_close * (1 + avg_hl%)  (avg daily range from real data)
        Low    = pred_close * (1 - avg_hl%)
        Volume = rolling mean of last 20 days (real data only)

    WHY ERROR COMPOUNDS:
    -------------------------------------------------------
    Day 1:  input = 60 real rows      → most accurate
    Day 2:  input = 59 real + 1 synth → small error
    Day 5:  input = 55 real + 5 synth → moderate error
    Day 10: input = 50 real + 10 synth → highest uncertainty

    The MAPE of 2.77% applies to day 1 ONLY. Days 5–10 carry
    significantly higher uncertainty and should be read as a
    directional trend, not a precise price target.

    Args:
        model     : Loaded GRU model.
        window_df : DataFrame with >= 60 OHLCV rows (real historical data).
        scaler    : Fitted MinMaxScaler.
        n_days    : Number of future days to forecast (1, 5, or 10).

    Returns:
        list[dict]: One dict per predicted day, each containing:
            {
              "day":        int   (1-indexed step),
              "date":       str   (estimated trading date, YYYY-MM-DD),
              "close":      float (predicted USD price),
              "is_real":    bool  (always False — these are predictions),
            }
    """
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")

    if len(window_df) < LOOKBACK_WINDOW:
        raise ValueError(
            f"window_df has {len(window_df)} rows — need at least {LOOKBACK_WINDOW}."
        )

    # ── Working buffer — starts as copy of the real data ──
    buffer = window_df[FEATURE_COLUMNS].copy().reset_index(drop=True)

    # ── Pre-compute statistics from REAL data only ──
    # Never updated with synthetic rows — prevents compounding stat errors.
    real_close  = buffer["Close"].values.astype(float)
    real_high   = buffer["High"].values.astype(float)
    real_low    = buffer["Low"].values.astype(float)
    real_volume = buffer["Volume"].values.astype(float)

    daily_hl_range = (real_high - real_low) / real_close
    avg_hl_pct = float(np.mean(daily_hl_range[-RANGE_ROLLING_WINDOW:]))
    avg_volume     = float(np.mean(real_volume[-VOLUME_ROLLING_WINDOW:]))
    last_real_close = float(real_close[-1])

    # ── Momentum correction: linear trend slope from recent real closes ──
    # Addresses mean-reversion drift in recursive forecasting.
    # Each synthetic row fed back into the model can reinforce a slight
    # downward signal, causing all predictions to converge downward.
    # We compute the recent price slope from real data and use it to
    # blend the raw model output with a trend-anchored value, with model
    # trust decaying as we step further into the future.
    trend_closes = real_close[-TREND_WINDOW:]
    slope = float(np.polyfit(np.arange(TREND_WINDOW), trend_closes, 1)[0])
    # slope is USD per trading day (positive = uptrend, negative = downtrend)

    # ── Estimate last real trading date ──
    if isinstance(window_df.index, pd.DatetimeIndex):
        last_date = window_df.index[-1].to_pydatetime()
    else:
        last_date = datetime.today()

    # ── Recursive prediction loop ──
    predictions = []

    for step in range(1, n_days + 1):
        current_window = buffer.tail(LOOKBACK_WINDOW).copy()
        current_window.reset_index(drop=True, inplace=True)

        # ── Raw model prediction ──
        model_input = preprocess_window(current_window, scaler)
        raw_output  = model.predict(model_input, verbose=0)
        raw_close   = inverse_transform_prediction(float(raw_output[0][0]), scaler)

        # ── Momentum-weighted blend ──
        # trend_anchor: where price "should" be if the recent slope continues
        # model_weight: how much we trust the model vs the trend at this step
        #
        # step=1:  model_weight = 1 - 1*0.08 = 0.92  (mostly model)
        # step=5:  model_weight = 1 - 5*0.08 = 0.60  (balanced)
        # step=10: model_weight = max(0.30, ...)= 0.30 (mostly trend)
        trend_anchor = last_real_close + step * slope
        model_weight = max(MIN_MODEL_WEIGHT, 1.0 - step * TREND_BLEND_DECAY)
        trend_weight = 1.0 - model_weight

        pred_close = round(
            model_weight * raw_close + trend_weight * trend_anchor, 2
        )

        # ── Next trading date (skip weekends) ──
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        last_date = next_date

        # ── Synthetic OHLCV row ──
        # Open uses last_real_close (not pred_close) to avoid feeding a
        # persistent red-candle signal back into the model window.
        # High/Low estimated from real data range stats.
        synth_open   = last_real_close
        synth_high   = round(pred_close * (1 + avg_hl_pct), 4)
        synth_low    = round(pred_close * (1 - avg_hl_pct), 4)
        synth_volume = avg_volume

        synthetic_row = pd.DataFrame([{
            "Open":   synth_open,
            "High":   synth_high,
            "Low":    synth_low,
            "Close":  pred_close,
            "Volume": synth_volume,
        }])

        buffer = pd.concat([buffer, synthetic_row], ignore_index=True)

        predictions.append({
            "day":     step,
            "date":    next_date.strftime("%Y-%m-%d"),
            "close":   pred_close,
            "is_real": False,
        })

    return predictions


# ── Metadata ──────────────────────────────────────────────────────────────────

def get_model_metadata() -> dict:
    """Return model config and performance metrics for the UI dashboard."""
    return MODEL_METADATA
