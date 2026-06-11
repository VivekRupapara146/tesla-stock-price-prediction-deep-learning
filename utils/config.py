"""
config.py
---------
Central configuration for the Tesla Stock Prediction project.

All magic numbers, constants, and shared settings live here.
Every other module imports from this file — never hardcode these values elsewhere.
"""

# ── Ticker ────────────────────────────────────────────────────────────────────
TICKER = "TSLA"

# ── Feature Engineering ───────────────────────────────────────────────────────
# Order is fixed and must match the training pipeline exactly.
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Position of 'Close' in FEATURE_COLUMNS — used for inverse transform dummy array
CLOSE_COL_INDEX = 3

# Total number of OHLCV features
N_FEATURES = len(FEATURE_COLUMNS)  # 5

# ── Sequence Window ───────────────────────────────────────────────────────────
# Must match the lookback window used during GRU training
LOOKBACK_WINDOW = 60

# ── Data Fetching ─────────────────────────────────────────────────────────────
# Fetch buffer in calendar days — must be larger than LOOKBACK_WINDOW
# to account for weekends + holidays (60 trading days ≈ 90 calendar days)
FETCH_DAYS_BUFFER = 120

# ── Model Files ───────────────────────────────────────────────────────────────
MODEL_FILENAME  = "tesla_gru_model.keras"
SCALER_FILENAME = "tesla_scaler.pkl"

# ── Multi-Step Forecasting ────────────────────────────────────────────────────
# Available prediction horizons shown in the UI selector.
PREDICTION_HORIZONS = {
    "Next Day (1)":    1,
    "Up to 5 Days":    5,
    "Up to 10 Days":  10,
}

# Number of recent days used to compute rolling Volume average for
# synthetic rows during recursive forecasting.
VOLUME_ROLLING_WINDOW = 20

# Number of recent days used to compute average High-Low daily range
# as a percentage of Close — used to synthesize High and Low for
# predicted days.
RANGE_ROLLING_WINDOW = 20
