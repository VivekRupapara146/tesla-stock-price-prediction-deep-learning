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
