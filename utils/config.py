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

# ── Momentum Correction (anti-drift) ─────────────────────────────────────────
# Recursive forecasting causes predictions to drift toward the training mean
# because each synthetic row reinforces the model's last signal.
# These constants control a momentum blend that anchors predictions to the
# recent real-data trend as uncertainty grows with each step.

# Number of recent real Close prices used to compute the linear trend slope.
TREND_WINDOW = 20

# How much model trust decays per forecast step (0.0 = no decay, 1.0 = instant).
# At 0.08: day-1 weight = 0.92, day-5 weight = 0.60, day-10 weight = 0.30.
TREND_BLEND_DECAY = 0.08

# Minimum model weight — model always has at least this influence even at day 10.
# Prevents the forecast from becoming a pure linear extrapolation.
MIN_MODEL_WEIGHT = 0.30
