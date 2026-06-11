"""
data_loader.py
--------------
Handles all data ingestion for the Tesla Stock Prediction app.

Responsibilities:
    - Fetch live Tesla OHLCV data from Yahoo Finance (yfinance).
    - Read and parse user-uploaded CSV files.
    - Validate DataFrame structure (columns, row count, nulls).
    - Return a clean, correctly ordered DataFrame ready for preprocessing.

Public API:
    fetch_live_data()          -> pd.DataFrame
    load_csv_data(file)        -> pd.DataFrame
    validate_dataframe(df)     -> (bool, str)   [is_valid, message]
    get_latest_window(df)      -> pd.DataFrame  [exactly 60 rows, 5 cols]
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


# ── Constants ────────────────────────────────────────────────────────────────

TICKER = "TSLA"
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
LOOKBACK_WINDOW = 60      # GRU was trained on 60 trading days
FETCH_DAYS_BUFFER = 120   # Fetch more to account for weekends + holidays


# ── Live Data ─────────────────────────────────────────────────────────────────

def fetch_live_data() -> pd.DataFrame:
    """
    Fetch the latest Tesla OHLCV data from Yahoo Finance.

    Fetches FETCH_DAYS_BUFFER calendar days to ensure at least
    LOOKBACK_WINDOW trading days are available after removing
    weekends and holidays.

    Returns:
        pd.DataFrame: Clean OHLCV DataFrame with DatetimeIndex,
                      containing at least 60 trading day rows.

    Raises:
        ConnectionError: If Yahoo Finance cannot be reached.
        ValueError: If fewer than 60 trading days are returned.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=FETCH_DAYS_BUFFER)

    try:
        df = yf.download(
            TICKER,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,   # suppress yfinance console output
            auto_adjust=True  # removes Adj Close; keeps OHLCV clean
        )
    except Exception as e:
        raise ConnectionError(
            f"Failed to fetch data from Yahoo Finance: {e}"
        )

    if df.empty:
        raise ValueError(
            "Yahoo Finance returned no data for TSLA. "
            "Check your internet connection or try again later."
        )

    # yfinance may return MultiIndex columns — flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only the required OHLCV columns
    df = df[REQUIRED_COLUMNS].copy()

    # Drop any rows with NaN (can occur at boundaries)
    df.dropna(inplace=True)

    if len(df) < LOOKBACK_WINDOW:
        raise ValueError(
            f"Only {len(df)} trading days fetched. "
            f"Minimum required: {LOOKBACK_WINDOW}. "
            "Try increasing FETCH_DAYS_BUFFER."
        )

    return df


# ── CSV Upload ────────────────────────────────────────────────────────────────

def load_csv_data(file) -> pd.DataFrame:
    """
    Read a user-uploaded CSV file from Streamlit's file uploader.

    Handles case-insensitive column names.
    Attempts to parse a 'Date' column as the index if present.

    Args:
        file: Streamlit UploadedFile object (file-like buffer).

    Returns:
        pd.DataFrame: Raw DataFrame parsed from CSV.

    Raises:
        ValueError: If the file cannot be parsed as a valid CSV.
    """
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Could not read the uploaded file as CSV: {e}")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    # Normalize column names to title-case (handles 'open' -> 'Open')
    df.columns = [col.strip().title() for col in df.columns]

    # If a Date column exists, set it as the index
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
        except Exception:
            # Non-critical — proceed without date index
            pass

    return df


# ── Validation ────────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate that a DataFrame meets all requirements for prediction.

    Checks:
        1. All required OHLCV columns are present.
        2. Minimum 60 rows are available.
        3. No NaN values in the required columns.
        4. All OHLCV columns are numeric.

    Args:
        df (pd.DataFrame): DataFrame to validate.

    Returns:
        tuple[bool, str]: (True, "Valid") if all checks pass,
                          (False, <error message>) on first failure.
    """
    # Check 1 — Required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, (
            f"Missing required columns: {missing_cols}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )

    # Check 2 — Minimum row count
    if len(df) < LOOKBACK_WINDOW:
        return False, (
            f"Dataset has only {len(df)} rows. "
            f"Minimum required: {LOOKBACK_WINDOW} trading days."
        )

    # Check 3 — No NaN values in OHLCV columns
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        return False, (
            f"NaN values found in columns: {cols_with_nulls.to_dict()}. "
            "Please clean the data before uploading."
        )

    # Check 4 — Numeric columns
    for col in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, (
                f"Column '{col}' contains non-numeric values. "
                "All OHLCV columns must be numeric."
            )

    return True, "Valid"


# ── Window Extraction ─────────────────────────────────────────────────────────

def get_latest_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the last LOOKBACK_WINDOW rows in the correct feature order.

    This is the final step before passing data to preprocessing.
    Enforces the exact column order the GRU model was trained on:
        [Open, High, Low, Close, Volume]

    Args:
        df (pd.DataFrame): Validated DataFrame with OHLCV columns.

    Returns:
        pd.DataFrame: Shape (60, 5) with correct column order.
    """
    window = df[REQUIRED_COLUMNS].tail(LOOKBACK_WINDOW).copy()
    window.reset_index(drop=True, inplace=True)
    return window

