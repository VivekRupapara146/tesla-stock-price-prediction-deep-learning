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
    validate_dataframe(df)     -> (bool, str)
    get_latest_window(df)      -> pd.DataFrame  [exactly 60 rows, 5 cols]
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from utils.config import (
    TICKER, FEATURE_COLUMNS, LOOKBACK_WINDOW, FETCH_DAYS_BUFFER
)

# Alias for readability — validation checks for these columns
REQUIRED_COLUMNS = FEATURE_COLUMNS


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal helper — standardise any raw yfinance DataFrame into a
    clean OHLCV DataFrame ready for preprocessing.

    Steps:
        1. Flatten MultiIndex columns (newer yfinance versions return these).
        2. Keep only OHLCV columns.
        3. Drop NaN rows.
        4. Sort index ascending (yfinance order is not guaranteed).
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"yfinance returned unexpected columns. Missing: {missing}. "
            f"Got: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_live_data() -> pd.DataFrame:
    """
    Fetch the latest Tesla OHLCV data from Yahoo Finance.

    WHY TWO METHODS?
    ----------------
    yf.download() suffers from a known URL-template bug in some versions:
        YFTzMissingError: $%ticker%: possibly delisted; No timezone found
    This happens because the ticker variable is never substituted into the
    Yahoo Finance API URL, returning an empty JSON response.

    Ticker.history() uses a different internal code path but first calls
    .info to resolve the timezone — an extra network request that can also
    fail independently.

    Solution: try Ticker.history() first. If it fails for any reason,
    fall back to yf.download() with period= syntax (more stable than
    start/end dates in some yfinance versions). Both paths clean through
    the same _clean_ohlcv() helper.

    Returns:
        pd.DataFrame: Clean OHLCV DataFrame with DatetimeIndex,
                      sorted ascending, containing >= 60 trading rows.

    Raises:
        ConnectionError: If both fetch methods fail.
        ValueError: If fewer than 60 trading days are returned.
    """
    df       = None
    errors   = []

    # ── Method 1: Ticker.history() ──
    # Preferred — avoids URL-template substitution bug in yf.download()
    try:
        ticker = yf.Ticker(TICKER)
        raw = ticker.history(
            period="6mo",
            auto_adjust=True,
            actions=False,      # drops Dividends + Stock Splits columns
        )
        if raw is not None and not raw.empty:
            df = _clean_ohlcv(raw)
    except Exception as e:
        errors.append(f"Ticker.history() failed: {e}")

    # ── Method 2: yf.download() with period= ──
    # Fallback — period= syntax is more stable than start/end in some versions
    if df is None or df.empty:
        try:
            raw = yf.download(
                TICKER,
                period="6mo",
                auto_adjust=True,
                progress=False,
                threads=False,   # single-threaded avoids some race conditions
            )
            if raw is not None and not raw.empty:
                df = _clean_ohlcv(raw)
        except Exception as e:
            errors.append(f"yf.download() failed: {e}")

    # ── Both methods failed ──
    if df is None or df.empty:
        error_detail = " | ".join(errors)
        raise ConnectionError(
            "Failed to fetch TSLA data from Yahoo Finance. "
            "Tried Ticker.history() and yf.download(). "
            f"Errors: {error_detail}. "
            "Fix: pip install --upgrade yfinance"
        )

    if len(df) < LOOKBACK_WINDOW:
        raise ValueError(
            f"Only {len(df)} trading days fetched. "
            f"Minimum required: {LOOKBACK_WINDOW}."
        )

    return df


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

    # Normalize to title-case: 'open' -> 'Open', 'HIGH' -> 'High'
    df.columns = [col.strip().title() for col in df.columns]

    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)   # FIX: same sort guarantee for CSVs
        except Exception:
            pass

    return df


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate that a DataFrame meets all requirements for prediction.

    Checks (in order of severity):
        1. All required OHLCV columns are present.
        2. Minimum 60 rows are available.
        3. No NaN values in the required columns.
        4. All OHLCV columns are numeric.

    Returns:
        tuple[bool, str]: (True, "Valid") or (False, <error message>).
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, (
            f"Missing required columns: {missing_cols}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )

    if len(df) < LOOKBACK_WINDOW:
        return False, (
            f"Dataset has only {len(df)} rows. "
            f"Minimum required: {LOOKBACK_WINDOW} trading days."
        )

    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        return False, (
            f"NaN values found in columns: {cols_with_nulls.to_dict()}. "
            "Please clean the data before uploading."
        )

    for col in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, (
                f"Column '{col}' contains non-numeric values. "
                "All OHLCV columns must be numeric."
            )

    return True, "Valid"


def get_latest_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the last LOOKBACK_WINDOW rows in the correct feature order.

    Column order [Open, High, Low, Close, Volume] is enforced here —
    this is the contract between data ingestion and preprocessing.

    Returns:
        pd.DataFrame: Shape (60, 5), correct column order, reset index.
    """
    window = df[REQUIRED_COLUMNS].tail(LOOKBACK_WINDOW).copy()
    window.reset_index(drop=True, inplace=True)
    return window
