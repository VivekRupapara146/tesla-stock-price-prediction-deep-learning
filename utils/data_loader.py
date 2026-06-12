"""
data_loader.py
--------------
Handles CSV data ingestion for the Tesla Stock Prediction app.

Live data fetching has been removed. This module handles user-uploaded
CSV files only. Live fetching can be re-integrated in a future release.

Public API:
    load_csv_data(file)        -> pd.DataFrame
    validate_dataframe(df)     -> tuple[bool, str]
    get_latest_window(df)      -> pd.DataFrame  [exactly 60 rows, 5 cols]
"""

import pandas as pd

from utils.config import FEATURE_COLUMNS, LOOKBACK_WINDOW

REQUIRED_COLUMNS = FEATURE_COLUMNS


def load_csv_data(file) -> pd.DataFrame:
    """
    Read a user-uploaded CSV file from Streamlit's file uploader.

    - Normalises column names to title-case ('open' → 'Open').
    - Parses 'Date' column as DatetimeIndex if present.
    - Sorts index ascending to guarantee chronological order.

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

    # Normalise column names — handles 'open' → 'Open', ' Close ' → 'Close'
    df.columns = [col.strip().title() for col in df.columns]

    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
        except Exception:
            pass   # non-critical — proceed without date index

    return df


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate that a DataFrame meets all prediction requirements.

    Checks (in order):
        1. All required OHLCV columns are present.
        2. Minimum 60 rows available.
        3. No NaN values in OHLCV columns.
        4. All OHLCV columns are numeric.

    Args:
        df (pd.DataFrame): DataFrame to validate.

    Returns:
        tuple[bool, str]: (True, "Valid") on success,
                          (False, <error message>) on first failure.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return False, (
            f"Missing columns: {missing_cols}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )

    if len(df) < LOOKBACK_WINDOW:
        return False, (
            f"Only {len(df)} rows. "
            f"Minimum required: {LOOKBACK_WINDOW} trading days."
        )

    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        return False, (
            f"NaN values found in: {cols_with_nulls.to_dict()}. "
            "Please clean the data before uploading."
        )

    for col in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, (
                f"Column '{col}' contains non-numeric values."
            )

    return True, "Valid"


def get_latest_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the last 60 rows in the correct OHLCV column order.

    Enforces the feature order [Open, High, Low, Close, Volume] that
    the GRU model was trained on, regardless of CSV column order.

    Args:
        df (pd.DataFrame): Validated DataFrame with OHLCV columns.

    Returns:
        pd.DataFrame: Shape (60, 5), reset integer index.
    """
    window = df[REQUIRED_COLUMNS].tail(LOOKBACK_WINDOW).copy()
    window.reset_index(drop=True, inplace=True)
    return window
