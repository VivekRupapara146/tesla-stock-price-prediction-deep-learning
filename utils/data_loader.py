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

import io
import requests
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
    Internal helper — standardise any raw DataFrame into a
    clean OHLCV DataFrame ready for preprocessing.

    Steps:
        1. Flatten MultiIndex columns (some yfinance versions return these).
        2. Keep only OHLCV columns in correct order.
        3. Drop NaN rows.
        4. Sort index ascending (yfinance order is not guaranteed).
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Unexpected columns from data source. Missing: {missing}. "
            f"Got: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df


# Browser headers — required by Yahoo Finance since 2024 to avoid 401/empty responses
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _fetch_via_direct_csv() -> pd.DataFrame:
    """
    Fetch OHLCV data by downloading Yahoo Finance CSV directly.

    WHY THIS EXISTS:
    Yahoo Finance changed its API in 2024 to require a crumb/cookie
    for JSON endpoints. yfinance tries to obtain this automatically
    but frequently fails, returning an empty JSON body — causing:
        JSONDecodeError: Expecting value: line 1 column 1 (char 0)

    The v7/finance/download CSV endpoint still works with just a
    proper User-Agent header and does not require a crumb cookie.
    This makes it the most reliable fallback for live data.
    """
    end_ts   = int(datetime.today().timestamp())
    start_ts = int((datetime.today() - timedelta(days=FETCH_DAYS_BUFFER)).timestamp())

    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{TICKER}"
        f"?period1={start_ts}&period2={end_ts}"
        f"&interval=1d&events=history&includeAdjustedClose=true"
    )

    session = requests.Session()
    session.headers.update(_HEADERS)

    response = session.get(url, timeout=15)
    response.raise_for_status()

    df = pd.read_csv(
        io.StringIO(response.text),
        parse_dates=["Date"],
        index_col="Date",
    )

    # CSV columns: Open, High, Low, Close, Adj Close, Volume
    # Rename Adj Close → Close (adjusted prices preferred), drop original Close
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Close"], errors="ignore")
        df = df.rename(columns={"Adj Close": "Close"})

    return df


def fetch_live_data() -> pd.DataFrame:
    """
    Fetch the latest Tesla OHLCV data from Yahoo Finance.

    THREE-METHOD FALLBACK STRATEGY:
    --------------------------------
    Yahoo Finance changed its API in 2024 to require session cookies
    (crumb tokens) for JSON endpoints. Both yfinance methods attempt
    to obtain this token automatically but fail in many environments,
    returning an empty response body that causes:
        JSONDecodeError: Expecting value: line 1 column 1 (char 0)

    Method 1 — yf.Ticker().history() with browser session
        Uses a custom requests.Session with real browser headers so
        Yahoo Finance accepts the crumb request.

    Method 2 — yf.download() with browser session
        Same session approach, different yfinance code path.

    Method 3 — Direct CSV download (most reliable)
        Hits the v7/finance/download endpoint directly with browser
        headers. Does not require a crumb cookie. Pure requests + pandas.

    Returns:
        pd.DataFrame: Clean OHLCV DataFrame with DatetimeIndex,
                      sorted ascending, containing >= 60 trading rows.

    Raises:
        ConnectionError: If all three methods fail.
        ValueError: If fewer than 60 trading days are returned.
    """
    df     = None
    errors = []

    # Shared browser session — passed to yfinance to avoid crumb failures
    session = requests.Session()
    session.headers.update(_HEADERS)

    # ── Method 1: Ticker.history() with browser session ──
    try:
        ticker = yf.Ticker(TICKER, session=session)
        raw = ticker.history(
            period="6mo",
            auto_adjust=True,
            actions=False,
        )
        if raw is not None and not raw.empty:
            df = _clean_ohlcv(raw)
    except Exception as e:
        errors.append(f"[1] Ticker.history(): {e}")

    # ── Method 2: yf.download() with browser session ──
    if df is None or df.empty:
        try:
            raw = yf.download(
                TICKER,
                period="6mo",
                auto_adjust=True,
                progress=False,
                threads=False,
                session=session,
            )
            if raw is not None and not raw.empty:
                df = _clean_ohlcv(raw)
        except Exception as e:
            errors.append(f"[2] yf.download(): {e}")

    # ── Method 3: Direct CSV download ──
    if df is None or df.empty:
        try:
            raw = _fetch_via_direct_csv()
            if raw is not None and not raw.empty:
                df = _clean_ohlcv(raw)
        except Exception as e:
            errors.append(f"[3] Direct CSV: {e}")

    # ── All methods failed ──
    if df is None or df.empty:
        raise ConnectionError(
            "All three fetch methods failed for TSLA. "
            + " | ".join(errors)
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
