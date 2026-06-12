"""
data_loader.py
--------------
Handles all data ingestion for the Tesla Stock Prediction app.

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

REQUIRED_COLUMNS = FEATURE_COLUMNS

# ── Browser headers ───────────────────────────────────────────────────────────
# Passed to yfinance and requests to avoid bot-detection rejections.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Stooq ticker symbol for TSLA (US-listed)
_STOOQ_SYMBOL = "tsla.us"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise any raw DataFrame into a clean OHLCV DataFrame.

    Steps:
        1. Flatten MultiIndex columns (some yfinance versions return these).
        2. Keep only OHLCV columns in correct order.
        3. Drop NaN rows.
        4. Sort index ascending.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Unexpected columns. Missing: {missing}. Got: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df


def _fetch_via_yfinance_ticker(session) -> pd.DataFrame:
    """Method 1 — yf.Ticker().history() with browser session."""
    ticker = yf.Ticker(TICKER, session=session)
    raw = ticker.history(period="6mo", auto_adjust=True, actions=False)
    if raw is None or raw.empty:
        raise ValueError("Ticker.history() returned empty DataFrame.")
    return _clean_ohlcv(raw)


def _fetch_via_yfinance_download(session) -> pd.DataFrame:
    """Method 2 — yf.download() with browser session."""
    raw = yf.download(
        TICKER,
        period="6mo",
        auto_adjust=True,
        progress=False,
        threads=False,
        session=session,
    )
    if raw is None or raw.empty:
        raise ValueError("yf.download() returned empty DataFrame.")
    return _clean_ohlcv(raw)


def _fetch_via_stooq() -> pd.DataFrame:
    """
    Method 3 — Fetch OHLCV data from Stooq (stooq.com).

    WHY STOOQ:
    ----------
    Yahoo Finance aggressively blocks automated access in 2025,
    requiring session cookies + crumb tokens that break in many
    environments. Stooq is a free financial data provider that
    serves historical OHLCV as plain CSV with no API key, no
    authentication, and no bot detection. It is the data source
    used internally by pandas_datareader.

    URL format:
        https://stooq.com/q/d/l/?s=tsla.us&i=d
        s = ticker symbol (tsla.us for TSLA on US exchange)
        i = interval (d = daily)

    Returns data in DESCENDING date order — we sort ascending.
    Columns returned: Date, Open, High, Low, Close, Volume
    """
    url = f"https://stooq.com/q/d/l/?s={_STOOQ_SYMBOL}&i=d"

    session = requests.Session()
    session.headers.update(_HEADERS)

    response = session.get(url, timeout=15)
    response.raise_for_status()

    # Guard: Stooq returns "No data" as plain text on bad symbols
    if "No data" in response.text or len(response.text) < 50:
        raise ValueError(
            f"Stooq returned no data for '{_STOOQ_SYMBOL}'. "
            "Check the symbol in config.py."
        )

    df = pd.read_csv(
        io.StringIO(response.text),
        parse_dates=["Date"],
        index_col="Date",
    )

    # Stooq returns columns: Open, High, Low, Close, Volume — perfect match
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_live_data() -> pd.DataFrame:
    """
    Fetch the latest Tesla OHLCV data using a 3-method fallback chain.

    METHOD CHAIN:
    -------------
    Method 1 — yf.Ticker().history() with browser session
        Most convenient when yfinance works. Fails when Yahoo Finance
        blocks the crumb request.

    Method 2 — yf.download() with browser session
        Different yfinance code path. Fails for the same reason as
        Method 1 when Yahoo Finance rejects the session.

    Method 3 — Stooq direct CSV (most reliable)
        Completely independent of Yahoo Finance. Stooq serves free
        historical OHLCV data as plain CSV with no authentication.
        This is the guaranteed fallback.

    Returns:
        pd.DataFrame: Clean OHLCV DataFrame, DatetimeIndex,
                      ascending order, >= 60 trading rows.

    Raises:
        ConnectionError: If all three methods fail.
        ValueError: If fewer than 60 trading days are returned.
    """
    df     = None
    errors = []

    # Shared browser session for yfinance methods
    session = requests.Session()
    session.headers.update(_HEADERS)

    # ── Method 1 ──
    try:
        df = _fetch_via_yfinance_ticker(session)
    except Exception as e:
        errors.append(f"[1] yf.Ticker: {e}")

    # ── Method 2 ──
    if df is None or df.empty:
        try:
            df = _fetch_via_yfinance_download(session)
        except Exception as e:
            errors.append(f"[2] yf.download: {e}")

    # ── Method 3: Stooq ──
    if df is None or df.empty:
        try:
            raw = _fetch_via_stooq()
            df  = _clean_ohlcv(raw)
        except Exception as e:
            errors.append(f"[3] Stooq: {e}")

    # ── All failed ──
    if df is None or df.empty:
        raise ConnectionError(
            "All data sources failed.\n" + "\n".join(errors)
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

    Normalises column names to title-case ('open' → 'Open').
    Parses 'Date' column as DatetimeIndex if present.
    """
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Could not read the uploaded file as CSV: {e}")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    df.columns = [col.strip().title() for col in df.columns]

    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
        except Exception:
            pass

    return df


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate that a DataFrame meets all prediction requirements.

    Checks (in order):
        1. All OHLCV columns present.
        2. Minimum 60 rows.
        3. No NaN values.
        4. All columns numeric.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return False, (
            f"Missing columns: {missing_cols}. Expected: {REQUIRED_COLUMNS}"
        )

    if len(df) < LOOKBACK_WINDOW:
        return False, (
            f"Only {len(df)} rows. Minimum required: {LOOKBACK_WINDOW}."
        )

    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        return False, (
            f"NaN values in: {cols_with_nulls.to_dict()}. "
            "Please clean the data before uploading."
        )

    for col in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, (
                f"Column '{col}' is not numeric."
            )

    return True, "Valid"


def get_latest_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the last 60 rows in the correct OHLCV column order.

    Returns:
        pd.DataFrame: Shape (60, 5), reset integer index.
    """
    window = df[REQUIRED_COLUMNS].tail(LOOKBACK_WINDOW).copy()
    window.reset_index(drop=True, inplace=True)
    return window
