"""
app.py
------
Main Streamlit entry point for the Tesla Stock Price Prediction app.

Responsibility:
    UI orchestration ONLY — no data logic, no model logic.

Structure:
    1. Page config + CSS
    2. Cached loaders (model + scaler)
    3. Header
    4. Model information dashboard
    5. Prediction tabs (Live / CSV), each with horizon selector

Run:
    streamlit run app.py
"""

import os
import streamlit as st

from utils.data_loader import (
    load_csv_data,
    validate_dataframe, get_latest_window,
)
from utils.preprocessing import load_scaler, preprocess_window
from utils.predictor import (
    load_model, predict,
    predict_multi_step, get_model_metadata,
)
from utils.visualizer import (
    plot_closing_price, plot_multi_step_forecast,
    render_model_metrics, render_multi_step_output,
    render_data_preview,
)
from utils.config import PREDICTION_HORIZONS


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "tesla_gru_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "tesla_scaler.pkl")


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tesla Stock Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .stApp { background-color: #0f0f0f; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background-color: #1a1a1a;
        border-radius: 12px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 8px 24px;
        color: #888; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E31937 !important; color: white !important;
    }
    [data-testid="stMetric"] {
        background-color: #1a1a1a; border: 1px solid #2a2a2a;
        border-radius: 12px; padding: 16px;
    }
    [data-testid="stMetricLabel"] { color: #888 !important; }
    [data-testid="stMetricValue"] { color: #fff !important; font-size: 1.4rem !important; }
    .stButton > button {
        background-color: #E31937; color: white; border: none;
        border-radius: 8px; padding: 10px 28px;
        font-weight: 600; font-size: 15px; width: 100%;
    }
    .stButton > button:hover { background-color: #b5132b; }
    hr { border-color: #2a2a2a !important; }
    h2, h3 { color: #ffffff !important; }

    /* Radio group for horizon selector */
    div[data-testid="stRadio"] > label {
        color: #aaa !important; font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached Loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading GRU model...")
def get_model():
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner="Loading scaler...")
def get_scaler():
    return load_scaler(SCALER_PATH)


# ── Prediction Pipeline Helper ────────────────────────────────────────────────

def run_prediction_pipeline(df, scaler, model, n_days: int) -> list[dict]:
    """
    Run either single-step or multi-step prediction.

    Always returns a list[dict] for uniform handling in both tabs,
    even for n_days=1 (list of one item).

    Args:
        df     : Validated OHLCV DataFrame (>= 60 rows).
        scaler : Fitted MinMaxScaler.
        model  : Loaded GRU model.
        n_days : Forecast horizon (1, 5, or 10).

    Returns:
        list[dict]: Each dict has keys: day, date, close, is_real.
    """
    window = get_latest_window(df)   # enforces (60, 5) + column order

    if n_days == 1:
        # Single step: use preprocess_window + predict directly
        model_input = preprocess_window(window, scaler)
        price = predict(model, model_input, scaler)
        from datetime import datetime, timedelta
        next_date = datetime.today()
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        return [{"day": 1, "date": next_date.strftime("%Y-%m-%d"),
                 "close": price, "is_real": False}]
    else:
        return predict_multi_step(model, window, scaler, n_days)


# ── Startup: load model + scaler ──────────────────────────────────────────────

model_error = scaler_error = None
model = scaler = None

try:
    model = get_model()
except Exception as e:
    model_error = str(e)

try:
    scaler = get_scaler()
except Exception as e:
    scaler_error = str(e)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 8px 0 24px 0;">
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:36px;">🚗</span>
        <div>
            <h1 style="margin:0; color:#E31937; font-size:2rem;
                       font-weight:700; letter-spacing:-0.5px;">
                Tesla Stock Price Prediction
            </h1>
            <p style="margin:4px 0 0 0; color:#888; font-size:14px;">
                GRU Deep Learning &nbsp;·&nbsp; 60-Day Lookback &nbsp;·&nbsp;
                OHLCV Features &nbsp;·&nbsp; Forecast up to 10 Days
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Error banners ──
if model_error:
    st.error(f"**Model file missing.** Place `tesla_gru_model.keras` in `model/`.\n\n`{model_error}`")
if scaler_error:
    st.error(f"**Scaler file missing.** Place `tesla_scaler.pkl` in `model/`.\n\n`{scaler_error}`")


# ── Model Info Dashboard ──────────────────────────────────────────────────────

with st.expander("Model Information Dashboard", expanded=True):
    meta = get_model_metadata()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Model Type**"); st.markdown(f"`{meta['model_name']}`")
    with c2:
        st.markdown("**Input Window**"); st.markdown("`60 Trading Days`")
    with c3:
        st.markdown("**Features**"); st.markdown("`Open · High · Low · Close · Volume`")
    with c4:
        st.markdown("**Target**"); st.markdown("`Next Day Closing Price`")
    st.markdown("##### Performance Metrics")
    render_model_metrics(meta)

st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_live, tab_csv = st.tabs([
    "📡  Live Tesla Data  (Yahoo Finance)",
    "📂  Upload CSV",
])


# ════════════════════════════════════════════════════════════════════════════
# SHARED PREDICTION SECTION — rendered identically in both tabs
# ════════════════════════════════════════════════════════════════════════════

def render_prediction_section(tab_key: str, source: str):
    """
    Render the full prediction UI for one tab.

    tab_key : 'live' or 'csv'  — used to namespace session_state keys.
    source  : 'Yahoo Finance' or 'Uploaded CSV' — shown in cards + charts.
    """
    df_key         = f"{tab_key}_df"
    pred_key       = f"{tab_key}_predictions"
    err_key        = f"{tab_key}_predict_error"
    horizon_key    = f"{tab_key}_horizon"

    df = st.session_state.get(df_key)
    if df is None:
        return   # nothing fetched/uploaded yet — nothing to show

    # ── Data preview + historical chart ──
    st.markdown("#### Last 60 Trading Days")
    render_data_preview(df.tail(60), source=source)

    st.markdown("#### Closing Price Trend")
    st.plotly_chart(
        plot_closing_price(df.tail(60), source=source),
        use_container_width=True,
    )

    st.divider()

    # ── Prediction horizon selector ──
    st.markdown("#### Prediction Horizon")
    horizon_label = st.radio(
        "Select forecast window:",
        options=list(PREDICTION_HORIZONS.keys()),
        horizontal=True,
        key=horizon_key,
        help=(
            "Next Day: single prediction using 60 real trading days.  \n"
            "5 / 10 Days: recursive forecast — error compounds with each step."
        ),
    )
    n_days = PREDICTION_HORIZONS[horizon_label]

    # Context note for multi-step
    if n_days > 1:
        st.info(
            f"**{horizon_label} forecast** uses recursive prediction: each predicted "
            f"day is fed back as input for the next. "
            f"Day 1 uses 60 real rows. By day {n_days}, {n_days - 1} synthetic "
            f"rows are in the window. Confidence decreases with each step."
        )

    # ── Predict button ──
    predict_disabled = (model is None or scaler is None)
    if predict_disabled:
        st.warning("Prediction unavailable — model or scaler file missing.")
        return

    btn_label = (
        "Predict Next Day Closing Price"
        if n_days == 1
        else f"Predict Next {n_days} Days"
    )

    if st.button(btn_label, key=f"btn_predict_{tab_key}"):
        spinner_msg = (
            "Running GRU inference..."
            if n_days == 1
            else f"Running recursive GRU forecast for {n_days} days..."
        )
        with st.spinner(spinner_msg):
            try:
                predictions = run_prediction_pipeline(df, scaler, model, n_days)
                st.session_state[pred_key]  = predictions
                st.session_state[err_key]   = None
            except Exception as e:
                st.session_state[err_key]   = str(e)
                st.session_state[pred_key]  = None

    # ── Error ──
    if st.session_state.get(err_key):
        st.error(f"Prediction failed: {st.session_state[err_key]}")

    # ── Results ──
    predictions = st.session_state.get(pred_key)
    if predictions:
        # Show forecast chart for multi-step, prediction card for single
        if len(predictions) > 1:
            st.markdown("#### Forecast Chart")
            st.plotly_chart(
                plot_multi_step_forecast(df.tail(60), predictions),
                use_container_width=True,
            )

        st.markdown("#### Prediction Result")
        render_multi_step_output(predictions, source=source)


# ════════════════════════════════════════════════════════════════════════════
# TAB A — Live Tesla Data (placeholder)
# ════════════════════════════════════════════════════════════════════════════

with tab_live:
    st.markdown("### Live Tesla Stock Prediction")

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #2a2a2a;
            border-radius: 16px;
            padding: 40px 32px;
            text-align: center;
            margin: 16px 0;
        ">
            <div style="font-size: 48px; margin-bottom: 16px;">📡</div>
            <h3 style="color: #ffffff; margin: 0 0 12px 0;">
                Live Data Feed — Coming Soon
            </h3>
            <p style="color: #888; font-size: 15px; max-width: 480px;
                       margin: 0 auto 20px auto; line-height: 1.6;">
                Automatic fetching of real-time Tesla OHLCV data from a
                live market data source will be integrated in a future release.
            </p>
            <div style="
                display: inline-block;
                background: #1e1e1e;
                border: 1px solid #333;
                border-radius: 10px;
                padding: 16px 28px;
                text-align: left;
                margin-top: 8px;
            ">
                <p style="color: #888; font-size: 12px; text-transform: uppercase;
                           letter-spacing: 1px; margin: 0 0 10px 0;">
                    Planned upgrades
                </p>
                <p style="color: #ccc; font-size: 13px; margin: 4px 0;">
                    ✦ &nbsp; Real-time OHLCV data via a stable market API
                </p>
                <p style="color: #ccc; font-size: 13px; margin: 4px 0;">
                    ✦ &nbsp; Auto-refresh on market open / close
                </p>
                <p style="color: #ccc; font-size: 13px; margin: 4px 0;">
                    ✦ &nbsp; One-click prediction without manual CSV upload
                </p>
            </div>
            <p style="color: #555; font-size: 12px; margin: 24px 0 0 0;">
                In the meantime, use the
                <strong style="color: #888;">Upload CSV</strong> tab
                to run predictions on your own historical data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB B — CSV Upload
# ════════════════════════════════════════════════════════════════════════════

with tab_csv:
    st.markdown("### Custom CSV Prediction")
    st.markdown(
        "Upload historical Tesla OHLCV data. "
        "The model will forecast up to **10 trading days** ahead."
    )

    st.info(
        "**CSV Requirements**\n"
        "- Columns: `Open`, `High`, `Low`, `Close`, `Volume`  (case-insensitive)\n"
        "- Minimum 60 rows\n"
        "- Optional `Date` column used as index"
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"], key="csv_uploader",
    )

    if uploaded_file is not None:
        with st.spinner("Reading and validating CSV..."):
            try:
                csv_df = load_csv_data(uploaded_file)
                is_valid, msg = validate_dataframe(csv_df)
                if not is_valid:
                    st.error(f"Validation failed: {msg}")
                else:
                    st.success(f"Loaded — {len(csv_df)} rows · {len(csv_df.columns)} columns")
                    st.session_state["csv_df"]           = csv_df
                    st.session_state["csv_predictions"]  = None
                    st.session_state["csv_predict_error"]= None
            except Exception as e:
                st.error(f"Could not read file: {e}")
                st.session_state["csv_df"] = None

    render_prediction_section(tab_key="csv", source="Uploaded CSV")


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<p style='text-align:center; color:#444; font-size:12px;'>"
    "Tesla Stock Price Prediction &nbsp;·&nbsp; GRU Deep Learning &nbsp;·&nbsp;"
    " For educational purposes only &nbsp;·&nbsp; Not financial advice"
    "</p>",
    unsafe_allow_html=True,
)
