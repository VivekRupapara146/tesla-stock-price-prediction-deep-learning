"""
app.py
------
Main Streamlit entry point for the Tesla Stock Price Prediction app.

Responsibility:
    UI orchestration ONLY — no data logic, no model logic.
    All heavy lifting is delegated to the utils/ modules.

Structure:
    1. Page configuration and custom CSS
    2. Cached resource loaders (model + scaler)
    3. Header and branding
    4. Model information dashboard
    5. Prediction tabs:
       - Tab A: Live Tesla data via Yahoo Finance
       - Tab B: User-uploaded CSV

Run:
    streamlit run app.py
"""

import os
import streamlit as st

from utils.data_loader import (
    fetch_live_data,
    load_csv_data,
    validate_dataframe,
    get_latest_window,
)
from utils.preprocessing import load_scaler, preprocess_window
from utils.predictor import load_model, predict, get_model_metadata
from utils.visualizer import (
    plot_closing_price,
    render_model_metrics,
    render_prediction_output,
    render_data_preview,
)


# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "tesla_gru_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "tesla_scaler.pkl")


# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tesla Stock Predictor",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Hide Streamlit default header/footer */
    #MainMenu, footer, header { visibility: hidden; }

    /* Main background */
    .stApp { background-color: #0f0f0f; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1a1a;
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 24px;
        color: #888;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E31937 !important;
        color: white !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] { color: #888 !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.4rem !important; }

    /* Buttons */
    .stButton > button {
        background-color: #E31937;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 15px;
        width: 100%;
        transition: background 0.2s ease;
    }
    .stButton > button:hover { background-color: #b5132b; }

    /* Info boxes */
    .stInfo {
        background-color: #1a1a2e;
        border-left: 4px solid #E31937;
        border-radius: 4px;
    }

    /* Divider */
    hr { border-color: #2a2a2a !important; }

    /* Section headers */
    h2, h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached Resource Loaders ───────────────────────────────────────────────────
# @st.cache_resource: loads ONCE per session, survives reruns.
# Without this, model + scaler reload on every button click (~3s latency each).

@st.cache_resource(show_spinner="Loading GRU model...")
def get_model():
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner="Loading scaler...")
def get_scaler():
    return load_scaler(SCALER_PATH)


# ── Helper: run full prediction pipeline ─────────────────────────────────────

def run_prediction(df, scaler, model, source: str) -> float:
    """
    Execute the full prediction pipeline on a validated DataFrame.

    Args:
        df     : Raw OHLCV DataFrame (>=60 rows).
        scaler : Loaded MinMaxScaler.
        model  : Loaded GRU model.
        source : 'Yahoo Finance' or 'Uploaded CSV'.

    Returns:
        float: Predicted next-day Tesla closing price in USD.
    """
    window      = get_latest_window(df)        # (60, 5) correct order
    model_input = preprocess_window(window, scaler)  # (1, 60, 5) scaled
    price       = predict(model, model_input, scaler) # float USD
    return price


# ── App Startup: load model and scaler once ───────────────────────────────────

model_load_error  = None
scaler_load_error = None
model  = None
scaler = None

try:
    model = get_model()
except Exception as e:
    model_load_error = str(e)

try:
    scaler = get_scaler()
except Exception as e:
    scaler_load_error = str(e)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 8px 0 24px 0;">
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:36px;">🚗</span>
        <div>
            <h1 style="margin:0; color:#E31937; font-size:2rem; font-weight:700; letter-spacing:-0.5px;">
                Tesla Stock Price Prediction
            </h1>
            <p style="margin:4px 0 0 0; color:#888; font-size:14px;">
                GRU Deep Learning Model &nbsp;·&nbsp; 60-Day Lookback &nbsp;·&nbsp;
                OHLCV Features &nbsp;·&nbsp; R² = 0.9603
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ── Model Load Error Banner ───────────────────────────────────────────────────
# Show a prominent error if model/scaler files are missing.
# App still renders — user can read instructions without crashing.

if model_load_error:
    st.error(
        f"**Model file not found.**\n\n"
        f"Place `tesla_gru_model.keras` inside the `model/` directory.\n\n"
        f"Details: `{model_load_error}`"
    )

if scaler_load_error:
    st.error(
        f"**Scaler file not found.**\n\n"
        f"Place `tesla_scaler.pkl` inside the `model/` directory.\n\n"
        f"Details: `{scaler_load_error}`"
    )


# ── Model Information Dashboard ───────────────────────────────────────────────

with st.expander("Model Information Dashboard", expanded=True):

    meta = get_model_metadata()

    # Architecture info row
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.markdown("**Model Type**")
        st.markdown(f"`{meta['model_name']}`")
    with info_col2:
        st.markdown("**Input Window**")
        st.markdown(f"`{meta['input_window']} Trading Days`")
    with info_col3:
        st.markdown("**Features**")
        st.markdown("`Open · High · Low · Close · Volume`")
    with info_col4:
        st.markdown("**Target**")
        st.markdown(f"`{meta['target']}`")

    st.markdown("##### Performance Metrics")
    render_model_metrics(meta)


st.divider()


# ── Prediction Tabs ───────────────────────────────────────────────────────────

tab_live, tab_csv = st.tabs([
    "📡  Live Tesla Data  (Yahoo Finance)",
    "📂  Upload CSV",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB A — Live Tesla Data
# ════════════════════════════════════════════════════════════════════════════

with tab_live:

    st.markdown("### Live Tesla Stock Prediction")
    st.markdown(
        "Fetches the latest Tesla OHLCV data from Yahoo Finance and predicts "
        "the **next trading day's closing price** using the last 60 trading days."
    )

    # ── Fetch data on button click ──
    if st.button("Fetch Latest Tesla Data", key="btn_fetch"):

        with st.spinner("Fetching data from Yahoo Finance..."):
            try:
                live_df = fetch_live_data()
                # Store in session state so data persists across reruns
                st.session_state["live_df"] = live_df
                st.session_state["live_fetch_error"] = None
            except Exception as e:
                st.session_state["live_fetch_error"] = str(e)
                st.session_state["live_df"] = None

    # ── Show fetch error if any ──
    if st.session_state.get("live_fetch_error"):
        st.error(f"Data fetch failed: {st.session_state['live_fetch_error']}")

    # ── Show data + prediction if fetch succeeded ──
    if st.session_state.get("live_df") is not None:
        live_df = st.session_state["live_df"]

        # Data preview
        st.markdown("#### Last 60 Trading Days")
        render_data_preview(live_df.tail(60), source="Yahoo Finance")

        # Closing price chart
        st.markdown("#### Closing Price Trend")
        fig = plot_closing_price(live_df.tail(60), source="Yahoo Finance")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Predict button
        predict_disabled = (model is None or scaler is None)
        if predict_disabled:
            st.warning("Prediction unavailable — model or scaler file missing.")
        else:
            if st.button("Predict Next Day Closing Price", key="btn_predict_live"):
                with st.spinner("Running GRU model inference..."):
                    try:
                        price = run_prediction(live_df, scaler, model, "Yahoo Finance")
                        st.session_state["live_prediction"] = price
                        st.session_state["live_predict_error"] = None
                    except Exception as e:
                        st.session_state["live_predict_error"] = str(e)
                        st.session_state["live_prediction"] = None

            if st.session_state.get("live_predict_error"):
                st.error(f"Prediction failed: {st.session_state['live_predict_error']}")

            if st.session_state.get("live_prediction") is not None:
                render_prediction_output(
                    price=st.session_state["live_prediction"],
                    source="Yahoo Finance",
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB B — CSV Upload
# ════════════════════════════════════════════════════════════════════════════

with tab_csv:

    st.markdown("### Custom CSV Prediction")
    st.markdown(
        "Upload a CSV file containing historical Tesla OHLCV data. "
        "The model will use the **last 60 rows** for prediction."
    )

    # Requirements callout
    st.info(
        "**CSV Requirements**\n"
        "- Required columns: `Open`, `High`, `Low`, `Close`, `Volume`\n"
        "- Minimum rows: 60 (trading days)\n"
        "- Column names are case-insensitive\n"
        "- Optional: a `Date` column (will be used as index)"
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="csv_uploader",
        help="Upload historical Tesla OHLCV data with at least 60 rows.",
    )

    if uploaded_file is not None:

        with st.spinner("Reading and validating CSV..."):
            try:
                csv_df = load_csv_data(uploaded_file)
                is_valid, validation_msg = validate_dataframe(csv_df)

                if not is_valid:
                    st.error(f"Validation failed: {validation_msg}")
                    csv_df = None
                else:
                    st.success(
                        f"File loaded successfully — "
                        f"{len(csv_df)} rows · {len(csv_df.columns)} columns"
                    )
                    st.session_state["csv_df"] = csv_df
                    st.session_state["csv_error"] = None

            except Exception as e:
                st.error(f"Could not read file: {e}")
                st.session_state["csv_df"] = None

        # ── Show data + prediction if CSV is valid ──
        if st.session_state.get("csv_df") is not None:
            csv_df = st.session_state["csv_df"]

            st.markdown("#### Last 60 Rows Preview")
            render_data_preview(csv_df.tail(60), source="Uploaded CSV")

            st.markdown("#### Closing Price Trend")
            fig_csv = plot_closing_price(csv_df.tail(60), source="Uploaded CSV")
            st.plotly_chart(fig_csv, use_container_width=True)

            st.divider()

            predict_disabled = (model is None or scaler is None)
            if predict_disabled:
                st.warning("Prediction unavailable — model or scaler file missing.")
            else:
                if st.button("Predict Next Day Closing Price", key="btn_predict_csv"):
                    with st.spinner("Running GRU model inference..."):
                        try:
                            price = run_prediction(csv_df, scaler, model, "Uploaded CSV")
                            st.session_state["csv_prediction"] = price
                            st.session_state["csv_predict_error"] = None
                        except Exception as e:
                            st.session_state["csv_predict_error"] = str(e)
                            st.session_state["csv_prediction"] = None

                if st.session_state.get("csv_predict_error"):
                    st.error(f"Prediction failed: {st.session_state['csv_predict_error']}")

                if st.session_state.get("csv_prediction") is not None:
                    render_prediction_output(
                        price=st.session_state["csv_prediction"],
                        source="Uploaded CSV",
                    )


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<p style='text-align:center; color:#444; font-size:12px;'>"
    "Tesla Stock Price Prediction &nbsp;·&nbsp; GRU Deep Learning &nbsp;·&nbsp; "
    "Built with Streamlit &nbsp;·&nbsp; For educational purposes only &nbsp;·&nbsp; "
    "Not financial advice"
    "</p>",
    unsafe_allow_html=True,
)
