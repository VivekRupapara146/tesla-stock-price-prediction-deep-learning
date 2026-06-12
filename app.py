"""
app.py
------
Main Streamlit entry point for the Tesla Stock Price Prediction app.

Responsibility: UI orchestration only.

Run:
    streamlit run app.py
"""

import os
import streamlit as st

from utils.data_loader import (
    load_csv_data, validate_dataframe, get_latest_window,
)
from utils.preprocessing import load_scaler, preprocess_window
from utils.predictor import load_model, predict, predict_multi_step, get_model_metadata
from utils.visualizer import (
    get_theme_colors,
    plot_closing_price, plot_multi_step_forecast,
    render_model_metrics, render_multi_step_output, render_data_preview,
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
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Theme State ───────────────────────────────────────────────────────────────

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

theme = st.session_state["theme"]
c     = get_theme_colors(theme)

is_dark = (theme == "dark")


# ── CSS injection ─────────────────────────────────────────────────────────────

def inject_css(theme: str, c: dict) -> None:
    """Inject full-app CSS for the active theme."""

    # Tab active color, button, metric cards adapt to theme
    tab_active_bg  = c["accent"]
    tab_inactive   = c["subtext"]
    metric_bg      = c["card_bg"]
    metric_border  = c["card_border"]
    metric_label   = c["subtext"]
    metric_val     = c["text"]
    btn_bg         = c["accent"]
    btn_hover      = "#b5132b"
    app_bg         = c["bg"]
    hr_color       = c["card_border"]

    st.markdown(
        f"""
        <style>
        /* ── Global ── */
        #MainMenu, footer, header {{ visibility: hidden; }}
        .stApp {{ background-color: {app_bg}; }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: {metric_bg};
            border-radius: 12px;
            padding: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 8px 24px;
            color: {tab_inactive};
            font-weight: 500;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {tab_active_bg} !important;
            color: white !important;
        }}

        /* ── Metric cards ── */
        [data-testid="stMetric"] {{
            background-color: {metric_bg};
            border: 1px solid {metric_border};
            border-radius: 12px;
            padding: 16px;
        }}
        [data-testid="stMetricLabel"] {{ color: {metric_label} !important; }}
        [data-testid="stMetricValue"] {{
            color: {metric_val} !important;
            font-size: 1.4rem !important;
        }}

        /* ── Primary button ── */
        .stButton > button {{
            background-color: {btn_bg};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 28px;
            font-weight: 600;
            font-size: 15px;
            width: 100%;
        }}
        .stButton > button:hover {{ background-color: {btn_hover}; }}

        /* ── Divider ── */
        hr {{ border-color: {hr_color} !important; }}

        /* ── Headings ── */
        h2, h3 {{ color: {c["text"]} !important; }}

        /* ── Expander ── */
        [data-testid="stExpander"] {{
            background-color: {metric_bg};
            border: 1px solid {metric_border};
            border-radius: 12px;
        }}

        /* ── File uploader ── */
        [data-testid="stFileUploader"] {{
            background-color: {metric_bg};
            border: 1px solid {metric_border};
            border-radius: 8px;
        }}

        /* ── Dataframe ── */
        [data-testid="stDataFrame"] {{
            background-color: {metric_bg};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css(theme, c)


# ── Cached Loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading GRU model...")
def get_model():
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner="Loading scaler...")
def get_scaler():
    return load_scaler(SCALER_PATH)


# ── Prediction Pipeline ───────────────────────────────────────────────────────

def run_prediction_pipeline(df, scaler, model, n_days: int) -> list[dict]:
    """Run single or multi-step prediction. Always returns list[dict]."""
    window = get_latest_window(df)

    if n_days == 1:
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


# ── Startup ───────────────────────────────────────────────────────────────────

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


# ── Header + Theme Toggle ─────────────────────────────────────────────────────

header_col, toggle_col = st.columns([11, 1])

with header_col:
    st.markdown(
        f"""
        <div style="padding:8px 0 24px 0;">
            <div style="display:flex;align-items:center;gap:12px;">
                <span style="font-size:36px;">🚗</span>
                <div>
                    <h1 style="margin:0;color:{c['accent']};font-size:2rem;
                               font-weight:700;letter-spacing:-0.5px;">
                        Tesla Stock Price Prediction
                    </h1>
                    <p style="margin:4px 0 0 0;color:{c['subtext']};font-size:14px;">
                        GRU Deep Learning &nbsp;·&nbsp; 60-Day Lookback
                        &nbsp;·&nbsp; OHLCV Features &nbsp;·&nbsp;
                        Forecast up to 10 Days
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with toggle_col:
    # Vertical spacer to align button with header text
    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    toggle_label = "☀️ Light" if is_dark else "🌙 Dark"
    if st.button(toggle_label, key="theme_toggle"):
        st.session_state["theme"] = "light" if is_dark else "dark"
        st.rerun()

st.divider()

# ── Load error banners ──
if model_error:
    st.error(
        f"**Model file missing.** Place `tesla_gru_model.keras` in `model/`.\n\n"
        f"`{model_error}`"
    )
if scaler_error:
    st.error(
        f"**Scaler file missing.** Place `tesla_scaler.pkl` in `model/`.\n\n"
        f"`{scaler_error}`"
    )


# ── Model Info Dashboard ──────────────────────────────────────────────────────

with st.expander("Model Information Dashboard", expanded=True):
    meta = get_model_metadata()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<span style='color:{c['subtext']};font-size:12px;'>MODEL TYPE</span>", unsafe_allow_html=True)
        st.markdown(f"`{meta['model_name']}`")
    with c2:
        st.markdown(f"<span style='color:{c['subtext']};font-size:12px;'>INPUT WINDOW</span>", unsafe_allow_html=True)
        st.markdown("`60 Trading Days`")
    with c3:
        st.markdown(f"<span style='color:{c['subtext']};font-size:12px;'>FEATURES</span>", unsafe_allow_html=True)
        st.markdown("`Open · High · Low · Close · Volume`")
    with c4:
        st.markdown(f"<span style='color:{c['subtext']};font-size:12px;'>TARGET</span>", unsafe_allow_html=True)
        st.markdown(f"`{meta['target']}`")
    st.markdown("##### Performance Metrics")
    render_model_metrics(meta)

st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_live, tab_csv = st.tabs([
    "📡  Live Tesla Data  (Yahoo Finance)",
    "📂  Upload CSV",
])


# ════════════════════════════════════════════════════════════════════════════
# SHARED PREDICTION UI — used by CSV tab
# ════════════════════════════════════════════════════════════════════════════

def render_prediction_section(tab_key: str, source: str) -> None:
    """Render data preview + horizon selector + prediction output for one tab."""
    df = st.session_state.get(f"{tab_key}_df")
    if df is None:
        return

    # Data preview + trend chart
    st.markdown("#### Last 60 Trading Days")
    render_data_preview(df.tail(60), source=source)

    st.markdown("#### Closing Price Trend")
    st.plotly_chart(
        plot_closing_price(df.tail(60), source=source, theme=theme),
        use_container_width=True,
    )

    st.divider()

    # Horizon selector
    st.markdown("#### Prediction Horizon")
    horizon_label = st.radio(
        "Select forecast window:",
        options=list(PREDICTION_HORIZONS.keys()),
        horizontal=True,
        key=f"{tab_key}_horizon",
        help=(
            "Next Day: single prediction using 60 real trading days.  \n"
            "5 / 10 Days: recursive forecast — error compounds with each step."
        ),
    )
    n_days = PREDICTION_HORIZONS[horizon_label]

    if n_days > 1:
        st.info(
            f"**{horizon_label} forecast** uses recursive prediction. "
            f"Day 1 uses 60 real rows. By day {n_days}, {n_days - 1} synthetic "
            f"rows are in the window. Confidence decreases with each step."
        )

    # Predict button
    if model is None or scaler is None:
        st.warning("Prediction unavailable — model or scaler file missing.")
        return

    btn_label = (
        "Predict Next Day Closing Price" if n_days == 1
        else f"Predict Next {n_days} Days"
    )

    if st.button(btn_label, key=f"btn_predict_{tab_key}"):
        with st.spinner(
            "Running GRU inference..." if n_days == 1
            else f"Running recursive GRU forecast for {n_days} days..."
        ):
            try:
                preds = run_prediction_pipeline(df, scaler, model, n_days)
                st.session_state[f"{tab_key}_predictions"] = preds
                st.session_state[f"{tab_key}_predict_error"] = None
            except Exception as e:
                st.session_state[f"{tab_key}_predict_error"] = str(e)
                st.session_state[f"{tab_key}_predictions"] = None

    if st.session_state.get(f"{tab_key}_predict_error"):
        st.error(f"Prediction failed: {st.session_state[f'{tab_key}_predict_error']}")

    preds = st.session_state.get(f"{tab_key}_predictions")
    if preds:
        if len(preds) > 1:
            st.markdown("#### Forecast Chart")
            st.plotly_chart(
                plot_multi_step_forecast(df.tail(60), preds, theme=theme),
                use_container_width=True,
            )
        st.markdown("#### Prediction Result")
        render_multi_step_output(preds, source=source, theme=theme)


# ════════════════════════════════════════════════════════════════════════════
# TAB A — Live Tesla Data (placeholder)
# ════════════════════════════════════════════════════════════════════════════

with tab_live:
    st.markdown("### Live Tesla Stock Prediction")
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{c['card_bg']} 0%,{c['bg']} 100%);
                    border:1px solid {c['card_border']};border-radius:16px;
                    padding:40px 32px;text-align:center;margin:16px 0;">
            <div style="font-size:48px;margin-bottom:16px;">📡</div>
            <h3 style="color:{c['text']};margin:0 0 12px 0;">
                Live Data Feed — Coming Soon
            </h3>
            <p style="color:{c['subtext']};font-size:15px;max-width:480px;
                       margin:0 auto 20px auto;line-height:1.6;">
                Automatic fetching of real-time Tesla OHLCV data from a
                live market data source will be integrated in a future release.
            </p>
            <div style="display:inline-block;background:{c['bg']};
                        border:1px solid {c['card_border']};border-radius:10px;
                        padding:16px 28px;text-align:left;margin-top:8px;">
                <p style="color:{c['subtext']};font-size:12px;text-transform:uppercase;
                           letter-spacing:1px;margin:0 0 10px 0;">Planned upgrades</p>
                <p style="color:{c['text']};font-size:13px;margin:4px 0;">
                    ✦ &nbsp; Real-time OHLCV data via a stable market API</p>
                <p style="color:{c['text']};font-size:13px;margin:4px 0;">
                    ✦ &nbsp; Auto-refresh on market open / close</p>
                <p style="color:{c['text']};font-size:13px;margin:4px 0;">
                    ✦ &nbsp; One-click prediction without manual CSV upload</p>
            </div>
            <p style="color:{c['subtext']};font-size:12px;margin:24px 0 0 0;">
                In the meantime, use the
                <strong>Upload CSV</strong> tab to run predictions.
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
                    st.success(
                        f"Loaded — {len(csv_df)} rows · {len(csv_df.columns)} columns"
                    )
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
    f"<p style='text-align:center;color:{c['subtext']};font-size:12px;'>"
    "Tesla Stock Price Prediction &nbsp;·&nbsp; GRU Deep Learning "
    "&nbsp;·&nbsp; For educational purposes only &nbsp;·&nbsp; Not financial advice"
    "</p>",
    unsafe_allow_html=True,
)
