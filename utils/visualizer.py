"""
visualizer.py
-------------
Generates all visual components for the Tesla Stock Prediction Streamlit app.

Responsibilities:
    - Interactive Plotly chart for the 60-day closing price trend.
    - Model performance metric cards (MAE, RMSE, MAPE, R2).
    - Prediction result display with price, timestamp, and metadata.
    - Styled data preview table with source label.

Design principle:
    Functions that produce charts return a plotly.graph_objects.Figure.
    app.py calls st.plotly_chart(fig) — keeps visualizer independently testable.
    Functions that render pure Streamlit UI components (metrics, markdown)
    call st.* directly and return None.

Public API:
    plot_closing_price(df, source)          -> go.Figure
    render_model_metrics(metadata)          -> None  (renders st.metric cards)
    render_prediction_output(price, source) -> None  (renders st.markdown cards)
    render_data_preview(df, source)         -> None  (renders st.dataframe)
"""

import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st


# ── Color palette — consistent across all charts ─────────────────────────────

COLORS = {
    "primary":      "#E31937",   # Tesla red
    "secondary":    "#1C1C1C",   # Dark background
    "accent":       "#FFFFFF",   # White text
    "chart_line":   "#E31937",   # Close price line
    "chart_fill":   "rgba(227, 25, 55, 0.08)",  # Subtle red fill under curve
    "grid":         "rgba(255,255,255,0.08)",
    "positive":     "#00C853",   # Green — for good metrics
    "card_bg":      "#1E1E1E",
}


# ── Closing Price Chart ───────────────────────────────────────────────────────

def plot_closing_price(df: pd.DataFrame, source: str = "Yahoo Finance") -> go.Figure:
    """
    Generate an interactive Plotly line chart of the 60-day closing price trend.

    Features:
        - Filled area under the curve for visual depth.
        - Hover tooltip showing date and exact closing price.
        - Highlighted last data point (most recent Close).
        - Clean dark theme consistent with the app.

    Args:
        df (pd.DataFrame): DataFrame with a 'Close' column and DatetimeIndex
                           or integer index (both handled).
        source (str): Data source label shown in chart subtitle.

    Returns:
        go.Figure: Plotly figure object. Render with st.plotly_chart(fig).
    """
    # ── Build x-axis labels ──
    # Use index if it's a DatetimeIndex, otherwise use row numbers
    if isinstance(df.index, pd.DatetimeIndex):
        x_values = df.index.strftime("%b %d, %Y")
    else:
        x_values = [f"Day {i+1}" for i in range(len(df))]

    close_values = df["Close"].values

    fig = go.Figure()

    # ── Area fill under the curve ──
    fig.add_trace(go.Scatter(
        x=x_values,
        y=close_values,
        fill="tozeroy",
        fillcolor=COLORS["chart_fill"],
        line=dict(color="rgba(0,0,0,0)"),  # invisible line for fill only
        showlegend=False,
        hoverinfo="skip",
        name="fill",
    ))

    # ── Main close price line ──
    fig.add_trace(go.Scatter(
        x=x_values,
        y=close_values,
        mode="lines",
        line=dict(color=COLORS["chart_line"], width=2.5),
        name="Close Price",
        hovertemplate="<b>%{x}</b><br>Close: $%{y:.2f}<extra></extra>",
    ))

    # ── Highlight the last data point ──
    fig.add_trace(go.Scatter(
        x=[x_values[-1]],
        y=[close_values[-1]],
        mode="markers",
        marker=dict(color=COLORS["chart_line"], size=10, symbol="circle"),
        name=f"Latest: ${close_values[-1]:.2f}",
        hovertemplate=f"<b>Latest Close</b><br>${close_values[-1]:.2f}<extra></extra>",
    ))

    # ── Layout ──
    fig.update_layout(
        title=dict(
            text=f"Tesla (TSLA) — Last 60 Trading Days Close Price<br>"
                 f"<sup style='color:gray'>Source: {source}</sup>",
            font=dict(size=16, color=COLORS["accent"]),
            x=0,
        ),
        xaxis=dict(
            title="Date",
            tickangle=-35,
            tickfont=dict(size=10, color="gray"),
            showgrid=False,
            zeroline=False,
            # Show only ~8 evenly spaced labels to avoid crowding
            tickmode="array",
            tickvals=x_values[::8],
        ),
        yaxis=dict(
            title="Price (USD)",
            tickprefix="$",
            tickfont=dict(color="gray"),
            gridcolor=COLORS["grid"],
            zeroline=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(color="gray"),
        ),
        margin=dict(l=10, r=10, t=70, b=10),
        hovermode="x unified",
        height=380,
    )

    return fig


# ── Model Metrics Cards ───────────────────────────────────────────────────────

def render_model_metrics(metadata: dict) -> None:
    """
    Render a 4-column row of st.metric cards for model performance.

    Displays: MAE, RMSE, MAPE, R2 from the metadata dict.

    Args:
        metadata (dict): Output of get_model_metadata() from predictor.py.
                         Must contain a 'metrics' key with MAE, RMSE, MAPE, R2.
    """
    metrics = metadata.get("metrics", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="MAE",
            value=metrics.get("MAE", "N/A"),
            help="Mean Absolute Error — average prediction error in USD",
        )
    with col2:
        st.metric(
            label="RMSE",
            value=metrics.get("RMSE", "N/A"),
            help="Root Mean Squared Error — penalizes large errors more",
        )
    with col3:
        st.metric(
            label="MAPE",
            value=metrics.get("MAPE", "N/A"),
            help="Mean Absolute Percentage Error — relative error",
        )
    with col4:
        st.metric(
            label="R² Score",
            value=metrics.get("R2", "N/A"),
            help="Coefficient of determination — 1.0 is perfect fit",
        )


# ── Prediction Output Card ────────────────────────────────────────────────────

def render_prediction_output(price: float, source: str = "Yahoo Finance") -> None:
    """
    Render the prediction result as a styled card with metadata.

    Displays:
        - Predicted price in large USD format.
        - Prediction timestamp.
        - Data source used.
        - Model name and input window.

    Args:
        price (float): Predicted Tesla closing price in USD.
        source (str): 'Yahoo Finance' or 'Uploaded CSV'.
    """
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid {COLORS['chart_line']};
            border-radius: 16px;
            padding: 32px;
            text-align: center;
            margin: 16px 0;
        ">
            <p style="color: gray; font-size: 14px; margin: 0 0 8px 0; letter-spacing: 2px; text-transform: uppercase;">
                Predicted Next Day Closing Price
            </p>
            <h1 style="
                color: {COLORS['chart_line']};
                font-size: 56px;
                font-weight: 700;
                margin: 0 0 16px 0;
                letter-spacing: -1px;
            ">
                ${price:,.2f}
            </h1>
            <hr style="border-color: #333; margin: 16px 0;" />
            <div style="display: flex; justify-content: center; gap: 40px; flex-wrap: wrap;">
                <div>
                    <p style="color: gray; font-size: 11px; margin: 0; text-transform: uppercase; letter-spacing: 1px;">Predicted On</p>
                    <p style="color: white; font-size: 13px; margin: 4px 0 0 0;">{timestamp}</p>
                </div>
                <div>
                    <p style="color: gray; font-size: 11px; margin: 0; text-transform: uppercase; letter-spacing: 1px;">Data Source</p>
                    <p style="color: white; font-size: 13px; margin: 4px 0 0 0;">{source}</p>
                </div>
                <div>
                    <p style="color: gray; font-size: 11px; margin: 0; text-transform: uppercase; letter-spacing: 1px;">Model</p>
                    <p style="color: white; font-size: 13px; margin: 4px 0 0 0;">GRU Neural Network</p>
                </div>
                <div>
                    <p style="color: gray; font-size: 11px; margin: 0; text-transform: uppercase; letter-spacing: 1px;">Input Window</p>
                    <p style="color: white; font-size: 13px; margin: 4px 0 0 0;">60 Trading Days</p>
                </div>
            </div>
            <p style="color:#555; font-size:11px; margin:16px 0 0 0;">
                ⚠ For educational purposes only. Not financial advice.
                Past performance does not guarantee future results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Data Preview Table ────────────────────────────────────────────────────────

def render_data_preview(df: pd.DataFrame, source: str = "Yahoo Finance") -> None:
    """
    Render a styled preview of the last 60 rows of OHLCV data.

    Shows:
        - Source label above the table.
        - Row count and date range if index is DatetimeIndex.
        - Formatted price columns (2dp) and volume with commas.
        - Full interactive st.dataframe with scroll.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV columns (60 rows).
        source (str): Data source label.
    """
    # ── Header ──
    col_label, col_info = st.columns([1, 2])
    with col_label:
        st.caption(f"**Data Source:** {source}")
    with col_info:
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            date_from = df.index[0].strftime("%b %d, %Y")
            date_to   = df.index[-1].strftime("%b %d, %Y")
            st.caption(f"**Period:** {date_from} → {date_to} | **Rows:** {len(df)}")
        else:
            st.caption(f"**Rows:** {len(df)}")

    # ── Format display copy — don't mutate the original df ──
    display_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    for col in ["Open", "High", "Low", "Close"]:
        display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}")

    display_df["Volume"] = display_df["Volume"].map(lambda x: f"{int(x):,}")

    st.dataframe(
        display_df,
        use_container_width=True,
        height=280,
    )

# ── Multi-Step Forecast Chart ─────────────────────────────────────────────────

def plot_multi_step_forecast(
    history_df:  pd.DataFrame,
    predictions: list,
    n_history:   int = 30,
) -> go.Figure:
    """
    Plot historical closing prices + multi-step forecast on the same chart.

    Shows the last n_history real days as a solid line, then extends with
    a dashed forecast line for each predicted day. A vertical divider marks
    the boundary between real and predicted data.

    Args:
        history_df  : DataFrame with 'Close' column and DatetimeIndex or
                      integer index (the 60-day real window).
        predictions : Output of predict_multi_step() — list of dicts with
                      keys: day, date, close.
        n_history   : How many recent real days to show (default: 30).

    Returns:
        go.Figure: Plotly figure. Render with st.plotly_chart(fig).
    """
    # ── Historical segment ──
    hist = history_df.tail(n_history).copy()

    if isinstance(hist.index, pd.DatetimeIndex):
        hist_x = hist.index.strftime("%b %d").tolist()
    else:
        hist_x = [f"Day -{n_history - i}" for i in range(len(hist))]

    hist_y = hist["Close"].values.tolist()

    # ── Forecast segment ──
    # Connect the last real point to the first forecast point for continuity
    forecast_x = [hist_x[-1]] + [p["date"] for p in predictions]
    forecast_y = [hist_y[-1]] + [p["close"] for p in predictions]

    fig = go.Figure()

    # Historical area fill
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y,
        fill="tozeroy",
        fillcolor=COLORS["chart_fill"],
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip", name="hist_fill",
    ))

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y,
        mode="lines",
        line=dict(color=COLORS["chart_line"], width=2.5),
        name="Historical Close",
        hovertemplate="<b>%{x}</b><br>Close: $%{y:.2f}<extra></extra>",
    ))

    # Forecast line — dashed to signal uncertainty
    fig.add_trace(go.Scatter(
        x=forecast_x, y=forecast_y,
        mode="lines+markers",
        line=dict(color="#FFA500", width=2.5, dash="dash"),
        marker=dict(color="#FFA500", size=8, symbol="circle-open"),
        name="Forecast (GRU)",
        hovertemplate="<b>%{x}</b><br>Forecast: $%{y:.2f}<extra></extra>",
    ))

    # Vertical divider — "today" boundary
    # add_vline requires numeric x on categorical axes — use add_shape instead
    last_real_idx = len(hist_x) - 1
    fig.add_shape(
        type="line",
        x0=last_real_idx, x1=last_real_idx,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
    )
    fig.add_annotation(
        x=last_real_idx, y=1.02,
        xref="x", yref="paper",
        text="Today", showarrow=False,
        font=dict(color="gray", size=11),
    )

    # Annotate each forecast point with its price
    for p in predictions:
        fig.add_annotation(
            x=p["date"],
            y=p["close"],
            text=f"${p['close']:,.2f}",
            showarrow=False,
            yshift=14,
            font=dict(color="#FFA500", size=10),
        )

    fig.update_layout(
        title=dict(
            text=f"Tesla (TSLA) — Historical + {len(predictions)}-Day Forecast<br>"
                 f"<sup style='color:gray'>Dashed line = recursive GRU forecast "
                 f"(uncertainty increases with each step)</sup>",
            font=dict(size=15, color=COLORS["accent"]),
            x=0,
        ),
        xaxis=dict(
            title="Date",
            tickangle=-35,
            tickfont=dict(size=10, color="gray"),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Price (USD)",
            tickprefix="$",
            tickfont=dict(color="gray"),
            gridcolor=COLORS["grid"],
            zeroline=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1, font=dict(color="gray"),
        ),
        margin=dict(l=10, r=10, t=80, b=10),
        hovermode="x unified",
        height=420,
    )

    return fig


def render_multi_step_output(predictions: list, source: str = "Yahoo Finance") -> None:
    """
    Render the multi-step forecast as a table + summary card.

    For n=1 day: shows a single large price card (same as render_prediction_output).
    For n>1 days: shows a table of all predictions + a summary card.

    Args:
        predictions : Output of predict_multi_step().
        source      : Data source label.
    """
    n = len(predictions)
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    if n == 1:
        # ── Single day — show big card ──
        price = predictions[0]["close"]
        date  = predictions[0]["date"]
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border: 1px solid {COLORS['chart_line']};
                border-radius: 16px;
                padding: 32px;
                text-align: center;
                margin: 16px 0;
            ">
                <p style="color:gray; font-size:14px; margin:0 0 8px 0;
                          letter-spacing:2px; text-transform:uppercase;">
                    Predicted Closing Price — {date}
                </p>
                <h1 style="color:{COLORS['chart_line']}; font-size:56px;
                           font-weight:700; margin:0 0 16px 0; letter-spacing:-1px;">
                    ${price:,.2f}
                </h1>
                <hr style="border-color:#333; margin:16px 0;" />
                <div style="display:flex; justify-content:center; gap:40px; flex-wrap:wrap;">
                    <div>
                        <p style="color:gray; font-size:11px; margin:0; text-transform:uppercase;">Predicted On</p>
                        <p style="color:white; font-size:13px; margin:4px 0 0 0;">{timestamp}</p>
                    </div>
                    <div>
                        <p style="color:gray; font-size:11px; margin:0; text-transform:uppercase;">Data Source</p>
                        <p style="color:white; font-size:13px; margin:4px 0 0 0;">{source}</p>
                    </div>
                    <div>
                        <p style="color:gray; font-size:11px; margin:0; text-transform:uppercase;">Model</p>
                        <p style="color:white; font-size:13px; margin:4px 0 0 0;">GRU Neural Network</p>
                    </div>
                    <div>
                        <p style="color:gray; font-size:11px; margin:0; text-transform:uppercase;">Input Window</p>
                        <p style="color:white; font-size:13px; margin:4px 0 0 0;">60 Trading Days</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        # ── Multi-day — show table + uncertainty warning ──
        st.warning(
            f"**Forecast Uncertainty Notice**  \n"
            f"Day 1 prediction uses 60 real trading days (MAPE ≈ 2.77%).  \n"
            f"Days 2–{n} use synthetic rows built from prior predictions — "
            f"error compounds with each step. Treat days 5–10 as a **directional "
            f"trend**, not a precise price target."
        )

        # Build forecast table
        forecast_data = {
            "Day":            [p["day"] for p in predictions],
            "Date (Est.)":    [p["date"] for p in predictions],
            "Predicted Close":[f"${p['close']:,.2f}" for p in predictions],
            "Confidence":     [
                "High" if p["day"] == 1
                else "Medium" if p["day"] <= 3
                else "Low"
                for p in predictions
            ],
        }
        import pandas as pd_inner
        forecast_df = pd_inner.DataFrame(forecast_data)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        # Summary card — show first and last prediction
        first_p = predictions[0]["close"]
        last_p  = predictions[-1]["close"]
        delta   = last_p - first_p
        arrow   = "↑" if delta >= 0 else "↓"
        color   = COLORS["positive"] if delta >= 0 else COLORS["chart_line"]

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border: 1px solid #2a2a2a;
                border-radius: 16px;
                padding: 24px 32px;
                margin: 12px 0;
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                gap: 20px;
            ">
                <div style="text-align:center;">
                    <p style="color:gray; font-size:11px; text-transform:uppercase; margin:0;">
                        Day 1 Forecast</p>
                    <p style="color:white; font-size:24px; font-weight:700; margin:4px 0;">
                        ${first_p:,.2f}</p>
                </div>
                <div style="text-align:center;">
                    <p style="color:gray; font-size:11px; text-transform:uppercase; margin:0;">
                        Day {n} Forecast</p>
                    <p style="color:white; font-size:24px; font-weight:700; margin:4px 0;">
                        ${last_p:,.2f}</p>
                </div>
                <div style="text-align:center;">
                    <p style="color:gray; font-size:11px; text-transform:uppercase; margin:0;">
                        {n}-Day Direction</p>
                    <p style="color:{color}; font-size:24px; font-weight:700; margin:4px 0;">
                        {arrow} ${abs(delta):,.2f}</p>
                </div>
                <div style="text-align:center;">
                    <p style="color:gray; font-size:11px; text-transform:uppercase; margin:0;">
                        Source</p>
                    <p style="color:white; font-size:13px; margin:4px 0;">{source}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
