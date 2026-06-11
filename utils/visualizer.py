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
