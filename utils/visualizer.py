"""
visualizer.py
-------------
Generates all visual components for the Tesla Stock Prediction Streamlit app.

Public API:
    get_theme_colors(theme)                    -> dict
    plot_closing_price(df, source, theme)      -> go.Figure
    plot_multi_step_forecast(df, preds, theme) -> go.Figure
    render_model_metrics(metadata)             -> None
    render_multi_step_output(preds, source, theme) -> None
    render_data_preview(df, source)            -> None
"""

import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st


# ── Theme palettes ────────────────────────────────────────────────────────────

_DARK = {
    "bg":           "#0f0f0f",
    "card_bg":      "#1a1a1a",
    "card_border":  "#2a2a2a",
    "text":         "#ffffff",
    "subtext":      "#888888",
    "accent":       "#E31937",          # Tesla red
    "chart_line":   "#E31937",
    "chart_fill":   "rgba(227,25,55,0.08)",
    "forecast":     "#FFA500",
    "grid":         "rgba(255,255,255,0.07)",
    "tick":         "#888888",
    "positive":     "#00C853",
    "divider_line": "rgba(255,255,255,0.15)",
}

_LIGHT = {
    "bg":           "#f5f5f5",
    "card_bg":      "#ffffff",
    "card_border":  "#e0e0e0",
    "text":         "#1a1a1a",
    "subtext":      "#666666",
    "accent":       "#E31937",
    "chart_line":   "#E31937",
    "chart_fill":   "rgba(227,25,55,0.06)",
    "forecast":     "#d97706",
    "grid":         "rgba(0,0,0,0.07)",
    "tick":         "#666666",
    "positive":     "#16a34a",
    "divider_line": "rgba(0,0,0,0.15)",
}

# Keep backward-compat alias used in old HTML strings
COLORS = _DARK


def get_theme_colors(theme: str = "dark") -> dict:
    """Return the color palette dict for the given theme ('dark' | 'light')."""
    return _DARK if theme == "dark" else _LIGHT


# ── Closing Price Chart ───────────────────────────────────────────────────────

def plot_closing_price(
    df:     pd.DataFrame,
    source: str = "Uploaded CSV",
    theme:  str = "dark",
) -> go.Figure:
    """
    Interactive Plotly line chart of the 60-day closing price trend.

    Args:
        df     : DataFrame with 'Close' column and DatetimeIndex or int index.
        source : Data source label shown in chart subtitle.
        theme  : 'dark' | 'light'

    Returns:
        go.Figure
    """
    c = get_theme_colors(theme)

    if isinstance(df.index, pd.DatetimeIndex):
        x_values = df.index.strftime("%b %d, %Y")
    else:
        x_values = [f"Day {i+1}" for i in range(len(df))]

    close_values = df["Close"].values

    fig = go.Figure()

    # Area fill
    fig.add_trace(go.Scatter(
        x=x_values, y=close_values,
        fill="tozeroy", fillcolor=c["chart_fill"],
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip", name="fill",
    ))

    # Price line
    fig.add_trace(go.Scatter(
        x=x_values, y=close_values,
        mode="lines",
        line=dict(color=c["chart_line"], width=2.5),
        name="Close Price",
        hovertemplate="<b>%{x}</b><br>Close: $%{y:.2f}<extra></extra>",
    ))

    # Latest point marker
    fig.add_trace(go.Scatter(
        x=[x_values[-1]], y=[close_values[-1]],
        mode="markers",
        marker=dict(color=c["chart_line"], size=10, symbol="circle"),
        name=f"Latest: ${close_values[-1]:.2f}",
        hovertemplate=f"<b>Latest Close</b><br>${close_values[-1]:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"Tesla (TSLA) — Last 60 Trading Days Close Price<br>"
                f"<sup style='color:{c['subtext']}'>Source: {source}</sup>"
            ),
            font=dict(size=16, color=c["text"]), x=0,
        ),
        xaxis=dict(
            title="Date", tickangle=-35,
            tickfont=dict(size=10, color=c["tick"]),
            showgrid=False, zeroline=False,
            tickmode="array", tickvals=x_values[::8],
        ),
        yaxis=dict(
            title="Price (USD)", tickprefix="$",
            tickfont=dict(color=c["tick"]),
            gridcolor=c["grid"], zeroline=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            font=dict(color=c["subtext"]),
        ),
        margin=dict(l=10, r=10, t=70, b=10),
        hovermode="x unified",
        height=380,
    )

    return fig


# ── Model Metrics Cards ───────────────────────────────────────────────────────

def render_model_metrics(metadata: dict) -> None:
    """Render 4-column st.metric row for MAE, RMSE, MAPE, R²."""
    metrics = metadata.get("metrics", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MAE",     metrics.get("MAE",  "N/A"),
                  help="Mean Absolute Error — average prediction error in USD")
    with col2:
        st.metric("RMSE",    metrics.get("RMSE", "N/A"),
                  help="Root Mean Squared Error — penalises large errors more")
    with col3:
        st.metric("MAPE",    metrics.get("MAPE", "N/A"),
                  help="Mean Absolute Percentage Error — relative error")
    with col4:
        st.metric("R² Score",metrics.get("R2",   "N/A"),
                  help="Coefficient of determination — 1.0 is perfect fit")


# ── Data Preview Table ────────────────────────────────────────────────────────

def render_data_preview(df: pd.DataFrame, source: str = "Uploaded CSV") -> None:
    """Render a formatted OHLCV preview table with source label."""
    col_label, col_info = st.columns([1, 2])
    with col_label:
        st.caption(f"**Data Source:** {source}")
    with col_info:
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            st.caption(
                f"**Period:** {df.index[0].strftime('%b %d, %Y')} → "
                f"{df.index[-1].strftime('%b %d, %Y')} | **Rows:** {len(df)}"
            )
        else:
            st.caption(f"**Rows:** {len(df)}")

    display_df = df[["Open","High","Low","Close","Volume"]].copy()
    for col in ["Open","High","Low","Close"]:
        display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}")
    display_df["Volume"] = display_df["Volume"].map(lambda x: f"{int(x):,}")
    st.dataframe(display_df, use_container_width=True, height=280)


# ── Multi-Step Forecast Chart ─────────────────────────────────────────────────

def plot_multi_step_forecast(
    history_df:  pd.DataFrame,
    predictions: list,
    n_history:   int = 30,
    theme:       str = "dark",
) -> go.Figure:
    """
    Historical close prices + multi-step GRU forecast on one chart.

    Args:
        history_df  : 60-row real OHLCV window.
        predictions : Output of predict_multi_step().
        n_history   : How many recent real days to display (default 30).
        theme       : 'dark' | 'light'

    Returns:
        go.Figure
    """
    c    = get_theme_colors(theme)
    hist = history_df.tail(n_history).copy()

    if isinstance(hist.index, pd.DatetimeIndex):
        hist_x = hist.index.strftime("%b %d").tolist()
    else:
        hist_x = [f"Day -{n_history - i}" for i in range(len(hist))]

    hist_y     = hist["Close"].values.tolist()
    forecast_x = [hist_x[-1]] + [p["date"] for p in predictions]
    forecast_y = [hist_y[-1]] + [p["close"] for p in predictions]

    fig = go.Figure()

    # Historical fill
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y,
        fill="tozeroy", fillcolor=c["chart_fill"],
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip", name="hist_fill",
    ))

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y,
        mode="lines",
        line=dict(color=c["chart_line"], width=2.5),
        name="Historical Close",
        hovertemplate="<b>%{x}</b><br>Close: $%{y:.2f}<extra></extra>",
    ))

    # Forecast line — dashed to signal uncertainty
    fig.add_trace(go.Scatter(
        x=forecast_x, y=forecast_y,
        mode="lines+markers",
        line=dict(color=c["forecast"], width=2.5, dash="dash"),
        marker=dict(color=c["forecast"], size=8, symbol="circle-open"),
        name="Forecast (GRU)",
        hovertemplate="<b>%{x}</b><br>Forecast: $%{y:.2f}<extra></extra>",
    ))

    # "Today" divider
    last_real_idx = len(hist_x) - 1
    fig.add_shape(
        type="line",
        x0=last_real_idx, x1=last_real_idx, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=c["divider_line"], width=1, dash="dot"),
    )
    fig.add_annotation(
        x=last_real_idx, y=1.02, xref="x", yref="paper",
        text="Today", showarrow=False,
        font=dict(color=c["subtext"], size=11),
    )

    # Price labels on forecast points
    for p in predictions:
        fig.add_annotation(
            x=p["date"], y=p["close"],
            text=f"${p['close']:,.2f}", showarrow=False,
            yshift=14, font=dict(color=c["forecast"], size=10),
        )

    fig.update_layout(
        title=dict(
            text=(
                f"Tesla (TSLA) — Historical + {len(predictions)}-Day Forecast<br>"
                f"<sup style='color:{c['subtext']}'>Dashed = recursive GRU "
                f"(uncertainty grows with each step)</sup>"
            ),
            font=dict(size=15, color=c["text"]), x=0,
        ),
        xaxis=dict(
            title="Date", tickangle=-35,
            tickfont=dict(size=10, color=c["tick"]),
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            title="Price (USD)", tickprefix="$",
            tickfont=dict(color=c["tick"]),
            gridcolor=c["grid"], zeroline=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1, font=dict(color=c["subtext"]),
        ),
        margin=dict(l=10, r=10, t=80, b=10),
        hovermode="x unified",
        height=420,
    )

    return fig


# ── Multi-Step Prediction Output ──────────────────────────────────────────────

def render_multi_step_output(
    predictions: list,
    source:      str = "Uploaded CSV",
    theme:       str = "dark",
) -> None:
    """
    Render the multi-step forecast as a styled card or table.

    n=1  → single large price card.
    n>1  → table with confidence labels + summary card.

    Args:
        predictions : Output of predict_multi_step().
        source      : Data source label.
        theme       : 'dark' | 'light'
    """
    c         = get_theme_colors(theme)
    n         = len(predictions)
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    if n == 1:
        price = predictions[0]["close"]
        date  = predictions[0]["date"]
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,{c['card_bg']} 0%,{c['bg']} 100%);
                        border:1px solid {c['accent']};border-radius:16px;
                        padding:32px;text-align:center;margin:16px 0;">
                <p style="color:{c['subtext']};font-size:14px;margin:0 0 8px 0;
                          letter-spacing:2px;text-transform:uppercase;">
                    Predicted Closing Price — {date}
                </p>
                <h1 style="color:{c['accent']};font-size:56px;font-weight:700;
                           margin:0 0 16px 0;letter-spacing:-1px;">
                    ${price:,.2f}
                </h1>
                <hr style="border-color:{c['card_border']};margin:16px 0;" />
                <div style="display:flex;justify-content:center;gap:40px;flex-wrap:wrap;">
                    <div>
                        <p style="color:{c['subtext']};font-size:11px;margin:0;
                                  text-transform:uppercase;letter-spacing:1px;">Predicted On</p>
                        <p style="color:{c['text']};font-size:13px;margin:4px 0 0 0;">{timestamp}</p>
                    </div>
                    <div>
                        <p style="color:{c['subtext']};font-size:11px;margin:0;
                                  text-transform:uppercase;letter-spacing:1px;">Source</p>
                        <p style="color:{c['text']};font-size:13px;margin:4px 0 0 0;">{source}</p>
                    </div>
                    <div>
                        <p style="color:{c['subtext']};font-size:11px;margin:0;
                                  text-transform:uppercase;letter-spacing:1px;">Model</p>
                        <p style="color:{c['text']};font-size:13px;margin:4px 0 0 0;">GRU Neural Network</p>
                    </div>
                    <div>
                        <p style="color:{c['subtext']};font-size:11px;margin:0;
                                  text-transform:uppercase;letter-spacing:1px;">Input Window</p>
                        <p style="color:{c['text']};font-size:13px;margin:4px 0 0 0;">60 Trading Days</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.warning(
            f"**Forecast Uncertainty** — Day 1 uses 60 real rows (MAPE ≈ 2.77%). "
            f"Days 2–{n} use synthetic rows; error compounds with each step. "
            f"Treat days 5–10 as a directional trend, not a precise target."
        )

        forecast_df = pd.DataFrame({
            "Day":             [p["day"]   for p in predictions],
            "Date (Est.)":     [p["date"]  for p in predictions],
            "Predicted Close": [f"${p['close']:,.2f}" for p in predictions],
            "Confidence":      [
                "High" if p["day"] == 1 else
                "Medium" if p["day"] <= 3 else "Low"
                for p in predictions
            ],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        first_p = predictions[0]["close"]
        last_p  = predictions[-1]["close"]
        delta   = last_p - first_p
        arrow   = "↑" if delta >= 0 else "↓"
        d_color = c["positive"] if delta >= 0 else c["accent"]

        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,{c['card_bg']} 0%,{c['bg']} 100%);
                        border:1px solid {c['card_border']};border-radius:16px;
                        padding:24px 32px;margin:12px 0;
                        display:flex;justify-content:space-around;flex-wrap:wrap;gap:20px;">
                <div style="text-align:center;">
                    <p style="color:{c['subtext']};font-size:11px;text-transform:uppercase;margin:0;">
                        Day 1 Forecast</p>
                    <p style="color:{c['text']};font-size:24px;font-weight:700;margin:4px 0;">
                        ${first_p:,.2f}</p>
                </div>
                <div style="text-align:center;">
                    <p style="color:{c['subtext']};font-size:11px;text-transform:uppercase;margin:0;">
                        Day {n} Forecast</p>
                    <p style="color:{c['text']};font-size:24px;font-weight:700;margin:4px 0;">
                        ${last_p:,.2f}</p>
                </div>
                <div style="text-align:center;">
                    <p style="color:{c['subtext']};font-size:11px;text-transform:uppercase;margin:0;">
                        {n}-Day Direction</p>
                    <p style="color:{d_color};font-size:24px;font-weight:700;margin:4px 0;">
                        {arrow} ${abs(delta):,.2f}</p>
                </div>
                <div style="text-align:center;">
                    <p style="color:{c['subtext']};font-size:11px;text-transform:uppercase;margin:0;">
                        Source</p>
                    <p style="color:{c['text']};font-size:13px;margin:4px 0;">{source}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
