"""
app_simple.py  —  Day 4: Streamlit UI
---------------------------------------
Simple, clean Streamlit dashboard that:
  - Lets you pick a stock from a dropdown
  - Shows an interactive price chart (close + MA50 + MA200)
  - Loads the saved RandomForest model and displays a UP / DOWN prediction
  - Shows model confidence as a progress bar

Run from project root:
    streamlit run app_simple.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup — must happen before any local imports ─────────────────────────
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense AI — Simple",
    page_icon="📈",
    layout="centered",
)

# ── Imports from our src modules ──────────────────────────────────────────────
from config import SYMBOLS, DATA_DIR, MODELS_DIR
from data_collection import download_historical_data


# ── Helper: compute MAs ───────────────────────────────────────────────────────

def _add_mas(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA50 and MA200 columns to a Close-price DataFrame."""
    df = df.copy()
    df["MA50"]  = df["Close"].rolling(window=50,  min_periods=1).mean()
    df["MA200"] = df["Close"].rolling(window=200, min_periods=1).mean()
    return df


# ── Helper: build price chart ─────────────────────────────────────────────────

def _price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Interactive Plotly chart showing:
      - Close price (area)
      - MA50  (dashed yellow)
      - MA200 (dashed red)
    """
    df = _add_mas(df)
    dates = df.index.strftime("%Y-%m-%d")

    fig = go.Figure()

    # Close price
    fig.add_trace(go.Scatter(
        x=dates, y=df["Close"],
        mode="lines",
        name="Close",
        line=dict(color="#7986cb", width=2),
        fill="tozeroy",
        fillcolor="rgba(121,134,203,0.10)",
    ))

    # MA 50
    fig.add_trace(go.Scatter(
        x=dates, y=df["MA50"],
        mode="lines",
        name="MA 50",
        line=dict(color="#ffd54f", width=1.5, dash="dash"),
    ))

    # MA 200 — only plot from day 200 onward to avoid the ramp-up
    ma200 = df["MA200"].copy()
    ma200.iloc[:199] = np.nan
    fig.add_trace(go.Scatter(
        x=dates, y=ma200,
        mode="lines",
        name="MA 200",
        line=dict(color="#ef9a9a", width=1.5, dash="dot"),
    ))

    fig.update_layout(
        title=f"{symbol} — Historical Close Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=30, l=10, r=10),
    )
    return fig


# ── Helper: load model + run prediction ──────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_rf_model(symbol: str):
    """
    Load the RandomForest model and scaler for *symbol* from models/.
    Cached so Streamlit only loads once per symbol.
    Returns (model, scaler) or (None, None) if not found.
    """
    import joblib

    model_path  = MODELS_DIR / f"rf_model_{symbol}.pkl"
    scaler_path = MODELS_DIR / f"rf_scaler_{symbol}.pkl"

    if not model_path.exists() or not scaler_path.exists():
        return None, None

    return joblib.load(model_path), joblib.load(scaler_path)


def _run_prediction(symbol: str) -> dict | None:
    """
    Build features for *symbol* and return a prediction dict.
    Returns None if no model exists.
    """
    from feature_engineering import build_feature_matrix
    from model import SIMPLE_FEATURES

    model, scaler = _load_rf_model(symbol)
    if model is None:
        return None

    raw     = download_historical_data(symbol)
    feat_df = build_feature_matrix(raw)

    used_cols = [c for c in SIMPLE_FEATURES if c in feat_df.columns]
    X_latest  = feat_df[used_cols].values[-1:].astype("float32")
    X_scaled  = scaler.transform(X_latest)

    pred_label = int(model.predict(X_scaled)[0])
    confidence = float(model.predict_proba(X_scaled)[0, 1])

    return {
        "direction":  "UP" if pred_label == 1 else "DOWN",
        "confidence": round(confidence * 100, 2),
        "label":      pred_label,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("📈 StockSense AI")
st.caption("Simple stock prediction dashboard — Day 4 build")
st.divider()

# ── Stock selector ────────────────────────────────────────────────────────────
symbol = st.selectbox(
    "Select a stock symbol",
    options=SYMBOLS,
    index=0,
    help="Historical data must be downloaded first (run data_collection.py)",
)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Loading data for {symbol}…"):
    try:
        df = download_historical_data(symbol)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

# ── Latest price summary ──────────────────────────────────────────────────────
latest       = df.iloc[-1]
prev         = df.iloc[-2]
daily_change = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
change_color = "green" if daily_change >= 0 else "red"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Close",  f"${latest['Close']:.2f}", f"{daily_change:+.2f}%")
col2.metric("Open",   f"${latest['Open']:.2f}")
col3.metric("High",   f"${latest['High']:.2f}")
col4.metric("Low",    f"${latest['Low']:.2f}")

st.divider()

# ── Price chart ───────────────────────────────────────────────────────────────
st.subheader("📊 Price Chart")
st.plotly_chart(_price_chart(df, symbol), use_container_width=True)

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
st.subheader("🔮 Next-Day Prediction")

with st.spinner("Running model…"):
    prediction = _run_prediction(symbol)

if prediction is None:
    # Model hasn't been trained yet — show instructions
    st.warning(
        f"No trained model found for **{symbol}**.\n\n"
        "Train it by running:\n"
        "```\ncd src\npython model.py --symbol "
        + symbol
        + "\n```"
    )
else:
    direction  = prediction["direction"]
    confidence = prediction["confidence"]

    # Big direction badge
    if direction == "UP":
        st.success(f"### ⬆️  Predicted: **UP**")
    else:
        st.error(f"### ⬇️  Predicted: **DOWN**")

    # Confidence bar
    st.markdown(f"**Model confidence:** {confidence:.1f}%")
    st.progress(int(confidence))

    # Human-readable explanation
    if confidence >= 80:
        note = "High confidence — strong signal."
    elif confidence >= 60:
        note = "Moderate confidence — treat with caution."
    else:
        note = "Low confidence — model is uncertain."

    st.caption(note)

st.divider()

# ── Raw data preview ──────────────────────────────────────────────────────────
with st.expander("📋 Raw data (last 10 rows)"):
    st.dataframe(df.tail(10), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center;color:gray;font-size:0.75rem'>"
    "StockSense AI · For educational use only · Not financial advice"
    "</p>",
    unsafe_allow_html=True,
)
