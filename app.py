"""
app.py — StockSense AI  |  Streamlit Frontend
----------------------------------------------
Professional dashboard for stock market prediction using
XGBoost, LSTM, and GRU models with FinBERT sentiment analysis.

Run from project root:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# ── Path setup — must happen before any local imports ─────────────────────────
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global font */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2130, #252840);
    border: 1px solid #3a3d5c;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.metric-label  { font-size: 0.78rem; color: #8b92b0; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
.metric-value  { font-size: 1.9rem; font-weight: 700; color: #e8eaf6; }
.metric-sub    { font-size: 0.8rem; color: #8b92b0; margin-top: 4px; }

/* Signal badges */
.badge-buy  { background:#00c851; color:#fff; padding:6px 20px; border-radius:20px; font-weight:700; font-size:1rem; }
.badge-sell { background:#ff4444; color:#fff; padding:6px 20px; border-radius:20px; font-weight:700; font-size:1rem; }
.badge-hold { background:#ffbb33; color:#000; padding:6px 20px; border-radius:20px; font-weight:700; font-size:1rem; }
.badge-up   { background:#00c851; color:#fff; padding:4px 14px; border-radius:14px; font-weight:600; font-size:0.85rem; }
.badge-down { background:#ff4444; color:#fff; padding:4px 14px; border-radius:14px; font-weight:600; font-size:0.85rem; }

/* Section headers */
.section-header {
    font-size: 1.1rem; font-weight: 600; color: #9fa8da;
    border-bottom: 2px solid #3a3d5c; padding-bottom: 8px; margin-bottom: 16px;
}

/* Sidebar */
[data-testid="stSidebar"] { background: #161828; }
[data-testid="stSidebar"] h1 { color: #9fa8da; }

/* Scrollable table */
.dataframe-container { overflow-x: auto; }
</style>
""", unsafe_allow_html=True)


# ── Model / config imports (lazy-cached so Streamlit doesn't reload every run) ─

@st.cache_resource(show_spinner="Loading models…")
def _load_backend():
    """
    Import and cache all backend modules + trained models.
    Returns a dict of callables so the rest of the app never re-imports.
    """
    from config import SYMBOLS, DEFAULT_SYMBOL, XGBOOST_MODEL_PATH, LSTM_MODEL_PATH, GRU_MODEL_PATH, SCALER_PATH
    from predict import (
        predict_stock,
        get_sentiment,
        get_confidence,
        get_recommendation,
        get_latest_stock_data,
        predict_all_models,
    )
    from data_collection import download_historical_data, download_latest_data
    from feature_engineering import build_feature_matrix

    available_models = []
    if XGBOOST_MODEL_PATH.exists(): available_models.append("XGBoost")
    if LSTM_MODEL_PATH.exists():    available_models.append("LSTM")
    if GRU_MODEL_PATH.exists():     available_models.append("GRU")

    return {
        "SYMBOLS": SYMBOLS,
        "DEFAULT_SYMBOL": DEFAULT_SYMBOL,
        "available_models": available_models,
        "predict_stock": predict_stock,
        "get_sentiment": get_sentiment,
        "get_confidence": get_confidence,
        "get_recommendation": get_recommendation,
        "get_latest_stock_data": get_latest_stock_data,
        "predict_all_models": predict_all_models,
        "download_historical_data": download_historical_data,
        "download_latest_data": download_latest_data,
        "build_feature_matrix": build_feature_matrix,
        "models_ready": len(available_models) > 0,
    }


# ── Helper renderers ─────────────────────────────────────────────────────────

def _badge(label: str) -> str:
    cls = {"BUY": "buy", "SELL": "sell", "HOLD": "hold",
           "UP": "up", "DOWN": "down"}.get(label.upper(), "hold")
    return f'<span class="badge-{cls}">{label}</span>'


def _delta_color(val: float) -> str:
    return "#00c851" if val >= 0 else "#ff4444"


def _fmt_volume(v: int) -> str:
    if v >= 1_000_000_000:
        return f"{v/1e9:.2f}B"
    if v >= 1_000_000:
        return f"{v/1e6:.2f}M"
    if v >= 1_000:
        return f"{v/1e3:.1f}K"
    return str(v)


def _confidence_gauge(confidence: float, title: str = "Confidence") -> go.Figure:
    """Plotly gauge chart for model confidence."""
    color = "#00c851" if confidence >= 80 else "#ffbb33" if confidence >= 60 else "#ff4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"size": 28, "color": "#e8eaf6"}},
        title={"text": title, "font": {"size": 14, "color": "#8b92b0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b92b0", "tickfont": {"color": "#8b92b0"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1e2130",
            "bordercolor": "#3a3d5c",
            "steps": [
                {"range": [0, 60],  "color": "#2a1f2f"},
                {"range": [60, 80], "color": "#2a2a1f"},
                {"range": [80, 100],"color": "#1f2a1f"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": confidence},
        },
    ))
    fig.update_layout(
        height=200,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#1e2130",
        font_color="#e8eaf6",
    )
    return fig


def _price_chart(price_history: list[dict], symbol: str) -> go.Figure:
    """Candlestick-style area chart from price history."""
    if not price_history:
        return go.Figure()

    dates  = [r["date"]  for r in price_history]
    closes = [r["close"] for r in price_history]
    vols   = [r["volume"] for r in price_history]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=(f"{symbol} — Close Price (30d)", "Volume"),
    )

    # Price area
    fig.add_trace(go.Scatter(
        x=dates, y=closes,
        mode="lines",
        name="Close",
        line=dict(color="#7986cb", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(121,134,203,0.12)",
    ), row=1, col=1)

    # Volume bars
    colors = ["#00c851" if i == 0 or closes[i] >= closes[i-1] else "#ff4444"
              for i in range(len(closes))]
    fig.add_trace(go.Bar(
        x=dates, y=vols,
        name="Volume",
        marker_color=colors,
        opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        height=420,
        paper_bgcolor="#1e2130",
        plot_bgcolor="#1e2130",
        font_color="#e8eaf6",
        showlegend=False,
        margin=dict(t=40, b=20, l=10, r=10),
        xaxis=dict(showgrid=False, color="#8b92b0"),
        yaxis=dict(showgrid=True, gridcolor="#2a2d45", color="#8b92b0"),
        xaxis2=dict(showgrid=False, color="#8b92b0"),
        yaxis2=dict(showgrid=True, gridcolor="#2a2d45", color="#8b92b0"),
    )
    fig.update_annotations(font_color="#8b92b0", font_size=12)
    return fig


def _comparison_chart(comparison_data: list[dict]) -> go.Figure:
    """Grouped bar chart for model metric comparison."""
    if not comparison_data:
        return go.Figure()

    models  = [d["model"]     for d in comparison_data]
    acc     = [d.get("accuracy",  0) * 100 for d in comparison_data]
    prec    = [d.get("precision", 0) * 100 for d in comparison_data]
    rec     = [d.get("recall",    0) * 100 for d in comparison_data]
    f1      = [d.get("f1_score",  0) * 100 for d in comparison_data]

    palette = ["#7986cb", "#4db6ac", "#ffb74d", "#ef9a9a"]
    fig = go.Figure()
    for name, vals, color in zip(
        ["Accuracy", "Precision", "Recall", "F1 Score"],
        [acc, prec, rec, f1],
        palette,
    ):
        fig.add_trace(go.Bar(name=name, x=models, y=vals, marker_color=color))

    fig.update_layout(
        barmode="group",
        height=340,
        paper_bgcolor="#1e2130",
        plot_bgcolor="#1e2130",
        font_color="#e8eaf6",
        legend=dict(bgcolor="#1e2130", bordercolor="#3a3d5c", orientation="h", y=1.12),
        yaxis=dict(range=[0, 105], showgrid=True, gridcolor="#2a2d45", title="Score (%)"),
        xaxis=dict(showgrid=False),
        margin=dict(t=50, b=20, l=10, r=10),
    )
    return fig


def _sentiment_donut(scores: dict) -> go.Figure:
    """Donut chart for sentiment breakdown."""
    labels = ["Positive", "Negative", "Neutral"]
    values = [
        scores.get("sentiment_positive", 0),
        scores.get("sentiment_negative", 0),
        scores.get("sentiment_neutral",  1),
    ]
    colors = ["#00c851", "#ff4444", "#8b92b0"]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=colors,
        textinfo="percent",
        textfont_size=12,
        textfont_color="#fff",
    ))
    fig.update_layout(
        height=220,
        paper_bgcolor="#1e2130",
        font_color="#e8eaf6",
        showlegend=True,
        legend=dict(bgcolor="#1e2130", font_size=11),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    return fig


# ── Load backend ──────────────────────────────────────────────────────────────

try:
    backend = _load_backend()
    MODELS_READY = backend["models_ready"]
    AVAILABLE_MODELS = backend["available_models"]
    SYMBOLS = backend["SYMBOLS"]
except Exception as _boot_err:
    st.error(f"Backend failed to load: {_boot_err}")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 StockSense AI")
    st.markdown("*Financial prediction powered by Deep Learning & Sentiment Analysis*")
    st.markdown("---")

    selected_symbol = st.selectbox(
        "Select Stock Symbol",
        options=SYMBOLS,
        index=0,
        help="Choose the stock you want to analyse",
    )

    if AVAILABLE_MODELS:
        selected_model = st.selectbox(
            "Prediction Model",
            options=AVAILABLE_MODELS,
            index=0,
            help="XGBoost = fastest  |  LSTM/GRU = sequence-aware",
        )
    else:
        selected_model = None
        st.warning("No trained models found.\nRun the training scripts first.")

    st.markdown("---")
    st.markdown("### News Headlines (optional)")
    headlines_input = st.text_area(
        "Enter headlines for sentiment analysis",
        placeholder="Apple hits all-time high on strong earnings\nFed signals rate pause ahead",
        height=110,
        help="One headline per line. Leave blank to skip sentiment enrichment.",
    )
    headlines = [h.strip() for h in headlines_input.strip().splitlines() if h.strip()]

    run_prediction = st.button(
        "🔮 Run Prediction",
        type="primary",
        use_container_width=True,
        disabled=not MODELS_READY,
    )

    st.markdown("---")
    st.markdown("### All Models Consensus")
    run_all = st.button(
        "⚡ Run All Models",
        use_container_width=True,
        disabled=not MODELS_READY,
    )

    st.markdown("---")
    with st.expander("ℹ️ Recommendation Logic"):
        st.markdown("""
| Signal | Condition |
|--------|-----------|
| 🟢 **BUY**  | UP + confidence ≥ 80% |
| 🟡 **HOLD** | UP + confidence 60–80% |
| 🔴 **SELL** | DOWN + confidence ≥ 80% |
| 🟡 **HOLD** | otherwise |
        """)

    st.markdown("---")
    st.caption("⚠️ For educational use only. Not financial advice.")


# ── Main content ──────────────────────────────────────────────────────────────

st.markdown(f"# 📈 StockSense AI")
st.markdown(f"**AI-powered stock market prediction** · {selected_symbol} · {datetime.now().strftime('%B %d, %Y')}")
st.markdown("---")

# ── Section 1: Latest Stock Data ─────────────────────────────────────────────

st.markdown('<div class="section-header">📊 Market Overview</div>', unsafe_allow_html=True)

with st.spinner(f"Fetching latest data for {selected_symbol}…"):
    try:
        stock_data = backend["get_latest_stock_data"](selected_symbol)
    except Exception as e:
        st.error(f"Could not fetch stock data: {e}")
        stock_data = {}

if stock_data:
    col1, col2, col3, col4, col5 = st.columns(5)
    ret = stock_data.get("daily_return_pct", 0)
    ret_color = _delta_color(ret)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Last Close</div>
            <div class="metric-value">${stock_data.get('latest_close', '—')}</div>
            <div class="metric-sub" style="color:{ret_color}">
                {'+' if ret >= 0 else ''}{ret:.2f}% today
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Open</div>
            <div class="metric-value">${stock_data.get('latest_open', '—')}</div>
            <div class="metric-sub">{stock_data.get('date', '')}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Day High</div>
            <div class="metric-value" style="color:#00c851">${stock_data.get('latest_high', '—')}</div>
            <div class="metric-sub">intraday max</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Day Low</div>
            <div class="metric-value" style="color:#ff4444">${stock_data.get('latest_low', '—')}</div>
            <div class="metric-sub">intraday min</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Volume</div>
            <div class="metric-value">{_fmt_volume(stock_data.get('latest_volume', 0))}</div>
            <div class="metric-sub">{stock_data.get('sector', 'N/A')}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Price chart
    ph = stock_data.get("price_history", [])
    if ph:
        st.plotly_chart(_price_chart(ph, selected_symbol), use_container_width=True)

st.markdown("---")

# ── Section 2: Prediction Panel ───────────────────────────────────────────────

st.markdown('<div class="section-header">🔮 Prediction Engine</div>', unsafe_allow_html=True)

# State container — prediction results live here
if "pred_result"     not in st.session_state: st.session_state.pred_result     = None
if "all_pred_result" not in st.session_state: st.session_state.all_pred_result = None
if "sentiment_result"not in st.session_state: st.session_state.sentiment_result= None

# Run single model prediction
if run_prediction and MODELS_READY and selected_model:
    with st.spinner(f"Running {selected_model} prediction…"):
        try:
            result = backend["predict_stock"](
                symbol=selected_symbol,
                model_name=selected_model.upper(),
                headlines=headlines or None,
            )
            st.session_state.pred_result = result
            if headlines:
                st.session_state.sentiment_result = backend["get_sentiment"](headlines)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Run all models prediction
if run_all and MODELS_READY:
    with st.spinner("Running all models…"):
        try:
            result = backend["predict_all_models"](
                symbol=selected_symbol,
                headlines=headlines or None,
            )
            st.session_state.all_pred_result = result
            if headlines:
                st.session_state.sentiment_result = backend["get_sentiment"](headlines)
        except Exception as e:
            st.error(f"All-model prediction failed: {e}")

# ── Display single-model result ───────────────────────────────────────────────

if st.session_state.pred_result:
    res = st.session_state.pred_result
    col_a, col_b, col_c = st.columns([1, 1, 1])

    with col_a:
        st.markdown("##### Prediction")
        st.markdown(
            f'<div style="text-align:center;margin-top:8px">'
            f'{_badge(res["prediction"])}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<p style="text-align:center;color:#8b92b0;font-size:0.8rem;margin-top:6px">'
            f'Next-day direction  ·  {res["model"]}</p>',
            unsafe_allow_html=True
        )

    with col_b:
        st.plotly_chart(
            _confidence_gauge(res["confidence"], "Model Confidence"),
            use_container_width=True,
        )

    with col_c:
        st.markdown("##### Recommendation")
        st.markdown(
            f'<div style="text-align:center;margin-top:8px">'
            f'{_badge(res["recommendation"])}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<p style="text-align:center;color:#8b92b0;font-size:0.8rem;margin-top:6px">'
            f'Confidence: {res["confidence"]:.1f}%  ·  '
            f'{res.get("timestamp","")[:16].replace("T"," ")}</p>',
            unsafe_allow_html=True
        )

# ── Display all-models result ─────────────────────────────────────────────────

if st.session_state.all_pred_result:
    st.markdown("---")
    all_res = st.session_state.all_pred_result
    st.markdown(
        f"##### Consensus Signal: {_badge(all_res['consensus'])}  "
        f"<span style='color:#8b92b0;font-size:0.85rem'>"
        f"({selected_symbol} · {all_res.get('timestamp','')[:16].replace('T',' ')})</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    model_cols = st.columns(len(all_res["models"]))
    comparison_rows = []
    for col, (mname, mdata) in zip(model_cols, all_res["models"].items()):
        with col:
            if "error" in mdata:
                st.error(f"{mname}: {mdata['error']}")
            else:
                st.markdown(f"**{mname}**")
                st.markdown(
                    _badge(mdata["prediction"]) + "&nbsp;&nbsp;" + _badge(mdata["recommendation"]),
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    _confidence_gauge(mdata["confidence"], ""),
                    use_container_width=True,
                )
                comparison_rows.append({
                    "Model": mname,
                    "Prediction": mdata["prediction"],
                    "Confidence": f"{mdata['confidence']:.1f}%",
                    "Signal": mdata["recommendation"],
                })

    if comparison_rows:
        st.dataframe(
            pd.DataFrame(comparison_rows).set_index("Model"),
            use_container_width=True,
        )

st.markdown("---")

# ── Section 3: Sentiment Analysis ────────────────────────────────────────────

st.markdown('<div class="section-header">🧠 Sentiment Analysis (FinBERT)</div>', unsafe_allow_html=True)

sent_col1, sent_col2 = st.columns([1, 1])

with sent_col1:
    if not headlines:
        st.info("Enter news headlines in the sidebar to activate sentiment analysis.")
    elif st.session_state.sentiment_result:
        s = st.session_state.sentiment_result
        st.plotly_chart(_sentiment_donut(s), use_container_width=True)
    else:
        if st.button("Analyse Headlines", use_container_width=True):
            with st.spinner("Running FinBERT…"):
                try:
                    st.session_state.sentiment_result = backend["get_sentiment"](headlines)
                    st.rerun()
                except Exception as e:
                    st.error(f"Sentiment analysis failed: {e}")

with sent_col2:
    if st.session_state.sentiment_result:
        s = st.session_state.sentiment_result
        st.markdown(f"""
| Metric | Score |
|--------|-------|
| Overall Label | **{s.get('overall_label','—').title()}** |
| Positive Score | {s.get('sentiment_positive', 0):.3f} |
| Negative Score | {s.get('sentiment_negative', 0):.3f} |
| Neutral Score  | {s.get('sentiment_neutral',  0):.3f} |
| Composite Score | **{s.get('sentiment_score', 0):.3f}** |
| Headlines analysed | {s.get('n_headlines', 0)} |
        """)

        score = s.get("sentiment_score", 0)
        if score > 0.3:
            st.success("📈 Strong positive sentiment — may support upward movement")
        elif score > 0:
            st.info("🔄 Mild positive sentiment")
        elif score > -0.3:
            st.warning("🔄 Mild negative sentiment")
        else:
            st.error("📉 Strong negative sentiment — may pressure prices")
    elif headlines:
        st.markdown("*Run a prediction or click 'Analyse Headlines' to see results.*")

st.markdown("---")

# ── Section 4: Model Comparison Table ────────────────────────────────────────

st.markdown('<div class="section-header">📋 Model Performance Comparison</div>', unsafe_allow_html=True)

COMPARISON_CSV = ROOT / "results" / "comparison.csv"
METRICS_CSV    = ROOT / "results" / "metrics.csv"

if COMPARISON_CSV.exists():
    try:
        comp_df = pd.read_csv(COMPARISON_CSV)
        # Round numeric columns for display
        num_cols = comp_df.select_dtypes(include="number").columns
        disp_df = comp_df.copy()
        for c in num_cols:
            if c not in ("rank",):
                disp_df[c] = disp_df[c].apply(lambda x: f"{x*100:.2f}%" if x <= 1.01 else f"{x:.4f}")

        st.dataframe(disp_df.set_index("model") if "model" in disp_df.columns else disp_df,
                     use_container_width=True)

        # Comparison bar chart
        plot_data = comp_df.to_dict(orient="records")
        st.plotly_chart(_comparison_chart(plot_data), use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load comparison table: {e}")
else:
    st.info("Run `python src/evaluate_models.py` to generate the model comparison table.")

    # If we have metrics.csv instead, use that
    if METRICS_CSV.exists():
        try:
            metrics_df = pd.read_csv(METRICS_CSV)
            st.markdown("**Available metrics (from training):**")
            st.dataframe(metrics_df, use_container_width=True)
            st.plotly_chart(_comparison_chart(metrics_df.to_dict(orient="records")),
                            use_container_width=True)
        except Exception:
            pass

st.markdown("---")

# ── Section 5: Technical Features Explorer ────────────────────────────────────

st.markdown('<div class="section-header">🔬 Technical Feature Explorer</div>', unsafe_allow_html=True)

with st.expander("View recent technical indicators", expanded=False):
    with st.spinner(f"Building feature matrix for {selected_symbol}…"):
        try:
            raw_df = backend["download_latest_data"](selected_symbol)
            feat_df = backend["build_feature_matrix"](raw_df)

            display_cols = [
                "Close", "Daily_Return", "RSI", "MACD", "MACD_Signal",
                "EMA_20", "SMA_10", "SMA_50", "Volatility", "ATR",
            ]
            available = [c for c in display_cols if c in feat_df.columns]
            show_df = feat_df[available].tail(20).copy()

            # Format for readability
            for col in show_df.columns:
                if col == "Close":
                    show_df[col] = show_df[col].apply(lambda x: f"${x:.2f}")
                elif col == "Daily_Return":
                    show_df[col] = show_df[col].apply(lambda x: f"{x*100:.2f}%")
                else:
                    show_df[col] = show_df[col].apply(lambda x: f"{x:.3f}")

            show_df.index = show_df.index.strftime("%Y-%m-%d")
            st.dataframe(show_df, use_container_width=True)

            # Mini RSI chart
            if "RSI" in feat_df.columns:
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(
                    x=feat_df.index.strftime("%Y-%m-%d"),
                    y=feat_df["RSI"],
                    mode="lines",
                    line=dict(color="#ffb74d", width=2),
                    name="RSI",
                ))
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="#ff4444", annotation_text="Overbought (70)")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="#00c851", annotation_text="Oversold (30)")
                rsi_fig.update_layout(
                    title="RSI (14)",
                    height=220,
                    paper_bgcolor="#1e2130",
                    plot_bgcolor="#1e2130",
                    font_color="#e8eaf6",
                    yaxis=dict(showgrid=True, gridcolor="#2a2d45"),
                    xaxis=dict(showgrid=False),
                    margin=dict(t=40, b=20, l=10, r=10),
                )
                st.plotly_chart(rsi_fig, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not build features: {e}")

st.markdown("---")

# ── Section 6: Recent Predictions Log ────────────────────────────────────────

PREDICTIONS_CSV = ROOT / "results" / "predictions.csv"

if PREDICTIONS_CSV.exists():
    with st.expander("📁 Recent Predictions Log", expanded=False):
        try:
            pred_log = pd.read_csv(PREDICTIONS_CSV)
            if "date" in pred_log.columns:
                pred_log = pred_log.sort_values("date", ascending=False).head(30)
            st.dataframe(pred_log, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load predictions log: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#4a4d6a;font-size:0.78rem">'
    'StockSense AI · Built with XGBoost, LSTM, GRU & FinBERT · '
    'For educational purposes only · Not financial advice'
    '</p>',
    unsafe_allow_html=True,
)
