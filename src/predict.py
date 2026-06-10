"""
predict.py
----------
Prediction service layer — the primary interface for the Streamlit frontend.

All public functions return JSON-friendly dictionaries so they can be
called directly from Streamlit without any serialisation boilerplate.

Public API
----------
predict_stock(symbol, model_name)         → prediction dict
get_sentiment(headlines)                  → sentiment dict
get_confidence(symbol, model_name)        → confidence dict
get_recommendation(symbol, model_name)    → recommendation dict
get_latest_stock_data(symbol)             → latest OHLCV dict
predict_all_models(symbol)                → combined dict from all models
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_SYMBOL,
    SYMBOLS,
    SEQUENCE_LENGTH,
    XGBOOST_MODEL_PATH,
    LSTM_MODEL_PATH,
    GRU_MODEL_PATH,
    SCALER_PATH,
    BUY_THRESHOLD,
    HOLD_THRESHOLD,
    LATEST_PERIOD,
)
from data_collection import download_latest_data, get_stock_info
from feature_engineering import build_feature_matrix, get_feature_columns
from sentiment_analysis import FinBERTSentiment, get_latest_sentiment_score
from utils import (
    get_logger,
    load_xgboost,
    load_keras_model,
    load_scaler,
    create_sequences,
    to_json_safe,
    validate_symbol,
)

_log = get_logger(__name__)

# ─── Lazy model cache (avoids re-loading on every Streamlit re-run) ───────────
_model_cache: dict[str, object] = {}
_scaler_cache: Optional[object] = None
_sentiment_analyser: Optional[FinBERTSentiment] = None


def _get_scaler():
    global _scaler_cache
    if _scaler_cache is None:
        _scaler_cache = load_scaler(SCALER_PATH)
    return _scaler_cache


def _get_model(model_name: str):
    """Load and cache a model by name."""
    global _model_cache
    name = model_name.upper()
    if name not in _model_cache:
        if name == "XGBOOST":
            _model_cache[name] = load_xgboost(XGBOOST_MODEL_PATH)
        elif name == "LSTM":
            _model_cache[name] = load_keras_model(LSTM_MODEL_PATH)
        elif name == "GRU":
            _model_cache[name] = load_keras_model(GRU_MODEL_PATH)
        else:
            raise ValueError(f"Unknown model: {model_name}. Choose XGBoost, LSTM, or GRU.")
    return _model_cache[name]


def _get_sentiment_analyser() -> FinBERTSentiment:
    global _sentiment_analyser
    if _sentiment_analyser is None:
        _sentiment_analyser = FinBERTSentiment()
    return _sentiment_analyser


# ─── Feature preparation for inference ───────────────────────────────────────

def _prepare_inference_features(
    symbol: str,
    sentiment_scores: Optional[dict] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Fetch latest data, engineer features, return scaled array.

    Returns
    -------
    X_scaled : np.ndarray  shape (N, features)
    feat_df  : pd.DataFrame  (for metadata)
    """
    raw = download_latest_data(symbol, period=LATEST_PERIOD)

    # Build a minimal sentiment DataFrame for merging (last row = latest sentiment)
    if sentiment_scores is not None:
        n = len(raw)
        sent_df = pd.DataFrame(
            {
                "sentiment_positive": [sentiment_scores.get("sentiment_positive", 0.0)] * n,
                "sentiment_negative": [sentiment_scores.get("sentiment_negative", 0.0)] * n,
                "sentiment_neutral": [sentiment_scores.get("sentiment_neutral", 1.0)] * n,
                "sentiment_score": [sentiment_scores.get("sentiment_score", 0.0)] * n,
            },
            index=raw.index,
        )
    else:
        sent_df = None

    feat_df = build_feature_matrix(raw, sentiment_df=sent_df)
    feature_cols = [c for c in get_feature_columns() if c in feat_df.columns]

    scaler = _get_scaler()
    X_scaled = scaler.transform(feat_df[feature_cols].values)
    return X_scaled, feat_df


def _run_model_inference(
    X_scaled: np.ndarray,
    model_name: str,
    seq_len: int = SEQUENCE_LENGTH,
) -> tuple[int, float]:
    """
    Run the latest window through the specified model.

    Returns
    -------
    prediction : int   1 = UP, 0 = DOWN
    confidence : float  probability of UP  (0.0 – 1.0)
    """
    model = _get_model(model_name)
    name = model_name.upper()

    if name == "XGBOOST":
        # Use the most recent single observation
        x = X_scaled[-1:, :]              # shape (1, features)
        confidence = float(model.predict_proba(x)[0, 1])
        prediction = int(model.predict(x)[0])

    else:  # LSTM or GRU
        if len(X_scaled) < seq_len:
            raise ValueError(
                f"Not enough data for sequence (need {seq_len}, got {len(X_scaled)})."
            )
        x_seq = X_scaled[-seq_len:, :][np.newaxis, :, :]  # (1, seq_len, features)
        confidence = float(model.predict(x_seq, verbose=0)[0, 0])
        prediction = int(confidence >= 0.5)

    return prediction, confidence


# ─── Recommendation engine ────────────────────────────────────────────────────

def _make_recommendation(
    prediction: int,
    confidence: float,
    buy_threshold: float = BUY_THRESHOLD,
    hold_threshold: float = HOLD_THRESHOLD,
) -> str:
    """
    Translate prediction + confidence into BUY / HOLD / SELL.

    Logic
    -----
    UP   + confidence ≥ buy_threshold   → BUY
    UP   + confidence ≥ hold_threshold  → HOLD
    DOWN + confidence ≥ buy_threshold   → SELL
    otherwise                           → HOLD
    """
    if prediction == 1:
        if confidence >= buy_threshold:
            return "BUY"
        elif confidence >= hold_threshold:
            return "HOLD"
        else:
            return "HOLD"
    else:  # prediction == 0 → DOWN
        if confidence >= buy_threshold:
            return "SELL"
        else:
            return "HOLD"


# ─── Public API ───────────────────────────────────────────────────────────────

def predict_stock(
    symbol: str = DEFAULT_SYMBOL,
    model_name: str = "XGBOOST",
    headlines: Optional[list[str]] = None,
) -> dict:
    """
    Predict next-day stock movement.

    Parameters
    ----------
    symbol : str
        Ticker symbol (must be in SYMBOLS list).
    model_name : str
        "XGBOOST", "LSTM", or "GRU".
    headlines : list[str] | None
        Optional current news headlines for sentiment enrichment.

    Returns
    -------
    dict
        {
          "stock": "AAPL",
          "model": "XGBOOST",
          "prediction": "UP",
          "confidence": 84.7,
          "recommendation": "BUY",
          "timestamp": "2024-06-01T15:30:00"
        }
    """
    symbol = validate_symbol(symbol, SYMBOLS)
    _log.info("predict_stock(%s, %s) …", symbol, model_name)

    # Optional sentiment enrichment
    sentiment_scores = None
    if headlines:
        analyser = _get_sentiment_analyser()
        sentiment_scores = get_latest_sentiment_score(headlines, analyser)

    X_scaled, _ = _prepare_inference_features(symbol, sentiment_scores)
    prediction, confidence = _run_model_inference(X_scaled, model_name)
    recommendation = _make_recommendation(prediction, confidence)

    result = {
        "stock": symbol,
        "model": model_name.upper(),
        "prediction": "UP" if prediction == 1 else "DOWN",
        "confidence": round(confidence * 100, 2),
        "recommendation": recommendation,
        "timestamp": datetime.utcnow().isoformat(),
    }
    _log.info("Result: %s", result)
    return to_json_safe(result)


def get_sentiment(headlines: list[str]) -> dict:
    """
    Score a list of financial news headlines with FinBERT.

    Parameters
    ----------
    headlines : list[str]

    Returns
    -------
    dict
        {
          "sentiment_positive": 0.72,
          "sentiment_negative": 0.08,
          "sentiment_neutral": 0.20,
          "sentiment_score": 0.64,
          "overall_label": "positive",
          "n_headlines": 5
        }
    """
    if not headlines:
        return {
            "sentiment_positive": 0.0,
            "sentiment_negative": 0.0,
            "sentiment_neutral": 1.0,
            "sentiment_score": 0.0,
            "overall_label": "neutral",
            "n_headlines": 0,
        }

    analyser = _get_sentiment_analyser()
    scores = get_latest_sentiment_score(headlines, analyser)

    # Determine dominant label
    label_scores = {
        "positive": scores["sentiment_positive"],
        "negative": scores["sentiment_negative"],
        "neutral": scores["sentiment_neutral"],
    }
    overall = max(label_scores, key=label_scores.get)

    result = {**scores, "overall_label": overall, "n_headlines": len(headlines)}
    return to_json_safe(result)


def get_confidence(
    symbol: str = DEFAULT_SYMBOL,
    model_name: str = "XGBOOST",
) -> dict:
    """
    Return only the raw confidence score for a symbol.

    Returns
    -------
    dict  {"stock", "model", "confidence", "prediction"}
    """
    pred = predict_stock(symbol=symbol, model_name=model_name)
    return to_json_safe(
        {
            "stock": pred["stock"],
            "model": pred["model"],
            "confidence": pred["confidence"],
            "prediction": pred["prediction"],
        }
    )


def get_recommendation(
    symbol: str = DEFAULT_SYMBOL,
    model_name: str = "XGBOOST",
    headlines: Optional[list[str]] = None,
) -> dict:
    """
    Full recommendation including sentiment context.

    Returns
    -------
    dict  {"stock", "model", "prediction", "confidence",
           "recommendation", "sentiment", "timestamp"}
    """
    symbol = validate_symbol(symbol, SYMBOLS)

    sentiment_result: dict = {}
    if headlines:
        sentiment_result = get_sentiment(headlines)

    pred = predict_stock(symbol=symbol, model_name=model_name, headlines=headlines)

    result = {
        "stock": pred["stock"],
        "model": pred["model"],
        "prediction": pred["prediction"],
        "confidence": pred["confidence"],
        "recommendation": pred["recommendation"],
        "sentiment": sentiment_result,
        "timestamp": pred["timestamp"],
    }
    return to_json_safe(result)


def get_latest_stock_data(symbol: str = DEFAULT_SYMBOL) -> dict:
    """
    Fetch the most recent OHLCV data and basic stock metadata.

    Returns
    -------
    dict
        {
          "symbol": "AAPL",
          "name": "Apple Inc.",
          "latest_close": 189.45,
          "latest_open": 188.10,
          "latest_high": 190.30,
          "latest_low": 187.60,
          "latest_volume": 54321000,
          "date": "2024-05-31",
          "daily_return_pct": 0.71,
          "price_history": [{"date": ..., "close": ...}, ...]
        }
    """
    symbol = validate_symbol(symbol, SYMBOLS)
    _log.info("get_latest_stock_data(%s) …", symbol)

    df = download_latest_data(symbol)
    info = get_stock_info(symbol)

    latest = df.iloc[-1]
    prev_close = df.iloc[-2]["Close"] if len(df) > 1 else latest["Close"]
    daily_return = ((latest["Close"] - prev_close) / prev_close) * 100

    # Build compact price history (last 30 rows)
    history = df.tail(30)[["Close", "Volume"]].copy()
    history.index = history.index.strftime("%Y-%m-%d")
    price_history = [
        {"date": d, "close": round(float(row["Close"]), 2), "volume": int(row["Volume"])}
        for d, row in history.iterrows()
    ]

    result = {
        "symbol": symbol,
        "name": info.get("name", symbol),
        "sector": info.get("sector", "N/A"),
        "latest_close": round(float(latest["Close"]), 2),
        "latest_open": round(float(latest["Open"]), 2),
        "latest_high": round(float(latest["High"]), 2),
        "latest_low": round(float(latest["Low"]), 2),
        "latest_volume": int(latest["Volume"]),
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "daily_return_pct": round(float(daily_return), 2),
        "price_history": price_history,
    }
    return to_json_safe(result)


def predict_all_models(
    symbol: str = DEFAULT_SYMBOL,
    headlines: Optional[list[str]] = None,
) -> dict:
    """
    Run prediction with all three models and return a combined summary.

    Returns
    -------
    dict
        {
          "stock": "AAPL",
          "timestamp": "...",
          "sentiment": {...},
          "models": {
            "XGBoost": {"prediction": "UP", "confidence": 84.7, "recommendation": "BUY"},
            "LSTM":    {"prediction": "UP", "confidence": 76.2, "recommendation": "HOLD"},
            "GRU":     {"prediction": "UP", "confidence": 80.1, "recommendation": "BUY"}
          },
          "consensus": "BUY"
        }
    """
    symbol = validate_symbol(symbol, SYMBOLS)
    _log.info("predict_all_models(%s) …", symbol)

    model_names = []
    if XGBOOST_MODEL_PATH.exists():
        model_names.append("XGBOOST")
    if LSTM_MODEL_PATH.exists():
        model_names.append("LSTM")
    if GRU_MODEL_PATH.exists():
        model_names.append("GRU")

    if not model_names:
        raise RuntimeError("No trained models found. Run training scripts first.")

    sentiment_result = get_sentiment(headlines) if headlines else {}

    model_results: dict[str, dict] = {}
    for name in model_names:
        try:
            pred = predict_stock(symbol=symbol, model_name=name, headlines=headlines)
            model_results[name] = {
                "prediction": pred["prediction"],
                "confidence": pred["confidence"],
                "recommendation": pred["recommendation"],
            }
        except Exception as exc:
            _log.error("[%s] Prediction failed: %s", name, exc)
            model_results[name] = {"error": str(exc)}

    # ── Consensus vote ────────────────────────────────────────────────
    recommendations = [
        v["recommendation"]
        for v in model_results.values()
        if "recommendation" in v
    ]
    consensus = "HOLD"
    if recommendations:
        from collections import Counter
        vote = Counter(recommendations).most_common(1)[0][0]
        consensus = vote

    result = {
        "stock": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "sentiment": sentiment_result,
        "models": model_results,
        "consensus": consensus,
    }
    return to_json_safe(result)


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run stock prediction")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument(
        "--model", default="XGBOOST", choices=["XGBOOST", "LSTM", "GRU", "ALL"]
    )
    parser.add_argument(
        "--headlines",
        nargs="*",
        default=[],
        help="Optional news headlines for sentiment",
    )
    args = parser.parse_args()

    if args.model == "ALL":
        output = predict_all_models(symbol=args.symbol, headlines=args.headlines or None)
    else:
        output = predict_stock(
            symbol=args.symbol,
            model_name=args.model,
            headlines=args.headlines or None,
        )

    print(json.dumps(output, indent=2))
