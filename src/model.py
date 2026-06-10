"""
model.py  —  Day 3: Prediction Model
--------------------------------------
Trains a RandomForest classifier to predict next-day stock direction
(UP = 1 / DOWN = 0) using features built from the existing
feature_engineering pipeline.

Features used
-------------
  - Daily Return
  - SMA 10, SMA 50
  - RSI (14)
  - MACD, MACD Signal, MACD Histogram
  - EMA 20
  - Volume Change
  - Volatility (10-day rolling std of returns)
  - Price Range  = (High - Low) / Close
  - Close to Open = (Close - Open) / Open

Target
------
  1  →  next-day Close > today's Close  (UP)
  0  →  next-day Close ≤ today's Close  (DOWN)

Saved artefacts
---------------
  models/rf_model_<SYMBOL>.pkl   — fitted RandomForest
  models/rf_scaler_<SYMBOL>.pkl  — fitted StandardScaler

Run from project root:
    python src/model.py                   # train on AAPL
    python src/model.py --symbol TSLA
    python src/model.py --all-symbols
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR,
    MODELS_DIR,
    SYMBOLS,
    DEFAULT_SYMBOL,
    TEST_SIZE,
    RANDOM_STATE,
)
from data_collection import download_historical_data
from feature_engineering import build_feature_matrix, get_feature_columns
from utils import get_logger

_log = get_logger(__name__)

# ── RandomForest hyperparameters ──────────────────────────────────────────────
RF_PARAMS: dict = {
    "n_estimators":   200,
    "max_depth":      8,
    "min_samples_split": 10,
    "min_samples_leaf":  5,
    "class_weight":   "balanced",   # handles class imbalance
    "random_state":   RANDOM_STATE,
    "n_jobs":         -1,
}

# Feature columns used by this simple model (subset of full pipeline)
SIMPLE_FEATURES: list[str] = [
    "Daily_Return",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "EMA_20",
    "SMA_10",
    "SMA_50",
    "Volume_Change",
    "Volatility",
    "Price_Range",
    "Close_to_Open",
]


# ── Data Preparation ──────────────────────────────────────────────────────────

def prepare_features(symbol: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Download data, build feature matrix, and extract SIMPLE_FEATURES.

    Parameters
    ----------
    symbol : str

    Returns
    -------
    X : np.ndarray  shape (N, n_features)
    y : np.ndarray  shape (N,)
    used_cols : list[str]  — ordered feature column names
    """
    _log.info("[%s] Building feature matrix …", symbol)

    # Load from cache (data_collection caches to parquet automatically)
    raw = download_historical_data(symbol)

    # build_feature_matrix adds all technical indicators + Target
    feat_df = build_feature_matrix(raw)

    # Keep only the features available in the dataframe
    used_cols = [c for c in SIMPLE_FEATURES if c in feat_df.columns]
    missing = [c for c in SIMPLE_FEATURES if c not in feat_df.columns]
    if missing:
        _log.warning("[%s] Missing features (will be skipped): %s", symbol, missing)

    X = feat_df[used_cols].values.astype(np.float32)
    y = feat_df["Target"].values.astype(int)

    _log.info("[%s] Features shape: %s  |  Target distribution: UP=%d DOWN=%d",
              symbol, X.shape, y.sum(), (y == 0).sum())
    return X, y, used_cols


# ── Training Pipeline ─────────────────────────────────────────────────────────

def train_model(symbol: str = DEFAULT_SYMBOL, save: bool = True) -> dict:
    """
    Train a RandomForest classifier for *symbol*.

    Parameters
    ----------
    symbol : str
    save : bool
        If True, persist model and scaler to models/.

    Returns
    -------
    dict  with keys: model, scaler, metrics, feature_names, confusion_matrix
    """
    _log.info("=== Training RandomForest [%s] ===", symbol)

    X, y, feature_names = prepare_features(symbol)

    # ── Split (no shuffle — preserve time order) ──────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        shuffle=False,          # time series: no shuffling
        random_state=RANDOM_STATE,
    )

    # ── Scale ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Fit ───────────────────────────────────────────────────────────
    _log.info("[%s] Fitting RandomForest (%d trees) …", symbol, RF_PARAMS["n_estimators"])
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train_s, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["DOWN", "UP"])

    _log.info(
        "[%s] Acc: %.4f  Prec: %.4f  Rec: %.4f  F1: %.4f",
        symbol, acc, prec, rec, f1,
    )
    _log.info("\n%s", report)

    metrics = {
        "model":     "RandomForest",
        "symbol":    symbol,
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_score":  round(f1,   4),
        "n_train":   len(X_train),
        "n_test":    len(X_test),
    }

    # ── Save artefacts ─────────────────────────────────────────────────
    if save:
        model_path  = MODELS_DIR / f"rf_model_{symbol}.pkl"
        scaler_path = MODELS_DIR / f"rf_scaler_{symbol}.pkl"
        joblib.dump(model,  model_path)
        joblib.dump(scaler, scaler_path)
        _log.info("Model saved  → %s", model_path)
        _log.info("Scaler saved → %s", scaler_path)

    return {
        "model":           model,
        "scaler":          scaler,
        "metrics":         metrics,
        "feature_names":   feature_names,
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ── Inference helper ──────────────────────────────────────────────────────────

def load_model(symbol: str) -> tuple:
    """
    Load the saved RandomForest model and scaler for *symbol*.

    Returns
    -------
    (model, scaler)  — or raises FileNotFoundError
    """
    model_path  = MODELS_DIR / f"rf_model_{symbol}.pkl"
    scaler_path = MODELS_DIR / f"rf_scaler_{symbol}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model for {symbol} not found at {model_path}.\n"
            "Run:  python src/model.py --symbol {symbol}  first."
        )

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    _log.info("[%s] Model loaded ← %s", symbol, model_path)
    return model, scaler


def predict_next_day(symbol: str) -> dict:
    """
    Load the saved model and predict next-day direction for *symbol*.

    Uses the most recent row from the feature matrix as input.

    Returns
    -------
    dict  {symbol, prediction, confidence, direction}
    """
    raw     = download_historical_data(symbol)
    feat_df = build_feature_matrix(raw)

    used_cols = [c for c in SIMPLE_FEATURES if c in feat_df.columns]
    X_latest  = feat_df[used_cols].values[-1:].astype(np.float32)

    model, scaler = load_model(symbol)
    X_scaled = scaler.transform(X_latest)

    pred_label = int(model.predict(X_scaled)[0])
    confidence = float(model.predict_proba(X_scaled)[0, 1])

    result = {
        "symbol":     symbol,
        "prediction": pred_label,           # 0 or 1
        "direction":  "UP" if pred_label == 1 else "DOWN",
        "confidence": round(confidence * 100, 2),
    }
    _log.info("[%s] Prediction: %s (confidence %.1f%%)",
              symbol, result["direction"], result["confidence"])
    return result


# ── Train all symbols ─────────────────────────────────────────────────────────

def train_all_symbols(symbols: list[str] = SYMBOLS) -> pd.DataFrame:
    """Train RandomForest for every symbol and return a metrics table."""
    rows = []
    for sym in symbols:
        try:
            result = train_model(sym, save=True)
            rows.append(result["metrics"])
        except Exception as e:
            _log.error("[%s] Training failed: %s", sym, e)
    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RandomForest prediction model")
    parser.add_argument("--symbol",      default=DEFAULT_SYMBOL, help="Single ticker")
    parser.add_argument("--all-symbols", action="store_true",    help="Train for all symbols")
    parser.add_argument("--predict",     action="store_true",    help="Run inference after training")
    args = parser.parse_args()

    if args.all_symbols:
        metrics_df = train_all_symbols()
        print("\n=== RandomForest — All Symbols ===")
        print(metrics_df.to_string(index=False))
    else:
        symbol = args.symbol.upper()
        result = train_model(symbol)

        print(f"\n=== RandomForest Metrics [{symbol}] ===")
        for k, v in result["metrics"].items():
            print(f"  {k:<15}: {v}")

        print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
        print("            DOWN   UP")
        cm = result["confusion_matrix"]
        print(f"  Actual DOWN  {cm[0][0]:>5}  {cm[0][1]:>5}")
        print(f"  Actual UP    {cm[1][0]:>5}  {cm[1][1]:>5}")

        print("\nClassification Report:")
        print(result["classification_report"])

        if args.predict:
            pred = predict_next_day(symbol)
            print(f"\n=== Next-Day Prediction [{symbol}] ===")
            print(f"  Direction  : {pred['direction']}")
            print(f"  Confidence : {pred['confidence']:.1f}%")
