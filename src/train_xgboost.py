"""
train_xgboost.py
----------------
Train an XGBoost binary classifier on the engineered feature matrix.

Steps:
  1. Download historical data
  2. Build feature matrix (technical + sentiment)
  3. Train / test split + scale
  4. Train XGBoost with early stopping
  5. Evaluate and save model + scaler
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_SYMBOL,
    SYMBOLS,
    XGBOOST_PARAMS,
    TEST_SIZE,
    RANDOM_STATE,
    XGBOOST_MODEL_PATH,
    SCALER_PATH,
    RESULTS_DIR,
    METRICS_CSV,
)
from data_collection import download_historical_data
from feature_engineering import build_feature_matrix, get_feature_columns
from utils import (
    get_logger,
    save_xgboost,
    save_scaler,
    save_dataframe,
    validate_dataframe,
)

_log = get_logger(__name__)


# ─── Training Pipeline ────────────────────────────────────────────────────────

def prepare_data(
    symbol: str = DEFAULT_SYMBOL,
    sentiment_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler]:
    """
    Download data, build features, split and scale.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, fitted_scaler
    """
    _log.info("[%s] Preparing XGBoost data …", symbol)

    raw = download_historical_data(symbol)
    feat_df = build_feature_matrix(raw, sentiment_df=sentiment_df)
    validate_dataframe(feat_df, ["Target"], name="feature_matrix")

    feature_cols = [c for c in get_feature_columns() if c in feat_df.columns]
    X = feat_df[feature_cols].values
    y = feat_df["Target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    _log.info(
        "Data split — train: %d, test: %d | features: %d",
        len(X_train),
        len(X_test),
        X_train.shape[1],
    )
    return X_train, X_test, y_train, y_test, feature_cols, scaler


def train_xgboost(
    symbol: str = DEFAULT_SYMBOL,
    sentiment_df: pd.DataFrame | None = None,
    save: bool = True,
) -> dict:
    """
    Full XGBoost training run.

    Parameters
    ----------
    symbol : str
    sentiment_df : pd.DataFrame | None
    save : bool
        Whether to persist model + scaler to disk.

    Returns
    -------
    dict  {model, scaler, metrics, feature_names}
    """
    _log.info("=== XGBoost Training [%s] ===", symbol)

    X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_data(
        symbol, sentiment_df
    )

    params = dict(XGBOOST_PARAMS)
    # early_stopping_rounds must be set at construction time in XGBoost 2.x
    params["early_stopping_rounds"] = 20
    model = XGBClassifier(**params)

    _log.info("Fitting XGBoost …")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluation ────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, target_names=["DOWN", "UP"])
    cm = confusion_matrix(y_test, y_pred)

    _log.info(
        "XGBoost results — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f",
        acc, prec, rec, f1,
    )
    _log.info("\n%s", report)

    metrics = {
        "model": "XGBoost",
        "symbol": symbol,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    }

    # ── Save artefacts ────────────────────────────────────────────────
    if save:
        save_xgboost(model, XGBOOST_MODEL_PATH)
        save_scaler(scaler, SCALER_PATH)

        metrics_df = pd.DataFrame([metrics])
        # Append to or create metrics.csv
        if METRICS_CSV.exists():
            existing = pd.read_csv(METRICS_CSV)
            # Remove any existing XGBoost row for this symbol, then append
            existing = existing[
                ~((existing["model"] == "XGBoost") & (existing["symbol"] == symbol))
            ]
            metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
        save_dataframe(metrics_df, METRICS_CSV, index=False)

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "feature_names": feature_cols,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def train_all_symbols(symbols: list[str] = SYMBOLS) -> pd.DataFrame:
    """
    Train XGBoost for every symbol and return a metrics comparison DataFrame.
    """
    all_metrics = []
    for sym in symbols:
        try:
            result = train_xgboost(symbol=sym, save=True)
            all_metrics.append(result["metrics"])
        except Exception as exc:
            _log.error("[%s] XGBoost training failed: %s", sym, exc)
    return pd.DataFrame(all_metrics)


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--all-symbols", action="store_true")
    args = parser.parse_args()

    if args.all_symbols:
        df = train_all_symbols()
        print("\n=== XGBoost Results (all symbols) ===")
        print(df.to_string(index=False))
    else:
        result = train_xgboost(symbol=args.symbol)
        print("\n=== XGBoost Metrics ===")
        for k, v in result["metrics"].items():
            print(f"  {k:15s}: {v}")
        print("\nConfusion Matrix:")
        print(result["confusion_matrix"])
