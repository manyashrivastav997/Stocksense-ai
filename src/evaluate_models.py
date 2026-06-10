"""
evaluate_models.py
------------------
Load all trained models, run evaluation on fresh test data,
generate a comparison table, confusion matrices, and save results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_SYMBOL,
    SEQUENCE_LENGTH,
    TEST_SIZE,
    RANDOM_STATE,
    XGBOOST_MODEL_PATH,
    LSTM_MODEL_PATH,
    GRU_MODEL_PATH,
    SCALER_PATH,
    METRICS_CSV,
    PREDICTIONS_CSV,
    COMPARISON_CSV,
    RESULTS_DIR,
)
from data_collection import download_historical_data
from feature_engineering import build_feature_matrix, get_feature_columns
from utils import (
    get_logger,
    load_xgboost,
    load_keras_model,
    load_scaler,
    save_dataframe,
    create_sequences,
    validate_dataframe,
)

_log = get_logger(__name__)


# ─── Shared data loader ───────────────────────────────────────────────────────

def _load_test_data(
    symbol: str,
    sentiment_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Return scaled test features, labels, and corresponding dates.

    Returns
    -------
    X_test_scaled : np.ndarray  (flat, for XGBoost)
    y_test        : np.ndarray
    test_dates    : pd.DatetimeIndex
    """
    raw = download_historical_data(symbol)
    feat_df = build_feature_matrix(raw, sentiment_df=sentiment_df)
    validate_dataframe(feat_df, ["Target"])

    feature_cols = [c for c in get_feature_columns() if c in feat_df.columns]
    X_raw = feat_df[feature_cols].values
    y_raw = feat_df["Target"].values
    dates = feat_df.index

    split_idx = int(len(X_raw) * (1 - TEST_SIZE))
    X_test_raw = X_raw[split_idx:]
    y_test = y_raw[split_idx:]
    test_dates = dates[split_idx:]

    scaler = load_scaler(SCALER_PATH)
    X_test_scaled = scaler.transform(X_test_raw)

    return X_test_scaled, y_test, test_dates


# ─── Per-model evaluation helpers ────────────────────────────────────────────

def _compute_metrics(
    model_name: str,
    symbol: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute a standardised metrics dictionary for one model."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    return {
        "model": model_name,
        "symbol": symbol,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
    }


def evaluate_xgboost(
    symbol: str = DEFAULT_SYMBOL,
    sentiment_df: pd.DataFrame | None = None,
) -> dict:
    """Evaluate the saved XGBoost model."""
    _log.info("[XGBoost] Evaluating on %s …", symbol)
    model = load_xgboost(XGBOOST_MODEL_PATH)

    X_test, y_test, test_dates = _load_test_data(symbol, sentiment_df)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = _compute_metrics("XGBoost", symbol, y_test, y_pred, y_prob)
    report = classification_report(y_test, y_pred, target_names=["DOWN", "UP"])
    cm = confusion_matrix(y_test, y_pred)

    _log.info("XGBoost — %s", metrics)
    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_test": y_test,
        "dates": test_dates,
        "report": report,
        "cm": cm,
    }


def evaluate_lstm(
    symbol: str = DEFAULT_SYMBOL,
    sentiment_df: pd.DataFrame | None = None,
    seq_len: int = SEQUENCE_LENGTH,
) -> dict:
    """Evaluate the saved LSTM model."""
    _log.info("[LSTM] Evaluating on %s …", symbol)
    model = load_keras_model(LSTM_MODEL_PATH)

    X_test_flat, y_test_flat, test_dates_flat = _load_test_data(symbol, sentiment_df)
    X_seq, y_seq = create_sequences(X_test_flat, y_test_flat, seq_len)

    y_prob = model.predict(X_seq, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    # Dates align to the end of each window
    seq_dates = test_dates_flat[seq_len:]

    metrics = _compute_metrics("LSTM", symbol, y_seq, y_pred, y_prob)
    report = classification_report(y_seq, y_pred, target_names=["DOWN", "UP"])
    cm = confusion_matrix(y_seq, y_pred)

    _log.info("LSTM — %s", metrics)
    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_test": y_seq,
        "dates": seq_dates,
        "report": report,
        "cm": cm,
    }


def evaluate_gru(
    symbol: str = DEFAULT_SYMBOL,
    sentiment_df: pd.DataFrame | None = None,
    seq_len: int = SEQUENCE_LENGTH,
) -> dict:
    """Evaluate the saved GRU model."""
    _log.info("[GRU] Evaluating on %s …", symbol)
    model = load_keras_model(GRU_MODEL_PATH)

    X_test_flat, y_test_flat, test_dates_flat = _load_test_data(symbol, sentiment_df)
    X_seq, y_seq = create_sequences(X_test_flat, y_test_flat, seq_len)

    y_prob = model.predict(X_seq, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    seq_dates = test_dates_flat[seq_len:]

    metrics = _compute_metrics("GRU", symbol, y_seq, y_pred, y_prob)
    report = classification_report(y_seq, y_pred, target_names=["DOWN", "UP"])
    cm = confusion_matrix(y_seq, y_pred)

    _log.info("GRU — %s", metrics)
    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_test": y_seq,
        "dates": seq_dates,
        "report": report,
        "cm": cm,
    }


# ─── Comparison & Visualisation ───────────────────────────────────────────────

def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a ranked comparison DataFrame from evaluate_* results.

    Parameters
    ----------
    results : dict  {model_name: evaluate_*() return dict}

    Returns
    -------
    pd.DataFrame sorted by F1 score descending
    """
    rows = [r["metrics"] for r in results.values()]
    df = pd.DataFrame(rows)
    df = df.sort_values("f1_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def plot_confusion_matrices(
    results: dict[str, dict],
    symbol: str,
    save_path: Path = RESULTS_DIR,
) -> None:
    """Plot and save confusion matrices for all models side by side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, results.items()):
        cm = res["cm"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["DOWN", "UP"],
            yticklabels=["DOWN", "UP"],
            ax=ax,
        )
        ax.set_title(f"{model_name} — {symbol}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    out = save_path / f"{symbol}_confusion_matrices.png"
    plt.savefig(out, dpi=120)
    plt.close()
    _log.info("Confusion matrix plot saved → %s", out)


def plot_comparison_bar(
    comparison_df: pd.DataFrame,
    symbol: str,
    save_path: Path = RESULTS_DIR,
) -> None:
    """Bar chart comparing accuracy / precision / recall / F1 across models."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    subset = comparison_df[["model"] + metrics_to_plot].set_index("model")

    ax = subset.plot(kind="bar", figsize=(10, 5), rot=0, width=0.7)
    ax.set_title(f"Model Comparison — {symbol}")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")

    plt.tight_layout()
    out = save_path / f"{symbol}_model_comparison.png"
    plt.savefig(out, dpi=120)
    plt.close()
    _log.info("Comparison bar chart saved → %s", out)


# ─── Main orchestrator ────────────────────────────────────────────────────────

def run_full_evaluation(
    symbol: str = DEFAULT_SYMBOL,
    sentiment_df: pd.DataFrame | None = None,
    generate_plots: bool = True,
) -> dict:
    """
    Evaluate all three models, generate comparison table, save results.

    Returns
    -------
    dict  {comparison_df, results, best_model}
    """
    _log.info("=== Full Model Evaluation [%s] ===", symbol)

    results: dict[str, dict] = {}

    # XGBoost
    if XGBOOST_MODEL_PATH.exists():
        results["XGBoost"] = evaluate_xgboost(symbol, sentiment_df)
    else:
        _log.warning("XGBoost model not found — skipping.")

    # LSTM
    if LSTM_MODEL_PATH.exists():
        results["LSTM"] = evaluate_lstm(symbol, sentiment_df)
    else:
        _log.warning("LSTM model not found — skipping.")

    # GRU
    if GRU_MODEL_PATH.exists():
        results["GRU"] = evaluate_gru(symbol, sentiment_df)
    else:
        _log.warning("GRU model not found — skipping.")

    if not results:
        raise RuntimeError("No trained models found. Run training scripts first.")

    comparison_df = build_comparison_table(results)
    save_dataframe(comparison_df, COMPARISON_CSV, index=False)

    # ── Save per-model metrics ────────────────────────────────────────
    all_metrics = pd.DataFrame([r["metrics"] for r in results.values()])
    save_dataframe(all_metrics, METRICS_CSV, index=False)

    # ── Save predictions (using XGBoost predictions as reference) ─────
    best_name = comparison_df.iloc[0]["model"]
    best_result = results[best_name]
    pred_df = pd.DataFrame(
        {
            "date": best_result["dates"],
            "y_true": best_result["y_test"],
            "y_pred": best_result["y_pred"],
            "probability": best_result["y_prob"],
        }
    )
    pred_df["direction_true"] = pred_df["y_true"].map({1: "UP", 0: "DOWN"})
    pred_df["direction_pred"] = pred_df["y_pred"].map({1: "UP", 0: "DOWN"})
    save_dataframe(pred_df, PREDICTIONS_CSV, index=False)

    if generate_plots:
        plot_confusion_matrices(results, symbol)
        plot_comparison_bar(comparison_df, symbol)

    _log.info("\n%s", comparison_df.to_string(index=False))
    _log.info("Best model: %s (F1=%.4f)", best_name, comparison_df.iloc[0]["f1_score"])

    return {
        "comparison_df": comparison_df,
        "results": results,
        "best_model": best_name,
    }


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    outcome = run_full_evaluation(
        symbol=args.symbol,
        generate_plots=not args.no_plots,
    )
    print("\n=== Model Comparison ===")
    print(outcome["comparison_df"].to_string(index=False))
    print(f"\nBest model: {outcome['best_model']}")
