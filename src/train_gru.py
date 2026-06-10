"""
train_gru.py
------------
Build and train a stacked GRU binary-classification model.

Architecture mirrors the LSTM model for fair comparison:
  GRU(128) → Dropout → GRU(64) → Dropout → Dense(32) → Dense(1, sigmoid)

The GRU model uses the scaler already fitted during LSTM training when
possible, but will fit a new one if needed.
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

import tensorflow as tf
# TF 2.16 bundles Keras 3 — import keras directly for stability
try:
    import keras
    from keras import layers, callbacks
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_SYMBOL,
    SYMBOLS,
    SEQUENCE_LENGTH,
    GRU_UNITS,
    GRU_DROPOUT,
    GRU_EPOCHS,
    GRU_BATCH_SIZE,
    GRU_PATIENCE,
    TEST_SIZE,
    RANDOM_STATE,
    GRU_MODEL_PATH,
    SCALER_PATH,
    METRICS_CSV,
    MODELS_DIR,
)
from data_collection import download_historical_data
from feature_engineering import build_feature_matrix, get_feature_columns
from utils import (
    get_logger,
    save_keras_model,
    save_scaler,
    load_scaler,
    save_dataframe,
    create_sequences,
    validate_dataframe,
)

_log = get_logger(__name__)

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ─── Model Definition ─────────────────────────────────────────────────────────

def build_gru_model(
    input_shape: tuple[int, int],
    units: list[int] = GRU_UNITS,
    dropout: float = GRU_DROPOUT,
) -> keras.Model:
    """
    Build a stacked GRU binary classifier.

    Parameters
    ----------
    input_shape : (sequence_length, n_features)
    units : list[int]
    dropout : float

    Returns
    -------
    keras.Model  (compiled)
    """
    model = keras.Sequential(name="GRU_Classifier")
    model.add(keras.Input(shape=input_shape))

    model.add(layers.GRU(units[0], return_sequences=True))
    model.add(layers.Dropout(dropout))

    model.add(layers.GRU(units[1], return_sequences=False))
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=_log.info)
    return model


# ─── Data Preparation ─────────────────────────────────────────────────────────

def prepare_data(
    symbol: str = DEFAULT_SYMBOL,
    sentiment_df: pd.DataFrame | None = None,
    seq_len: int = SEQUENCE_LENGTH,
    load_existing_scaler: bool = False,
) -> tuple:
    """
    Download, feature-engineer, scale, and create sequences for GRU input.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    _log.info("[%s] Preparing GRU data (seq_len=%d) …", symbol, seq_len)

    raw = download_historical_data(symbol)
    feat_df = build_feature_matrix(raw, sentiment_df=sentiment_df)
    validate_dataframe(feat_df, ["Target"], name="feature_matrix")

    feature_cols = [c for c in get_feature_columns() if c in feat_df.columns]
    X_raw = feat_df[feature_cols].values
    y_raw = feat_df["Target"].values

    split_idx = int(len(X_raw) * (1 - TEST_SIZE))
    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train_raw, y_test_raw = y_raw[:split_idx], y_raw[split_idx:]

    if load_existing_scaler and SCALER_PATH.exists():
        scaler = load_scaler(SCALER_PATH)
        X_train_scaled = scaler.transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

    X_train, y_train = create_sequences(X_train_scaled, y_train_raw, seq_len)
    X_test, y_test = create_sequences(X_test_scaled, y_test_raw, seq_len)

    _log.info("Sequences — train: %s, test: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test, scaler


# ─── Training Pipeline ────────────────────────────────────────────────────────

def train_gru(
    symbol: str = DEFAULT_SYMBOL,
    sentiment_df: pd.DataFrame | None = None,
    save: bool = True,
) -> dict:
    """
    Full GRU training pipeline.

    Parameters
    ----------
    symbol : str
    sentiment_df : pd.DataFrame | None
    save : bool

    Returns
    -------
    dict  {model, scaler, metrics, history}
    """
    _log.info("=== GRU Training [%s] ===", symbol)

    X_train, X_test, y_train, y_test, scaler = prepare_data(symbol, sentiment_df)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_gru_model(input_shape)

    # ── Callbacks ─────────────────────────────────────────────────────
    best_weights_path = str(MODELS_DIR / "gru_best_weights.keras")
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=GRU_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            best_weights_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    _log.info("Training GRU …")
    history = model.fit(
        X_train,
        y_train,
        epochs=GRU_EPOCHS,
        batch_size=GRU_BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=cbs,
        verbose=1,
    )

    # ── Evaluation ────────────────────────────────────────────────────
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, target_names=["DOWN", "UP"])
    cm = confusion_matrix(y_test, y_pred)

    _log.info(
        "GRU results — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f",
        acc, prec, rec, f1,
    )
    _log.info("\n%s", report)

    metrics = {
        "model": "GRU",
        "symbol": symbol,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    }

    # ── Save artefacts ────────────────────────────────────────────────
    if save:
        save_keras_model(model, GRU_MODEL_PATH)
        save_scaler(scaler, SCALER_PATH)

        metrics_df = pd.DataFrame([metrics])
        if METRICS_CSV.exists():
            existing = pd.read_csv(METRICS_CSV)
            existing = existing[
                ~((existing["model"] == "GRU") & (existing["symbol"] == symbol))
            ]
            metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
        save_dataframe(metrics_df, METRICS_CSV, index=False)

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "history": history.history,
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GRU model")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    args = parser.parse_args()

    result = train_gru(symbol=args.symbol)
    print("\n=== GRU Metrics ===")
    for k, v in result["metrics"].items():
        print(f"  {k:15s}: {v}")
    print("\nConfusion Matrix:")
    print(result["confusion_matrix"])
