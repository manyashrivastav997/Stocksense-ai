"""
utils.py
--------
Shared utility functions: logging, validation, model I/O, feature helpers.
"""

import logging
import sys
import os
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import joblib

from config import (
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    XGBOOST_MODEL_PATH,
    LSTM_MODEL_PATH,
    GRU_MODEL_PATH,
    SCALER_PATH,
)


# ─── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger configured to write to stdout.

    Parameters
    ----------
    name : str
        Typically __name__ of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger


_log = get_logger(__name__)


# ─── Data Validation ──────────────────────────────────────────────────────────

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    name: str = "DataFrame",
) -> bool:
    """
    Assert that a DataFrame contains required columns and is non-empty.

    Parameters
    ----------
    df : pd.DataFrame
    required_columns : list[str]
    name : str
        Label used in error messages.

    Returns
    -------
    bool
        True if valid, raises ValueError otherwise.
    """
    if df is None or df.empty:
        raise ValueError(f"{name} is None or empty.")
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    return True


def validate_symbol(symbol: str, valid_symbols: list[str]) -> str:
    """
    Validate and normalise a ticker symbol.

    Parameters
    ----------
    symbol : str
    valid_symbols : list[str]

    Returns
    -------
    str
        Upper-cased symbol.

    Raises
    ------
    ValueError
        If symbol not in valid_symbols.
    """
    symbol = symbol.upper().strip()
    if symbol not in valid_symbols:
        raise ValueError(
            f"Symbol '{symbol}' not supported. Choose from: {valid_symbols}"
        )
    return symbol


# ─── Feature Processing Helpers ───────────────────────────────────────────────

def drop_na_rows(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Drop rows with any NaN, logging the count removed."""
    before = len(df)
    df = df.dropna()
    removed = before - len(df)
    if removed:
        _log.debug("%s — dropped %d NaN rows, %d remain.", context, removed, len(df))
    return df


def clip_outliers(
    df: pd.DataFrame,
    columns: list[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """
    Winsorise extreme values in the specified columns using quantile clipping.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
    lower_q : float
    upper_q : float

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            lo = df[col].quantile(lower_q)
            hi = df[col].quantile(upper_q)
            df[col] = df[col].clip(lo, hi)
    return df


def safe_divide(
    numerator: Union[pd.Series, np.ndarray, float],
    denominator: Union[pd.Series, np.ndarray, float],
    fill_value: float = 0.0,
) -> Union[pd.Series, np.ndarray, float]:
    """Division with zero-denominator protection."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / denominator, fill_value)
    if isinstance(numerator, pd.Series):
        return pd.Series(result, index=numerator.index)
    return result


# ─── Sequence Generation ──────────────────────────────────────────────────────

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over X/y to produce 3-D sequences for LSTM/GRU.

    Parameters
    ----------
    X : np.ndarray  shape (N, features)
    y : np.ndarray  shape (N,)
    sequence_length : int

    Returns
    -------
    X_seq : np.ndarray  shape (N - seq_len, seq_len, features)
    y_seq : np.ndarray  shape (N - seq_len,)
    """
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i - sequence_length : i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# ─── Model I/O ────────────────────────────────────────────────────────────────

def save_scaler(scaler: Any, path: Path = SCALER_PATH) -> None:
    """Persist a fitted scaler to disk."""
    joblib.dump(scaler, path)
    _log.info("Scaler saved → %s", path)


def load_scaler(path: Path = SCALER_PATH) -> Any:
    """Load a previously saved scaler."""
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found at {path}. Run training first.")
    scaler = joblib.load(path)
    _log.info("Scaler loaded ← %s", path)
    return scaler


def save_xgboost(model: Any, path: Path = XGBOOST_MODEL_PATH) -> None:
    """Persist XGBoost model."""
    joblib.dump(model, path)
    _log.info("XGBoost model saved → %s", path)


def load_xgboost(path: Path = XGBOOST_MODEL_PATH) -> Any:
    """Load XGBoost model."""
    if not path.exists():
        raise FileNotFoundError(f"XGBoost model not found at {path}.")
    model = joblib.load(path)
    _log.info("XGBoost model loaded ← %s", path)
    return model


def save_keras_model(model: Any, path: Path) -> None:
    """Persist a Keras model."""
    model.save(str(path))
    _log.info("Keras model saved → %s", path)


def load_keras_model(path: Path) -> Any:
    """Load a Keras model (compatible with TF 2.16 / Keras 3)."""
    if not path.exists():
        raise FileNotFoundError(f"Keras model not found at {path}.")
    try:
        import keras
        model = keras.models.load_model(str(path))
    except ImportError:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(path))
    _log.info("Keras model loaded ← %s", path)
    return model


# ─── Results I/O ──────────────────────────────────────────────────────────────

def save_dataframe(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    """Save a DataFrame to CSV, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    _log.info("DataFrame saved → %s  (%d rows)", path, len(df))


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a CSV into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    _log.info("DataFrame loaded ← %s  (%d rows)", path, len(df))
    return df


# ─── JSON helpers ─────────────────────────────────────────────────────────────

def to_json_safe(obj: Any) -> Any:
    """
    Recursively convert numpy / pandas types to native Python so that
    json.dumps() never raises TypeError.
    """
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def pretty_json(obj: Any) -> str:
    """Return a pretty-printed JSON string from any dict/list."""
    return json.dumps(to_json_safe(obj), indent=2)
