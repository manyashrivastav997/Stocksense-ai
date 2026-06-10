"""
feature_engineering.py
-----------------------
Compute technical indicators, daily returns, and volatility.
Merge with optional sentiment features to produce the final feature matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Import ta (Technical Analysis library)
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    EMA_PERIOD,
    SMA_SHORT,
    SMA_LONG,
    VOLATILITY_WINDOW,
    VOLUME_CHANGE_PERIOD,
    DATA_DIR,
)
from utils import get_logger, drop_na_rows, clip_outliers

_log = get_logger(__name__)


# ─── Technical Indicators ─────────────────────────────────────────────────────

def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'Daily_Return' = percentage change in Close."""
    df = df.copy()
    df["Daily_Return"] = df["Close"].pct_change()
    return df


def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Relative Strength Index."""
    df = df.copy()
    rsi = RSIIndicator(close=df["Close"], window=period)
    df["RSI"] = rsi.rsi()
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """MACD line, Signal line, and Histogram."""
    df = df.copy()
    macd = MACD(
        close=df["Close"],
        window_fast=fast,
        window_slow=slow,
        window_sign=signal,
    )
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    return df


def add_ema(df: pd.DataFrame, period: int = EMA_PERIOD) -> pd.DataFrame:
    """Exponential Moving Average."""
    df = df.copy()
    ema = EMAIndicator(close=df["Close"], window=period)
    df[f"EMA_{period}"] = ema.ema_indicator()
    return df


def add_sma(
    df: pd.DataFrame,
    short: int = SMA_SHORT,
    long: int = SMA_LONG,
) -> pd.DataFrame:
    """Short and Long Simple Moving Averages."""
    df = df.copy()
    df[f"SMA_{short}"] = SMAIndicator(close=df["Close"], window=short).sma_indicator()
    df[f"SMA_{long}"] = SMAIndicator(close=df["Close"], window=long).sma_indicator()
    return df


def add_volume_change(
    df: pd.DataFrame, periods: int = VOLUME_CHANGE_PERIOD
) -> pd.DataFrame:
    """Percentage change in Volume."""
    df = df.copy()
    df["Volume_Change"] = df["Volume"].pct_change(periods=periods)
    return df


def add_volatility(
    df: pd.DataFrame, window: int = VOLATILITY_WINDOW
) -> pd.DataFrame:
    """
    Rolling standard deviation of daily returns as a proxy for volatility.
    Also adds Average True Range (ATR).
    """
    df = df.copy()
    df["Volatility"] = df["Daily_Return"].rolling(window=window).std()
    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=window,
    )
    df["ATR"] = atr.average_true_range()
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalised price-position features:
    - Price_Range: (High - Low) / Close
    - Close_to_Open: (Close - Open) / Open
    """
    df = df.copy()
    df["Price_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Close_to_Open"] = (df["Close"] - df["Open"]) / df["Open"]
    return df


# ─── Target Variable ─────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target = 1 if next-day Close > today's Close, else 0.
    The last row will have NaN and is subsequently dropped.
    """
    df = df.copy()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def compute_all_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply every technical indicator in sequence and return the enriched DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data with columns: Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame
        Original columns + all technical features + Target.
    """
    _log.info("Computing technical features …")
    df = add_daily_returns(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_ema(df)
    df = add_sma(df)
    df = add_volume_change(df)
    df = add_volatility(df)
    df = add_price_features(df)
    df = add_target(df)
    _log.info("Technical features computed. Shape: %s", df.shape)
    return df


def merge_sentiment_features(
    tech_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Left-join sentiment scores onto the technical-feature DataFrame by Date.
    If sentiment_df is None or has no overlap, synthetic neutral scores are used
    so the pipeline never breaks.

    Parameters
    ----------
    tech_df : pd.DataFrame
        DatetimeIndex, all technical features.
    sentiment_df : pd.DataFrame | None
        DatetimeIndex (or 'Date' column), columns:
        sentiment_positive, sentiment_negative, sentiment_neutral, sentiment_score.

    Returns
    -------
    pd.DataFrame
    """
    _log.info("Merging sentiment features …")

    SENTIMENT_COLS = [
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_neutral",
        "sentiment_score",
    ]

    if sentiment_df is None or sentiment_df.empty:
        _log.warning("No sentiment data provided — using neutral placeholders.")
        for col in SENTIMENT_COLS:
            tech_df[col] = 0.0
        return tech_df

    # Ensure DatetimeIndex on sentiment side
    if "Date" in sentiment_df.columns:
        sentiment_df = sentiment_df.set_index("Date")
    sentiment_df.index = pd.to_datetime(sentiment_df.index)

    merged = tech_df.join(sentiment_df[SENTIMENT_COLS], how="left")
    merged[SENTIMENT_COLS] = merged[SENTIMENT_COLS].fillna(0.0)
    _log.info("Sentiment merge complete. Shape: %s", merged.shape)
    return merged


def get_feature_columns() -> list[str]:
    """
    Return the ordered list of feature columns expected by the models.
    Must match the columns produced by build_feature_matrix().
    """
    return [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Daily_Return",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        f"EMA_{EMA_PERIOD}",
        f"SMA_{SMA_SHORT}",
        f"SMA_{SMA_LONG}",
        "Volume_Change",
        "Volatility",
        "ATR",
        "Price_Range",
        "Close_to_Open",
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_neutral",
        "sentiment_score",
    ]


def build_feature_matrix(
    df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
    clip: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: technical features → merge sentiment → drop NaN → clip outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV (from data_collection).
    sentiment_df : pd.DataFrame | None
    clip : bool
        Whether to winsorise outlier feature values.

    Returns
    -------
    pd.DataFrame
        Clean feature matrix with 'Target' column.
    """
    df = compute_all_technical_features(df)
    df = merge_sentiment_features(df, sentiment_df)
    df = drop_na_rows(df, context="build_feature_matrix")

    if clip:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_safe = [c for c in numeric_cols if c != "Target"]
        df = clip_outliers(df, target_safe)

    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        _log.warning("Some expected features are absent: %s", missing)

    result = df[available + ["Target"]].copy()
    _log.info("Feature matrix ready. Shape: %s", result.shape)
    return result


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from data_collection import download_historical_data

    parser = argparse.ArgumentParser(description="Build feature matrix for a symbol")
    parser.add_argument("--symbol", default="AAPL")
    args = parser.parse_args()

    raw = download_historical_data(args.symbol)
    feat = build_feature_matrix(raw)
    print(feat.tail())
    print("\nFeature columns:", feat.columns.tolist())
    print("Shape:", feat.shape)

    out = DATA_DIR / f"{args.symbol}_features.parquet"
    feat.to_parquet(out)
    print(f"Saved → {out}")
