"""
data_collection.py
------------------
Download historical and latest stock data via yfinance.
All results are cached as Parquet for fast re-use.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# Import project config and utils
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR,
    SYMBOLS,
    HISTORICAL_START,
    HISTORICAL_END,
    LATEST_PERIOD,
    INTERVAL,
)
from utils import get_logger, validate_dataframe

_log = get_logger(__name__)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _parquet_path(symbol: str, tag: str) -> Path:
    """Return the Parquet cache path for a given symbol and data tag."""
    return DATA_DIR / f"{symbol}_{tag}.parquet"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance ≥ 0.2.x returns a MultiIndex (field, ticker) even for a single
    ticker when using yf.download().  Flatten to the field name only.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Level 0 = field name (Close, Open …), level 1 = ticker symbol
        df.columns = [col[0] for col in df.columns]
    # Strip any residual whitespace
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _clean_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Standardise OHLCV data:
    - Flatten multi-index columns
    - Rename to title-case
    - Drop rows with missing OHLCV values
    - Add 'Symbol' column
    """
    df = _flatten_columns(df)

    # Normalise column names to title-case (Open, High, Low, Close, Volume)
    rename_map = {c: c.strip().title() for c in df.columns}
    df = df.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{symbol}] Missing OHLCV columns after normalisation: {missing}")

    df = df[required].dropna()
    df["Symbol"] = symbol
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


# ─── Public API ───────────────────────────────────────────────────────────────

def download_historical_data(
    symbol: str,
    start: str = HISTORICAL_START,
    end: str = HISTORICAL_END,
    interval: str = INTERVAL,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download historical OHLCV data for *symbol* between *start* and *end*.

    Results are cached as Parquet.  Pass ``force_refresh=True`` to re-download.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. "AAPL".
    start : str
        ISO date string "YYYY-MM-DD".
    end : str
        ISO date string "YYYY-MM-DD".
    interval : str
        Bar interval, default "1d".
    force_refresh : bool
        When True the cache is ignored.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns: Open, High, Low, Close, Volume, Symbol.
    """
    cache = _parquet_path(symbol, "historical")

    if cache.exists() and not force_refresh:
        _log.info("[%s] Loading historical data from cache: %s", symbol, cache)
        return pd.read_parquet(cache)

    _log.info("[%s] Downloading historical data %s → %s …", symbol, start, end)
    raw = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise RuntimeError(
            f"yfinance returned empty data for {symbol} ({start} → {end})."
        )

    df = _clean_ohlcv(raw, symbol)
    df.to_parquet(cache)
    _log.info("[%s] Historical data saved (%d rows) → %s", symbol, len(df), cache)
    return df


def download_latest_data(
    symbol: str,
    period: str = LATEST_PERIOD,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    """
    Download the most recent *period* of data for *symbol* (used at inference).

    This call always hits the network — no caching — so predictions are fresh.

    Parameters
    ----------
    symbol : str
    period : str
        yfinance period string such as "60d", "1mo", "3mo".
    interval : str

    Returns
    -------
    pd.DataFrame
    """
    _log.info("[%s] Downloading latest data (period=%s) …", symbol, period)
    raw = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise RuntimeError(f"yfinance returned empty latest data for {symbol}.")

    df = _clean_ohlcv(raw, symbol)
    _log.info("[%s] Latest data fetched (%d rows).", symbol, len(df))
    return df


def download_all_symbols(
    symbols: list[str] = SYMBOLS,
    start: str = HISTORICAL_START,
    end: str = HISTORICAL_END,
    force_refresh: bool = False,
    delay_seconds: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """
    Download historical data for every symbol in *symbols*.

    Parameters
    ----------
    symbols : list[str]
    start : str
    end : str
    force_refresh : bool
    delay_seconds : float
        Polite delay between API calls to avoid rate-limiting.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping symbol → DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = download_historical_data(
                sym, start=start, end=end, force_refresh=force_refresh
            )
            results[sym] = df
        except Exception as exc:
            _log.error("[%s] Failed to download: %s", sym, exc)
        time.sleep(delay_seconds)
    _log.info("Downloaded %d / %d symbols successfully.", len(results), len(symbols))
    return results


def get_stock_info(symbol: str) -> dict:
    """
    Fetch basic metadata for a ticker (name, sector, market cap, etc.).

    Returns
    -------
    dict
        Subset of yfinance's ``info`` dictionary.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "symbol": symbol,
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", None),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
        }
    except Exception as exc:
        _log.warning("[%s] Could not fetch info: %s", symbol, exc)
        return {"symbol": symbol}


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download stock data")
    parser.add_argument(
        "--symbols", nargs="+", default=SYMBOLS, help="Ticker symbols"
    )
    parser.add_argument("--start", default=HISTORICAL_START)
    parser.add_argument("--end", default=HISTORICAL_END)
    parser.add_argument(
        "--refresh", action="store_true", help="Force re-download"
    )
    args = parser.parse_args()

    data = download_all_symbols(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        force_refresh=args.refresh,
    )
    for sym, df in data.items():
        print(f"  {sym}: {len(df)} rows  |  {df.index[0].date()} → {df.index[-1].date()}")
