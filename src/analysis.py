"""
analysis.py  —  Day 2: Visualization & Analysis
------------------------------------------------
Loads cached Parquet files from data/ and produces three chart types
per stock symbol:

  1. closing_price_<SYMBOL>.png  — close price with 50-day & 200-day MAs
  2. volume_trend_<SYMBOL>.png   — daily volume bar chart
  3. returns_dist_<SYMBOL>.png   — daily-return histogram + KDE

All plots are saved to the outputs/ folder.

Run from project root:
    python src/analysis.py                    # all symbols
    python src/analysis.py --symbol AAPL     # single symbol
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, SYMBOLS, DEFAULT_SYMBOL
from utils import get_logger

_log = get_logger(__name__)

# Where to save plots
OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d2e",
    "axes.edgecolor":    "#3a3d5c",
    "axes.labelcolor":   "#c5cae9",
    "axes.titlecolor":   "#e8eaf6",
    "xtick.color":       "#8b92b0",
    "ytick.color":       "#8b92b0",
    "grid.color":        "#2a2d45",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "text.color":        "#e8eaf6",
    "legend.facecolor":  "#1a1d2e",
    "legend.edgecolor":  "#3a3d5c",
    "legend.labelcolor": "#c5cae9",
    "font.size":         11,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_parquet(symbol: str) -> pd.DataFrame:
    """
    Load the cached Parquet file for *symbol*.
    Raises FileNotFoundError with a helpful message if not found.
    """
    path = DATA_DIR / f"{symbol}_historical.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No data found for {symbol} at {path}.\n"
            "Run:  python src/data_collection.py  first."
        )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    _log.info("[%s] Loaded %d rows from %s", symbol, len(df), path)
    return df


def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 50-day and 200-day simple moving averages on Close."""
    df = df.copy()
    df["MA50"]  = df["Close"].rolling(window=50,  min_periods=1).mean()
    df["MA200"] = df["Close"].rolling(window=200, min_periods=1).mean()
    return df


def _save(fig: plt.Figure, name: str) -> Path:
    """Save figure and return the path."""
    out = OUTPUTS_DIR / name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _log.info("Saved → %s", out)
    return out


# ── Chart 1: Closing Price + Moving Averages ─────────────────────────────────

def plot_closing_price(df: pd.DataFrame, symbol: str) -> Path:
    """
    Line chart of daily close price with 50-day and 200-day MAs overlaid.

    Parameters
    ----------
    df : pd.DataFrame   — OHLCV with DatetimeIndex
    symbol : str

    Returns
    -------
    Path  — saved file path
    """
    df = _add_moving_averages(df)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Close price filled area
    ax.fill_between(df.index, df["Close"], alpha=0.15, color="#7986cb")
    ax.plot(df.index, df["Close"], color="#7986cb", linewidth=1.4, label="Close")

    # 50-day MA
    ax.plot(df.index, df["MA50"],  color="#ffd54f", linewidth=1.2,
            linestyle="--", label="MA 50")

    # 200-day MA  (only draw where we have enough history)
    ma200_valid = df["MA200"].where(
        pd.Series(range(len(df)), index=df.index) >= 199
    )
    ax.plot(df.index, ma200_valid, color="#ef9a9a", linewidth=1.2,
            linestyle="-.", label="MA 200")

    # Formatting
    ax.set_title(f"{symbol} — Closing Price with Moving Averages", fontsize=14, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    return _save(fig, f"closing_price_{symbol}.png")


# ── Chart 2: Volume Trend ─────────────────────────────────────────────────────

def plot_volume_trend(df: pd.DataFrame, symbol: str) -> Path:
    """
    Daily volume bar chart coloured green (up day) / red (down day).
    A 20-day rolling average of volume is overlaid.

    Parameters
    ----------
    df : pd.DataFrame
    symbol : str

    Returns
    -------
    Path
    """
    df = df.copy()
    df["Vol_MA20"] = df["Volume"].rolling(window=20, min_periods=1).mean()

    # Colour bars by price direction
    colors = [
        "#00c851" if c >= o else "#ff4444"
        for c, o in zip(df["Close"], df["Open"])
    ]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(df.index, df["Volume"], color=colors, alpha=0.6, width=1.0, label="Volume")
    ax.plot(df.index, df["Vol_MA20"], color="#80cbc4", linewidth=1.5,
            label="Vol MA-20")

    ax.set_title(f"{symbol} — Volume Trend", fontsize=14, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    # Format y-axis in millions
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )

    return _save(fig, f"volume_trend_{symbol}.png")


# ── Chart 3: Daily Returns Distribution ──────────────────────────────────────

def plot_returns_distribution(df: pd.DataFrame, symbol: str) -> Path:
    """
    Histogram of daily percentage returns with a KDE overlay and
    ±1-std shaded region.

    Parameters
    ----------
    df : pd.DataFrame
    symbol : str

    Returns
    -------
    Path
    """
    returns = df["Close"].pct_change().dropna() * 100  # in percent

    mean_r = returns.mean()
    std_r  = returns.std()

    fig, ax = plt.subplots(figsize=(10, 4))

    # Histogram
    n, bins, patches = ax.hist(
        returns, bins=80, density=True, alpha=0.55,
        color="#7986cb", edgecolor="#3a3d5c", linewidth=0.3,
    )

    # Simple KDE using numpy
    kde_x = np.linspace(returns.min(), returns.max(), 300)
    kde_y = (
        (1 / (std_r * np.sqrt(2 * np.pi)))
        * np.exp(-0.5 * ((kde_x - mean_r) / std_r) ** 2)
    )
    ax.plot(kde_x, kde_y, color="#ffd54f", linewidth=2, label="Normal fit")

    # ±1 std shaded
    ax.axvspan(mean_r - std_r, mean_r + std_r,
               alpha=0.12, color="#7986cb", label="±1 std")
    ax.axvline(mean_r, color="#80cbc4", linewidth=1.5, linestyle="--",
               label=f"Mean {mean_r:.3f}%")
    ax.axvline(0, color="#ef9a9a", linewidth=1.0, linestyle=":")

    ax.set_title(f"{symbol} — Daily Returns Distribution", fontsize=14, pad=12)
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, axis="y")

    return _save(fig, f"returns_dist_{symbol}.png")


# ── Master runner ─────────────────────────────────────────────────────────────

def analyse_symbol(symbol: str) -> dict[str, Path]:
    """
    Run all three charts for one symbol and return a dict of saved paths.
    """
    _log.info("=== Analysing %s ===", symbol)
    df = _load_parquet(symbol)

    paths = {
        "closing_price":      plot_closing_price(df, symbol),
        "volume_trend":       plot_volume_trend(df, symbol),
        "returns_dist":       plot_returns_distribution(df, symbol),
    }

    _log.info("[%s] All charts saved to %s", symbol, OUTPUTS_DIR)
    return paths


def analyse_all(symbols: list[str] = SYMBOLS) -> dict[str, dict[str, Path]]:
    """Run analysis for every symbol and return nested path dict."""
    results: dict[str, dict[str, Path]] = {}
    for sym in symbols:
        try:
            results[sym] = analyse_symbol(sym)
        except FileNotFoundError as e:
            _log.warning(str(e))
        except Exception as e:
            _log.error("[%s] Analysis failed: %s", sym, e)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise stock data")
    parser.add_argument(
        "--symbol", default=None,
        help="Single ticker (omit to run all symbols)"
    )
    args = parser.parse_args()

    if args.symbol:
        paths = analyse_symbol(args.symbol.upper())
        for chart, p in paths.items():
            print(f"  {chart:20s} → {p}")
    else:
        all_paths = analyse_all()
        for sym, paths in all_paths.items():
            for chart, p in paths.items():
                print(f"  {sym} {chart:20s} → {p}")
