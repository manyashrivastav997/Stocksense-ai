"""
config.py
---------
Centralized configuration for StockSense AI.
All hyperparameters, paths, thresholds, and symbols live here.
"""

import os
from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
RESULTS_DIR: Path = ROOT_DIR / "results"
SRC_DIR: Path = ROOT_DIR / "src"

# Ensure directories exist at import time
for _dir in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Stock Symbols ────────────────────────────────────────────────────────────
SYMBOLS: list[str] = ["AAPL", "TSLA", "NVDA", "MSFT", "SPY"]
DEFAULT_SYMBOL: str = "AAPL"

# ─── Data Collection ─────────────────────────────────────────────────────────
HISTORICAL_START: str = "2018-01-01"
HISTORICAL_END: str = "2024-12-31"
INTERVAL: str = "1d"                   # daily bars
LATEST_PERIOD: str = "60d"            # for inference: last 60 days

# ─── Feature Engineering ─────────────────────────────────────────────────────
RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
EMA_PERIOD: int = 20
SMA_SHORT: int = 10
SMA_LONG: int = 50
VOLATILITY_WINDOW: int = 10
VOLUME_CHANGE_PERIOD: int = 1

# ─── Sequence / LSTM / GRU ────────────────────────────────────────────────────
SEQUENCE_LENGTH: int = 60             # look-back window (days)

# ─── Train / Test Split ──────────────────────────────────────────────────────
TEST_SIZE: float = 0.20               # 80/20 split
RANDOM_STATE: int = 42

# ─── XGBoost Hyperparameters ─────────────────────────────────────────────────
# NOTE: use_label_encoder was removed in XGBoost 2.x — do not add it back.
XGBOOST_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ─── LSTM Hyperparameters ─────────────────────────────────────────────────────
LSTM_UNITS: list[int] = [128, 64]
LSTM_DROPOUT: float = 0.30
LSTM_EPOCHS: int = 100
LSTM_BATCH_SIZE: int = 32
LSTM_PATIENCE: int = 15              # early stopping patience

# ─── GRU Hyperparameters ──────────────────────────────────────────────────────
GRU_UNITS: list[int] = [128, 64]
GRU_DROPOUT: float = 0.30
GRU_EPOCHS: int = 100
GRU_BATCH_SIZE: int = 32
GRU_PATIENCE: int = 15

# ─── Sentiment (FinBERT) ──────────────────────────────────────────────────────
FINBERT_MODEL: str = "ProsusAI/finbert"
SENTIMENT_MAX_LENGTH: int = 512
SENTIMENT_BATCH_SIZE: int = 16

# ─── Recommendation Thresholds ────────────────────────────────────────────────
BUY_THRESHOLD: float = 0.80          # confidence ≥ 80 % → BUY
HOLD_THRESHOLD: float = 0.60         # confidence 60–80 % → HOLD
# confidence < 60 % → SELL

# ─── Model File Paths ─────────────────────────────────────────────────────────
XGBOOST_MODEL_PATH: Path = MODELS_DIR / "xgboost_model.pkl"
LSTM_MODEL_PATH: Path = MODELS_DIR / "lstm_model.keras"
GRU_MODEL_PATH: Path = MODELS_DIR / "gru_model.keras"
SCALER_PATH: Path = MODELS_DIR / "scaler.pkl"

# ─── Results File Paths ───────────────────────────────────────────────────────
METRICS_CSV: Path = RESULTS_DIR / "metrics.csv"
PREDICTIONS_CSV: Path = RESULTS_DIR / "predictions.csv"
COMPARISON_CSV: Path = RESULTS_DIR / "comparison.csv"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
