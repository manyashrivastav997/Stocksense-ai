# StockSense AI
### Financial Market Prediction using Sentiment Analysis and Deep Learning

A complete end-to-end stock market prediction system combining technical analysis,
FinBERT sentiment scoring, and three machine-learning models (XGBoost, LSTM, GRU).

---

## Project Structure

```
StockSenseAI/
├── data/                        # Auto-generated Parquet caches
├── models/                      # Saved model artefacts
│   ├── xgboost_model.pkl
│   ├── lstm_model.keras
│   ├── gru_model.keras
│   └── scaler.pkl
├── results/                     # CSVs + evaluation plots
│   ├── metrics.csv
│   ├── predictions.csv
│   └── comparison.csv
├── src/
│   ├── config.py                # All hyperparameters & paths
│   ├── data_collection.py       # yfinance downloader
│   ├── feature_engineering.py  # Technical indicators + target
│   ├── sentiment_analysis.py   # FinBERT wrapper
│   ├── train_xgboost.py        # XGBoost training
│   ├── train_lstm.py           # LSTM training
│   ├── train_gru.py            # GRU training
│   ├── evaluate_models.py      # Model comparison + plots
│   ├── predict.py              # Streamlit-ready prediction API
│   └── utils.py                # Shared utilities
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+.  
> GPU acceleration for TensorFlow/PyTorch is optional but recommended.

---

### 2. Download Historical Data

```bash
cd src
python data_collection.py --symbols AAPL TSLA NVDA MSFT SPY
```

Downloads and caches OHLCV data from 2018-01-01 → 2024-12-31.

---

### 3. Build Feature Matrix (optional standalone check)

```bash
python feature_engineering.py --symbol AAPL
```

---

### 4. Run Sentiment Analysis

Score a single headline:
```bash
python sentiment_analysis.py --text "Apple beats earnings expectations."
```

Validate the pipeline against Financial PhraseBank:
```bash
python sentiment_analysis.py --validate --n-samples 200
```

---

### 5. Train Models

```bash
# Train all three models for AAPL
python train_xgboost.py --symbol AAPL
python train_lstm.py    --symbol AAPL
python train_gru.py     --symbol AAPL

# Train XGBoost for all symbols
python train_xgboost.py --all-symbols
```

---

### 6. Evaluate & Compare

```bash
python evaluate_models.py --symbol AAPL
```

Outputs:
- `results/metrics.csv`
- `results/comparison.csv`
- `results/AAPL_confusion_matrices.png`
- `results/AAPL_model_comparison.png`

---

### 7. Make Predictions

```bash
# Single model
python predict.py --symbol AAPL --model XGBOOST

# All models with headlines
python predict.py --symbol AAPL --model ALL \
  --headlines "Apple surges on strong iPhone sales" "AAPL hits all-time high"
```

---

## Streamlit Integration

```python
import sys
sys.path.insert(0, "src")

from predict import (
    predict_stock,
    get_sentiment,
    get_confidence,
    get_recommendation,
    get_latest_stock_data,
    predict_all_models,
)

# Stock data
data = get_latest_stock_data("AAPL")

# Single prediction
result = predict_stock("AAPL", model_name="XGBOOST")

# Sentiment
sentiment = get_sentiment([
    "Apple reports record revenue",
    "Fed signals rate cuts ahead"
])

# Full recommendation with sentiment
rec = get_recommendation("AAPL", model_name="LSTM", headlines=[...])

# All models + consensus
all_preds = predict_all_models("AAPL")
```

---

## Example JSON Responses

### predict_stock()
```json
{
  "stock": "AAPL",
  "model": "XGBOOST",
  "prediction": "UP",
  "confidence": 84.7,
  "recommendation": "BUY",
  "timestamp": "2024-06-01T15:30:00"
}
```

### get_sentiment()
```json
{
  "sentiment_positive": 0.72,
  "sentiment_negative": 0.08,
  "sentiment_neutral": 0.20,
  "sentiment_score": 0.64,
  "overall_label": "positive",
  "n_headlines": 3
}
```

### get_latest_stock_data()
```json
{
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "sector": "Technology",
  "latest_close": 189.45,
  "latest_open": 188.10,
  "latest_high": 190.30,
  "latest_low": 187.60,
  "latest_volume": 54321000,
  "date": "2024-05-31",
  "daily_return_pct": 0.71,
  "price_history": [...]
}
```

### predict_all_models()
```json
{
  "stock": "AAPL",
  "timestamp": "2024-06-01T15:30:00",
  "sentiment": { "overall_label": "positive", "sentiment_score": 0.64 },
  "models": {
    "XGBOOST": { "prediction": "UP", "confidence": 84.7, "recommendation": "BUY" },
    "LSTM":    { "prediction": "UP", "confidence": 76.2, "recommendation": "HOLD" },
    "GRU":     { "prediction": "UP", "confidence": 80.1, "recommendation": "BUY" }
  },
  "consensus": "BUY"
}
```

---

## Recommendation Logic

| Condition                              | Signal |
|----------------------------------------|--------|
| Prediction = UP  + confidence ≥ 80 %  | BUY    |
| Prediction = UP  + confidence 60–80 % | HOLD   |
| Prediction = DOWN + confidence ≥ 80 % | SELL   |
| Otherwise                              | HOLD   |

Thresholds are configurable in `src/config.py`.

---

## Models

| Model    | Input          | Architecture                          |
|----------|----------------|---------------------------------------|
| XGBoost  | Flat features  | 300 trees, depth 6, lr 0.05           |
| LSTM     | 60-day window  | LSTM(128) → LSTM(64) → Dense(1)       |
| GRU      | 60-day window  | GRU(128)  → GRU(64)  → Dense(1)      |

All models include Dropout (0.30), BatchNormalization, EarlyStopping.

---

## Features Used

| Category   | Features                                              |
|------------|-------------------------------------------------------|
| Price      | Open, High, Low, Close, Volume                       |
| Returns    | Daily Return, Price Range, Close-to-Open             |
| Momentum   | RSI(14), MACD, MACD Signal, MACD Histogram           |
| Trend      | EMA(20), SMA(10), SMA(50)                            |
| Volatility | Rolling Volatility(10), ATR(10)                      |
| Volume     | Volume Change                                         |
| Sentiment  | Positive, Negative, Neutral Score, Composite Score   |

---

## Supported Symbols

`AAPL` · `TSLA` · `NVDA` · `MSFT` · `SPY`

---

## Tech Stack

- **Data**: yfinance, pandas, numpy
- **Features**: ta (Technical Analysis library)
- **Sentiment**: FinBERT (ProsusAI/finbert), HuggingFace Transformers
- **ML**: XGBoost, scikit-learn
- **DL**: TensorFlow / Keras, PyTorch (FinBERT inference)
- **Visualisation**: matplotlib, seaborn
- **Frontend**: Streamlit

---

## Disclaimer

This project is for educational and portfolio purposes only.
It does not constitute financial advice. Past model performance
does not guarantee future results.
