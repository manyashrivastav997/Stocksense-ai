# StockSense AI — Complete Setup Guide
## Copy-paste-ready commands, beginner-friendly

---

## STEP 0 — Prerequisites

You need Python 3.10 or 3.11 installed.
Check your version:
```
python --version
```

---

## STEP 1 — Create a virtual environment

```
cd StockSenseAI
python -m venv venv
```

Activate it:

On Windows:
```
venv\Scripts\activate
```

On Mac/Linux:
```
source venv/bin/activate
```

You should see `(venv)` in your terminal. Keep this active for ALL steps below.

---

## STEP 2 — Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

This takes 3–8 minutes. It downloads TensorFlow, PyTorch, XGBoost, FinBERT etc.
If you see any red errors, run:
```
pip install -r requirements.txt --no-cache-dir
```

---

## STEP 3 — Download historical stock data

```
cd src
python data_collection.py
```

Expected output:
```
  AAPL: 1761 rows  |  2018-01-02 → 2024-12-31
  TSLA: 1761 rows  |  2018-01-02 → 2024-12-31
  NVDA: 1761 rows  |  2018-01-02 → 2024-12-31
  MSFT: 1761 rows  |  2018-01-02 → 2024-12-31
  SPY:  1761 rows  |  2018-01-02 → 2024-12-31
```

Files created in `data/`:
- AAPL_historical.parquet
- TSLA_historical.parquet
- NVDA_historical.parquet
- MSFT_historical.parquet
- SPY_historical.parquet

---

## STEP 4 — Train XGBoost (fastest, ~2 minutes)

```
python train_xgboost.py --symbol AAPL
```

Expected output (last lines):
```
XGBoost results — Acc: 0.5XXX | Prec: 0.5XXX | Rec: 0.5XXX | F1: 0.5XXX
XGBoost model saved → .../models/xgboost_model.pkl
Scaler saved → .../models/scaler.pkl
```

---

## STEP 5 — Train LSTM (~15–40 minutes on CPU)

```
python train_lstm.py --symbol AAPL
```

You will see epoch-by-epoch progress. Training stops early when validation loss
stops improving (EarlyStopping patience=15 epochs).

Expected final lines:
```
LSTM results — Acc: 0.5XXX | Prec: 0.5XXX | Rec: 0.5XXX | F1: 0.5XXX
Keras model saved → .../models/lstm_model.keras
```

---

## STEP 6 — Train GRU (~12–35 minutes on CPU)

```
python train_gru.py --symbol AAPL
```

Expected final lines:
```
GRU results — Acc: 0.5XXX | Prec: 0.5XXX | Rec: 0.5XXX | F1: 0.5XXX
Keras model saved → .../models/gru_model.keras
```

---

## STEP 7 — Evaluate all models

```
python evaluate_models.py --symbol AAPL
```

Creates in `results/`:
- metrics.csv
- comparison.csv
- predictions.csv
- AAPL_confusion_matrices.png
- AAPL_model_comparison.png

---

## STEP 8 — Verify everything works

```
python -c "
import sys; sys.path.insert(0, '.')
from utils import load_xgboost, load_keras_model, load_scaler
from config import XGBOOST_MODEL_PATH, LSTM_MODEL_PATH, GRU_MODEL_PATH, SCALER_PATH
s = load_scaler(SCALER_PATH)
x = load_xgboost(XGBOOST_MODEL_PATH)
l = load_keras_model(LSTM_MODEL_PATH)
g = load_keras_model(GRU_MODEL_PATH)
print('ALL MODELS LOADED OK')
print('Scaler features:', s.n_features_in_)
print('XGBoost classes:', x.n_classes_)
print('LSTM output:', l.output_shape)
print('GRU output:', g.output_shape)
"
```

---

## STEP 9 — Test prediction CLI

```
python predict.py --symbol AAPL --model ALL
```

You should see a JSON block like:
```json
{
  "stock": "AAPL",
  "timestamp": "2024-06-01T15:30:00",
  "sentiment": {},
  "models": {
    "XGBOOST": {"prediction": "UP", "confidence": 63.4, "recommendation": "HOLD"},
    "LSTM":    {"prediction": "UP", "confidence": 57.8, "recommendation": "HOLD"},
    "GRU":     {"prediction": "UP", "confidence": 61.2, "recommendation": "HOLD"}
  },
  "consensus": "HOLD"
}
```

---

## STEP 10 — Launch Streamlit

```
cd ..
streamlit run app.py
```

Your browser opens at:  http://localhost:8501

---

## Expected final folder structure after all steps

```
StockSenseAI/
├── data/
│   ├── AAPL_historical.parquet
│   ├── TSLA_historical.parquet
│   ├── NVDA_historical.parquet
│   ├── MSFT_historical.parquet
│   └── SPY_historical.parquet
│
├── models/
│   ├── xgboost_model.pkl       ← from Step 4
│   ├── scaler.pkl              ← from Step 4
│   ├── lstm_model.keras        ← from Step 5
│   ├── gru_model.keras         ← from Step 6
│   ├── lstm_best_weights.keras ← checkpoint (safe to delete)
│   └── gru_best_weights.keras  ← checkpoint (safe to delete)
│
├── results/
│   ├── metrics.csv
│   ├── comparison.csv
│   ├── predictions.csv
│   ├── AAPL_confusion_matrices.png
│   └── AAPL_model_comparison.png
│
├── src/
│   ├── config.py
│   ├── data_collection.py
│   ├── feature_engineering.py
│   ├── sentiment_analysis.py
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── train_gru.py
│   ├── evaluate_models.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── README.md
└── SETUP.md
```

---

## Deploying to Streamlit Community Cloud

1. Push your project to a public GitHub repository
2. Include `models/`, `results/`, `data/` folders with your trained files
   (use Git LFS for large .keras files if needed)
3. Go to https://share.streamlit.io
4. Connect your GitHub repo
5. Set main file path to: `app.py`
6. Click Deploy

No code changes needed — the app auto-detects trained models on startup.

---

## Troubleshooting

**"No module named config"**
→ Make sure you are running scripts from inside the `src/` folder.

**"yfinance returned empty data"**
→ Check your internet connection. Try: `python -c "import yfinance as yf; print(yf.download('AAPL', period='5d'))"`

**TensorFlow CUDA warnings**
→ Normal on CPU-only machines. Training still works, just slower.

**"FileNotFoundError: scaler.pkl"**
→ You haven't run Step 4 yet. XGBoost must be trained before prediction.

**Streamlit "ModuleNotFoundError"**
→ Make sure your venv is activated and requirements.txt was installed successfully.
