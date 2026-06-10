"""
sentiment_analysis.py
---------------------
FinBERT-based financial sentiment analysis.

Two modes:
  1. Batch-score a list of sentences (e.g. Financial PhraseBank validation).
  2. Score a single headline → returns pos/neg/neu scores + composite score.

The composite sentiment_score is:  positive - negative  ∈ [-1, +1]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FINBERT_MODEL,
    SENTIMENT_MAX_LENGTH,
    SENTIMENT_BATCH_SIZE,
    DATA_DIR,
)
from utils import get_logger

_log = get_logger(__name__)


# ─── FinBERT Wrapper ──────────────────────────────────────────────────────────

class FinBERTSentiment:
    """
    Thin wrapper around the ProsusAI/finbert HuggingFace model.

    Usage
    -----
    >>> analyser = FinBERTSentiment()
    >>> result = analyser.score("Apple earnings beat expectations.")
    >>> print(result)
    {'positive': 0.92, 'negative': 0.03, 'neutral': 0.05, 'sentiment_score': 0.89}
    """

    # FinBERT label ordering from its config
    _LABEL_MAP: dict[int, str] = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(self, model_name: str = FINBERT_MODEL) -> None:
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _log.info("Loading FinBERT from '%s' on %s …", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        _log.info("FinBERT ready.")

    # ------------------------------------------------------------------
    def _forward(self, texts: list[str]) -> np.ndarray:
        """
        Run a single batch through the model.

        Returns
        -------
        np.ndarray  shape (batch, 3)  — softmax probabilities [pos, neg, neu]
        """
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=SENTIMENT_MAX_LENGTH,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self.model(**encoding).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs  # (batch, 3)

    # ------------------------------------------------------------------
    def score(self, text: str) -> dict[str, float]:
        """
        Score a single text string.

        Returns
        -------
        dict with keys: positive, negative, neutral, sentiment_score
        """
        probs = self._forward([text])[0]  # shape (3,)
        pos, neg, neu = float(probs[0]), float(probs[1]), float(probs[2])
        return {
            "positive": round(pos, 4),
            "negative": round(neg, 4),
            "neutral": round(neu, 4),
            "sentiment_score": round(pos - neg, 4),
        }

    # ------------------------------------------------------------------
    def score_batch(
        self,
        texts: list[str],
        batch_size: int = SENTIMENT_BATCH_SIZE,
    ) -> list[dict[str, float]]:
        """
        Score a list of texts in mini-batches.

        Parameters
        ----------
        texts : list[str]
        batch_size : int

        Returns
        -------
        list[dict]  one dict per input text
        """
        results: list[dict[str, float]] = []
        total = len(texts)

        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            probs = self._forward(batch)  # (batch_size, 3)
            for row in probs:
                pos, neg, neu = float(row[0]), float(row[1]), float(row[2])
                results.append(
                    {
                        "positive": round(pos, 4),
                        "negative": round(neg, 4),
                        "neutral": round(neu, 4),
                        "sentiment_score": round(pos - neg, 4),
                    }
                )
            pct = min(start + batch_size, total)
            _log.debug("Scored %d / %d texts …", pct, total)

        return results

    # ------------------------------------------------------------------
    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "sentence",
    ) -> pd.DataFrame:
        """
        Add sentiment columns to a DataFrame that has a text column.

        Parameters
        ----------
        df : pd.DataFrame
        text_column : str
            Column containing the raw text to score.

        Returns
        -------
        pd.DataFrame
            Original df + [sentiment_positive, sentiment_negative,
                           sentiment_neutral, sentiment_score]
        """
        texts = df[text_column].fillna("").tolist()
        scores = self.score_batch(texts)
        scores_df = pd.DataFrame(scores)
        scores_df.columns = [
            "sentiment_positive",
            "sentiment_negative",
            "sentiment_neutral",
            "sentiment_score",
        ]
        return pd.concat([df.reset_index(drop=True), scores_df], axis=1)


# ─── Financial PhraseBank Helper ──────────────────────────────────────────────

def load_financial_phrasebank(
    agreement: str = "75agree",
) -> pd.DataFrame:
    """
    Load the Financial PhraseBank dataset via HuggingFace datasets.

    Parameters
    ----------
    agreement : str
        One of: "50agree", "66agree", "75agree", "allagree".

    Returns
    -------
    pd.DataFrame  columns: sentence, label (0=negative,1=neutral,2=positive)
    """
    try:
        from datasets import load_dataset

        _log.info("Loading Financial PhraseBank (%s) …", agreement)
        dataset = load_dataset(
            "financial_phrasebank",
            agreement,
            trust_remote_code=True,
        )
        df = dataset["train"].to_pandas()
        # Rename to a consistent schema
        df = df.rename(columns={"label": "true_label"})
        label_names = {0: "negative", 1: "neutral", 2: "positive"}
        df["true_label_str"] = df["true_label"].map(label_names)
        _log.info("Financial PhraseBank loaded: %d sentences.", len(df))
        return df
    except Exception as exc:
        _log.error("Failed to load Financial PhraseBank: %s", exc)
        raise


def validate_sentiment_pipeline(
    analyser: FinBERTSentiment,
    n_samples: int = 200,
) -> dict[str, float]:
    """
    Score a sample of Financial PhraseBank sentences and compare against
    the ground-truth labels to report pipeline accuracy.

    Parameters
    ----------
    analyser : FinBERTSentiment
    n_samples : int
        How many sentences to evaluate (capped at dataset size).

    Returns
    -------
    dict with 'accuracy' and 'n_samples'.
    """
    from sklearn.metrics import accuracy_score

    df = load_financial_phrasebank()
    df = df.sample(min(n_samples, len(df)), random_state=42).reset_index(drop=True)

    scored = analyser.score_dataframe(df, text_column="sentence")

    # Map FinBERT prediction to integer label
    label_int = {"positive": 2, "neutral": 1, "negative": 0}

    def _pred_label(row: pd.Series) -> int:
        scores = {
            "positive": row["sentiment_positive"],
            "negative": row["sentiment_negative"],
            "neutral": row["sentiment_neutral"],
        }
        return label_int[max(scores, key=scores.get)]

    scored["predicted_label"] = scored.apply(_pred_label, axis=1)
    acc = accuracy_score(scored["true_label"], scored["predicted_label"])
    _log.info("Sentiment pipeline accuracy on %d samples: %.2f%%", len(scored), acc * 100)
    return {"accuracy": round(acc, 4), "n_samples": len(scored)}


# ─── Aggregation for time-series ──────────────────────────────────────────────

def aggregate_daily_sentiment(
    scored_df: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Aggregate per-sentence scores to one row per day.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Must contain [date_column, sentiment_positive, sentiment_negative,
                      sentiment_neutral, sentiment_score].
    date_column : str

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, daily-averaged sentiment columns.
    """
    df = scored_df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    agg = (
        df.groupby(date_column)[
            [
                "sentiment_positive",
                "sentiment_negative",
                "sentiment_neutral",
                "sentiment_score",
            ]
        ]
        .mean()
        .round(4)
    )
    agg.index.name = "Date"
    return agg


def get_latest_sentiment_score(
    headlines: list[str],
    analyser: Optional[FinBERTSentiment] = None,
) -> dict[str, float]:
    """
    Score a list of current news headlines and return the averaged scores.

    Parameters
    ----------
    headlines : list[str]
        Recent news headlines for a stock.
    analyser : FinBERTSentiment | None
        If None a new analyser is instantiated (slower; reuse when possible).

    Returns
    -------
    dict  with averaged pos/neg/neu/sentiment_score
    """
    if not headlines:
        return {
            "sentiment_positive": 0.0,
            "sentiment_negative": 0.0,
            "sentiment_neutral": 1.0,
            "sentiment_score": 0.0,
        }

    if analyser is None:
        analyser = FinBERTSentiment()

    scores = analyser.score_batch(headlines)
    arr = np.array(
        [
            [s["positive"], s["negative"], s["neutral"], s["sentiment_score"]]
            for s in scores
        ]
    )
    avg = arr.mean(axis=0)
    return {
        "sentiment_positive": round(float(avg[0]), 4),
        "sentiment_negative": round(float(avg[1]), 4),
        "sentiment_neutral": round(float(avg[2]), 4),
        "sentiment_score": round(float(avg[3]), 4),
    }


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FinBERT sentiment analysis")
    parser.add_argument(
        "--text",
        type=str,
        default="Apple reports record quarterly earnings, beating analyst expectations.",
        help="Single headline to score",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate pipeline on Financial PhraseBank",
    )
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()

    analyser = FinBERTSentiment()

    if args.validate:
        result = validate_sentiment_pipeline(analyser, n_samples=args.n_samples)
        print(f"\nValidation accuracy: {result['accuracy']*100:.1f}%  "
              f"({result['n_samples']} samples)")
    else:
        result = analyser.score(args.text)
        print(f"\nText   : {args.text}")
        print(f"Scores : {result}")
