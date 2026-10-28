"""Model construction and data validation for toxic-comment classification."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

TEXT_COLUMN = "comment_text"
LABEL_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)


def validate_training_frame(frame: pd.DataFrame, text_column: str = TEXT_COLUMN,
                            label_columns: Sequence[str] = LABEL_COLUMNS) -> None:
    """Raise a clear error when a training dataframe has missing columns or invalid labels."""
    required = [text_column, *label_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "Training data is missing required column(s): " + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("Training data contains no rows.")
    if frame[text_column].isna().any():
        raise ValueError(f"Column '{text_column}' contains missing values.")
    for column in label_columns:
        values = set(pd.Series(frame[column]).dropna().unique().tolist())
        if not values.issubset({0, 1, False, True}):
            raise ValueError(f"Label column '{column}' must contain only 0/1 values.")
        if frame[column].isna().any():
            raise ValueError(f"Label column '{column}' contains missing values.")


def build_model() -> Pipeline:
    """Create a baseline pipeline; TF-IDF is fitted only on training data."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=200_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1_000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                    n_jobs=None,
                ),
            ),
        ]
    )
