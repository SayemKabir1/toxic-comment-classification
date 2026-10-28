"""Run a saved toxic-comment classifier on text supplied on the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from toxic_comment_classifier.model import LABEL_COLUMNS


def predict_comments(model_path: Path, comments: list[str]) -> list[dict[str, object]]:
    """Return predicted labels and, when supported, per-label probabilities."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not comments or any(not isinstance(text, str) or not text.strip() for text in comments):
        raise ValueError("Provide at least one non-empty comment.")

    model = joblib.load(model_path)
    predictions = model.predict(pd.Series(comments))
    probabilities = model.predict_proba(pd.Series(comments)) if hasattr(model, "predict_proba") else None

    results: list[dict[str, object]] = []
    for row_index, comment in enumerate(comments):
        active_labels = [
            label for label, value in zip(LABEL_COLUMNS, predictions[row_index]) if int(value) == 1
        ]
        item: dict[str, object] = {"comment_text": comment, "predicted_labels": active_labels}
        if probabilities is not None:
            item["label_probabilities"] = {
                label: float(probabilities[row_index][col_index])
                for col_index, label in enumerate(LABEL_COLUMNS)
            }
        results.append(item)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/toxic_comment_model.joblib"))
    parser.add_argument("--text", action="append", required=True,
                        help="Comment to classify. Supply this option more than once for multiple comments.")
    args = parser.parse_args()
    try:
        results = predict_comments(args.model, args.text)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
