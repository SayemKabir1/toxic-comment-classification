"""Train and evaluate a TF-IDF plus logistic-regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from toxic_comment_classifier.model import (
    LABEL_COLUMNS,
    TEXT_COLUMN,
    build_model,
    validate_training_frame,
)


def train_model(data_path: Path, model_path: Path, report_path: Path,
                test_size: float = 0.2, random_state: int = 42) -> dict:
    """Fit the model, save a holdout report, and persist the fitted pipeline."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Could not find training CSV: {data_path}. Download the dataset and pass its path."
        )
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")

    frame = pd.read_csv(data_path)
    validate_training_frame(frame)
    x_train, x_valid, y_train, y_valid = train_test_split(
        frame[TEXT_COLUMN].astype(str),
        frame[list(LABEL_COLUMNS)].astype(int),
        test_size=test_size,
        random_state=random_state,
    )

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_valid)
    report = classification_report(
        y_valid,
        predictions,
        target_names=list(LABEL_COLUMNS),
        output_dict=True,
        zero_division=0,
    )
    report_metadata = {
        "dataset": str(data_path),
        "rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_valid)),
        "test_size": test_size,
        "random_state": random_state,
        "text_column": TEXT_COLUMN,
        "label_columns": list(LABEL_COLUMNS),
        "metrics": report,
        "note": "Holdout metrics from a random split; these are not independent external test results.",
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    report_path.write_text(json.dumps(report_metadata, indent=2), encoding="utf-8")
    return report_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to Jigsaw train.csv")
    parser.add_argument("--model-out", type=Path, default=Path("models/toxic_comment_model.joblib"))
    parser.add_argument("--report-out", type=Path, default=Path("results/validation_report.json"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        report = train_model(args.data, args.model_out, args.report_out,
                             args.test_size, args.random_state)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Model saved to: {args.model_out}")
    print(f"Validation report saved to: {args.report_out}")
    print("Per-label F1 scores:")
    for label in LABEL_COLUMNS:
        metrics = report["metrics"][label]
        print(f"  {label}: {metrics['f1-score']:.4f}")


if __name__ == "__main__":
    main()
