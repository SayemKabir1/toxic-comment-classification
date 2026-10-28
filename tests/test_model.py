from pathlib import Path

import joblib
import pandas as pd
import pytest

from toxic_comment_classifier.model import LABEL_COLUMNS, TEXT_COLUMN, build_model, validate_training_frame
from toxic_comment_classifier.predict import predict_comments
from toxic_comment_classifier.train import train_model


def toy_frame() -> pd.DataFrame:
    texts = [
        "thank you for sharing this", "you are helpful", "have a nice day", "great answer",
        "you are an idiot", "this is disgusting", "I will hurt you", "you are toxic",
        "I hate this group", "shut up stupid", "that was useful", "nice explanation",
        "you are awful", "I will attack you", "wonderful post", "thanks friend",
        "you are a moron", "I dislike your hateful comment", "good morning", "excellent work",
        "I hate this group", "that hateful group is awful", "this group deserves no hate",
    ]
    rows = []
    for i, text in enumerate(texts):
        is_toxic = int(any(word in text for word in ("idiot", "disgusting", "hurt", "toxic", "stupid", "awful", "attack", "moron", "hateful")))
        row = {TEXT_COLUMN: text}
        for label in LABEL_COLUMNS:
            row[label] = 0
        row["toxic"] = is_toxic
        if "disgusting" in text:
            row["obscene"] = 1
        if "hurt" in text or "attack" in text:
            row["threat"] = 1
            row["severe_toxic"] = 1
        if "idiot" in text or "stupid" in text or "moron" in text:
            row["insult"] = 1
        if "hateful" in text or "hate this group" in text:
            row["identity_hate"] = 1
        rows.append(row)
    return pd.DataFrame(rows)


def test_validate_training_frame_accepts_expected_columns():
    validate_training_frame(toy_frame())


def test_validate_training_frame_reports_missing_columns():
    with pytest.raises(ValueError, match="missing required"):
        validate_training_frame(pd.DataFrame({TEXT_COLUMN: ["hello"]}))


def test_validate_training_frame_rejects_invalid_labels():
    frame = toy_frame()
    frame.loc[0, "toxic"] = 2
    with pytest.raises(ValueError, match="only 0/1"):
        validate_training_frame(frame)


def test_model_can_fit_and_predict():
    frame = toy_frame()
    model = build_model()
    model.fit(frame[TEXT_COLUMN], frame[list(LABEL_COLUMNS)])
    predictions = model.predict(["you are an idiot", "thanks for your help"])
    assert predictions.shape == (2, len(LABEL_COLUMNS))
    assert set(predictions.ravel()).issubset({0, 1})


def test_training_saves_model_and_report(tmp_path: Path):
    data_path = tmp_path / "train.csv"
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"
    toy_frame().to_csv(data_path, index=False)
    report = train_model(data_path, model_path, report_path, test_size=0.25)
    assert model_path.is_file()
    assert report_path.is_file()
    assert report["train_rows"] + report["validation_rows"] == len(toy_frame())
    assert "toxic" in report["metrics"]
    assert joblib.load(model_path) is not None


def test_predict_comments_returns_labels_and_probabilities(tmp_path: Path):
    frame = toy_frame()
    model_path = tmp_path / "model.joblib"
    model = build_model().fit(frame[TEXT_COLUMN], frame[list(LABEL_COLUMNS)])
    joblib.dump(model, model_path)
    results = predict_comments(model_path, ["thanks for the answer"])
    assert results[0]["comment_text"] == "thanks for the answer"
    assert isinstance(results[0]["predicted_labels"], list)
    assert set(results[0]["label_probabilities"]) == set(LABEL_COLUMNS)


def test_predict_rejects_empty_comment(tmp_path: Path):
    frame = toy_frame()
    model_path = tmp_path / "model.joblib"
    joblib.dump(build_model().fit(frame[TEXT_COLUMN], frame[list(LABEL_COLUMNS)]), model_path)
    with pytest.raises(ValueError, match="non-empty comment"):
        predict_comments(model_path, ["  "])
