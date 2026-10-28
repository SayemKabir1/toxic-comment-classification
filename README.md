# Toxic Comment Classification

A reproducible baseline for identifying multiple categories of potentially toxic online comments using TF-IDF features and one-vs-rest logistic regression with scikit-learn.

## Project status

The baseline has been trained locally on the Jigsaw Toxic Comment Classification Challenge training CSV. The metrics below come from a random 80/20 train-validation split with `random_state=42`. They are **validation metrics, not an independent benchmark or production-readiness claim**. The dataset and fitted model are not included in this repository.

## Results

| Label | Precision | Recall | F1-score | Positive examples in validation set |
|---|---:|---:|---:|---:|
| Toxic | 0.6661 | 0.8472 | 0.7458 | 3,056 |
| Severe toxic | 0.3026 | 0.7975 | 0.4387 | 321 |
| Obscene | 0.7269 | 0.8706 | 0.7923 | 1,715 |
| Threat | 0.3022 | 0.7432 | 0.4297 | 74 |
| Insult | 0.5991 | 0.8631 | 0.7073 | 1,614 |
| Identity hate | 0.2897 | 0.7075 | 0.4111 | 294 |

| Aggregate metric | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Micro average | 0.5987 | 0.8473 | 0.7016 |
| Macro average | 0.4811 | 0.8048 | 0.5875 |
| Weighted average | 0.6296 | 0.8473 | 0.7171 |

The macro F1-score is the unweighted mean across the six labels. These values are **not accuracy**. Recall is higher than precision, so this configuration catches many labelled toxic comments but also produces false positives. Rare categories, especially `threat`, have comparatively few positive validation examples, so their estimates are less stable. Threshold tuning and error analysis are important next steps.

The full machine-readable report from the local run is in [`results/validation_report.json`](results/validation_report.json).

## What it predicts

The expected Jigsaw training CSV contains a `comment_text` column and six binary label columns:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

This is a multi-label classification problem: one comment can receive more than one label.

## Project structure

```text
src/toxic_comment_classifier/  Model, training, and prediction modules
 tests/                        Automated tests using synthetic text
 examples/                     Small example inputs
 results/validation_report.json  Validation metrics from the recorded run
 README.md                     Project documentation
 pyproject.toml                Package metadata and dependencies
```

## Setup on Windows

Python 3.10 or newer is required. The original training run was completed in a local Python 3.13 virtual environment.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If PowerShell blocks virtual-environment activation, use the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```
## Get the data

Obtain the Jigsaw Toxic Comment Classification dataset from the official Kaggle competition page:

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

Review and follow the dataset terms before use. Place the downloaded `train.csv` in a local `data/` directory, or pass its location to the training command. The dataset is deliberately not bundled in this repository.

## Train and evaluate

```powershell
python -m toxic_comment_classifier.train --data data/train.csv
```

By default, the command saves the fitted model to `models/toxic_comment_model.joblib` and a JSON holdout report to `results/validation_report.json`. The fitted model and local dataset are ignored by Git; the committed validation report records the results above.

To set a different holdout size or output locations:

```powershell
python -m toxic_comment_classifier.train --data data/train.csv --test-size 0.2 --random-state 42 --model-out models/model.joblib --report-out results/new_report.json
```

The script uses a random train-validation split with a fixed seed. It reports per-label precision, recall, F1-score, and support. Since toxicity labels can be imbalanced, do not rely on accuracy alone. The default report is a preliminary holdout evaluation, not an independent benchmark or proof of production readiness.

## Predict on comments

After training:

```powershell
python -m toxic_comment_classifier.predict --model models/toxic_comment_model.joblib --text "Thank you for your help" --text "You are an idiot"
```

The command prints predicted labels and per-label model probability estimates as JSON. A score is a model estimate, not a guarantee that a comment violates a policy. Because this baseline uses class weighting and default decision thresholds, its scores should not be treated as calibrated probabilities without further evaluation. Any real moderation workflow requires threshold selection, error analysis, appropriate human review, and monitoring.

## Run tests

```powershell
python -m pytest -q
```

The test suite uses a small synthetic dataset so tests do not require Kaggle credentials or a large data download. Seven tests passed during local verification. Passing tests confirms basic code behavior; it does not validate real-world predictive performance.

## Method

The pipeline converts comments to unigram and bigram TF-IDF features, then fits a one-vs-rest logistic regression classifier for each label. TF-IDF is inside the scikit-learn pipeline, so the vocabulary and inverse-document-frequency weights are fitted on the training portion only. Logistic regression uses `class_weight="balanced"` to account for imbalance in each label.

## Limitations and next steps

- The reported result is a single random holdout split, not a separate untouched test set or external evaluation.
- No per-label thresholds have been tuned.
- The low precision for rare labels means false positives need review; high recall alone is not sufficient for a moderation system.
- The model may produce false positives and false negatives, especially for rare labels or context-dependent language.
- It should not be used as an automated enforcement system without further evaluation, calibration, safeguards, and monitoring.
- This baseline uses conventional NLP rather than a transformer or an LLM agent.
- Useful next work includes reviewing false positives/negatives, evaluating a separately held-out test set, selecting thresholds on validation data, and comparing against a transformer baseline.

## Attribution

The dataset is from the Jigsaw Toxic Comment Classification Challenge on Kaggle. Follow the dataset's terms and cite the source when using the data. This repository is a portfolio implementation of a baseline approach; it is not an official Jigsaw model or an endorsement by Kaggle or Quora.
