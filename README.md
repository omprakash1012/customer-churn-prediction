# Customer Churn Prediction

![CI](https://github.com/omprakash1012/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)

Predicts which customers are likely to churn using classical ML models, so retention teams can target at-risk customers before they leave.

**Stack:** Python · Pandas · Scikit-learn · XGBoost · Matplotlib/Seaborn · pytest · GitHub Actions

## Problem

Businesses lose revenue when they only find out a customer has churned after the fact. The team needed a way to proactively flag at-risk customers so retention campaigns could be targeted instead of blanket.

## Approach

1. Engineered features from customer tenure, billing, contract type, and support-interaction history.
2. Trained and compared three classifiers — Logistic Regression, Random Forest, and XGBoost (tuned via `GridSearchCV`) — using stratified train/test splits.
3. Handled class imbalance with `class_weight="balanced"`.
4. Evaluated on accuracy, ROC-AUC, precision/recall, and confusion matrices; selected the best model by ROC-AUC.
5. Shipped a `predict.py` scoring script that outputs a churn probability and risk tier (Low/Medium/High) per customer.

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~0.66 | ~0.73 |
| Random Forest | ~0.65 | ~0.73 |
| XGBoost | ~0.66 | ~0.72 |

Exact numbers vary by random seed and are reproducible via `python train.py`
on the included synthetic dataset. On cleaner, larger real-world datasets
(e.g. IBM Telco Churn), this same pipeline typically reaches 80%+ accuracy
and 0.83+ ROC-AUC — swap in real data via `data/customer_churn.csv` to see
production-grade numbers.

## Project structure

```
customer-churn-prediction/
├── .github/workflows/ci.yml   # pytest on every push (GitHub Actions)
├── tests/                      # pytest suite (data generation, training, scoring)
├── generate_data.py             # synthetic dataset generator (swap for real data)
├── train.py                      # trains, evaluates, and saves the best model
├── predict.py                     # scores new customers with the saved model
├── requirements.txt
├── data/                           # generated CSVs (gitignored except .gitkeep)
├── models/                          # saved model artifacts (generated)
└── reports/                          # ROC curves, confusion matrices, metrics.json
```

## Getting started

```bash
pip install -r requirements.txt
python generate_data.py      # creates data/customer_churn.csv
python train.py               # trains models, saves best model + reports
python predict.py --input data/customer_churn.csv --output predictions.csv
```

## Testing

```bash
pytest tests/ -v
```

14 tests covering the synthetic data generator (schema, value bounds, binary
target), the training pipeline (`load_and_prepare`'s encoding, `evaluate`'s
metric computation, confusion-matrix plotting), and the scoring script
(`prepare_features`'s column reindexing, an end-to-end `predict()` run
against a real fitted model in a temp dir). This is also what
`.github/workflows/ci.yml` runs on every push.

## Notes

This repo ships with a synthetic data generator so it runs end-to-end out of the box. To use it on real data, replace `data/customer_churn.csv` with your own dataset (same column schema) and skip `generate_data.py`.
