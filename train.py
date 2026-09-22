"""
Customer Churn Prediction
--------------------------
Trains and compares Logistic Regression, Random Forest, and XGBoost
classifiers to predict customer churn, then saves the best model.

Usage:
    python generate_data.py     # creates data/customer_churn.csv
    python train.py             # trains models, prints metrics, saves best model
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_PATH = "data/customer_churn.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"
CATEGORICAL_COLS = [
    "contract_type", "internet_service", "tech_support",
    "paperless_billing", "payment_method", "partner", "dependents",
]
NUMERIC_COLS = [
    "tenure_months", "monthly_charges", "total_charges",
    "support_calls", "late_payments", "senior_citizen",
]


def load_and_prepare(path=DATA_PATH):
    """Load the churn CSV and split it into a one-hot-encoded feature matrix.

    Categorical columns are one-hot encoded (drop_first=True), and the
    identifier and target columns are dropped from the feature matrix.

    Args:
        path: Path to the churn CSV (must contain "customer_id" and "churn").

    Returns:
        A tuple (X, y) of the feature DataFrame and the churn target Series.
    """
    df = pd.read_csv(path)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]
    return X, y


def train_models(X_train, y_train):
    """Fit Logistic Regression, Random Forest, and XGBoost on the training set.

    Logistic Regression is trained on standardized features (tree-based
    models are trained on the raw features, since scaling doesn't affect
    them); XGBoost's hyperparameters are tuned with a small grid search
    over ROC-AUC.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.

    Returns:
        A dict mapping model name -> (fitted model, fitted StandardScaler
        or None). The scaler is only present for Logistic Regression and
        must be applied to any data scored with that model.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {}

    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = (log_reg, scaler)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = (rf, None)

    xgb_grid = GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42),
        param_grid={
            "n_estimators": [200, 400],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
    )
    xgb_grid.fit(X_train, y_train)
    models["XGBoost"] = (xgb_grid.best_estimator_, None)

    return models


def evaluate(models, X_test, y_test):
    """Score every trained model on the held-out test set.

    Prints accuracy, ROC-AUC, and a classification report per model, and
    saves a combined ROC curve comparison to
    "{REPORT_DIR}/roc_comparison.png".

    Args:
        models: Dict as returned by train_models (name -> (model, scaler)).
        X_test: Test feature DataFrame.
        y_test: Test target Series.

    Returns:
        A dict mapping model name -> {"accuracy": float, "roc_auc": float}.
    """
    results = {}
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    for name, (model, scaler) in models.items():
        X_eval = scaler.transform(X_test) if scaler else X_test
        preds = model.predict(X_eval)
        probs = model.predict_proba(X_eval)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        results[name] = {"accuracy": round(acc, 4), "roc_auc": round(auc, 4)}

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, preds, target_names=["Retained", "Churned"]))

        RocCurveDisplay.from_predictions(y_test, probs, name=name, ax=ax)

    ax.set_title("ROC Curve Comparison")
    os.makedirs(REPORT_DIR, exist_ok=True)
    plt.savefig(f"{REPORT_DIR}/roc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


def plot_confusion_matrix(model, scaler, X_test, y_test, name):
    """Save a confusion-matrix heatmap for one model to REPORT_DIR.

    Args:
        model: A fitted classifier with a predict() method.
        scaler: Fitted StandardScaler to apply before predicting, or None
            if the model was trained on unscaled features.
        X_test: Test feature DataFrame.
        y_test: Test target Series.
        name: Model name, used in the plot title and output filename.
    """
    X_eval = scaler.transform(X_test) if scaler else X_test
    preds = model.predict(X_eval)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(f"{REPORT_DIR}/confusion_matrix_{name.replace(' ', '_').lower()}.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Run the full pipeline: load data, train, evaluate, and save the best model.

    Requires data/customer_churn.csv to already exist (run
    generate_data.py first). Saves the best model (by ROC-AUC), its
    scaler (if any), and the training feature columns to MODEL_DIR, plus
    per-model confusion matrices, an ROC comparison plot, and a
    metrics.json summary to REPORT_DIR.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Run `python generate_data.py` first to create the dataset.")

    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = train_models(X_train, y_train)
    results = evaluate(models, X_test, y_test)

    for name, (model, scaler) in models.items():
        plot_confusion_matrix(model, scaler, X_test, y_test, name)

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model, best_scaler = models[best_name]

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    if best_scaler:
        joblib.dump(best_scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_columns.pkl")

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(f"{REPORT_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest model: {best_name} (saved to {MODEL_DIR}/best_model.pkl)")
    print(f"Metrics saved to {REPORT_DIR}/metrics.json")


if __name__ == "__main__":
    main()
