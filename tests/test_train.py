import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from generate_data import generate_customers
from train import evaluate, load_and_prepare, plot_confusion_matrix


@pytest.fixture
def sample_csv(tmp_path):
    df = generate_customers(n=300)
    path = tmp_path / "customer_churn.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_and_prepare_drops_id_and_target_columns(sample_csv):
    X, y = load_and_prepare(sample_csv)
    assert "customer_id" not in X.columns
    assert "churn" not in X.columns
    assert len(X) == len(y)


def test_load_and_prepare_one_hot_encodes_categoricals(sample_csv):
    X, _ = load_and_prepare(sample_csv)
    # one-hot encoding should replace raw categorical columns with dummy columns
    assert "contract_type" not in X.columns
    assert any(col.startswith("contract_type_") for col in X.columns)


def test_load_and_prepare_all_columns_numeric(sample_csv):
    X, y = load_and_prepare(sample_csv)
    assert all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns)
    assert pd.api.types.is_numeric_dtype(y)


def test_evaluate_returns_accuracy_and_auc_for_each_model(sample_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = load_and_prepare(sample_csv)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000).fit(X_scaled, y)

    results = evaluate({"Logistic Regression": (model, scaler)}, X, y)

    assert "Logistic Regression" in results
    assert 0.0 <= results["Logistic Regression"]["accuracy"] <= 1.0
    assert 0.0 <= results["Logistic Regression"]["roc_auc"] <= 1.0
    assert (tmp_path / "reports" / "roc_comparison.png").exists()


def test_plot_confusion_matrix_saves_png(sample_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()
    X, y = load_and_prepare(sample_csv)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000).fit(X_scaled, y)

    plot_confusion_matrix(model, scaler, X, y, "Logistic Regression")

    assert (tmp_path / "reports" / "confusion_matrix_logistic_regression.png").exists()
