import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from generate_data import generate_customers
from predict import load_artifacts, prepare_features, predict
from train import load_and_prepare


@pytest.fixture
def trained_artifacts(tmp_path):
    """Fit a tiny real model and save it in the same shape train.py would,
    so predict.py can be tested end-to-end without a full training run."""
    df = generate_customers(n=300)
    data_csv = tmp_path / "customer_churn.csv"
    df.to_csv(data_csv, index=False)

    X, y = load_and_prepare(str(data_csv))
    model = LogisticRegression(max_iter=1000).fit(X, y)

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    joblib.dump(model, model_dir / "best_model.pkl")
    joblib.dump(list(X.columns), model_dir / "feature_columns.pkl")

    return str(model_dir), df


def test_prepare_features_reindexes_to_expected_columns():
    feature_columns = ["tenure_months", "contract_type_One year", "contract_type_Two year"]
    # Include multiple contract_type values so pd.get_dummies(drop_first=True)
    # actually emits dummy columns instead of dropping the only category present.
    df = pd.DataFrame({
        "tenure_months": [10, 20, 30],
        "contract_type": ["Month-to-month", "One year", "Two year"],
        "internet_service": ["DSL", "DSL", "DSL"],
        "tech_support": ["Yes", "Yes", "Yes"],
        "paperless_billing": ["No", "No", "No"],
        "payment_method": ["Credit card", "Credit card", "Credit card"],
        "partner": ["No", "No", "No"],
        "dependents": ["No", "No", "No"],
    })
    result = prepare_features(df, feature_columns)
    assert list(result.columns) == feature_columns
    assert result["contract_type_One year"].iloc[1] == 1
    assert result["contract_type_Two year"].iloc[1] == 0
    assert result["contract_type_Two year"].iloc[2] == 1


def test_load_artifacts_returns_model_scaler_and_columns(trained_artifacts):
    model_dir, _ = trained_artifacts
    model, scaler, feature_columns = load_artifacts(model_dir)
    assert hasattr(model, "predict")
    assert scaler is None  # no scaler was saved in the fixture
    assert isinstance(feature_columns, list)


def test_predict_writes_output_with_expected_columns(trained_artifacts, tmp_path):
    model_dir, df = trained_artifacts
    input_csv = tmp_path / "new_customers.csv"
    output_csv = tmp_path / "predictions.csv"
    df.drop(columns=["churn"]).to_csv(input_csv, index=False)

    predict(str(input_csv), str(output_csv), model_dir=model_dir)

    out = pd.read_csv(output_csv)
    assert len(out) == len(df)
    assert {"customer_id", "churn_probability", "predicted_churn", "risk_tier"}.issubset(out.columns)
    assert out["churn_probability"].between(0, 1).all()
    assert set(out["risk_tier"].unique()).issubset({"Low", "Medium", "High"})
