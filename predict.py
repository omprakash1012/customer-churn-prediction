"""
Loads the trained churn model and scores new customers.

Usage:
    python predict.py --input data/new_customers.csv --output predictions.csv
"""
import argparse

import joblib
import pandas as pd

CATEGORICAL_COLS = [
    "contract_type", "internet_service", "tech_support",
    "paperless_billing", "payment_method", "partner", "dependents",
]


def load_artifacts(model_dir="models"):
    """Load the saved model, feature columns, and optional scaler from disk.

    Args:
        model_dir: Directory containing best_model.pkl, feature_columns.pkl,
            and (optionally) scaler.pkl, as saved by train.py.

    Returns:
        A tuple (model, scaler, feature_columns), where scaler is None if
        the model wasn't trained on scaled features (e.g. tree-based models).
    """
    model = joblib.load(f"{model_dir}/best_model.pkl")
    feature_columns = joblib.load(f"{model_dir}/feature_columns.pkl")
    try:
        scaler = joblib.load(f"{model_dir}/scaler.pkl")
    except FileNotFoundError:
        scaler = None
    return model, scaler, feature_columns


def prepare_features(df, feature_columns):
    """One-hot encode categorical columns and align to the training feature set.

    Args:
        df: Raw customer DataFrame (without the customer_id column).
        feature_columns: The exact column set/order the model was trained
            on, as saved by train.py.

    Returns:
        A DataFrame one-hot encoded and reindexed to match feature_columns,
        with any missing columns filled with 0.
    """
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    df_enc = df_enc.reindex(columns=feature_columns, fill_value=0)
    return df_enc


def predict(input_path, output_path, model_dir="models"):
    """Score customers in input_path and write predictions to output_path.

    Loads the saved model, prepares features, and computes a churn
    probability, a binary prediction, and a risk tier (Low/Medium/High)
    per customer.

    Args:
        input_path: Path to a CSV of customers to score (same schema as
            the training data, minus the "churn" column).
        output_path: Path to write the output CSV of predictions to.
        model_dir: Directory containing the saved model artifacts.

    Returns:
        None. Writes results to output_path and prints a summary to stdout.
    """
    model, scaler, feature_columns = load_artifacts(model_dir)
    df = pd.read_csv(input_path)
    ids = df["customer_id"] if "customer_id" in df else pd.Series(range(len(df)))

    X = prepare_features(df.drop(columns=["customer_id"], errors="ignore"), feature_columns)
    X_eval = scaler.transform(X) if scaler else X

    probs = model.predict_proba(X_eval)[:, 1]
    preds = model.predict(X_eval)

    out = pd.DataFrame({
        "customer_id": ids,
        "churn_probability": probs.round(4),
        "predicted_churn": preds,
        "risk_tier": pd.cut(
            probs, bins=[-0.01, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"]
        ),
    })
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} predictions -> {output_path}")
    print(out["risk_tier"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    predict(args.input, args.output, args.model_dir)
