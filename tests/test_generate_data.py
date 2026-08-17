from generate_data import generate_customers


def test_generate_customers_returns_requested_row_count():
    df = generate_customers(n=500)
    assert len(df) == 500


def test_generate_customers_has_expected_columns():
    df = generate_customers(n=50)
    expected = {
        "customer_id", "tenure_months", "monthly_charges", "total_charges",
        "contract_type", "support_calls", "late_payments", "internet_service",
        "tech_support", "paperless_billing", "payment_method", "senior_citizen",
        "partner", "dependents", "churn",
    }
    assert expected.issubset(set(df.columns))


def test_churn_column_is_binary():
    df = generate_customers(n=500)
    assert set(df["churn"].unique()).issubset({0, 1})


def test_tenure_months_within_expected_bounds():
    df = generate_customers(n=500)
    assert df["tenure_months"].min() >= 1
    assert df["tenure_months"].max() <= 72


def test_customer_ids_are_unique():
    df = generate_customers(n=500)
    assert df["customer_id"].nunique() == len(df)


def test_contract_type_only_has_known_categories():
    df = generate_customers(n=500)
    assert set(df["contract_type"].unique()).issubset(
        {"Month-to-month", "One year", "Two year"}
    )
