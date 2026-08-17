"""
Generates a synthetic customer churn dataset that mimics real-world
telecom/subscription behavior data (tenure, usage, billing, support calls).
Run this first if you don't have your own dataset.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

def generate_customers(n=N):
      tenure_months = np.random.gamma(shape=2.0, scale=12, size=n).clip(1, 72).astype(int)
      monthly_charges = np.random.normal(70, 25, n).clip(15, 150)
      total_charges = monthly_charges * tenure_months * np.random.uniform(0.9, 1.1, n)
      contract_type = np.random.choice(
          ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]
      )
      support_calls = np.random.poisson(1.5, n)
      late_payments = np.random.poisson(0.8, n)
      internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
      tech_support = np.random.choice(["Yes", "No"], n, p=[0.4, 0.6])
      paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.6, 0.4])
      payment_method = np.random.choice(
          ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
      )
      senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
      partner = np.random.choice(["Yes", "No"], n, p=[0.5, 0.5])
      dependents = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])

    churn_score = (
              -0.04 * tenure_months
              + 0.015 * monthly_charges
              + 0.35 * support_calls
              + 0.5 * late_payments
              + (contract_type == "Month-to-month") * 1.2
              + (internet_service == "Fiber optic") * 0.4
              + (tech_support == "No") * 0.5
              + (paperless_billing == "Yes") * 0.2
              + (payment_method == "Electronic check") * 0.4
              - (partner == "Yes") * 0.3
              - (dependents == "Yes") * 0.3
              + np.random.normal(0, 1.0, n)
    )
    churn_prob = 1 / (1 + np.exp(-(churn_score - 2.4)))
    churn = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
              "customer_id": [f"CUST-{i:05d}" for i in range(n)],
              "tenure_months": tenure_months,
              "monthly_charges": monthly_charges.round(2),
              "total_charges": total_charges.round(2),
              "contract_type": contract_type,
              "support_calls": support_calls,
              "late_payments": late_payments,
              "internet_service": internet_service,
              "tech_support": tech_support,
              "paperless_billing": paperless_billing,
              "payment_method": payment_method,
              "senior_citizen": senior_citizen,
              "partner": partner,
              "dependents": dependents,
              "churn": churn,
    })
    return df

if __name__ == "__main__":
      df = generate_customers()
      df.to_csv("data/customer_churn.csv", index=False)
      print(f"Generated {len(df)} rows -> data/customer_churn.csv")
      print(f"Churn rate: {df['churn'].mean():.2%}")
