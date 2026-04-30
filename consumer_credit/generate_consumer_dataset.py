import os
import numpy as np
import pandas as pd

from consumer_xgb_config import DATA_PATH, DATA_DIR, FEATURES, TARGET


def _clip(values, low, high):
    return np.minimum(np.maximum(values, low), high)


def generate_dataset(n=3000, seed=42):
    rng = np.random.default_rng(seed)

    cibil_score = rng.integers(300, 901, size=n)
    income = rng.integers(20000, 200001, size=n)
    dependents = rng.integers(0, 6, size=n)
    loan_amount = rng.integers(50000, 1000001, size=n)
    tenure = rng.integers(6, 61, size=n)
    down_payment = rng.integers(0, 300001, size=n)
    existing_emi = rng.integers(0, 40001, size=n)
    credit_card = rng.integers(0, 2, size=n)
    employment_type = rng.integers(0, 2, size=n)

    emi = np.maximum(loan_amount / tenure, 1000)
    account_age = rng.integers(3, 121, size=n)
    overdrafts = rng.integers(0, 6, size=n)
    bounced_payments = rng.integers(0, 4, size=n)
    late_emi_ratio = rng.uniform(0.0, 0.8, size=n)
    income_volatility = rng.uniform(0.0, 1.0, size=n)
    avg_balance = rng.integers(2000, 150001, size=n)
    transaction_frequency = rng.integers(5, 201, size=n)
    monthly_spending = rng.integers(5000, 150001, size=n)

    statement_score = (
        50
        + (cibil_score - 600) / 6
        + (avg_balance / 10000)
        + (transaction_frequency / 10)
        - (overdrafts * 6)
        - (bounced_payments * 10)
        - (late_emi_ratio * 40)
        - (income_volatility * 15)
    )
    statement_score = _clip(statement_score + rng.normal(0, 8, size=n), 0, 100)

    dti = (existing_emi + emi) / np.maximum(income, 1)
    affordability = loan_amount / np.maximum(income * 12, 1)
    down_payment_ratio = down_payment / np.maximum(loan_amount, 1)

    score = np.zeros(n)
    score += np.where(cibil_score >= 720, 28, np.where(cibil_score >= 650, 18, 6))
    score += np.where(dti < 0.4, 24, np.where(dti < 0.6, 10, -12))
    score += np.where(statement_score >= 65, 22, np.where(statement_score >= 50, 10, -14))
    score += np.where(affordability < 0.35, 10, np.where(affordability < 0.6, 2, -10))
    score += np.where(down_payment_ratio >= 0.2, 8, 0)
    score -= np.where(overdrafts + bounced_payments > 2, 10, 0)
    score -= np.where(dependents >= 4, 6, 0)
    score += rng.normal(0, 6, size=n)

    approved = (score >= 50).astype(int)

    data = pd.DataFrame({
        "cibil_score": cibil_score,
        "income": income,
        "dependents": dependents,
        "loan_amount": loan_amount,
        "tenure": tenure,
        "down_payment": down_payment,
        "existing_emi": existing_emi,
        "credit_card": credit_card,
        "employment_type": employment_type,
        "statement_score": statement_score,
        "emi": emi,
        "account_age": account_age,
        "overdrafts": overdrafts,
        "bounced_payments": bounced_payments,
        "late_emi_ratio": late_emi_ratio,
        "income_volatility": income_volatility,
        "avg_balance": avg_balance,
        "transaction_frequency": transaction_frequency,
        "monthly_spending": monthly_spending,
        TARGET: approved,
    })

    return data


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Consumer synthetic dataset saved to: {DATA_PATH}")
    print(df.head(5).to_string(index=False))
