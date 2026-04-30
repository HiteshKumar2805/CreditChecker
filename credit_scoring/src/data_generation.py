import numpy as np
import pandas as pd
import os


def generate_data(n=2000):
    np.random.seed(42)

    data = pd.DataFrame({
        "cibil_score": np.random.randint(300, 900, n),
        "income": np.random.randint(20000, 200000, n),
        "loan_amount": np.random.randint(50000, 1000000, n),
        "emi": np.random.randint(2000, 50000, n),
        "account_age": np.random.randint(1, 120, n),

        "overdrafts": np.random.randint(0, 5, n),
        "bounced_payments": np.random.randint(0, 3, n),
        "late_emi_ratio": np.random.rand(n),

        "income_volatility": np.random.rand(n),
        "avg_balance": np.random.randint(1000, 100000, n),

        "transaction_frequency": np.random.randint(5, 200, n),
        "monthly_spending": np.random.randint(10000, 150000, n),
    })

    # Target variable
    data["good_credit_behavior"] = (
        (data["cibil_score"] > 650).astype(int) +
        (data["late_emi_ratio"] < 0.2).astype(int) +
        (data["overdrafts"] == 0).astype(int)
    )

    data["good_credit_behavior"] = (data["good_credit_behavior"] >= 2).astype(int)

    return data


if __name__ == "__main__":
    df = generate_data()

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_data.csv", index=False)

    print("Dataset created successfully ✔")