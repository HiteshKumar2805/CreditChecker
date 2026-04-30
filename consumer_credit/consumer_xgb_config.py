import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_PATH = os.path.join(DATA_DIR, "consumer_synthetic.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "consumer_xgb.pkl")
META_PATH = os.path.join(MODEL_DIR, "consumer_xgb_meta.json")

FEATURES = [
    "cibil_score",
    "income",
    "dependents",
    "loan_amount",
    "tenure",
    "down_payment",
    "existing_emi",
    "credit_card",
    "employment_type",
    "statement_score",
    "emi",
    "account_age",
    "overdrafts",
    "bounced_payments",
    "late_emi_ratio",
    "income_volatility",
    "avg_balance",
    "transaction_frequency",
    "monthly_spending",
]

TARGET = "approved"
