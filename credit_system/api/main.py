"""
Credit Decisioning — FastAPI Application
==========================================
Endpoints:
  POST /kyc/extract       → OCR extract Aadhaar + PAN PDF data
    POST /consumer/evaluate → Evaluate a consumer loan application
    POST /vehicle/evaluate  → Evaluate a vehicle loan application
  POST /evaluate          → Evaluate a loan application
  GET  /audit/{app_id}   → Fetch audit record by application ID
  GET  /audit/recent      → Recent decisions (officer view)
  GET  /audit/verify      → Verify tamper-evidence of the full chain
  GET  /fairness/report   → Run fairness report on full dataset
  GET  /health            → Service health check
"""
import os, sys, json, uuid, time, io

# Make sure sibling packages resolve
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, REPO_ROOT)

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from explainability.shap_explainer import SHAPExplainer
from audit.audit_logger import log_decision, verify_chain, get_decision, get_recent_decisions
from api.kyc_extractor import extract_aadhaar_data, extract_pan_data, cross_verify
from api.payslip_extractor import extract_payslip_data
import api.secure_vault as vault

try:
    from consumer_credit.main import (
        load_dataset as consumer_load_dataset,
        preprocess_data as consumer_preprocess_data,
        train_model as consumer_train_model,
        predict_loan as consumer_predict_loan,
    )
except Exception as e:
    consumer_load_dataset = None
    consumer_preprocess_data = None
    consumer_train_model = None
    consumer_predict_loan = None
    print(f"[API] Could not import consumer_credit pipeline: {e}")

try:
    from credit_scoring.src.document_reader import (
        extract_text_from_pdf as vehicle_extract_text_from_pdf,
        extract_text_from_image as vehicle_extract_text_from_image,
    )
    from credit_scoring.src.transaction_parser import parse_transactions as vehicle_parse_transactions
    from credit_scoring.src.feature_engineering import add_features as vehicle_add_features
except Exception as e:
    vehicle_extract_text_from_pdf = None
    vehicle_extract_text_from_image = None
    vehicle_parse_transactions = None
    vehicle_add_features = None
    print(f"[API] Could not import credit_scoring pipeline: {e}")

try:
    import pytesseract

    _tesseract_cmd = os.environ.get("TESSERACT_CMD")
    _default_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if _tesseract_cmd and os.path.exists(_tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
    elif os.path.exists(_default_tesseract):
        pytesseract.pytesseract.tesseract_cmd = _default_tesseract
except Exception as e:
    print(f"[API] Tesseract not configured: {e}")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Credit Decisioning API",
    description="Explainable AI-powered loan decision engine with SHAP + audit trail",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load model artifacts at startup ──────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
_model      = joblib.load(os.path.join(MODEL_DIR, "lgbm_credit.pkl"))
_features   = json.load(open(os.path.join(MODEL_DIR, "feature_list.json")))
_meta       = json.load(open(os.path.join(MODEL_DIR, "model_meta.json")))
_threshold  = _meta["optimal_threshold"]
_explainer  = SHAPExplainer(model_dir=MODEL_DIR)

try:
    _personal_model = joblib.load(os.path.join(MODEL_DIR, "xgb_personal_loan.pkl"))
    print("[API] Personal loan model loaded.")
except Exception as e:
    _personal_model = None
    print(f"[API] Could not load personal loan model: {e}")

STATEMENT_FEATURES = [
    "cibil_score",
    "income",
    "loan_amount",
    "emi",
    "account_age",
    "overdrafts",
    "bounced_payments",
    "late_emi_ratio",
    "income_volatility",
    "avg_balance",
    "transaction_frequency",
    "monthly_spending",
    "debt_to_income_ratio",
    "spending_ratio",
    "income_stability",
    "emi_burden",
    "transaction_score",
]

_statement_model = None
try:
    statement_model_path = os.path.join(REPO_ROOT, "credit_scoring", "models", "xgboost_model.pkl")
    _statement_model = joblib.load(statement_model_path)
    print("[API] Bank statement XGBoost model loaded.")
except Exception as e:
    _statement_model = None
    print(f"[API] Could not load bank statement model: {e}")

_consumer_model = None
_consumer_scaler = None
_consumer_feature_columns = None
_consumer_xgb_model = None
_consumer_xgb_features = None
_consumer_xgb_threshold = 0.5
_consumer_xgb_meta = {}

try:
    if consumer_load_dataset and consumer_preprocess_data and consumer_train_model:
        consumer_data_path = os.path.join(REPO_ROOT, "consumer_credit", "loan_data.csv")
        consumer_df = consumer_load_dataset(consumer_data_path)
        X_c, y_c, scaler_c, _, feature_cols_c = consumer_preprocess_data(consumer_df)
        _consumer_model = consumer_train_model(X_c, y_c)
        _consumer_scaler = scaler_c
        _consumer_feature_columns = list(feature_cols_c)
        print("[API] Consumer loan model loaded.")
except Exception as e:
    _consumer_model = None
    print(f"[API] Could not load consumer loan model: {e}")

try:
    consumer_model_dir = os.path.join(REPO_ROOT, "consumer_credit", "models")
    _consumer_xgb_model = joblib.load(os.path.join(consumer_model_dir, "consumer_xgb.pkl"))
    _consumer_xgb_meta = json.load(open(os.path.join(consumer_model_dir, "consumer_xgb_meta.json")))
    _consumer_xgb_features = _consumer_xgb_meta.get("features")
    _consumer_xgb_threshold = float(_consumer_xgb_meta.get("optimal_threshold", 0.5))
    print("[API] Consumer XGBoost model loaded.")
except Exception as e:
    _consumer_xgb_model = None
    print(f"[API] Could not load consumer XGBoost model: {e}")

_personal_shap_explainer = None
_statement_shap_explainer = None
_consumer_shap_explainer = None
try:
    import shap

    if _personal_model:
        _personal_shap_explainer = shap.TreeExplainer(_personal_model)
        print("[API] Personal loan SHAP explainer ready.")
    if _statement_model:
        _statement_shap_explainer = shap.TreeExplainer(_statement_model)
    if _consumer_xgb_model:
        _consumer_shap_explainer = shap.TreeExplainer(_consumer_xgb_model)
except Exception as e:
    _personal_shap_explainer = None
    _statement_shap_explainer = None
    _consumer_shap_explainer = None
    print(f"[API] SHAP not available: {e}")

print(f"[API] Model loaded — threshold={_threshold:.4f}  version={_meta['version']}")


# ─── Request / Response Schemas ───────────────────────────────────────────────
class LoanApplication(BaseModel):
    # Optional — will be auto-generated if not provided
    application_id:          Optional[str]  = Field(None,  description="Unique application ID")
    officer_id:              Optional[str]  = Field(None,  description="Officer/agent ID")

    # Credit profile
    cibil_score:             int   = Field(..., ge=300,  le=900,    example=720)
    dti_ratio:               float = Field(..., ge=0.0,  le=5.0,    example=0.35)
    net_monthly_surplus:     float = Field(...,                     example=25000)
    days_past_due:           int   = Field(..., ge=0,    le=365,    example=0)
    bounced_cheques_12m:     int   = Field(..., ge=0,    le=20,     example=0)
    credit_enquiries_6m:     int   = Field(..., ge=0,    le=20,     example=2)
    credit_history_years:    int   = Field(..., ge=0,    le=40,     example=5)
    num_credit_cards:        int   = Field(..., ge=0,    le=20,     example=2)
    credit_utilization:      float = Field(..., ge=0.0,  le=1.0,    example=0.3)

    # Income & loan
    monthly_income:          float = Field(..., ge=0,               example=80000)
    co_applicant_income:     float = Field(0.0, ge=0,               example=0)
    loan_amount:             float = Field(..., ge=1000,            example=500000)
    loan_tenure_months:      int   = Field(..., ge=6,    le=360,    example=60)
    proposed_emi:            float = Field(..., ge=0,               example=12000)
    loan_to_income:          float = Field(..., ge=0,               example=6.25)
    loan_to_value:           float = Field(..., ge=0.0,  le=1.5,    example=0.7)
    loan_purpose:            int   = Field(..., ge=0,    le=10,     example=1)

    # Assets & existing obligations
    property_value:          float = Field(0.0, ge=0,               example=1000000)
    existing_loans:          int   = Field(..., ge=0,    le=20,     example=1)
    existing_emi_monthly:    float = Field(..., ge=0,               example=5000)

    # Banking behaviour
    avg_monthly_balance:     float = Field(..., ge=0,               example=50000)
    balance_volatility:      float = Field(..., ge=0.0,  le=1.0,    example=0.2)
    monthly_inflow:          float = Field(..., ge=0,               example=90000)
    salary_credit_regular:   int   = Field(..., ge=0,    le=1,      example=1)

    # Employment
    employment_type:         int   = Field(..., ge=0,    le=5,      example=1)
    employment_tenure_years: float = Field(..., ge=0,    le=50,     example=3.5)
    employment_tenure_days:  int   = Field(..., ge=0,               example=1277)

    # Personal (excluded from model, kept for fairness testing)
    age:                     int   = Field(..., ge=18,   le=80,     example=32)
    gender:                  int   = Field(..., ge=0,    le=1,      example=1)
    education:               int   = Field(..., ge=0,    le=5,      example=2)
    marital_status:          int   = Field(..., ge=0,    le=3,      example=1)
    num_dependents:          int   = Field(..., ge=0,    le=20,     example=2)
    num_children:            int   = Field(..., ge=0,    le=15,     example=1)
    city_tier:               int   = Field(..., ge=1,    le=3,      example=2)


class DecisionResponse(BaseModel):
    application_id:   str
    approved:         bool
    probability:      float
    confidence_level: str       # HIGH / MEDIUM / LOW
    applicant_message: dict     # Plain-language explanation
    shap_factors:     list      # Top-10 SHAP factors
    model_version:    str
    threshold_used:   float
    block_hash:       str       # Audit chain entry hash
    timestamp_iso:    str
    base_value:       Optional[float] = None
    counterfactuals:  Optional[list]  = None


class PersonalLoanApplication(BaseModel):
    cibil_score: int
    monthly_salary: float
    company_tier: int
    loan_amount: float
    loan_tenure_months: int


class ConsumerDecisionResponse(BaseModel):
    application_id:         str
    approved:               bool
    decision:               str
    score:                  float
    txn_confidence_score:   float
    cibil_normalized_score: float
    reason:                 Optional[str] = None
    statement_ml_probability: Optional[float] = None
    shap_factors:           list
    model_version:          str
    threshold_used:         float
    block_hash:             str
    timestamp_iso:          str
    applicant_message:      dict
    base_value:             Optional[float] = None
    counterfactuals:        Optional[list]  = None


class VehicleDecisionResponse(BaseModel):
    application_id: str
    approved:       bool
    decision:       str
    final_score:    float
    risk_score:     float
    signals:        dict
    statement_ml_probability: Optional[float] = None
    shap_factors:  list
    model_version:  str
    block_hash:     str
    timestamp_iso:  str
    applicant_message: dict
    base_value:     Optional[float] = None
    counterfactuals: Optional[list] = None


# ─── Helper ───────────────────────────────────────────────────────────────────
def _confidence(prob: float) -> str:
    distance = abs(prob - 0.5)
    if distance > 0.35:  return "HIGH"
    if distance > 0.15:  return "MEDIUM"
    return "LOW"


def _count_keywords(text: str, keywords: list) -> int:
    if not text:
        return 0
    lowered = text.lower()
    return sum(lowered.count(keyword) for keyword in keywords)


def _estimate_loan_amount(loan_amount: Optional[float], income: Optional[float]) -> float:
    if loan_amount and loan_amount > 0:
        return float(loan_amount)
    if income and income > 0:
        return float(income) * 10.0
    return 100000.0


def _estimate_emi(loan_amount: float, tenure: Optional[int], existing_emi: Optional[float]) -> float:
    if existing_emi and existing_emi > 0:
        return float(existing_emi)
    if tenure and tenure > 0:
        return float(loan_amount) / float(tenure)
    return max(1000.0, float(loan_amount) * 0.03)


def _pretty_label(name: str) -> str:
    return name.replace("_", " ").title()


def _encode_yes_no(value: Optional[str]) -> int:
    if not value:
        return 0
    return 1 if str(value).strip().lower() in ("yes", "y", "true", "1") else 0


def _encode_employment_type(value: Optional[str]) -> int:
    if not value:
        return 0
    lowered = str(value).strip().lower()
    if lowered in ("self-employed", "self employed", "selfemployed"):
        return 1
    return 0


def _build_shap_factors(explainer, feature_df: pd.DataFrame, feature_names: list, top_k: int = 6) -> list:
    if not explainer:
        return []

    shap_values = explainer.shap_values(feature_df)
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    values = np.array(values).reshape(-1)
    raw_values = feature_df.iloc[0].to_dict()
    order = np.argsort(np.abs(values))[::-1][:top_k]

    factors = []
    for idx in order:
        name = feature_names[idx]
        factors.append({
            "feature": name,
            "label": _pretty_label(name),
            "shap_value": float(values[idx]),
            "raw_value": raw_values.get(name),
        })
    return factors


def _get_base_value(explainer) -> float:
    """Extract the SHAP base (expected) value from a TreeExplainer."""
    if not explainer:
        return None
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        return float(ev[1]) if len(ev) > 1 else float(ev[0])
    return float(ev)


# ── Counterfactual guidance per feature ─────────────────────────────────────
_CF_GUIDANCE = {
    "cibil_score":        {"good": 750, "dir": "increase", "tip": "Pay all EMIs on time and reduce credit card balances for 6-12 months."},
    "dti_ratio":          {"good": 0.30, "dir": "decrease", "tip": "Clear existing debts or increase income before re-applying."},
    "net_monthly_surplus": {"good": 30000, "dir": "increase", "tip": "Reduce monthly expenses or increase income to improve surplus."},
    "monthly_income":     {"good": 80000, "dir": "increase", "tip": "Consider adding a co-applicant or waiting for a salary increment."},
    "monthly_salary":     {"good": 80000, "dir": "increase", "tip": "A higher documented salary strengthens the application."},
    "bounced_cheques_12m": {"good": 0, "dir": "decrease", "tip": "Maintain adequate account balance to prevent cheque bounces."},
    "credit_utilization":  {"good": 0.30, "dir": "decrease", "tip": "Pay down credit card balances to below 30% utilization."},
    "credit_enquiries_6m": {"good": 1, "dir": "decrease", "tip": "Avoid new credit applications for at least 6 months."},
    "days_past_due":       {"good": 0, "dir": "decrease", "tip": "Clear all overdue payments immediately."},
    "existing_loans":      {"good": 1, "dir": "decrease", "tip": "Close at least one existing loan before applying."},
    "loan_amount":         {"good": None, "dir": "decrease", "tip": "Consider applying for a smaller loan amount."},
    "company_tier":        {"good": 1, "dir": "decrease", "tip": "Tier 1 company employment strengthens applications."},
    "income_volatility":   {"good": 0.1, "dir": "decrease", "tip": "Maintain consistent income credits in your account."},
    "overdrafts":          {"good": 0, "dir": "decrease", "tip": "Avoid overdrafts by maintaining sufficient balance."},
    "bounced_payments":    {"good": 0, "dir": "decrease", "tip": "Ensure all payments clear without bouncing."},
    "spending_ratio":      {"good": 0.5, "dir": "decrease", "tip": "Reduce discretionary spending relative to income."},
    "debt_to_income_ratio": {"good": 0.3, "dir": "decrease", "tip": "Reduce total debt obligations relative to income."},
    "emi_burden":          {"good": 0.3, "dir": "decrease", "tip": "Reduce EMI obligations or increase income."},
    "avg_balance":         {"good": 50000, "dir": "increase", "tip": "Maintain a higher average bank balance."},
    "transaction_score":   {"good": 80, "dir": "increase", "tip": "Improve transaction patterns — avoid overdrafts and bounces."},
}


def _compute_counterfactuals(shap_factors: list, raw_values: dict = None) -> list:
    """Generate counterfactual suggestions for features that hurt the application."""
    counterfactuals = []
    negative_factors = [f for f in shap_factors if f["shap_value"] < 0 and not f["feature"].endswith("_rule")]

    for factor in sorted(negative_factors, key=lambda f: f["shap_value"])[:5]:
        feat = factor["feature"]
        guidance = _CF_GUIDANCE.get(feat)
        if not guidance:
            continue

        current = factor.get("raw_value")
        if current is None:
            continue

        target = guidance["good"]
        if target is None:
            if guidance["dir"] == "decrease":
                target = current * 0.7
            else:
                target = current * 1.5

        # Format display values
        def _fmt(v):
            if isinstance(v, float):
                return f"{v:,.2f}" if v < 10 else f"{v:,.0f}"
            return str(v)

        counterfactuals.append({
            "feature": feat,
            "label": factor["label"],
            "current_value": current,
            "target_value": target,
            "current_display": _fmt(current),
            "target_display": _fmt(target),
            "direction": guidance["dir"],
            "tip": guidance["tip"],
        })

    return counterfactuals


def _build_consumer_feature_row(
    input_data: dict,
    statement_features: dict,
    statement_score: Optional[float],
):
    features = {
        "cibil_score": float(input_data.get("cibil_score", 0)),
        "income": float(input_data.get("income", 0)),
        "dependents": float(input_data.get("dependents", 0)),
        "loan_amount": float(input_data.get("loan_amount", 0)),
        "tenure": float(input_data.get("tenure", 0)),
        "down_payment": float(input_data.get("down_payment", 0)),
        "existing_emi": float(input_data.get("existing_emi", 0)),
        "credit_card": float(_encode_yes_no(input_data.get("credit_card"))),
        "employment_type": float(_encode_employment_type(input_data.get("employment_type"))),
        "statement_score": float(statement_score or 0.0),
    }

    statement_defaults = {
        "emi": 0.0,
        "account_age": 0.0,
        "overdrafts": 0.0,
        "bounced_payments": 0.0,
        "late_emi_ratio": 0.0,
        "income_volatility": 0.0,
        "avg_balance": 0.0,
        "transaction_frequency": 0.0,
        "monthly_spending": 0.0,
    }

    for key, fallback in statement_defaults.items():
        features[key] = float(statement_features.get(key, fallback))

    feature_names = _consumer_xgb_features or list(features.keys())
    feature_row = {name: float(features.get(name, 0.0)) for name in feature_names}
    return pd.DataFrame([feature_row]), features


def _features_from_text(
    text: str,
    cibil_score: int,
    income: float,
    loan_amount: Optional[float],
    tenure: Optional[int],
    existing_emi: Optional[float],
):
    parsed = vehicle_parse_transactions(text or "") if vehicle_parse_transactions else []
    signals = vehicle_add_features(parsed) if vehicle_add_features else {}

    transaction_frequency = len(parsed)
    expense_score = float(signals.get("expense_score", 0))
    income_score = float(signals.get("income_score", 0))
    balance_score = float(signals.get("balance_score", 0))

    overdrafts = _count_keywords(text, ["overdraft", "od"])
    bounced = _count_keywords(text, ["bounce", "bounced", "return", "nsf"])
    late = _count_keywords(text, ["late", "overdue", "delayed"])

    denom = max(1.0, income_score + expense_score)
    income_volatility = min(1.0, max(0.0, expense_score / denom))

    monthly_spending = max(0.0, expense_score * 1000.0)
    avg_balance = max(1000.0, (float(income) * 0.25) + (balance_score * 100.0))

    account_age = min(120.0, max(1.0, 12.0 + (transaction_frequency // 10)))
    loan_amount_val = _estimate_loan_amount(loan_amount, income)
    emi = _estimate_emi(loan_amount_val, tenure, existing_emi)

    late_emi_ratio = float(late / max(1, transaction_frequency))

    income_value = max(1.0, float(income))
    existing_emi_value = float(existing_emi or 0.0)
    debt_to_income_ratio = (emi + existing_emi_value) / income_value
    spending_ratio = monthly_spending / income_value
    income_stability = max(0.0, min(1.0, 1.0 - income_volatility))
    emi_burden = emi / income_value
    transaction_score = 60.0 + (transaction_frequency / 3.0)
    transaction_score -= (overdrafts * 6.0)
    transaction_score -= (bounced * 8.0)
    transaction_score -= (late_emi_ratio * 40.0)
    transaction_score -= (spending_ratio * 20.0)
    transaction_score = max(0.0, min(100.0, transaction_score))

    features = {
        "cibil_score": float(cibil_score),
        "income": float(income),
        "loan_amount": float(loan_amount_val),
        "emi": float(emi),
        "account_age": float(account_age),
        "overdrafts": float(overdrafts),
        "bounced_payments": float(bounced),
        "late_emi_ratio": late_emi_ratio,
        "income_volatility": float(income_volatility),
        "avg_balance": float(avg_balance),
        "transaction_frequency": float(transaction_frequency),
        "monthly_spending": float(monthly_spending),
        "debt_to_income_ratio": float(debt_to_income_ratio),
        "spending_ratio": float(spending_ratio),
        "income_stability": float(income_stability),
        "emi_burden": float(emi_burden),
        "transaction_score": float(transaction_score),
    }

    return features, signals


def _features_from_csv(
    txn_df: pd.DataFrame,
    cibil_score: int,
    income: float,
    loan_amount: Optional[float],
    tenure: Optional[int],
    existing_emi: Optional[float],
):
    txn_df = txn_df.copy()
    if "txn_type" in txn_df.columns:
        txn_df["txn_type"] = txn_df["txn_type"].astype(str).str.lower()

    transaction_frequency = len(txn_df)
    debit_mask = txn_df.get("txn_type", pd.Series([], dtype=str)).str.contains("debit", na=False)
    credit_mask = txn_df.get("txn_type", pd.Series([], dtype=str)).str.contains("credit", na=False)

    debit_amounts = txn_df.loc[debit_mask, "amount"] if "amount" in txn_df.columns else pd.Series([], dtype=float)
    credit_amounts = txn_df.loc[credit_mask, "amount"] if "amount" in txn_df.columns else pd.Series([], dtype=float)

    monthly_spending = float(debit_amounts.abs().sum()) if not debit_amounts.empty else 0.0
    avg_balance = float(txn_df["balance"].mean()) if "balance" in txn_df.columns else float(income) * 0.25

    overdrafts = float((txn_df["balance"] < 0).sum()) if "balance" in txn_df.columns else 0.0

    income_volatility = 0.0
    if not credit_amounts.empty:
        mean_credit = credit_amounts.mean()
        if mean_credit:
            income_volatility = float(credit_amounts.std(ddof=0) / mean_credit)
    income_volatility = min(1.0, max(0.0, income_volatility))

    loan_amount_val = _estimate_loan_amount(loan_amount, income)
    emi = _estimate_emi(loan_amount_val, tenure, existing_emi)

    bounced_payments = 0.0
    late_emi_ratio = 0.0

    income_value = max(1.0, float(income))
    existing_emi_value = float(existing_emi or 0.0)
    debt_to_income_ratio = (emi + existing_emi_value) / income_value
    spending_ratio = monthly_spending / income_value
    income_stability = max(0.0, min(1.0, 1.0 - income_volatility))
    emi_burden = emi / income_value
    transaction_score = 60.0 + (transaction_frequency / 3.0)
    transaction_score -= (overdrafts * 6.0)
    transaction_score -= (bounced_payments * 8.0)
    transaction_score -= (late_emi_ratio * 40.0)
    transaction_score -= (spending_ratio * 20.0)
    transaction_score = max(0.0, min(100.0, transaction_score))

    features = {
        "cibil_score": float(cibil_score),
        "income": float(income),
        "loan_amount": float(loan_amount_val),
        "emi": float(emi),
        "account_age": 12.0,
        "overdrafts": float(overdrafts),
        "bounced_payments": bounced_payments,
        "late_emi_ratio": late_emi_ratio,
        "income_volatility": float(income_volatility),
        "avg_balance": float(avg_balance),
        "transaction_frequency": float(transaction_frequency),
        "monthly_spending": float(monthly_spending),
        "debt_to_income_ratio": float(debt_to_income_ratio),
        "spending_ratio": float(spending_ratio),
        "income_stability": float(income_stability),
        "emi_burden": float(emi_burden),
        "transaction_score": float(transaction_score),
    }

    return features, {}


def _classify_bank_statement(
    *,
    ext: str,
    file_bytes: bytes,
    cibil_score: int,
    income: float,
    loan_amount: Optional[float] = None,
    tenure: Optional[int] = None,
    existing_emi: Optional[float] = None,
):
    if not _statement_model:
        return None, {}, {}

    if ext == ".csv":
        txn_df = pd.read_csv(io.BytesIO(file_bytes))
        features, signals = _features_from_csv(
            txn_df,
            cibil_score,
            income,
            loan_amount,
            tenure,
            existing_emi,
        )
    elif ext in (".pdf", ".png", ".jpg", ".jpeg"):
        if ext == ".pdf" and not vehicle_extract_text_from_pdf:
            raise HTTPException(status_code=500, detail="PDF statement extraction not available.")
        if ext in (".png", ".jpg", ".jpeg") and not vehicle_extract_text_from_image:
            raise HTTPException(status_code=500, detail="Image statement extraction not available.")

        if ext == ".pdf":
            text = vehicle_extract_text_from_pdf(io.BytesIO(file_bytes))
        else:
            text = vehicle_extract_text_from_image(io.BytesIO(file_bytes))

        features, signals = _features_from_text(
            text,
            cibil_score,
            income,
            loan_amount,
            tenure,
            existing_emi,
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, PNG, JPG, or CSV.")

    feature_df = pd.DataFrame([{name: features.get(name, 0.0) for name in STATEMENT_FEATURES}])
    prob = float(_statement_model.predict_proba(feature_df)[0][1])
    return prob, features, signals


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_version": _meta["version"],
        "threshold":     _threshold,
        "cv_auc":        _meta["cv_auc"],
    }


# ─── KYC Document OCR Endpoint ────────────────────────────────────────────────

@app.post("/kyc/extract")
async def kyc_extract(
    aadhaar_pdf: UploadFile = File(..., description="Aadhaar card PDF/image (front + back)"),
    pan_pdf:     UploadFile = File(..., description="PAN card PDF/image (front + back)"),
):
    """
    OCR-extract structured KYC data from Aadhaar and PAN card PDFs/images.
    Returns name, Aadhaar number, DOB, gender, address, mobile (from Aadhaar)
    and PAN number + cross-verification results.
    """
    # Validate file types
    for upload, label in [(aadhaar_pdf, "Aadhaar"), (pan_pdf, "PAN")]:
        if upload.content_type not in ("application/pdf", "application/octet-stream"):
            # Accept any binary — user may not set correct MIME
            pass

    try:
        print("[KYC] Reading uploaded files...")
        aadhaar_bytes = await aadhaar_pdf.read()
        pan_bytes     = await pan_pdf.read()
        print(f"[KYC] Files read: Aadhaar={len(aadhaar_bytes)} bytes, PAN={len(pan_bytes)} bytes")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded files: {e}")

    try:
        print("[KYC] Starting Aadhaar OCR extraction (this may take 30-60 seconds)...")
        aadhaar_data = extract_aadhaar_data(aadhaar_bytes)
        print("[KYC] ✓ Aadhaar OCR complete")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Aadhaar OCR failed: {e}")

    try:
        print("[KYC] Starting PAN OCR extraction (this may take 15-30 seconds)...")
        pan_data = extract_pan_data(pan_bytes)
        print("[KYC] ✓ PAN OCR complete")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PAN OCR failed: {e}")

    aadhaar_has_data = any(aadhaar_data.get(key) for key in ("name", "aadhaar_number", "dob", "gender", "address", "mobile"))
    pan_has_data = any(pan_data.get(key) for key in ("name", "pan_number", "dob"))
    if not aadhaar_has_data or not pan_has_data:
        missing = []
        if not aadhaar_has_data:
            missing.append("Aadhaar")
        if not pan_has_data:
            missing.append("PAN")
        raise HTTPException(
            status_code=422,
            detail=f"Could not detect readable KYC data from {' and '.join(missing)}. Upload a clear PDF/image, crop to the document, and avoid blur or glare.",
        )

    verification = cross_verify(aadhaar_data, pan_data)

    payload = {
        "aadhaar": {
            "name":           aadhaar_data.get("name"),
            "aadhaar_number": aadhaar_data.get("aadhaar_number"),
            "dob":            aadhaar_data.get("dob"),
            "gender":         aadhaar_data.get("gender"),
            "address":        aadhaar_data.get("address"),
            "mobile":         aadhaar_data.get("mobile"),
        },
        "pan": {
            "name":       pan_data.get("name"),
            "pan_number": pan_data.get("pan_number"),
            "dob":        pan_data.get("dob"),
        },
        "verification": verification,
    }

    # Store encrypted in DB
    session_id = str(uuid.uuid4())
    vault.store_sensitive_data(session_id, payload)

    return {
        "session_id": session_id,
        "message": "Data extracted and securely stored in encrypted vault.",
        "aadhaar": payload["aadhaar"],
        "pan": payload["pan"],
        "verification": payload["verification"],
        "data": payload
    }


@app.post("/payslip/extract")
async def payslip_extract(
    payslip_img: UploadFile = File(..., description="Payslip Image"),
):
    """
    OCR-extract salary and company from payslip using EasyOCR.
    """
    try:
        img_bytes = await payslip_img.read()
        extracted = extract_payslip_data(img_bytes)
        if not extracted.get("salary"):
            raise HTTPException(
                status_code=422,
                detail="Could not find a salary amount in this payslip. Please upload a clearer image or a text-readable PDF.",
            )
        return extracted
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Payslip extraction failed: {e}")


@app.post("/evaluate_personal", response_model=DecisionResponse)
def evaluate_personal_loan(application: PersonalLoanApplication):
    if not _personal_model:
        raise HTTPException(status_code=500, detail="Personal loan model not loaded.")
        
    # Prepare features
    features = ['cibil_score', 'monthly_salary', 'company_tier', 'loan_amount']
    df = pd.DataFrame([{
        'cibil_score': application.cibil_score,
        'monthly_salary': application.monthly_salary,
        'company_tier': application.company_tier,
        'loan_amount': application.loan_amount
    }])
    
    # Predict
    prob = _personal_model.predict_proba(df)[0][1]
    approved = bool(prob > 0.5)
    
    app_id = str(uuid.uuid4())
    
    # Real SHAP explanation using TreeExplainer on the personal loan XGBoost model
    personal_features = ['cibil_score', 'monthly_salary', 'company_tier', 'loan_amount']
    shap_factors = _build_shap_factors(
        _personal_shap_explainer,
        df,
        personal_features,
        top_k=4,
    )
    base_value = _get_base_value(_personal_shap_explainer)
    counterfactuals = _compute_counterfactuals(shap_factors) if not approved else []
    
    # Generate data-driven applicant message from SHAP factors
    if approved:
        positive = [f for f in shap_factors if f["shap_value"] > 0]
        reasons = [
            f"{f['label']} positively influenced the decision."
            for f in positive[:3]
        ] or ["Strong overall profile."]
    else:
        negative = [f for f in shap_factors if f["shap_value"] < 0]
        reasons = [
            f"{f['label']} negatively influenced the decision."
            for f in negative[:3]
        ] or ["Profile did not meet the required criteria."]
    
    msg = {
        "summary": f"Personal loan result: {'You May Be Eligible' if approved else 'You May Get Rejected'}.",
        "primary_reasons": reasons,
        "actionable_tips": [
            f"{f['label']} — improve this to strengthen future applications"
            for f in shap_factors if f["shap_value"] < 0
        ][:3] if not approved else []
    }
    
    log_record = log_decision(
        application_id=app_id,
        inputs=application.dict(),
        decision={
            "approved": approved,
            "probability": float(prob),
            "threshold": 0.5,
        },
        shap_output={
            "factors": shap_factors,
            "base_value": base_value or 0.0,
        },
        officer_id="system"
    )
    
    return DecisionResponse(
        application_id=app_id,
        approved=approved,
        probability=float(prob),
        confidence_level=_confidence(float(prob)),
        applicant_message=msg,
        shap_factors=shap_factors,
        model_version="xgb_personal_1.0",
        threshold_used=0.5,
        block_hash=log_record["block_hash"],
        timestamp_iso=log_record["timestamp_iso"],
        base_value=base_value,
        counterfactuals=counterfactuals,
    )


@app.post("/consumer/evaluate", response_model=ConsumerDecisionResponse)
async def evaluate_consumer_loan(
    customer_name: Optional[str] = Form(None),
    product_price: float = Form(...),
    down_payment: float = Form(...),
    loan_amount: float = Form(...),
    tenure: int = Form(...),
    income: float = Form(...),
    dependents: int = Form(...),
    bank_name: str = Form(...),
    account_number: str = Form(...),
    ifsc: str = Form(...),
    cibil_score: int = Form(...),
    existing_emi: float = Form(...),
    credit_card: str = Form(...),
    employment_type: str = Form(...),
    statement: UploadFile = File(..., description="Bank statement (PDF/PNG/JPG/CSV)"),
):
    if not _consumer_xgb_model or not _consumer_xgb_features:
        raise HTTPException(status_code=500, detail="Consumer XGBoost model not loaded.")

    filename = statement.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    file_bytes = await statement.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded statement file is empty.")

    statement_prob, statement_features, statement_signals = _classify_bank_statement(
        ext=ext,
        file_bytes=file_bytes,
        cibil_score=cibil_score,
        income=income,
        loan_amount=loan_amount,
        tenure=tenure,
        existing_emi=existing_emi,
    )
    if statement_prob is None:
        raise HTTPException(status_code=500, detail="Bank statement model not loaded.")
    statement_score = float(statement_prob * 100)

    input_data = {
        "product_price": product_price,
        "down_payment": down_payment,
        "loan_amount": loan_amount,
        "tenure": tenure,
        "bank_name": bank_name,
        "account_number": account_number,
        "IFSC": ifsc,
        "cibil_score": cibil_score,
        "existing_emi": existing_emi,
        "credit_card": credit_card,
        "income": income,
        "employment_type": employment_type,
        "dependents": dependents,
    }

    consumer_df, consumer_features = _build_consumer_feature_row(
        input_data,
        statement_features,
        statement_score,
    )
    prob = float(_consumer_xgb_model.predict_proba(consumer_df)[0][1])
    approved = prob >= _consumer_xgb_threshold
    status = "You May Be Eligible" if approved else "You May Get Rejected"
    score = round(prob * 100, 2)
    cibil_normalized_score = max(0.0, min(100.0, (float(cibil_score) - 300.0) / 600.0 * 100.0))
    reason = None
    if not approved:
        reason = f"Probability ({score:.1f}/100) is below the {(_consumer_xgb_threshold * 100):.1f} threshold."

    shap_factors = _build_shap_factors(
        _consumer_shap_explainer,
        consumer_df,
        _consumer_xgb_features or list(consumer_df.columns),
    )

    app_id = str(uuid.uuid4())
    log_record = log_decision(
        application_id=app_id,
        inputs={
            **input_data,
            "customer_name": customer_name,
            "statement_file": filename,
            "statement_ml_probability": statement_prob,
            "statement_features": statement_features,
            "statement_signals": statement_signals,
            "consumer_features": consumer_features,
        },
        decision={
            "approved": approved,
            "probability": round(prob, 4),
            "threshold": _consumer_xgb_threshold,
            "status": status,
        },
        shap_output={
            "factors": shap_factors,
            "base_value": 0.0,
        },
        officer_id="system",
    )

    consumer_base_value = _get_base_value(_consumer_shap_explainer)
    consumer_counterfactuals = _compute_counterfactuals(shap_factors) if not approved else []

    if approved:
        positive = [f for f in shap_factors if f["shap_value"] > 0]
        reasons = [
            f"{f['label']} positively influenced the decision."
            for f in positive[:3]
        ] or ["Strong overall profile."]
    else:
        negative = [f for f in shap_factors if f["shap_value"] < 0]
        reasons = [
            f"{f['label']} negatively influenced the decision."
            for f in negative[:3]
        ] or ["Profile did not meet the required criteria."]

    msg = {
        "summary": f"Consumer loan result: {'You May Be Eligible' if approved else 'You May Get Rejected'}.",
        "primary_reasons": reasons,
        "actionable_tips": [
            f"{f['label']} — improve this to strengthen future applications"
            for f in shap_factors if f["shap_value"] < 0
        ][:3] if not approved else []
    }

    return ConsumerDecisionResponse(
        application_id=app_id,
        approved=approved,
        decision=status,
        score=float(score),
        txn_confidence_score=float(statement_score),
        cibil_normalized_score=float(cibil_normalized_score),
        reason=reason,
        statement_ml_probability=float(statement_prob),
        shap_factors=shap_factors,
        model_version="consumer_xgb_1.0",
        threshold_used=_consumer_xgb_threshold,
        block_hash=log_record["block_hash"],
        timestamp_iso=log_record["timestamp_iso"],
        applicant_message=msg,
        base_value=consumer_base_value,
        counterfactuals=consumer_counterfactuals,
    )


@app.post("/vehicle/evaluate", response_model=VehicleDecisionResponse)
async def evaluate_vehicle_loan(
    cibil_score: int = Form(...),
    income: float = Form(...),
    vehicle_company: str = Form(""),
    vehicle_model: str = Form(""),
    statement: UploadFile = File(..., description="Bank statement (PDF/PNG/JPG)"),
):
    if not vehicle_extract_text_from_pdf or not vehicle_extract_text_from_image:
        raise HTTPException(status_code=500, detail="Vehicle loan pipeline not available.")

    filename = statement.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    file_bytes = await statement.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded statement file is empty.")

    statement_prob, statement_features, signals = _classify_bank_statement(
        ext=ext,
        file_bytes=file_bytes,
        cibil_score=cibil_score,
        income=income,
    )

    if statement_prob is None:
        if ext == ".pdf":
            if not vehicle_extract_text_from_pdf:
                raise HTTPException(status_code=500, detail="PDF statement extraction not available.")
            text = vehicle_extract_text_from_pdf(io.BytesIO(file_bytes))
        elif ext in (".png", ".jpg", ".jpeg"):
            if not vehicle_extract_text_from_image:
                raise HTTPException(status_code=500, detail="Image statement extraction not available.")
            text = vehicle_extract_text_from_image(io.BytesIO(file_bytes))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF or image.")

        parsed = vehicle_parse_transactions(text or "") if vehicle_parse_transactions else []
        signals = vehicle_add_features(parsed) if vehicle_add_features else {}

    risk_score = 0
    if signals.get("salary_flag", 0) == 0:
        risk_score += 25
    if signals.get("fraud_risk", 0) > 0:
        risk_score += 40
    if signals.get("balance_score", 0) < 0:
        risk_score += 20
    if signals.get("transaction_volume", 0) < 5:
        risk_score += 15

    statement_score = float(statement_prob * 100.0) if statement_prob is not None else None
    if statement_score is None:
        final_score = max(0.0, 100.0 - float(risk_score))
    else:
        final_score = max(0.0, min(100.0, (statement_score * 0.7) + ((100.0 - float(risk_score)) * 0.3)))

    statement_feature_df = pd.DataFrame(
        [{name: statement_features.get(name, 0.0) for name in STATEMENT_FEATURES}]
    )
    shap_factors = _build_shap_factors(
        _statement_shap_explainer,
        statement_feature_df,
        STATEMENT_FEATURES,
    )

    # Append rule-based attribution factors (covers the 30% rule-based portion of the score)
    rule_factors = []
    if signals.get("salary_flag", 0) == 0:
        rule_factors.append({
            "feature": "salary_flag_rule",
            "label": "No Regular Salary (Rule-Based)",
            "shap_value": -0.25,
            "raw_value": "No regular salary credits detected in statement",
        })
    if signals.get("fraud_risk", 0) > 0:
        rule_factors.append({
            "feature": "fraud_risk_rule",
            "label": "Fraud Risk Detected (Rule-Based)",
            "shap_value": -0.40,
            "raw_value": f"Fraud risk level: {signals.get('fraud_risk', 0)}",
        })
    if signals.get("balance_score", 0) < 0:
        rule_factors.append({
            "feature": "balance_score_rule",
            "label": "Negative Balance Score (Rule-Based)",
            "shap_value": -0.20,
            "raw_value": f"Balance score: {signals.get('balance_score', 0)}",
        })
    if signals.get("transaction_volume", 0) < 5:
        rule_factors.append({
            "feature": "txn_volume_rule",
            "label": "Low Transaction Volume (Rule-Based)",
            "shap_value": -0.15,
            "raw_value": f"Only {signals.get('transaction_volume', 0)} transactions found",
        })
    shap_factors.extend(rule_factors)

    if final_score >= 70:
        decision = "You May Be Eligible"
    elif final_score >= 40:
        decision = "MANUAL_REVIEW"
    else:
        decision = "You May Get Rejected"

    approved = decision == "You May Be Eligible"
    app_id = str(uuid.uuid4())
    log_record = log_decision(
        application_id=app_id,
        inputs={
            "cibil_score": cibil_score,
            "income": income,
            "statement_file": filename,
            "signals": signals,
            "statement_ml_probability": statement_prob,
            "statement_features": statement_features,
        },
        decision={
            "approved": approved,
            "probability": round(final_score / 100, 4),
            "threshold": 0.7,
            "status": decision,
        },
        shap_output={
            "factors": shap_factors,
            "base_value": 0.0,
        },
        officer_id="system",
    )

    vehicle_base_value = _get_base_value(_statement_shap_explainer)
    vehicle_counterfactuals = _compute_counterfactuals(shap_factors) if not approved else []

    if approved:
        positive = [f for f in shap_factors if f["shap_value"] > 0]
        reasons = [
            f"{f['label']} positively influenced the decision."
            for f in positive[:3]
        ] or ["Strong overall profile."]
    else:
        negative = [f for f in shap_factors if f["shap_value"] < 0]
        reasons = [
            f"{f['label']} negatively influenced the decision."
            for f in negative[:3]
        ] or ["Profile did not meet the required criteria."]

    msg = {
        "summary": f"Vehicle loan result: {'You May Be Eligible' if approved else 'You May Get Rejected'}.",
        "primary_reasons": reasons,
        "actionable_tips": [
            f"{f['label']} — improve this to strengthen future applications"
            for f in shap_factors if f["shap_value"] < 0
        ][:3] if not approved else []
    }

    return VehicleDecisionResponse(
        application_id=app_id,
        approved=approved,
        decision=decision,
        final_score=round(final_score, 2),
        risk_score=float(risk_score),
        signals=signals,
        statement_ml_probability=float(statement_prob) if statement_prob is not None else None,
        shap_factors=shap_factors,
        model_version="vehicle_xgb_blend_1.0",
        block_hash=log_record["block_hash"],
        timestamp_iso=log_record["timestamp_iso"],
        applicant_message=msg,
        base_value=vehicle_base_value,
        counterfactuals=vehicle_counterfactuals,
    )


@app.post("/evaluate", response_model=DecisionResponse)
def evaluate_loan(application: LoanApplication):
    """
    Main decision endpoint. Returns:
    - Approval / Rejection
    - Plain-language explanation for applicant
    - SHAP feature attributions for officer
    - Audit block hash for tamper evidence
    """
    app_dict = application.dict()
    app_id   = app_dict.pop("application_id") or str(uuid.uuid4())
    officer  = app_dict.pop("officer_id")

    # ── Predict ────────────────────────────────────────────────────────────
    X       = pd.DataFrame([app_dict])[_features]
    prob    = float(_model.predict_proba(X)[0, 1])
    approved = prob >= _threshold

    # ── Explain (SHAP) ─────────────────────────────────────────────────────
    shap_output = _explainer.explain(app_dict)

    # ── Applicant message ──────────────────────────────────────────────────
    applicant_msg = _explainer.generate_applicant_message(approved, shap_output)

    # ── Anomaly flag — flag if confidence is LOW (borderline decisions) ────
    confidence = _confidence(prob)
    if confidence == "LOW":
        applicant_msg["anomaly_flag"] = (
            "This decision is borderline — flagged for human review."
        )

    # ── Audit log ──────────────────────────────────────────────────────────
    audit_entry = log_decision(
        application_id=app_id,
        inputs=app_dict,
        decision={
            "approved":   approved,
            "probability": round(prob, 6),
            "threshold":   _threshold,
            "confidence":  confidence,
        },
        shap_output=shap_output,
        officer_id=officer,
    )

    base_value = shap_output.get("base_value")
    counterfactuals = _compute_counterfactuals(shap_output["top_factors"]) if not approved else []

    return DecisionResponse(
        application_id=app_id,
        approved=approved,
        probability=round(prob, 4),
        confidence_level=confidence,
        applicant_message=applicant_msg,
        shap_factors=shap_output["top_factors"],
        model_version=_meta["version"],
        threshold_used=_threshold,
        block_hash=audit_entry["block_hash"],
        timestamp_iso=audit_entry["timestamp_iso"],
        base_value=base_value,
        counterfactuals=counterfactuals,
    )


@app.get("/audit/verify")
def audit_verify():
    """Verify the integrity of the full audit chain."""
    return verify_chain()


@app.get("/audit/recent")
def audit_recent(limit: int = 20):
    """Return the most recent N audit decisions (officer dashboard)."""
    return get_recent_decisions(limit=limit)


@app.get("/audit/{application_id}")
def audit_get(application_id: str):
    """Fetch a specific audit record by application ID."""
    entry = get_decision(application_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Application ID not found in audit log")
    return entry


@app.get("/fairness/report")
def fairness_report():
    """Run the full fairness report on the training dataset."""
    from fairness.fairness_report import run_fairness_on_dataset
    from fairness.drift_calculator import compute_model_drift
    
    base = os.path.join(os.path.dirname(__file__), "..")
    report = run_fairness_on_dataset(
        data_path=os.path.join(base, "..", "synthetic_credit_dataset.csv"),
        model_dir=os.path.join(base, "model"),
    )
    
    # Calculate drift
    try:
        import numpy as np
        model_dir = os.path.join(base, "model")
        with open(os.path.join(model_dir, "model_meta.json"), "r") as f:
            meta = json.load(f)
        
        # We need historical predictions from training. For now, mock a normal distribution around the threshold
        # In a real scenario, these would be saved probabilities from the validation set
        threshold = meta.get("optimal_threshold", 0.7)
        training_probs = np.random.normal(threshold, 0.15, 1000)
        training_probs = np.clip(training_probs, 0.0, 1.0)
        
        # Fetch recent audit decisions to get recent probs
        recent_audits = get_recent_decisions(limit=100)
        recent_probs = np.array([float(a["decision"]["probability"]) for a in recent_audits if "probability" in a["decision"]])
        
        drift = compute_model_drift(training_probs, recent_probs)
    except Exception as e:
        drift = {"status": "ERROR", "message": str(e)}

    return {"fairness": report, "drift": drift}


@app.get("/compliance/report", response_class=HTMLResponse)
def export_compliance_report():
    """Export the formal compliance report as HTML."""
    from fairness.compliance_exporter import generate_compliance_report_html
    data = fairness_report()
    html = generate_compliance_report_html(data.get("fairness", {}), data.get("drift", {}))
    return HTMLResponse(content=html)
