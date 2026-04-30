"""
SHAP Explainability Module
============================
Uses SHAP TreeExplainer for fast, exact explanations on the LightGBM model.
Translates SHAP values into plain-language text for applicants.
"""
import json, joblib, os
import shap
import numpy as np
import pandas as pd

# ─── Feature → Human-readable text map ──────────────────────────────────────
FEATURE_LABELS = {
    "cibil_score":              "CIBIL Credit Score",
    "dti_ratio":                "Debt-to-Income Ratio",
    "net_monthly_surplus":      "Net Monthly Surplus",
    "days_past_due":            "Days Past Due on Existing Loans",
    "bounced_cheques_12m":      "Bounced Cheques (Last 12 Months)",
    "monthly_income":           "Monthly Income",
    "co_applicant_income":      "Co-applicant Income",
    "loan_amount":              "Requested Loan Amount",
    "loan_tenure_months":       "Loan Tenure",
    "proposed_emi":             "Proposed Monthly EMI",
    "loan_to_income":           "Loan-to-Income Ratio",
    "loan_to_value":            "Loan-to-Value Ratio",
    "loan_purpose":             "Loan Purpose",
    "property_value":           "Property / Asset Value",
    "existing_loans":           "Number of Existing Loans",
    "existing_emi_monthly":     "Existing Monthly EMI Obligations",
    "credit_enquiries_6m":      "Credit Enquiries (Last 6 Months)",
    "credit_history_years":     "Length of Credit History",
    "num_credit_cards":         "Number of Credit Cards",
    "credit_utilization":       "Credit Utilization Rate",
    "avg_monthly_balance":      "Average Monthly Bank Balance",
    "balance_volatility":       "Bank Balance Stability",
    "monthly_inflow":           "Monthly Bank Inflow",
    "salary_credit_regular":    "Regular Salary Credits",
    "employment_type":          "Employment Type",
    "employment_tenure_years":  "Employment Tenure (Years)",
    "employment_tenure_days":   "Employment Tenure (Days)",
    "marital_status":           "Marital Status",
    "num_dependents":           "Number of Dependents",
    "num_children":             "Number of Children",
    "city_tier":                "City Tier",
    "education":                "Education Level",
}

ACTIONABLE_FEATURES = {
    "cibil_score",
    "dti_ratio",
    "net_monthly_surplus",
    "bounced_cheques_12m",
    "credit_utilization",
    "credit_enquiries_6m",
    "existing_loans",
    "existing_emi_monthly",
    "avg_monthly_balance",
    "salary_credit_regular",
}

APPLICANT_MESSAGES = {
    "cibil_score": {
        "negative": (
            "Your CIBIL score ({value}) is below the acceptable threshold. "
            "You can improve it by paying all EMIs on time, reducing credit card balances, "
            "and avoiding new loan applications for 6-12 months."
        ),
        "positive": "Your strong CIBIL score ({value}) positively influenced the decision.",
    },
    "dti_ratio": {
        "negative": (
            "Your Debt-to-Income ratio ({value:.2f}) is high, indicating that a large "
            "portion of your income is already committed to existing debts. "
            "Reducing or clearing existing loans before applying can improve your profile."
        ),
        "positive": "Your healthy Debt-to-Income ratio ({value:.2f}) supported your application.",
    },
    "net_monthly_surplus": {
        "negative": (
            "Your net monthly surplus after expenses is low (Rs. {value:,.0f}), "
            "raising concerns about your ability to service the new EMI. "
            "Increasing income or reducing expenses can help."
        ),
        "positive": "Your monthly surplus of Rs. {value:,.0f} demonstrates good repayment capacity.",
    },
    "days_past_due": {
        "negative": (
            "You have {value} days past due on existing obligations. "
            "Clearing all overdue amounts and maintaining a clean payment record "
            "for 6+ months will significantly improve your chances."
        ),
        "positive": "Your clean repayment history (0 days past due) strongly supported your application.",
    },
    "bounced_cheques_12m": {
        "negative": (
            "{value} bounced cheque(s) in the last 12 months negatively impacted your application. "
            "Maintaining adequate account balances to avoid future bounces is essential."
        ),
        "positive": "No bounced cheques in the last 12 months — this helped your application.",
    },
    "monthly_income": {
        "negative": (
            "Your monthly income (Rs. {value:,.0f}) relative to the requested loan amount "
            "raises repayment concerns. Consider applying for a smaller loan amount."
        ),
        "positive": "Your income level of Rs. {value:,.0f} is adequate for the requested loan.",
    },
    "credit_utilization": {
        "negative": (
            "Your credit utilization rate ({value:.0%}) is high. "
            "Aim to keep it below 30% by paying down credit card balances."
        ),
        "positive": "Your low credit utilization ({value:.0%}) reflects responsible credit usage.",
    },
    "existing_loans": {
        "negative": (
            "You currently have {value} active loan(s), which increases your overall debt burden. "
            "Closing at least one loan before applying may improve your profile."
        ),
        "positive": "Your existing loan count ({value}) is within acceptable limits.",
    },
    "avg_monthly_balance": {
        "negative": (
            "Your average monthly bank balance (Rs. {value:,.0f}) is low relative to the "
            "proposed EMI. Maintaining a higher balance demonstrates financial stability."
        ),
        "positive": "Your strong average bank balance (Rs. {value:,.0f}) supported your application.",
    },
    "credit_enquiries_6m": {
        "negative": (
            "{value} credit enquiry(ies) in the last 6 months indicates active credit-seeking "
            "behaviour, which can reduce your score. Avoid new applications for a few months."
        ),
        "positive": "Low recent credit enquiries ({value}) is a positive indicator.",
    },
    "loan_to_income": {
        "negative": (
            "The requested loan amount is {value:.1f}x your annual income, which is above "
            "comfortable limits. Consider a smaller loan or providing a co-applicant."
        ),
        "positive": "The loan-to-income ratio ({value:.1f}x) is within acceptable range.",
    },
    "loan_to_value": {
        "negative": (
            "The loan-to-value ratio ({value:.0%}) is high, meaning the loan covers most of "
            "the asset value with little margin. A larger down payment would help."
        ),
        "positive": "The loan-to-value ratio ({value:.0%}) is comfortable.",
    },
    "salary_credit_regular": {
        "negative": (
            "Irregular salary credits in your bank account raise concerns about income stability. "
            "Ensure your salary is credited directly to your bank account consistently."
        ),
        "positive": "Regular salary credits in your account demonstrate income stability.",
    },
}

DEFAULT_MESSAGE = {
    "negative": "The factor '{label}' negatively influenced the decision.",
    "positive": "The factor '{label}' positively influenced the decision.",
}


class SHAPExplainer:
    """Post-hoc SHAP explainer for the LightGBM credit model."""

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
        model_dir = os.path.abspath(model_dir)

        self.model    = joblib.load(os.path.join(model_dir, "lgbm_credit.pkl"))
        self.features = json.load(open(os.path.join(model_dir, "feature_list.json")))
        self.meta     = json.load(open(os.path.join(model_dir, "model_meta.json")))
        self.explainer = shap.TreeExplainer(self.model)
        print(f"[SHAPExplainer] Loaded model v{self.meta['version']} | threshold={self.meta['optimal_threshold']:.4f}")

    # ── Core SHAP computation ─────────────────────────────────────────────────
    def _compute_shap(self, input_dict: dict):
        X = pd.DataFrame([input_dict])[self.features]
        sv = self.explainer.shap_values(X)
        # LightGBM binary: sv is ndarray shape (1, n_features)
        if isinstance(sv, list):
            vals = sv[1][0]      # positive class
            base = self.explainer.expected_value[1]
        else:
            vals = sv[0]
            base = float(self.explainer.expected_value)
        return vals, float(base), X

    # ── Full explanation object ───────────────────────────────────────────────
    def explain(self, input_dict: dict, top_n: int = 10) -> dict:
        vals, base_value, X = self._compute_shap(input_dict)

        attributions = sorted(
            zip(self.features, vals, X.iloc[0].tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        top_factors = []
        for feat, shap_val, raw_val in attributions[:top_n]:
            direction = "positive" if shap_val > 0 else "negative"
            top_factors.append({
                "feature":      feat,
                "label":        FEATURE_LABELS.get(feat, feat),
                "shap_value":   round(float(shap_val), 6),
                "raw_value":    raw_val,
                "direction":    direction,
                "actionable":   feat in ACTIONABLE_FEATURES,
            })

        return {
            "base_value":  round(base_value, 6),
            "shap_sum":    round(float(np.sum(vals)), 6),
            "top_factors": top_factors,
            "all_shap":    {f: round(float(v), 6) for f, v in zip(self.features, vals)},
        }

    # ── Plain-language message for applicant ─────────────────────────────────
    def generate_applicant_message(self, approved: bool, shap_output: dict, top_n: int = 3) -> dict:
        # Filter to negative factors for rejection / positive for approval
        target_dir = "positive" if approved else "negative"
        relevant = [f for f in shap_output["top_factors"] if f["direction"] == target_dir][:top_n]

        reasons = []
        for factor in relevant:
            feat = factor["feature"]
            raw  = factor["raw_value"]
            tmpl = APPLICANT_MESSAGES.get(feat, DEFAULT_MESSAGE).get(target_dir, "")
            try:
                reasons.append(tmpl.format(value=raw, label=factor["label"]))
            except Exception:
                reasons.append(DEFAULT_MESSAGE[target_dir].format(label=factor["label"]))

        actionable_tips = [
            f for f in shap_output["top_factors"]
            if f["direction"] == "negative" and f["actionable"]
        ][:3]

        return {
            "status": "You May Be Eligible" if approved else "You May Get Rejected",
            "summary": (
                "Loan result: You May Be Eligible based on your strong financial profile."
                if approved else
                "Loan result: You May Get Rejected based on the following factors:"
            ),
            "primary_reasons": reasons,
            "actionable_tips": [
                f["label"] + " — improve this to strengthen future applications"
                for f in actionable_tips
            ] if not approved else [],
        }
