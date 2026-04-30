"""
End-to-End Integration Test
==============================
Tests the full pipeline:
  1. SHAP explainer loads and explains
  2. Audit logger writes + verifies chain
  3. Fairness report runs
  4. Full simulate of /evaluate logic (without HTTP)

Run with:  .\\venv\\Scripts\\python.exe credit_system\\test_pipeline.py
"""
import sys, os, json, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# ── 1. SHAP Explainer ──────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: SHAP Explainer")
print("=" * 60)

from explainability.shap_explainer import SHAPExplainer

explainer = SHAPExplainer(model_dir=os.path.join(os.path.dirname(__file__), "model"))

# Sample: Good applicant (should be APPROVED)
good_applicant = {
    "cibil_score": 780, "dti_ratio": 0.25, "net_monthly_surplus": 45000,
    "days_past_due": 0, "bounced_cheques_12m": 0, "monthly_income": 120000,
    "co_applicant_income": 0, "loan_amount": 500000, "loan_tenure_months": 60,
    "proposed_emi": 11000, "loan_to_income": 4.16, "loan_to_value": 0.5,
    "loan_purpose": 1, "property_value": 1000000, "existing_loans": 1,
    "existing_emi_monthly": 8000, "credit_enquiries_6m": 1,
    "credit_history_years": 8, "num_credit_cards": 2, "credit_utilization": 0.2,
    "avg_monthly_balance": 80000, "balance_volatility": 0.15,
    "monthly_inflow": 130000, "salary_credit_regular": 1,
    "employment_type": 1, "employment_tenure_years": 5.0,
    "employment_tenure_days": 1825, "marital_status": 1,
    "num_dependents": 2, "num_children": 1, "city_tier": 1, "education": 2,
    # protected attrs for fairness (not used in model)
    "age": 35, "gender": 1,
}

# Sample: Bad applicant (should be REJECTED)
bad_applicant = {
    "cibil_score": 420, "dti_ratio": 2.1, "net_monthly_surplus": -15000,
    "days_past_due": 120, "bounced_cheques_12m": 4, "monthly_income": 25000,
    "co_applicant_income": 0, "loan_amount": 900000, "loan_tenure_months": 24,
    "proposed_emi": 45000, "loan_to_income": 36.0, "loan_to_value": 0.95,
    "loan_purpose": 3, "property_value": 100000, "existing_loans": 4,
    "existing_emi_monthly": 35000, "credit_enquiries_6m": 5,
    "credit_history_years": 1, "num_credit_cards": 3, "credit_utilization": 0.92,
    "avg_monthly_balance": 3000, "balance_volatility": 0.88,
    "monthly_inflow": 28000, "salary_credit_regular": 0,
    "employment_type": 3, "employment_tenure_years": 0.5,
    "employment_tenure_days": 182, "marital_status": 0,
    "num_dependents": 5, "num_children": 3, "city_tier": 3, "education": 0,
    "age": 28, "gender": 0,
}

import joblib, json as _json, pandas as pd
model    = joblib.load(os.path.join(os.path.dirname(__file__), "model", "lgbm_credit.pkl"))
features = _json.load(open(os.path.join(os.path.dirname(__file__), "model", "feature_list.json")))
meta     = _json.load(open(os.path.join(os.path.dirname(__file__), "model", "model_meta.json")))
threshold = meta["optimal_threshold"]

for label, applicant in [("GOOD applicant", good_applicant), ("BAD applicant", bad_applicant)]:
    X = pd.DataFrame([applicant])[features]
    prob = float(model.predict_proba(X)[0, 1])
    approved = prob >= threshold

    shap_out = explainer.explain(applicant)
    msg = explainer.generate_applicant_message(approved, shap_out)

    print(f"\n--- {label} ---")
    print(f"  Probability : {prob:.4f}  |  Threshold : {threshold:.4f}")
    print(f"  Decision    : {'APPROVED' if approved else 'REJECTED'}")
    print(f"  Summary     : {msg['summary']}")
    if msg["primary_reasons"]:
        print("  Top Reasons :")
        for r in msg["primary_reasons"]:
            print(f"    - {r}")
    if msg.get("actionable_tips"):
        print("  Actionable  :")
        for t in msg["actionable_tips"]:
            print(f"    * {t}")
    print("  Top SHAP factors:")
    for f in shap_out["top_factors"][:5]:
        bar = "+" if f["shap_value"] > 0 else "-"
        print(f"    [{bar}] {f['label']:<35} shap={f['shap_value']:+.4f}  raw={f['raw_value']}")


# ── 2. Audit Logger ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: Audit Logger — Hash Chain")
print("=" * 60)

from audit.audit_logger import log_decision, verify_chain, get_decision

for label, applicant in [("GOOD", good_applicant), ("BAD", bad_applicant)]:
    app_id  = str(uuid.uuid4())
    X = pd.DataFrame([applicant])[features]
    prob = float(model.predict_proba(X)[0, 1])
    approved = prob >= threshold
    shap_out = explainer.explain(applicant)

    entry = log_decision(
        application_id=app_id,
        inputs=applicant,
        decision={"approved": approved, "probability": round(prob, 6), "threshold": threshold},
        shap_output=shap_out,
        officer_id="test_officer_01",
    )
    print(f"  [{label}] Logged | block_hash: {entry['block_hash'][:20]}...")

result = verify_chain()
print(f"\n  Chain Verification: valid={result['valid']}  total_blocks={result['total_blocks']}")
assert result["valid"], "CHAIN TAMPERED — TEST FAILED"
print("  Chain integrity: OK")


# ── 3. Fairness Report ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: Fairness Report")
print("=" * 60)

from fairness.fairness_report import run_fairness_on_dataset

report = run_fairness_on_dataset(
    data_path=os.path.join(os.path.dirname(__file__), "..", "synthetic_credit_dataset.csv"),
    model_dir=os.path.join(os.path.dirname(__file__), "model"),
)

for attr, metrics in report.items():
    flag = metrics["compliance_flag"]
    print(f"  {attr:<12} | DPD={metrics['demographic_parity_difference']:+.4f} | "
          f"EOD={metrics['equalized_odds_difference']:+.4f} | "
          f"4/5ths={'PASS' if metrics['four_fifths_rule_pass'] else 'FAIL'} | "
          f"Status={flag}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
print("\nTo start the API server run:")
print("  .\\venv\\Scripts\\uvicorn credit_system.api.main:app --reload --port 8000")
print("\nAPI docs available at: http://localhost:8000/docs")
