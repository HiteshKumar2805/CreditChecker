"""Quick smoke test against the live API at http://localhost:8080"""
import urllib.request, json, pprint

BASE = "http://localhost:8080"

def post(path, body):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(f"{BASE}{path}", data=data,
                                   headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def get(path):
    with urllib.request.urlopen(f"{BASE}{path}") as r:
        return json.loads(r.read())

# 1. Health
print("=== /health ===")
pprint.pprint(get("/health"))

# 2. Evaluate – APPROVED applicant
print("\n=== /evaluate (GOOD) ===")
resp = post("/evaluate", {
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
    "age": 35, "gender": 1,
})
print(f"  approved        : {resp['approved']}")
print(f"  probability     : {resp['probability']}")
print(f"  confidence      : {resp['confidence_level']}")
print(f"  block_hash      : {resp['block_hash'][:24]}...")
print(f"  message summary : {resp['applicant_message']['summary'][:60]}...")

# 3. Evaluate – REJECTED applicant
print("\n=== /evaluate (BAD) ===")
resp2 = post("/evaluate", {
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
})
print(f"  approved        : {resp2['approved']}")
print(f"  probability     : {resp2['probability']}")
print(f"  reasons         : {resp2['applicant_message']['primary_reasons']}")

# 4. Audit verify
print("\n=== /audit/verify ===")
pprint.pprint(get("/audit/verify"))

print("\nAll smoke tests passed.")
