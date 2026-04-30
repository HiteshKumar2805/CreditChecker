# CreditChecker

CreditChecker is an explainable AI credit decisioning and compliance platform for digital lending. It helps lenders evaluate loan applications with machine learning while making every decision explainable, fair, auditable, and regulator-ready.

High-level concept:

> CreditChecker is Sentry for AI lending decisions.

It does not replace credit managers. It works as a credit manager co-pilot that automates document extraction, credit scoring, explainability, fairness checks, and tamper-evident audit logging.

## Problem

AI credit models can process hundreds of borrower, income, transaction, and credit-history signals, but their reasoning is often opaque. When an applicant is rejected, the lender may know that the model prediction is statistically valid, but the applicant, compliance officer, and regulator still need to understand why.

CreditChecker solves this by adding an explainability and audit layer around AI-based lending decisions.

## Unique Value Proposition

CreditChecker turns black-box loan decisions into:

- Plain-language applicant explanations
- SHAP-based feature attribution
- Compliance officer audit trails
- Fairness reports across protected groups
- Model drift indicators
- Tamper-evident decision logs
- Exportable compliance reports

In short:

> We make AI lending decisions explainable and regulator-ready.

## Key Features

- **KYC OCR and verification**
  - Extracts Aadhaar and PAN details from PDFs or images.
  - Cross-verifies name and date of birth.

- **Multiple loan products**
  - Personal loan evaluation
  - Consumer loan evaluation
  - Vehicle loan evaluation

- **Document intelligence**
  - Payslip salary extraction
  - Bank statement upload support
  - Credit and affordability feature generation

- **Real ML models**
  - LightGBM credit model
  - XGBoost personal loan model
  - Bank-statement and consumer-loan scoring models

- **Explainability**
  - SHAP-based model explanations
  - Feature impact visualization
  - Counterfactual suggestions
  - Applicant-friendly reasons and improvement tips

- **Compliance officer dashboard**
  - Recent decisions
  - Search by application ID
  - Probability, confidence, threshold, inputs, and SHAP factors
  - Audit hash chain verification

- **Fairness and regulator reporting**
  - Demographic parity difference
  - Equalized odds difference
  - 4/5ths rule
  - Protected attributes including gender, age group, city tier, and income band
  - Compliance report export

- **Tamper-evident audit trail**
  - SHA-256 hash chaining
  - Detects post-hoc modification of decisions, timestamps, inputs, or SHAP outputs

## Tech Stack

**Backend**

- Python
- FastAPI
- LightGBM
- XGBoost
- SHAP
- Fairlearn
- Pandas / NumPy
- EasyOCR
- PyMuPDF

**Frontend**

- React
- Vite
- CSS modules/stylesheets

**Audit and Compliance**

- JSONL hash-chain audit log
- Fairness metrics
- PSI-based drift indicator
- HTML compliance report export

## Project Structure

```text
credit_system/
  api/
    main.py                 # FastAPI application and API routes
    kyc_extractor.py         # Aadhaar/PAN OCR extraction
    payslip_extractor.py     # Payslip salary extraction
  audit/
    audit_logger.py          # Tamper-evident hash-chain audit log
    audit_chain.jsonl        # Decision audit ledger
  explainability/
    shap_explainer.py        # SHAP explanation layer
  fairness/
    fairness_report.py       # Fairness metrics
    drift_calculator.py      # PSI drift monitoring
    compliance_exporter.py   # HTML compliance report export
  frontend/
    src/components/          # React UI components
  model/
    lgbm_credit.pkl
    xgb_personal_loan.pkl
    feature_list.json
    model_meta.json
  requirements.txt
  smoke_test.py
  test_pipeline.py
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Backend health check |
| `POST` | `/kyc/extract` | Extract Aadhaar and PAN data |
| `POST` | `/payslip/extract` | Extract salary and company from payslip |
| `POST` | `/evaluate_personal` | Evaluate personal loan |
| `POST` | `/consumer/evaluate` | Evaluate consumer loan |
| `POST` | `/vehicle/evaluate` | Evaluate vehicle loan |
| `POST` | `/evaluate` | Evaluate full credit application |
| `GET` | `/audit/recent` | Recent audit decisions |
| `GET` | `/audit/{application_id}` | Fetch one audit record |
| `GET` | `/audit/verify` | Verify audit hash chain |
| `GET` | `/fairness/report` | Fairness and drift report |
| `GET` | `/compliance/report` | Export HTML compliance report |

## How To Run

Run these commands from `D:\savee`.

### 1. Start Backend

```powershell
D:\savee\venv\Scripts\python.exe -m uvicorn credit_system.api.main:app --host 127.0.0.1 --port 8080
```

Backend URL:

```text
http://127.0.0.1:8080
```

API docs:

```text
http://127.0.0.1:8080/docs
```

### 2. Start Frontend

```powershell
cd D:\savee\credit_system\frontend
npm run dev -- --host 127.0.0.1
```

Frontend URL:

```text
http://127.0.0.1:5173
```

## Verification

Python syntax check:

```powershell
D:\savee\venv\Scripts\python.exe -B -m py_compile D:\savee\credit_system\api\main.py
```

Frontend build:

```powershell
cd D:\savee\credit_system\frontend
npm run build
```

Integration test:

```powershell
D:\savee\venv\Scripts\python.exe D:\savee\credit_system\test_pipeline.py
```

## Business Model

CreditChecker can be sold as a B2B SaaS and API platform for banks, NBFCs, fintech lenders, and loan marketplaces.

Potential revenue streams:

- Monthly SaaS subscription for lending teams
- Per-decision API pricing
- White-label borrower portal
- KYC, payslip, and bank statement extraction add-ons
- Compliance report exports
- Premium audit vault / decision ledger

Positioning:

> CreditChecker is the trust layer for AI-powered lending.

## Real-World Impact

CreditChecker helps:

- Borrowers understand why they may or may not be eligible
- Credit managers review decisions faster
- Lenders reduce compliance and reputational risk
- Auditors inspect every decision with SHAP and audit logs
- Regulators monitor fairness, drift, and disparate impact

The goal is not to remove human judgment from lending. The goal is to make AI-assisted credit decisions transparent, contestable, and accountable.

## Current Completion Status

Completed:

- Real ML models for credit decisioning
- SHAP-based explainability
- Applicant-facing explanations
- Officer dashboard
- Tamper-evident audit chain
- Fairness metrics
- Compliance report endpoint
- KYC and document extraction flows

Remaining production improvements:

- Dedicated regulator frontend dashboard
- Stronger production drift baseline
- PDF/CSV compliance export bundle
- More robust statistical anomaly detection
- Full model-agnostic SHAP/LIME support for every model type

## Disclaimer

This project is intended as a prototype/demo of explainable AI lending infrastructure. It should not be used as the sole basis for real-world credit decisions without legal, regulatory, security, and model-risk validation.
