# CreditChecker: Complete System Architecture

CreditChecker is an enterprise-grade, regulatory-compliant AI credit decisioning platform. It bridges the gap between high-accuracy machine learning (XGBoost/LightGBM) and strict banking regulations by implementing an extensive **Explainable AI (XAI)**, **Tamper-Evident Auditing**, and **Zero-Knowledge Encryption** pipeline.

---

## 1. High-Level Architecture Diagram

```mermaid
graph TD
    %% Define Styles
    classDef frontend fill:#1e293b,stroke:#64748b,color:#e2e8f0;
    classDef api fill:#0f172a,stroke:#3b82f6,color:#bfdbfe,stroke-width:2px;
    classDef ml fill:#4c1d95,stroke:#a78bfa,color:#ede9fe;
    classDef sec fill:#064e3b,stroke:#10b981,color:#d1fae5;
    classDef ext fill:#78350f,stroke:#f59e0b,color:#fef3c7;

    %% Components
    subgraph Frontend [React SPA UI]
        UI1(KYC Portal)
        UI2(Loan Portals: Personal, Consumer, Vehicle)
        UI3(Explainability Dashboard)
        UI4(Compliance Officer Portal)
    end

    subgraph BackendAPI [FastAPI Python Backend]
        API1(KYC / OCR Engine)
        API2(Loan Evaluation Engine)
        API3(Fairness & Drift Monitor)
    end

    subgraph ML_XAI [AI & Explainability Engine]
        ML1(XGBoost / LightGBM)
        ML2(SHAP TreeExplainer)
        ML3(Counterfactual Generator)
    end

    subgraph Security_Audit [Security & Compliance]
        SEC1(AES-256 Fernet Vault)
        SEC2(SHA-256 Hash Chain Logger)
        SEC3(Fairlearn / PSI Evaluator)
    end

    subgraph Database [Storage]
        DB1[(Supabase PostgreSQL)]
        DB2[(Local Audit JSON)]
    end

    %% Data Flow
    UI1 -- "Aadhaar/PAN PDFs" --> API1
    API1 -- "Extract via EasyOCR" --> API1
    API1 -- "Raw PII" --> SEC1
    SEC1 -- "Encrypted Ciphertext" --> DB1
    
    UI2 -- "Loan Data + Statements" --> API2
    API2 -- "Feature Vectors" --> ML1
    ML1 -- "Probability Score" --> API2
    API2 -- "Feature Vectors" --> ML2
    ML2 -- "SHAP Values" --> ML3
    ML3 -- "Plain English Tips" --> API2

    API2 -- "Logs Prediction + SHAP" --> SEC2
    SEC2 -- "Tamper-Evident Block" --> DB2

    UI3 <-- "XAI Visualizations & Results" --> API2
    
    UI4 -- "Generate Reports" --> API3
    API3 -- "Check Demographics" --> SEC3
    SEC3 -- "Compliance HTML/PDF" --> UI4

    class UI1,UI2,UI3,UI4 frontend;
    class API1,API2,API3 api;
    class ML1,ML2,ML3 ml;
    class SEC1,SEC2,SEC3 sec;
    class DB1,DB2 ext;
```

---

## 2. Component Breakdown

### A. Frontend Layer (React + Vite)
The user interface is completely decoupled from the machine learning logic, focusing on state management and premium visualizations.
*   **KYC Portal:** Handles drag-and-drop ingestion of identity documents. It does *not* store PII in browser cookies or local storage.
*   **Specialized Loan Portals:** Separate workflows for Personal, Consumer, and Vehicle loans. They dynamically switch between "Lite Mode" (clean, text-based explanations) and "Pro Mode" (deep SHAP waterfall charts).
*   **Explainability Panel:** Renders SHAP data into dynamic SVG gauges, diverging bar charts, and human-readable counterfactuals (e.g., *"Reduce existing EMI to increase approval odds"*).
*   **Officer Dashboard:** A secure internal view for compliance officers to view recent decisions, verify hash-chain integrity, and export regulatory reports.

### B. API Layer (FastAPI)
Acts as the central nervous system, handling all HTTP requests, file uploads, and routing to the ML engines.
*   **OCR Pipelines (`kyc_extractor.py`, `payslip_extractor.py`):** Uses **EasyOCR** and **PyMuPDF** to automatically extract text from PDFs and images using regex and spatial bounding boxes.
*   **Evaluation Endpoints:** `/evaluate_personal`, `/consumer/evaluate`, etc., handle feature engineering (converting raw user inputs into structured pandas DataFrames).

### C. Machine Learning & Explainability (XAI)
The core predictive engine built to outperform traditional scorecards while remaining fully transparent.
*   **Predictive Models:** Utilizes heavily optimized **XGBoost** and LightGBM models trained on diverse datasets.
*   **SHAP Engine (`shap_explainer.py`):** Wraps the models in a `shap.TreeExplainer`. Instead of a black-box "Computer says no," it calculates exactly how much each feature (e.g., CIBIL score, Company Tier) pushed the probability up or down from the baseline expected value.
*   **Counterfactual Engine:** Translates raw SHAP outputs into actionable steps for rejected applicants, meeting modern "Right to Explanation" consumer protection laws.

### D. Zero-Knowledge Security Vault
The system guarantees that the cloud provider (Supabase) cannot read customer data.
*   **`secure_vault.py`:** Before any PII (Aadhaar/PAN data) leaves the server, it is encrypted using **Symmetric AES-256 (Fernet)**.
*   **Key Isolation:** The encryption key (`.vault_key`) is stored strictly on the local backend file system. The Supabase database only receives raw ciphertext.

### E. Regulatory Compliance & Tamper-Evident Auditing
Built specifically to pass RBI (India) and GDPR regulatory audits.
*   **Hash Chain Auditing (`audit_logger.py`):** Every AI decision is saved as a "block". Each block contains the cryptographic SHA-256 hash of the *previous* block. If anyone alters a past decision in the database, the entire chain breaks, proving tampering occurred.
*   **Fairness Testing (`fairness_report.py`):** Uses Microsoft's `fairlearn` library to evaluate the model against protected classes (e.g., Gender, Income Bands) to ensure **Demographic Parity** and **Equalized Odds**.
*   **Drift Detection (`drift_calculator.py`):** Calculates the **Population Stability Index (PSI)**. If the demographic of applicants today shifts drastically from the data the model was trained on, the system flags it to prevent degrading accuracy.

---

## 3. Data Flow: The Lifecycle of a Loan Application

1.  **Ingestion:** User uploads Aadhaar/PAN. The Frontend sends the files to the `/kyc/extract` API.
2.  **Extraction & Encryption:** FastAPI runs OCR, extracts text, encrypts the PII via the Vault, saves the ciphertext to Supabase, and returns a secure `session_id` to the frontend.
3.  **Application:** User fills out loan details (e.g., CIBIL, Salary) and uploads bank statements. Frontend sends this to `/consumer/evaluate`.
4.  **Inference:** FastAPI engineers the features and passes the array to XGBoost. XGBoost returns a probability (e.g., `74.3%`).
5.  **Explainability:** The same feature array is passed to the SHAP explainer, which breaks down the `74.3%` into constituent parts (+15% due to high salary, -2% due to dependents).
6.  **Auditing:** The input features, the probability, and the SHAP factors are hashed, linked to the previous transaction, and stored in the tamper-evident audit log.
7.  **Response:** The frontend receives the probability, the SHAP factors, and the generated plain-English tips, rendering the Explainability Panel.
