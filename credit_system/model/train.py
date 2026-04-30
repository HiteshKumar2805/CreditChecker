"""
Credit Decisioning Model — Trainer
====================================
Trains a LightGBM classifier on the synthetic credit dataset.
Handles severe class imbalance via scale_pos_weight.
Saves model, feature list, and threshold to disk.
"""
import json, joblib, os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve
)

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "synthetic_credit_dataset.csv")
MODEL_DIR   = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(MODEL_DIR, "lgbm_credit.pkl")
FEAT_PATH   = os.path.join(MODEL_DIR, "feature_list.json")
META_PATH   = os.path.join(MODEL_DIR, "model_meta.json")

# Features used for prediction (excludes protected attributes gender/age from MODEL)
# Gender and age kept only for fairness TESTING, not for model decisions
FEATURES = [
    "cibil_score",
    "dti_ratio",
    "net_monthly_surplus",
    "days_past_due",
    "bounced_cheques_12m",
    "monthly_income",
    "co_applicant_income",
    "loan_amount",
    "loan_tenure_months",
    "proposed_emi",
    "loan_to_income",
    "loan_to_value",
    "loan_purpose",
    "property_value",
    "existing_loans",
    "existing_emi_monthly",
    "credit_enquiries_6m",
    "credit_history_years",
    "num_credit_cards",
    "credit_utilization",
    "avg_monthly_balance",
    "balance_volatility",
    "monthly_inflow",
    "salary_credit_regular",
    "employment_type",
    "employment_tenure_years",
    "employment_tenure_days",
    "marital_status",
    "num_dependents",
    "num_children",
    "city_tier",
    "education",
]
TARGET = "loan_approved"

# ─── Load Data ────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Class distribution:\n{df[TARGET].value_counts()}")

X = df[FEATURES]
y = df[TARGET]

# ─── Class Imbalance: scale_pos_weight ────────────────────────────────────────
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"\n  scale_pos_weight = {scale_pos_weight:.2f}  (handles {pos_count} positives vs {neg_count} negatives)")

# ─── Model ────────────────────────────────────────────────────────────────────
model = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)

# ─── Cross-Validation ─────────────────────────────────────────────────────────
print("\nRunning 5-fold stratified cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
cv_auc = roc_auc_score(y, cv_proba)
print(f"  CV AUC: {cv_auc:.4f}")

# ─── Optimal Threshold (max F1 on CV predictions) ─────────────────────────────
precision, recall, thresholds = precision_recall_curve(y, cv_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])
print(f"  Optimal threshold (max F1): {best_threshold:.4f}")

# ─── Final Training on Full Data ──────────────────────────────────────────────
print("\nFitting final model on full dataset...")
model.fit(X, y)

# ─── Evaluation on Full Data ─────────────────────────────────────────────────
y_proba = model.predict_proba(X)[:, 1]
y_pred  = (y_proba >= best_threshold).astype(int)
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=["Rejected", "Approved"]))
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

# ─── Feature Importance ───────────────────────────────────────────────────────
importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop 10 Features:")
print(importance.head(10).to_string())

# ─── Save Artifacts ───────────────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
json.dump(FEATURES, open(FEAT_PATH, "w"), indent=2)
meta = {
    "model_type": "LightGBM",
    "n_estimators": 800,
    "cv_auc": round(cv_auc, 4),
    "optimal_threshold": best_threshold,
    "scale_pos_weight": round(scale_pos_weight, 2),
    "n_features": len(FEATURES),
    "train_rows": len(df),
    "pos_class_count": int(pos_count),
    "neg_class_count": int(neg_count),
    "version": "1.0.0",
}
json.dump(meta, open(META_PATH, "w"), indent=2)

print(f"\n✅ Model saved      → {MODEL_PATH}")
print(f"✅ Feature list     → {FEAT_PATH}")
print(f"✅ Model metadata   → {META_PATH}")
