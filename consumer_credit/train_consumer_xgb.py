import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, classification_report
from xgboost import XGBClassifier

from consumer_xgb_config import DATA_PATH, MODEL_DIR, MODEL_PATH, META_PATH, FEATURES, TARGET


def train_xgb(data_path=DATA_PATH):
    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred = (proba >= best_threshold).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "model_type": "XGBoost",
        "version": "1.0.0",
        "features": FEATURES,
        "optimal_threshold": best_threshold,
        "train_rows": int(len(df)),
        "pos_class_count": int((y == 1).sum()),
        "neg_class_count": int((y == 0).sum()),
    }
    json.dump(meta, open(META_PATH, "w"), indent=2)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {META_PATH}")


if __name__ == "__main__":
    train_xgb()
