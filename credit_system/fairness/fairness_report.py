"""
Fairness Reporting Module
===========================
Computes demographic fairness metrics using Fairlearn:
  - Demographic Parity Difference
  - Equalized Odds Difference
  - 4/5ths Rule (EEOC guideline)

Protected attributes tested: gender, city_tier, age_group
(These are NOT model features — used only for fairness auditing)
"""
import json, os, joblib
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score


def _age_group(age: int) -> str:
    if age < 30:   return "21-29"
    if age < 40:   return "30-39"
    if age < 50:   return "40-49"
    return "50+"


def _income_band(income: float) -> str:
    if income < 30000:   return "< 30k"
    if income < 60000:   return "30k - 60k"
    if income < 100000:  return "60k - 100k"
    return "> 100k"


def compute_fairness_report(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Compute fairness metrics across protected attributes.

    Parameters
    ----------
    df     : Full dataframe including protected attributes
    y_true : Ground-truth labels
    y_pred : Model binary predictions (0/1)
    y_prob : Model predicted probabilities

    Returns dict of fairness report per attribute.
    """

    # ── Prepare sensitive attribute series ──────────────────────────────────
    sensitive = {
        "gender":      df["gender"].map({0: "Female", 1: "Male"}),
        "city_tier":   df["city_tier"].map({1: "Metro (T1)", 2: "Tier-2", 3: "Tier-3/Rural"}),
        "age_group":   df["age"].apply(_age_group),
        "income_band": df["income"].apply(_income_band),
    }

    report = {}

    for attr_name, attr_values in sensitive.items():
        # Core metrics per group
        mf = MetricFrame(
            metrics={
                "accuracy":      accuracy_score,
                "precision":     lambda yt, yp: precision_score(yt, yp, zero_division=0),
                "recall":        lambda yt, yp: recall_score(yt, yp, zero_division=0),
                "selection_rate": selection_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=attr_values,
        )

        # Disparity metrics
        dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=attr_values)
        eod = equalized_odds_difference(y_true, y_pred, sensitive_features=attr_values)

        # 4/5ths Rule (EEOC)
        sel_rates = mf.by_group["selection_rate"]
        max_rate  = sel_rates.max()
        min_rate  = sel_rates.min()
        four_fifths_ratio = float(min_rate / max_rate) if max_rate > 0 else 1.0
        four_fifths_pass  = four_fifths_ratio >= 0.8

        # Identify disadvantaged group
        disadvantaged_group = str(sel_rates.idxmin())

        report[attr_name] = {
            "demographic_parity_difference": round(float(dpd), 4),
            "equalized_odds_difference":     round(float(eod), 4),
            "four_fifths_ratio":             round(four_fifths_ratio, 4),
            "four_fifths_rule_pass":         four_fifths_pass,
            "disadvantaged_group":           disadvantaged_group,
            "selection_rate_by_group":       {k: round(float(v), 4) for k, v in sel_rates.items()},
            "accuracy_by_group":             {k: round(float(v), 4) for k, v in mf.by_group["accuracy"].items()},
            "recall_by_group":               {k: round(float(v), 4) for k, v in mf.by_group["recall"].items()},
            "overall_selection_rate":        round(float(y_pred.mean()), 4),
            "compliance_flag": (
                "PASS" if (abs(dpd) < 0.1 and four_fifths_pass)
                else "REVIEW_NEEDED"
            ),
        }

    return report


def run_fairness_on_dataset(data_path: str, model_dir: str) -> dict:
    """
    Convenience function: loads the dataset + model and runs the full fairness report.
    """
    import json as _json

    model    = joblib.load(os.path.join(model_dir, "lgbm_credit.pkl"))
    features = _json.load(open(os.path.join(model_dir, "feature_list.json")))
    meta     = _json.load(open(os.path.join(model_dir, "model_meta.json")))
    threshold = meta["optimal_threshold"]

    df     = pd.read_csv(data_path)
    X      = df[features]
    y_true = df["loan_approved"]
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    report = compute_fairness_report(df, y_true, y_pred, y_prob)

    print("\n=== FAIRNESS REPORT ===")
    for attr, metrics in report.items():
        print(f"\n[{attr.upper()}]")
        print(f"  Demographic Parity Diff : {metrics['demographic_parity_difference']}")
        print(f"  Equalized Odds Diff     : {metrics['equalized_odds_difference']}")
        print(f"  4/5ths Ratio            : {metrics['four_fifths_ratio']} ({'PASS' if metrics['four_fifths_rule_pass'] else 'FAIL'})")
        print(f"  Disadvantaged Group     : {metrics['disadvantaged_group']}")
        print(f"  Selection Rates         : {metrics['selection_rate_by_group']}")
        print(f"  Compliance Flag         : {metrics['compliance_flag']}")

    return report


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..")
    run_fairness_on_dataset(
        data_path=os.path.join(base, "..", "synthetic_credit_dataset.csv"),
        model_dir=os.path.join(base, "model"),
    )
