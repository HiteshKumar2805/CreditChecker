import numpy as np

def rule_based_score(row):

    score = 0

    # CIBIL
    if row["cibil_score"] > 750:
        score += 30
    elif row["cibil_score"] > 650:
        score += 20
    else:
        score += 5

    # Income stability
    if row["income_stability"] > 0.7:
        score += 20

    # Spending ratio
    if row["spending_ratio"] < 0.5:
        score += 20

    # Debt ratio
    if row["debt_to_income_ratio"] < 0.4:
        score += 30

    return np.clip(score, 0, 100)


def risk_penalty(row):

    penalty = 0
    penalty += row["overdrafts"] * 10
    penalty += row["bounced_payments"] * 15

    if row["debt_to_income_ratio"] > 0.6:
        penalty += 20

    return np.clip(penalty, 0, 100)


def final_confidence(ml_prob, rule_score, risk_score):

    return (
        ml_prob * 70 +
        rule_score * 0.2 +
        (100 - risk_score) * 0.1
    )