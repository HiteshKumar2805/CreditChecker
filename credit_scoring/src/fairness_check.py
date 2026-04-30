import pandas as pd

def fairness_report(df):

    groups = pd.cut(df["cibil_score"], bins=[300,500,650,750,900],
                    labels=["low","mid","high","premium"])

    df["group"] = groups

    report = df.groupby("group")["good_credit_behavior"].mean()

    print("\nApproval Rate by Group:")
    print(report)

    return report
    