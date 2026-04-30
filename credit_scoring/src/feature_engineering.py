import re

def add_features(transactions_text):
    """
    Extract financial behavior signals from bank statement text
    """

    text = " ".join(transactions_text).lower()

    # =========================
    # 💰 SALARY DETECTION
    # =========================
    salary_keywords = ["salary", "net salary", "payroll", "credit salary", "monthly salary"]

    salary_flag = 0
    for kw in salary_keywords:
        if kw in text:
            salary_flag = 1
            break

    # =========================
    # 💸 TRANSACTION ANALYSIS
    # =========================
    debit_keywords = ["debit", "withdraw", "upi", "spent", "purchase"]
    credit_keywords = ["credit", "deposit", "salary", "refund"]

    expense_score = sum(text.count(k) for k in debit_keywords)
    income_score = sum(text.count(k) for k in credit_keywords)

    # =========================
    # ⚖️ CASHFLOW BALANCE
    # =========================
    balance_score = income_score - expense_score

    # =========================
    # 🚨 FRAUD SIGNAL DETECTION
    # =========================
    fraud_keywords = ["bet", "casino", "gambling", "crypto", "loan app", "quick loan"]

    fraud_risk = sum(1 for k in fraud_keywords if k in text)

    # =========================
    # 📊 TRANSACTION VOLATILITY
    # =========================
    transaction_volume = len(transactions_text)

    # =========================
    # 📦 RETURN FEATURES
    # =========================
    return {
        "salary_flag": salary_flag,
        "expense_score": expense_score,
        "income_score": income_score,
        "balance_score": balance_score,
        "fraud_risk": fraud_risk,
        "transaction_volume": transaction_volume
    }