import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -------------------------------
# 1. LOAD DATASET
# -------------------------------
def load_dataset(path):
    df = pd.read_csv(path)
    print("Dataset Loaded Successfully")
    print("Columns:", df.columns.tolist())
    return df


# -------------------------------
# 2. PREPROCESS DATA (FIXED)
# -------------------------------
def preprocess_data(df):
    df = df.copy()

    # -------------------------------
    # FIX COLUMN NAME
    # -------------------------------
    if "monthly_income" in df.columns:
        df.rename(columns={"monthly_income": "income"}, inplace=True)

    # -------------------------------
    # ADD MISSING FEATURES (SIMULATION)
    # -------------------------------
    if "total_credit" not in df.columns:
        df["total_credit"] = df["income"]

    if "total_debit" not in df.columns:
        df["total_debit"] = df["income"] * 0.6

    if "txn_count" not in df.columns:
        df["txn_count"] = 10

    if "min_balance" not in df.columns:
        df["min_balance"] = df["income"] * 0.2

    if "savings_ratio" not in df.columns:
        df["savings_ratio"] = (
            (df["total_credit"] - df["total_debit"]) / df["total_credit"]
        )

    # -------------------------------
    # CREATE TARGET IF MISSING
    # -------------------------------
    if "target" not in df.columns:
        print("Creating target column")

        df["target"] = (
            (df["cibil_score"] >= 700) &
            (df["income"] > df["total_debit"])
        ).astype(int)

    # -------------------------------
    # SELECT REQUIRED FEATURES
    # -------------------------------
    required_cols = [
        "total_credit", "total_debit", "txn_count",
        "min_balance", "savings_ratio",
        "income", "cibil_score", "dependents"
    ]

    df = df[required_cols + ["target"]]

    # -------------------------------
    # ENCODING (SAFE)
    # -------------------------------
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'string']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # -------------------------------
    # SPLIT
    # -------------------------------
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoders, X.columns


# -------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------
def extract_features(transactions, profile):
    total_credit = sum(t['amount'] for t in transactions if t['txn_type'] == 'credit')
    total_debit = sum(t['amount'] for t in transactions if t['txn_type'] == 'debit')
    txn_count = len(transactions)
    min_balance = min(t['balance'] for t in transactions)

    savings_ratio = (total_credit - total_debit) / total_credit if total_credit != 0 else 0

    return {
        "total_credit": total_credit,
        "total_debit": total_debit,
        "txn_count": txn_count,
        "min_balance": min_balance,
        "savings_ratio": savings_ratio,
        "income": profile["income"],
        "cibil_score": profile["cibil_score"],
        "dependents": profile["dependents"]
    }


# -------------------------------
# 4. RULE ENGINE
# -------------------------------
def rule_engine(input_data, features):
    if input_data["cibil_score"] < 600:
        return False, "CIBIL score below 600"

    if input_data["loan_amount"] > 0.5 * input_data["income"]:
        return False, "Loan amount exceeds 50% of income"

    if features["total_debit"] > features["total_credit"]:
        return False, "Expenses exceed income"

    return True, "Passed"


# -------------------------------
# 5. TRAIN MODEL
# -------------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = SGDClassifier(loss="log_loss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {acc:.2f}")
    return model


# -------------------------------
# 6. PREDICTION PIPELINE
# -------------------------------
def categorize_transaction(desc):
    desc = str(desc).lower()
    # NLP Mock Rule Engine
    if any(keyword in desc for keyword in ['zerodha', 'groww', 'mutual fund', 'sip', 'lic', 'investment']):
        return 'investment'
    if any(keyword in desc for keyword in ['dream11', 'stake', 'bet', 'casino', 'rummy']):
        return 'high_risk'
    if any(keyword in desc for keyword in ['zomato', 'swiggy', 'amazon', 'flipkart', 'myntra', 'netflix']):
        return 'discretionary'
    if any(keyword in desc for keyword in ['rent', 'electricity', 'water', 'emi', 'loan', 'bill']):
        return 'essential'
    return 'other'

def evaluate_transaction_history(transactions, employment_type="salaried"):
    total_credit = sum(t['amount'] for t in transactions if t['txn_type'] == 'credit')
    total_debit = sum(t['amount'] for t in transactions if t['txn_type'] == 'debit')
    
    savings_ratio = (total_credit - total_debit) / total_credit if total_credit > 0 else 0
    
    # Base score
    score = 50 
    
    # 1. Savings Pattern
    if savings_ratio > 0.3:
        score += 30
    elif savings_ratio > 0.1:
        score += 15
    elif savings_ratio < 0:
        score -= 30
        
    # 2. Persona-based Activity & Velocity
    num_txns = len(transactions)
    num_debits = len([t for t in transactions if t['txn_type'] == 'debit'])
    avg_debit = total_debit / num_debits if num_debits > 0 else 0
    
    if employment_type == "self-employed":
        # Business / Self-Employed Persona rules (Tolerates more transactions)
        if num_txns < 5:
            score -= 10 # Self-employed should have higher velocity
        elif num_txns > 100 and avg_debit < 200:
            score -= 10
        elif 10 <= num_txns <= 80:
            score += 15
    else:
        # Salaried Persona rules
        if num_txns < 4:
            score -= 5 
        elif num_txns > 50 and avg_debit < 500:
            score -= 10 # Micro-spending penalty
        elif 5 <= num_txns <= 30:
            score += 10
            
    # 3. NLP Category Analysis
    investment_count = 0
    high_risk_count = 0
    for t in transactions:
        cat = categorize_transaction(t.get('description', ''))
        if cat == 'investment':
            investment_count += 1
        elif cat == 'high_risk':
            high_risk_count += 1
            
    # Reward financial discipline
    if investment_count > 0:
        score += (investment_count * 5) # +5 for each investment
        
    # Penalize risky behavior (gambling, etc.)
    if high_risk_count > 0:
        score -= (high_risk_count * 15) # Heavy penalty for gambling/betting
        
    # 4. Cash Flow Volume
    if total_credit > 25000:
        score += 10
        
    return max(0, min(100, score))


def predict_loan(
    model,
    scaler,
    feature_columns,
    input_data,
    transactions,
    profile,
    statement_ml_score=None,
):

    features = extract_features(transactions, profile)

    # --- NEW LOGIC: Calculate Scores ---
    employment_type = input_data.get('employment_type', 'salaried')
    if statement_ml_score is not None:
        txn_confidence_score = max(0.0, min(100.0, float(statement_ml_score)))
    else:
        txn_confidence_score = evaluate_transaction_history(transactions, employment_type)
    cibil_val = input_data.get("cibil_score", 300)
    cibil_normalized_score = max(0, min(100, (cibil_val - 300) / 600 * 100))
    
    blended_score = (txn_confidence_score * 0.6) + (cibil_normalized_score * 0.4)
    
    base_result = {
        "txn_confidence_score": txn_confidence_score,
        "cibil_normalized_score": cibil_normalized_score,
        "final_confidence_score": blended_score
    }

    # Rule Engine check
    passed, reason = rule_engine(input_data, features)
    if not passed:
        base_result.update({"status": "REJECTED", "reason": reason})
        return base_result

    # Run ML Model as an additional safety check
    input_df = pd.DataFrame([features])
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(input_df)

    # Using predict_proba since loss='log_loss' is used
    ml_prob = model.predict_proba(input_scaled)[0][1] * 100

    # Final Decision Logic: Blend heuristic and ML
    overall_score = (blended_score + ml_prob) / 2

    if overall_score >= 60:
        base_result.update({
            "status": "APPROVED",
            "loan_amount": input_data["loan_amount"],
            "score": overall_score
        })
    else:
        base_result.update({
            "status": "REJECTED",
            "reason": f"Overall Confidence Score ({overall_score:.1f}/100) is below the required 60 threshold."
        })
        
    return base_result


# -------------------------------
# 7. MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":

    df = load_dataset("loan_data.csv")

    X, y, scaler, label_encoders, feature_columns = preprocess_data(df)

    model = train_model(X, y)

    # SAMPLE INPUT
    input_data = {
        "product_price": 50000,
        "down_payment": 10000,
        "loan_amount": 40000,
        "tenure": 12,
        "bank_name": "HDFC",
        "account_number": "123456",
        "IFSC": "HDFC0001",
        "cibil_score": 720,
        "existing_emi": 2000,
        "credit_card": "yes",
        "income": 30000,
        "employment_type": "salaried",
        "dependents": 2
    }

    transactions = [
        {"txn_type": "credit", "amount": 30000, "balance": 35000},
        {"txn_type": "debit", "amount": 10000, "balance": 25000},
        {"txn_type": "debit", "amount": 5000, "balance": 20000}
    ]

    profile = {
        "income": input_data["income"],
        "cibil_score": input_data["cibil_score"],
        "dependents": input_data["dependents"]
    }

    result = predict_loan(
        model, scaler, feature_columns,
        input_data, transactions, profile
    )

    print("\n📊 FINAL RESULT:")
    print(result)