import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

def generate_data(num_samples=3000):
    np.random.seed(42)
    
    # Generate features
    cibil_score = np.random.randint(300, 900, num_samples)
    monthly_salary = np.random.randint(15000, 300000, num_samples)
    company_tier = np.random.choice([1, 2, 3], num_samples, p=[0.2, 0.5, 0.3])
    loan_amount = np.random.randint(50000, 2000000, num_samples)
    
    # Calculate derived features for logic
    # Assume typical tenure of 36 months, interest ~12%. EMI is approx loan_amount * 0.033
    estimated_emi = loan_amount * 0.033
    dti = estimated_emi / monthly_salary
    
    # Rule based approval to create target variable
    # 1. CIBIL must be > 600 (if tier 1) or > 650 (tier 2) or > 700 (tier 3)
    # 2. DTI must be < 0.5 (or < 0.6 for tier 1)
    approved = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        c = cibil_score[i]
        t = company_tier[i]
        d = dti[i]
        
        cibil_req = 700
        if t == 1: cibil_req = 600
        elif t == 2: cibil_req = 650
        
        dti_req = 0.5
        if t == 1: dti_req = 0.6
        elif t == 3: dti_req = 0.4
        
        # Add some random noise so it's not 100% deterministic (machine learning realistic)
        noise = np.random.normal(0, 20) 
        if (c + noise) > cibil_req and d < dti_req:
            approved[i] = 1
        else:
            # 5% chance of random override just for variance
            if np.random.rand() < 0.05:
                approved[i] = 1
                
    df = pd.DataFrame({
        'cibil_score': cibil_score,
        'monthly_salary': monthly_salary,
        'company_tier': company_tier,
        'loan_amount': loan_amount,
        'approved': approved
    })
    
    return df

def train_and_save():
    print("Generating 3000 rows of personal loan data...")
    df = generate_data(3000)
    
    df.to_csv(os.path.join(os.path.dirname(__file__), 'personal_loan_dataset.csv'), index=False)
    
    X = df[['cibil_score', 'monthly_salary', 'company_tier', 'loan_amount']]
    y = df['approved']
    
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X, y)
    
    model_path = os.path.join(os.path.dirname(__file__), 'xgb_personal_loan.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model saved to {model_path}")
    print(f"Accuracy on training set: {model.score(X, y):.4f}")

if __name__ == '__main__':
    train_and_save()
