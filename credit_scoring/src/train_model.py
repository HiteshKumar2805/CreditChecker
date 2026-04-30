import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from feature_engineering import add_features

df = pd.read_csv("data/synthetic_data.csv")
df = add_features(df)

X = df.drop("good_credit_behavior", axis=1)
y = df["good_credit_behavior"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1
)

model.fit(X_train, y_train)

joblib.dump(model, "models/xgboost_model.pkl")

print("Model trained successfully!")