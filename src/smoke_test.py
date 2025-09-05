# src/smoke_test.py
import json, pandas as pd
from joblib import load

pipe = load("models/churn_pipeline.pkl")
# Build a minimal one-row dataframe using columns from reports/feature_columns.json
with open("reports/feature_columns.json") as f:
    schema = json.load(f)

# Reasonable dummy example (edit values as you like)
row = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1020.0,
}

# Ensure all columns present
for col in schema.keys():
    if col not in row:
        row[col] = None

df = pd.DataFrame([row])
proba = pipe.predict_proba(df)[:,1][0]
pred = int(proba >= 0.5)
print({"churn_probability": float(proba), "prediction": pred})
