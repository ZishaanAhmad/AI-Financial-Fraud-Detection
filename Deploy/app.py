import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Title
st.title("🚨 Fraud Detection App (XGBoost)")

# Load model
MODEL_FILE = "xgb_fraud_model2.joblib"

# @st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

model = load_model()

# --- Feature Engineering (same as training) ---
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Drop identifiers
    d.drop(columns=["nameOrig", "nameDest"], inplace=True, errors="ignore")

    # Ensure numeric types
    for c in ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0.0)
        else:
            d[c] = 0.0

    # One-hot encode type
    if 'type' in d.columns:
        type_dummies = pd.get_dummies(d['type'].astype(str), prefix="type")
        d = pd.concat([d, type_dummies], axis=1)
    else:
        d['type'] = "UNKNOWN"
        d = pd.concat([d, pd.get_dummies(d['type'], prefix="type")], axis=1)

    # Derived calculation-based features
    d['orig_delta'] = d['oldbalanceOrg'] - d['newbalanceOrig'] - d['amount']
    d['dest_delta'] = d['newbalanceDest'] - d['oldbalanceDest'] - d['amount']

    d['logAmount'] = np.log1p(d['amount'].clip(lower=0))
    d['origBalanceRatio'] = np.where(d['oldbalanceOrg'] != 0, d['amount'] / d['oldbalanceOrg'], 0.0)
    d['destBalanceRatio'] = np.where(d['oldbalanceDest'] != 0, d['amount'] / d['oldbalanceDest'], 0.0)

    d['origZeroBalance'] = (d['oldbalanceOrg'] == 0).astype(int)
    d['destZeroBalance'] = (d['oldbalanceDest'] == 0).astype(int)

    d['rule_orig_inconsistent'] = (d['orig_delta'].abs() > 1e-9).astype(int)
    d['rule_dest_inconsistent'] = (d['dest_delta'].abs() > 1e-9).astype(int)
    d['rule_zero_origin_drain'] = ((d['newbalanceOrig'].abs() <= 1e-9) &
                                   (d['oldbalanceOrg'].sub(d['amount']).abs() <= 1e-9)).astype(int)
    d['rule_zero_dest_firstload'] = ((d['oldbalanceDest'].abs() <= 1e-9) &
                                     (d['newbalanceDest'].sub(d['amount']).abs() <= 1e-9)).astype(int)

    return d

# File uploader
uploaded_file = st.file_uploader("📂 Upload your transaction CSV file", type=["csv"])

if uploaded_file is not None:
    # Load CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Uploaded Data (First 5 rows)")
    st.write(df.head())

    # Apply feature engineering
    df_processed = feature_engineering(df)

    # Final feature set (must match training features)
    model_features = [
        'step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest',
        'orig_delta','dest_delta','logAmount',
        'origBalanceRatio','destBalanceRatio',
        'origZeroBalance','destZeroBalance',
        'rule_orig_inconsistent','rule_dest_inconsistent',
        'rule_zero_origin_drain','rule_zero_dest_firstload'
    ]

    # Add one-hot encoded type features dynamically
    type_features = [c for c in df_processed.columns if c.startswith("type_")]
    model_features.extend(type_features)

    # Ensure all exist (fill missing with 0.0)
    for f in model_features:
        if f not in df_processed.columns:
            df_processed[f] = 0.0

    df_model = df_processed[model_features]

    # Predict
    preds = model.predict(df_model)

    # Map predictions to YES/NO
    df["isFraud"] = np.where(preds == 1, "YES", "NO")

    # Show results
    st.subheader("✅ Predictions")
    st.dataframe(df[["step","type","amount","oldbalanceOrg","newbalanceOrig",
                     "oldbalanceDest","newbalanceDest","isFraud"]])

    # Option to download results
    csv_download = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Predictions CSV", csv_download,
                       "fraud_predictions.csv", "text/csv")
else:
    st.info("👆 Please upload a CSV file to start.")
