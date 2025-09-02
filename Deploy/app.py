import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Title
st.title("🚨 Fraud Detection App (XGBoost)")

# Load model
MODEL_FILE = "xgb_fraud_model1.joblib"

# @st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📂 Upload your transaction CSV file", type=["csv"])

if uploaded_file is not None:
    # Load CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Uploaded Data (First 5 rows)")
    st.write(df.head())

    # --- Feature Engineering ---
    df_processed = df.copy()

    # Encode type (map unseen types to 0)
    df_processed['type'] = df_processed['type'].map({
        "CASH_IN": 1,
        "CASH_OUT": 2,
        "DEBIT": 3,
        "TRANSFER": 4,
        "PAYMENT": 5
    }).fillna(0).astype(int)

    # Derived features
    df_processed['origBalanceDiff'] = df_processed['oldbalanceOrg'] - df_processed['newbalanceOrig']
    df_processed['destBalanceDiff'] = df_processed['oldbalanceDest'] - df_processed['newbalanceDest']

    df_processed['origBalanceRatio'] = df_processed.apply(
        lambda x: x['amount'] / (x['oldbalanceOrg']+1) if x['oldbalanceOrg'] > 0 else 0, axis=1
    )
    df_processed['destBalanceRatio'] = df_processed.apply(
        lambda x: x['amount'] / (x['oldbalanceDest']+1) if x['oldbalanceDest'] > 0 else 0, axis=1
    )

    df_processed['origZeroBalance'] = (df_processed['oldbalanceOrg'] == 0).astype(int)
    df_processed['destZeroBalance'] = (df_processed['oldbalanceDest'] == 0).astype(int)

    df_processed['logAmount'] = (df_processed['amount'] + 1).apply(lambda x: np.log(x))
    df_processed['amountToOrigBalance'] = df_processed.apply(
        lambda x: x['amount'] / (x['oldbalanceOrg']+1), axis=1
    )

    # Final feature set (must match training features)
    final_features = [
        'step','type','amount','oldbalanceOrg','newbalanceOrig',
        'oldbalanceDest','newbalanceDest','origBalanceDiff',
        'destBalanceDiff','origBalanceRatio','destBalanceRatio',
        'origZeroBalance','destZeroBalance','logAmount','amountToOrigBalance'
    ]

    df_model = df_processed[final_features]

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
    st.download_button("⬇️ Download Predictions CSV", csv_download, "fraud_predictions.csv", "text/csv")
else:
    st.info("👆 Please upload a CSV file to start.")
