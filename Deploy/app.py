import streamlit as st
import pandas as pd
import joblib

# --- Config ---
MODEL_FILES = {
    "XGBoost": "xgb_fraud_model1.joblib"
}

# --- UI Setup ---
st.set_page_config(page_title="AI Financial Fraud Detector", layout="wide")
st.title("🔍 AI Financial Fraud Detection App")
st.markdown("Upload a CSV file with transactions and check if they are **Fraud** or **Not Fraud**.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# --- Model Selection ---
selected_model_name = st.selectbox("Select ML Model", list(MODEL_FILES.keys()))
model = joblib.load(MODEL_FILES[selected_model_name])

if uploaded_file is not None:
    # Load CSV
    df = pd.read_csv(uploaded_file)

    st.markdown("### 📑 Uploaded Data (first 10 rows)")
    st.dataframe(df.head(10))

    # --- Preprocess ---
    required_cols = [
        "step","type","amount","nameOrig","oldbalanceOrg",
        "newbalanceOrig","nameDest","oldbalanceDest","newbalanceDest"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
    else:
        # Drop ID-like columns
        df_processed = df.copy()
        drop_cols = ["nameOrig", "nameDest"]
        df_processed.drop(columns=[col for col in drop_cols if col in df_processed.columns], inplace=True, errors="ignore")

        # Encode type
        if df_processed['type'].dtype == 'object':
            df_processed['type'] = df_processed['type'].map({
                "CASH_IN": 1,
                "CASH_OUT": 2,
                "DEBIT": 3,
                "TRANSFER": 4,
                "PAYMENT": 5
            }).fillna(0).astype(int)

        # --- Run Predictions ---
        preds = model.predict(df_processed)
        proba = model.predict_proba(df_processed)[:, 1]

        # Add results to original df
        df['isFraud'] = preds
        df['fraud_proba'] = proba

        # Replace 1/0 with YES/NO
        df['isFraud'] = df['isFraud'].map({1: "YES", 0: "NO"})

        # --- Show Results ---
        st.markdown("### 🔎 Prediction Results")
        st.dataframe(df[required_cols + ['isFraud', 'fraud_proba']].head(50))

        # --- Download Button ---
        st.download_button(
            label="⬇️ Download Results CSV",
            data=df.to_csv(index=False),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )
