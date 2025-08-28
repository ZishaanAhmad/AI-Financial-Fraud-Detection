import os
import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np


MODEL_PATH = "XGBoost_GridSearchCV_3.joblib"  # Match notebook save name

# --- Transaction Type Mapping ---
TYPE_MAP = {"TRANSFER": 1, "CASH_OUT": 2, "PAYMENT": 3, "DEBIT": 4, "CASH_IN": 5}

# --- UI Setup ---
st.set_page_config(page_title="AI Financial Fraud Detector", layout="wide")
st.title("AI Financial Fraud Detection (Single Input) | XGBoost Model")
st.markdown("Enter transaction details to check if it may be **fraudulent**.")

# --- Load Model + Metadata ---
# @st.cache_resource
def load_model_with_meta():
    saved = joblib.load(MODEL_PATH)
    if isinstance(saved, dict):
        return saved['model'], saved.get('features'), saved.get('scaler'), saved.get('amount_mean'), saved.get('amount_std'), saved.get('best_threshold', 0.5)
    else:
        return saved, None, None, None, None, 0.5

model, FEATURES, scaler, amount_mean, amount_std, optimal_threshold = load_model_with_meta()

if FEATURES is None:
    st.error("Model features not found in saved metadata. Ensure notebook saved them.")
    st.stop()

# --- Feature Engineering (match notebook) ---
def feature_engineering(df):
    df = df.copy()

    # Map transaction type
    if 'TX_TYPE' in df.columns:
        if df['TX_TYPE'].dtype == object:
            df['TX_TYPE'] = df['TX_TYPE'].map(TYPE_MAP).fillna(0).astype(int)
        else:
            df['TX_TYPE'] = pd.to_numeric(df['TX_TYPE'], errors='coerce').fillna(0).astype(int)
    else:
        df['TX_TYPE'] = 0

    # Ensure numeric TX_AMOUNT
    if 'TX_AMOUNT' in df.columns:
        df['TX_AMOUNT'] = pd.to_numeric(df['TX_AMOUNT'], errors='coerce').fillna(0.0)
    else:
        df['TX_AMOUNT'] = 0.0

    # Sender frequency placeholder
    df['is_frequent_sender'] = 0

    # Micro / round / zero amounts
    df['is_micro_tx'] = (df['TX_AMOUNT'] < 20).astype(int)
    df['is_round_amount'] = ((df['TX_AMOUNT'] % 1) == 0).astype(int)
    df['is_zero_amount'] = (df['TX_AMOUNT'] == 0).astype(int)

    # Log amount
    df['logAmount'] = np.log1p(df['TX_AMOUNT'])

    # Amount zscore (use training mean/std)
    if amount_mean is not None and amount_std is not None:
        df['amount_zscore'] = (df['TX_AMOUNT'] - amount_mean) / (amount_std + 1e-9)
    else:
        df['amount_zscore'] = 0.0

    # Hour of day from timestamp
    if 'TIMESTAMP' in df.columns:
        try:
            df['hour_of_day'] = ((df['TIMESTAMP'] // 3600) % 24).astype(int)
        except Exception:
            df['hour_of_day'] = (df['TIMESTAMP'] % 24).astype(int)
    else:
        df['hour_of_day'] = 0

    # Add any missing training features with 0
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    # Enforce exact feature order
    return df[FEATURES]

# --- Input Form ---
with st.form("fraud_check_form"):
    col1, col2 = st.columns(2)
    with col1:
        sender_id = st.text_input("Sender Account ID", value="0")
        tx_type = st.selectbox("Transaction Type", list(TYPE_MAP.keys()))
    with col2:
        receiver_id = st.text_input("Receiver Account ID", value="0")
        amount = st.number_input("Amount (TX_AMOUNT)", min_value=0.0, value=100.00)

    timestamp = st.number_input("Timestamp (TIMESTAMP)", min_value=0, value=0, step=1)

    submitted = st.form_submit_button("Check for Fraud")

# --- Prediction ---
if submitted:
    with st.spinner("Analyzing transaction..."):
        time.sleep(0.5)

        # Raw input DataFrame
        input_df = pd.DataFrame([{
            "SENDER_ACCOUNT_ID": sender_id,
            "RECEIVER_ACCOUNT_ID": receiver_id,
            "TX_TYPE": tx_type,
            "TX_AMOUNT": amount,
            "TIMESTAMP": timestamp
        }])

        processed_data = feature_engineering(input_df)

        with st.expander("View engineered features"):
            st.dataframe(processed_data)
            st.write(f"Number of features: {len(processed_data.columns)}")
            st.write("Feature names:", list(processed_data.columns))

        try:
            probability = model.predict_proba(processed_data)[0][1]
            prediction = int(probability >= optimal_threshold)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.stop()

    # Display results
    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error(f"""
        ⚠️ **Fraud Alert**  
        Confidence: {probability:.1%}  
        This transaction shows strong signs of potential fraud.
        """)
    else:
        st.success(f"""
        ✅ **Legitimate Transaction**  
        Confidence: {1-probability:.1%}  
        This transaction appears normal based on our analysis.
        """)

    st.markdown("### Transaction Details")
    summary_data = {
        "Field": ["Sender ID", "Receiver ID", "Type", "Amount", "Timestamp"],
        "Value": [sender_id, receiver_id, tx_type, f"${amount:,.2f}", timestamp]
    }
    st.table(pd.DataFrame(summary_data))
