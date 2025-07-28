import streamlit as st
import pandas as pd
import joblib
import os

# Paths
DATA_PATH = "Synthetic_Financial_datasets_log.csv"
model_files = {
    "CatBoost": "new_models/CatBoost_balanced_ManualCV.joblib",
    "XGBoost": "new_models/XGBoost_ManualCV.joblib",
    "LightGBM": "new_models/LightGBM_balanced_ManualCV.joblib"
}

# UI setup
st.set_page_config(page_title="🚨 Fraud Detection (Batch)", layout="wide")
st.title("🚨 Bulk Fraud Detection App")
st.markdown("Using **local dataset** with selected machine learning model.")

# Select model
selected_model_name = st.selectbox("Select Model", list(model_files.keys()))
model = joblib.load(model_files[selected_model_name])

# Load dataset
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Drop unused columns
    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud"]
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Map transaction type if needed
    if df['type'].dtype == 'object':
        df['type'] = df['type'].map({
            "CASH_IN": 1,
            "CASH_OUT": 2,
            "DEBIT": 3,
            "TRANSFER": 4,
            "PAYMENT": 5
        }).fillna(0).astype(int)

    return df

df = load_data()
st.success(f"✅ Loaded dataset with {len(df):,} records.")
st.markdown("### 📊 Preview of Raw Data")
st.dataframe(df.head())

# Predict
if st.button("🚀 Run Fraud Prediction"):
    preds = model.predict(df)
    proba = model.predict_proba(df)[:, 1]

    df['isFraud_pred'] = preds
    df['fraud_proba'] = proba

    fraud_count = (df['isFraud_pred'] == 1).sum()
    fraud_percent = (fraud_count / len(df)) * 100

    st.markdown(f"### 🔍 Results for **{selected_model_name}**")
    st.markdown(f"**🧾 Total Fraudulent Records Predicted:** {fraud_count:,} / {len(df):,} ({fraud_percent:.2f}%)")

    # Tabs to separate fraud and non-fraud
    tab1, tab2, tab3 = st.tabs(["📁 All Results", "🚨 Fraudulent Only", "✅ Not Fraudulent"])

    def paginate_dataframe(data, label):
        page_size = 1000
        total_rows = data.shape[0]
        total_pages = (total_rows // page_size) + 1

        page = st.number_input(f"{label} - Page", 1, total_pages, 1, key=label)
        start = (page - 1) * page_size
        end = start + page_size
        st.dataframe(data.iloc[start:end])

    with tab1:
        paginate_dataframe(df, "All Results")

    with tab2:
        paginate_dataframe(df[df['isFraud_pred'] == 1], "Fraudulent Records")

    with tab3:
        paginate_dataframe(df[df['isFraud_pred'] == 0], "Not Fraudulent Records")

    # Download button
    st.download_button(
        "⬇️ Download Results as CSV",
        data=df.to_csv(index=False),
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )
