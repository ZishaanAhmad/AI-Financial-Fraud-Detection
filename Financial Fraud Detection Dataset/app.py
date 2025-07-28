import streamlit as st
import pandas as pd
import joblib

# --- Config ---
DATA_PATH = "Synthetic_Financial_datasets_log.csv"
MODEL_FILES = {
    "CatBoost": "new_models/CatBoost_balanced_ManualCV.joblib" # ,
    # "XGBoost": "new_models/XGBoost_ManualCV.joblib",
    # "LightGBM": "new_models/LightGBM_balanced_ManualCV.joblib"
}
PAGE_SIZE = 1000

# --- UI Setup ---
st.set_page_config(page_title="AI Financial Fraud Detector", layout="wide")
st.title("AI Financial Fraud Detection App")
st.markdown("Detect fraudulent transactions on a **full dataset** using your selected ML model.")

# --- Model Selector ---
selected_model_name = st.selectbox("Select ML Model", list(MODEL_FILES.keys()))
model = joblib.load(MODEL_FILES[selected_model_name])

# --- Load Dataset ---
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Drop unused columns
    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

    # Encode 'type'
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
st.success(f"Loaded dataset with **{len(df):,} records**")
st.dataframe(df.head())

# --- Predict Button ---
if st.button("Run Fraud Prediction on Full Dataset"):
    preds = model.predict(df)
    proba = model.predict_proba(df)[:, 1]

    df['isFraud_pred'] = preds
    df['fraud_proba'] = proba

    fraud_count = (df['isFraud_pred'] == 1).sum()
    fraud_percent = (fraud_count / len(df)) * 100

    st.markdown(f"###**Total Records Predicted:** {fraud_count:,} / {len(df):,} ({fraud_percent:.2f}%)")

    # --- Pagination ---
    total_rows = df.shape[0]
    total_pages = (total_rows // PAGE_SIZE) + (1 if total_rows % PAGE_SIZE > 0 else 0)

    st.markdown(f"**Total Pages:** {total_pages}")
    st.markdown("Enter page number to view that chunk of results.")

    page = st.number_input("Enter Page Number", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    paged_df = df.iloc[start:end]

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["All Records", "Fraud Only", "Not Fraud"])

    with tab1:
        st.dataframe(paged_df)

    with tab2:
        st.dataframe(paged_df[paged_df['isFraud_pred'] == 1])

    with tab3:
        st.dataframe(paged_df[paged_df['isFraud_pred'] == 0])

    # --- Download ---
    st.download_button(
        label="⬇️ Download Full Results as CSV",
        data=df.to_csv(index=False),
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )
