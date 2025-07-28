import streamlit as st
import pandas as pd
import joblib

# --- Config ---
DATA_PATH = "Synthetic_Financial_datasets_log.csv"
MODEL_FILES = {
    "CatBoost": "new_models/CatBoost_balanced_ManualCV.joblib"
}
PAGE_SIZE = 1000

# --- UI Setup ---
st.set_page_config(page_title="AI Financial Fraud Detector", layout="wide")
st.title("AI Financial Fraud Detection App")
st.markdown("Detect fraudulent transactions on a **full dataset** using your selected ML model.")

# --- Load Dataset ---
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(DATA_PATH)
    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')
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
st.success(f"Loaded dataset with **{len(df):,} records**.")
st.dataframe(df.head())

# --- Model Selection ---
selected_model_name = st.selectbox("Select ML Model", list(MODEL_FILES.keys()))
model = joblib.load(MODEL_FILES[selected_model_name])

# --- Run Predictions ---
if st.button("Run Fraud Prediction on Full Dataset"):
    preds = model.predict(df)
    proba = model.predict_proba(df)[:, 1]

    df['isFraud_pred'] = preds
    df['fraud_proba'] = proba

    # Store in session state
    st.session_state['predicted_df'] = df
    st.session_state['page'] = 1  # reset pagination

# --- If Predictions Exist ---
if 'predicted_df' in st.session_state:
    df = st.session_state['predicted_df']

    total_records = len(df)
    total_frauds = (df['isFraud_pred'] == 1).sum()
    total_not_frauds = total_records - total_frauds
    fraud_percent = (total_frauds / total_records) * 100

    # --- Pagination ---
    total_pages = (total_records // PAGE_SIZE) + (1 if total_records % PAGE_SIZE > 0 else 0)

    st.markdown(f"### Prediction Results using **{selected_model_name}**")
    st.markdown(f"- **Total Records:** {total_records:,}")
    st.markdown(f"- **Frauds:** {total_frauds:,}")
    st.markdown(f"- **Not Frauds:** {total_not_frauds:,}")
    st.markdown(f"- **Fraud %:** {fraud_percent:.2f}%")
    st.markdown(f"- **Total Pages:** {total_pages}")

    col1, col2 = st.columns([1, 4])
    with col1:
        page_input = st.number_input("**Page**", min_value=1, max_value=total_pages, value=st.session_state.get('page', 1), step=1)
        st.session_state['page'] = page_input

    start = (st.session_state['page'] - 1) * PAGE_SIZE
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
