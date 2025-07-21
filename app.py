import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# Load the trained XGBoost model
model = joblib.load("xgboost_loan_model.pkl")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Predictor")
st.write("Enter applicant details to check if the loan will be approved.")

# --- Collect user input ---
no_of_dependents = st.number_input("No. of Dependents", min_value=0, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income (KES)", min_value=0)
loan_amount = st.number_input("Loan Amount (KES)", min_value=0)
loan_term = st.number_input("Loan Term (Months)", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# --- Preprocess input ---
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

input_data = {
    "no_of_dependents": no_of_dependents,
    "education": education,
    "self_employed": self_employed,
    "income_annum": income_annum,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": residential_assets_value,
    "commercial_assets_value": commercial_assets_value,
    "luxury_assets_value": luxury_assets_value,
    "bank_asset_value": bank_asset_value
}

input_df = pd.DataFrame([input_data])

# --- Predict ---
if st.button("🔎 Predict"):
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][int(prediction)]

    if prediction == 1:
        st.success(f"✅ Loan Approved with confidence: {confidence:.2f}")
    else:
        st.error(f"❌ Loan Rejected with confidence: {confidence:.2f}")

        # --- SHAP Explanation ---
    st.subheader("🧠 SHAP Explanation: Why this decision?")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    shap_df = pd.DataFrame({
        "Feature": input_df.columns,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=True)

    # Color code: green for positive, red for negative
    colors = shap_df["SHAP Value"].apply(lambda x: 'green' if x > 0 else 'red')

    # Plot with Matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
    ax.set_title("Feature Impact (SHAP values)")
    ax.set_xlabel("Contribution to Prediction")

    st.pyplot(fig)

