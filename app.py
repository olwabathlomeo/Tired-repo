import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# Load trained model and scaler
model = joblib.load('xgboost_loan_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Predictor")
st.markdown("Enter applicant details to check if the loan will be approved.")

# Input form
with st.form("loan_form"):
    dependents = st.number_input("No. of Dependents", min_value=0, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Annual Income (KES)", min_value=0, step=1000)
    loan_amount = st.number_input("Loan Amount (KES)", min_value=0, step=1000)
    loan_term = st.number_input("Loan Term (Months)", min_value=1, step=1)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
    res_assets = st.number_input("Residential Assets Value", min_value=0, step=1000)
    com_assets = st.number_input("Commercial Assets Value", min_value=0, step=1000)
    lux_assets = st.number_input("Luxury Assets Value", min_value=0, step=1000)
    bank_assets = st.number_input("Bank Asset Value", min_value=0, step=1000)

    submitted = st.form_submit_button("Check Approval")

if submitted:
    # Encode categorical variables
    education_encoded = 1 if education == "Graduate" else 0
    self_employed_encoded = 1 if self_employed == "Yes" else 0

    # Create input DataFrame
    raw_input = pd.DataFrame([[
        dependents,
        education_encoded,
        self_employed_encoded,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        res_assets,
        com_assets,
        lux_assets,
        bank_assets
    ]], columns=[
        "no_of_dependents", "education", "self_employed", "income_annum", "loan_amount",
        "loan_term", "cibil_score", "residential_assets_value", "commercial_assets_value",
        "luxury_assets_value", "bank_asset_value"
    ])

    # Scale input
    processed_input = scaler.transform(raw_input)

    # Predict
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]

    # Display prediction
    if prediction == 1:
        st.success(f"✅ Loan Approved with confidence: {probability:.2f}")
    else:
        st.error(f"❌ Loan Rejected with confidence: {1 - probability:.2f}")

    # SHAP explanation
    st.subheader("📊 Why This Decision?")

    # Use the raw (unscaled) input for SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(raw_input)

    # Prepare SHAP DataFrame
    shap_df = pd.DataFrame({
        "Feature": raw_input.columns,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=True)

    # Color based on impact
    bar_colors = ['green' if val > 0 else 'red' for val in shap_df["SHAP Value"]]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=bar_colors)
    ax.set_title("Feature Impact (SHAP Values)")
    st.pyplot(fig)

    # Explanation text
    if prediction == 1:
        st.info("🔍 This chart explains the features that pushed the decision **towards loan approval**.")
    else:
        st.info("🔍 This chart explains the features that pushed the decision **towards loan rejection**.")
