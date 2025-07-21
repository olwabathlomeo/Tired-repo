import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load the XGBoost model
model = joblib.load("xgboost_loan_model.pkl")

# Page title
st.title("🏦 Loan Approval Predictor")
st.markdown("Enter applicant details to check if the loan will be approved.")

# Input fields
no_of_dependents = st.number_input("No. of Dependents", min_value=0, max_value=20, value=0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income (KES)", min_value=0, value=100000)
loan_amount = st.number_input("Loan Amount (KES)", min_value=0, value=100000)
loan_term = st.number_input("Loan Term (Months)", min_value=1, value=12)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=500)
res_assets = st.number_input("Residential Assets Value", min_value=0, value=50000)
com_assets = st.number_input("Commercial Assets Value", min_value=0, value=20000)
lux_assets = st.number_input("Luxury Assets Value", min_value=0, value=10000)
bank_assets = st.number_input("Bank Asset Value", min_value=0, value=10000)

# Encode categorical features
education_encoded = 1 if education == "Graduate" else 0
self_employed_encoded = 1 if self_employed == "Yes" else 0

# Collect input
input_data = {
    "no_of_dependents": no_of_dependents,
    "education": education_encoded,
    "self_employed": self_employed_encoded,
    "income_annum": income_annum,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": res_assets,
    "commercial_assets_value": com_assets,
    "luxury_assets_value": lux_assets,
    "bank_asset_value": bank_assets
}

input_df = pd.DataFrame([input_data])

# Predict button
if st.button("Predict Loan Status"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Loan Approved with confidence: {prediction_proba:.2f}")
    else:
        st.error(f"❌ Loan Rejected with confidence: {prediction_proba:.2f}")

    # SHAP Explanation
    st.subheader("📊 Why This Decision?")
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(input_df)

    # Use SHAP values of predicted class
    shap_values = shap_values_all[prediction]

    # Plot
    shap_df = pd.DataFrame({
        "Feature": input_df.columns,
        "SHAP Value": shap_values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=True)

    bar_colors = ['green' if val > 0 else 'red' for val in shap_df["SHAP Value"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=bar_colors)
    ax.set_title("Feature Impact (SHAP Values)")
    st.pyplot(fig)

    # Adaptive explanation
    if prediction == 1:
        st.info("This chart explains the features that pushed the decision **towards approval**.")
    else:
        st.info("This chart shows the features that contributed **most to rejection**.")

