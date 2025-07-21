import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# Page config
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

# Load model and scaler
model = joblib.load("xgboost_loan_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Approval Predictor")
st.write("Enter applicant details to check if the loan will be approved.")

# Input form
dependents = st.number_input("No. of Dependents", min_value=0, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income = st.number_input("Annual Income (KES)", min_value=0)
loan_amount = st.number_input("Loan Amount (KES)", min_value=0)
loan_term = st.number_input("Loan Term (Months)", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=0, max_value=900)
res_assets = st.number_input("Residential Assets Value", min_value=0)
com_assets = st.number_input("Commercial Assets Value", min_value=0)
lux_assets = st.number_input("Luxury Assets Value", min_value=0)
bank_assets = st.number_input("Bank Asset Value", min_value=0)

# Prepare input
input_data = {
    "no_of_dependents": dependents,
    "education": 1 if education == "Graduate" else 0,
    "self_employed": 1 if self_employed == "Yes" else 0,
    "income_annum": income,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": res_assets,
    "commercial_assets_value": com_assets,
    "luxury_assets_value": lux_assets,
    "bank_asset_value": bank_assets
}

input_df = pd.DataFrame([input_data])

# Prediction button
if st.button("Predict Loan Status"):
    # Scale input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    confidence = model.predict_proba(scaled_input)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Loan Approved with confidence: {confidence:.2f}")
    else:
        st.error(f"❌ Loan Rejected with confidence: {confidence:.2f}")

    # SHAP Explanation
    st.subheader("🧠 SHAP Explanation: Why was this decision made?")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    shap_df = pd.DataFrame({
        "Feature": input_df.columns,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=True)

    colors = shap_df["SHAP Value"].apply(lambda x: 'green' if x > 0 else 'red')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
    ax.set_title("Feature Impact (SHAP values)")
    ax.set_xlabel("Contribution to Prediction")
    st.pyplot(fig)

    # Adaptive explanation
    if prediction == 1:
        st.markdown(f"""
    🧾 **Interpretation Guide (Approved ✅)**

    - The model is **{confidence:.2%} confident** that the loan should be approved.
    - ✅ **Green bars** = Features that helped with approval.
    - ❌ **Red bars** = Features that pulled down the approval score.
    """)
    else:
        st.markdown(f"""
    🧾 **Interpretation Guide (Rejected ❌)**

    - The model is **{confidence:.2%} confident** that the loan should be rejected.
    - ❌ **Red bars** = Features that pushed the model toward rejection.
    - ✅ **Green bars** = Helpful features, but not strong enough to reverse the decision.
    """)

