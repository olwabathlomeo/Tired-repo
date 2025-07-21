import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('xgboost_loan_model.pkl')
scaler = joblib.load('scaler.pkl')

# Input feature names
feature_names = [
    'no_of_dependents', 'education', 'self_employed', 'income_annum',
    'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value'
]

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Predictor")
st.markdown("Enter applicant details to check if the loan will be approved.")

# User input form
with st.form("loan_form"):
    no_of_dependents = st.number_input("No. of Dependents", min_value=0, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Annual Income (KES)", min_value=0)
    loan_amount = st.number_input("Loan Amount (KES)", min_value=0)
    loan_term = st.number_input("Loan Term (Months)", min_value=1)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
    luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
    bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

    submit = st.form_submit_button("Predict")

# Encode inputs and predict
if submit:
    # Convert categorical inputs
    education_encoded = 0 if education == "Graduate" else 1
    self_employed_encoded = 0 if self_employed == "No" else 1

    user_input = [[
        no_of_dependents, education_encoded, self_employed_encoded,
        income_annum, loan_amount, loan_term, cibil_score,
        residential_assets_value, commercial_assets_value,
        luxury_assets_value, bank_asset_value
    ]]

    input_df = pd.DataFrame(user_input, columns=feature_names)

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][prediction]

    # Display result
    if prediction == 1:
        st.success(f"✅ Loan Approved with confidence: {proba:.2f}")
    else:
        st.error(f"❌ Loan Rejected with confidence: {proba:.2f}")
