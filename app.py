import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")

st.title("🏦 Loan Approval Predictor")
st.markdown("Enter applicant details to check if the loan will be approved.")

# --- User Input ---
no_of_dependents = st.number_input("No. of Dependents", min_value=0, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income (KES)", min_value=0, step=1000)
loan_amount = st.number_input("Loan Amount (KES)", min_value=0, step=1000)
loan_term = st.number_input("Loan Term (Months)", min_value=1, step=1)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0, step=1000)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, step=1000)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, step=1000)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0, step=1000)

# --- Convert Inputs ---
input_dict = {
    'no_of_dependents': no_of_dependents,
    'education': 1 if education == "Graduate" else 0,
    'self_employed': 1 if self_employed == "Yes" else 0,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value
}

input_df = pd.DataFrame([input_dict])

# --- Load Models ---
model = joblib.load("xgboost_loan_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Predict ---
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]
confidence = model.predict_proba(scaled_input)[0][prediction]

# --- Result Display ---
if prediction == 1:
    st.success(f"✅ Loan Approved with confidence: {confidence:.2f}")
else:
    st.error(f"❌ Loan Rejected with confidence: {confidence:.2f}")

# --- SHAP Explanation ---
st.subheader("🧠 SHAP Explanation: Why was this decision made?")
explainer = shap.Explainer(model)
shap_values = explainer(input_df)

shap_df = pd.DataFrame({
    "Feature": input_df.columns,
    "SHAP Value": shap_values.values[0]
}).sort_values(by="SHAP Value", key=abs, ascending=True)

# Color coding: green for positive, red for negative impact
colors = shap_df["SHAP Value"].apply(lambda x: 'green' if x > 0 else 'red')

# Plot SHAP values
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
ax.set_title("Feature Impact (SHAP values)")
ax.set_xlabel("Contribution to Prediction")
st.pyplot(fig)

# --- Adaptive Explanation ---
if prediction == 1:
    st.markdown(f"""
🧾 **Interpretation Guide (Approved ✅)**

- The model is **{confidence:.2%} confident** that the loan should be approved.
- ✅ Green bars represent features that **boosted** the approval decision.
- ❌ Red bars show features that had a **negative effect** but weren't strong enough to lead to rejection.
- Strong positive influence from features like good CIBIL score, assets, or reasonable loan size helped secure approval.
""")
else:
    st.markdown(f"""
🧾 **Interpretation Guide (Rejected ❌)**

- The model is **{confidence:.2%} confident** that the loan should be rejected.
- ❌ Red bars show features that **strongly pulled the prediction toward rejection** (e.g. high loan amount, low income, poor credit score).
- ✅ Green bars show features that had a **positive influence**, but not enough to overturn the negative ones.
- Consider improving the negatively impacting areas before reapplying.
""")
