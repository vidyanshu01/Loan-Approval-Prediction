import sys
import os

# Add the root directory to the python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.append(repo_root)


import streamlit as st
import pandas as pd
import joblib
from src.config_loader import load_config

# ======================
# LOAD CONFIG & INITIAL OBJECTS
# ======================
config = load_config()

# Set up page configurations once right at the start
st.set_page_config(
    page_title=config["streamlit"]["page_title"],
    page_icon=config["streamlit"]["page_icon"],
    layout=config["streamlit"]["layout"]
)

# Load layout styling rules safely
def load_css():
    try:
        with open("app/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Skip silently if css file is missing

load_css()

# Load project binaries
model = joblib.load(config["artifacts"]["model_path"])
trained_features = joblib.load(config["artifacts"]["feature_columns_path"])
le = joblib.load(config["artifacts"]["encoder_path"])

st.title(f"🏦 {config['project']['name']}")

st.markdown("""
This application predicts whether a loan application is likely to be approved based on real-time data inputs.
""")

st.divider()

# ======================
# INPUT SELECTION SECTION
# ======================
col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    ApplicantIncome = st.number_input("Applicant Income ($)", min_value=0, value=5000)
    CoapplicantIncome = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=1, value=150)
    Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value=1, value=360)
    Credit_History = st.selectbox("Credit History Score", [1.0, 0.0])

# ======================
# INFERENCE PIPELINE PROCESSING
# ======================
if st.button("Predict Loan Status", type="primary"):

    # 1. Build dictionary matching initial CSV format structure 
    raw_input = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area
    }
    
    df = pd.DataFrame([raw_input])

    # 2. Extract engineered features exactly like training/prediction phase
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"].replace(0, 1)
    df["BalanceIncome"] = df["TotalIncome"] - (df["EMI"] * 1000)

    # 3. Handle model categorical one-hot formatting structures 
    categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 4. Reindex variables layout blueprint mapping matrix to match features exactly
    df = df.reindex(columns=trained_features, fill_value=0)

    # 5. Model execution matrix calculation processing
    pred_encoded = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    
    # Convert numerical 0 or 1 target predictions back to readable labels ('Y' / 'N')
    pred_label = le.inverse_transform([pred_encoded])[0]

    st.divider()

    # 6. Display visual UI output based on structural output maps
    if pred_label == 'Y':
        st.success(f"### ✅ Loan Approved! (Confidence: {probability[1]*100:.2f}%)")
    else:
        st.error(f"### ❌ Loan Rejected (Confidence: {probability[0]*100:.2f}%)")

    # Display clean analytics metrics presentation 
    st.subheader("📊 Prediction Probabilities Breakdowns")
    metrics_col1, metrics_col2 = st.columns(2)
    metrics_col1.metric(label="Approval Chance", value=f"{probability[1]*100:.2f}%")
    metrics_col2.metric(label="Rejection Chance", value=f"{probability[0]*100:.2f}%")
