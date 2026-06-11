import streamlit as st
import pandas as pd
import joblib

st.title("📂 Batch Loan Prediction")

# 1. Load trained artifacts and features mapping matrix
model = joblib.load("artifacts/model.pkl")
trained_features = joblib.load("artifacts/feature_columns.pkl")
le = joblib.load("artifacts/encoder.pkl")

uploaded_file = st.file_uploader(
    "Upload CSV File containing applicant features", 
    type=["csv"]
)

if uploaded_file is not None:
    # 2. Read the raw user file
    df_raw = pd.read_csv(uploaded_file)
    
    st.subheader("Uploaded Data Preview")
    st.dataframe(df_raw.head())
    
    # 3. Create a clean working copy for feature formatting
    df_processed = df_raw.copy()
    
    # Remove metadata tracking identifiers if they exist
    if "Loan_ID" in df_processed.columns:
        df_processed = df_processed.drop(["Loan_ID"], axis=1)
    if "Loan_Status" in df_processed.columns:
        df_processed = df_processed.drop(["Loan_Status"], axis=1)

    # 4. Generate identical engineered variables
    if "TotalIncome" not in df_processed.columns:
        df_processed["TotalIncome"] = df_processed["ApplicantIncome"] + df_processed["CoapplicantIncome"]
    if "EMI" not in df_processed.columns:
        df_processed["EMI"] = df_processed["LoanAmount"] / df_processed["Loan_Amount_Term"].replace(0, 1)
    if "BalanceIncome" not in df_processed.columns:
        df_processed["BalanceIncome"] = df_processed["TotalIncome"] - (df_processed["EMI"] * 1000)

    # 5. One-Hot Encode categorical strings into 0 and 1 columns
    categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    # 6. Critical Alignment: Reindex matrix to match the model training structure exactly
    df_processed = df_processed.reindex(columns=trained_features, fill_value=0)

    # 7. Execute Batch Analytics Inference
    batch_preds = model.predict(df_processed)
    batch_probs = model.predict_proba(df_processed)[:, 1] # Probability of approval (class 1)

    # 8. Decode binary integer classification targets back into clean tracking strings ('Y' or 'N')
    decoded_labels = le.inverse_transform(batch_preds)
    
    # 9. Map indicators into user-friendly UI presentation outputs
    df_raw["Prediction_Status"] = ["Approved" if x == 'Y' else "Rejected" for x in decoded_labels]
    df_raw["Approval_Probability"] = [f"{p * 100:.2f}%" for p in batch_probs]

    st.subheader("📊 Prediction Results")
    st.dataframe(df_raw)

    # 10. Direct download stream generation
    csv = df_raw.to_csv(index=False)
    st.download_button(
        label="📥 Download Enriched Predictions CSV",
        data=csv,
        file_name="loan_batch_predictions.csv",
        mime="text/csv"
    )
