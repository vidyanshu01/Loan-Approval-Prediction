import joblib
import pandas as pd
from .config_loader import load_config


config = load_config()
model = joblib.load(config["artifacts"]["model_path"])
trained_features = joblib.load(config["artifacts"]["feature_columns_path"])
le = joblib.load(config["artifacts"]["encoder_path"])

def predict_loan(sample_dict):

    df = pd.DataFrame([sample_dict])
    
    # Handle engineered columns if they are not already sent in the input
    if "TotalIncome" not in df.columns:
        df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    if "EMI" not in df.columns:
        # Match whatever fallback logic you have in data_preprocessing
        df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"].replace(0, 1)
    if "BalanceIncome" not in df.columns:
        df["BalanceIncome"] = df["TotalIncome"] - (df["EMI"] * 1000) # Ensure scales match your prep script


    if "Loan_ID" in df.columns:
        df = df.drop(["Loan_ID"], axis=1)
    if "Loan_Status" in df.columns:
        df = df.drop(["Loan_Status"], axis=1)

    # Apply the exact same One-Hot Encoding transformation
    categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Critical Step: Align sample columns to match model's expected features exactly
    df = df.reindex(columns=trained_features, fill_value=0)
  
    pred_encoded = model.predict(df)[0]
    prob = model.predict_proba(df)[0][pred_encoded]

    pred_label = le.inverse_transform([pred_encoded])[0]
    
    return pred_label, prob

if __name__ == "__main__":
    sample = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban"
    }
    
    pred, prob = predict_loan(sample)
    print(f"\nPrediction Outcome: {pred}")
    print(f"Confidence Level: {prob * 100:.2f}%\n")
