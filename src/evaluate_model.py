import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.utils.logger import logger
from .config_loader import load_config

# 1. Load configuration
config = load_config()
logger.info("Evaluation Started")

# 2. Load the trained model and the list of trained feature columns
model = joblib.load(config["artifacts"]["model_path"])
trained_features = joblib.load(config["artifacts"]["feature_columns_path"])

# 3. Read and preprocess the data exactly like the training script
df = pd.read_csv(config["data"]["processed_data_path"])
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)

# Encode Target
le = joblib.load(config["artifacts"]["encoder_path"])
y = le.transform(df["Loan_Status"])

# One-Hot Encode features
categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Align columns to match training exactly (adds missing columns as 0, removes extras)
X = X.reindex(columns=trained_features, fill_value=0)

# 4. Generate the exact same Test Split using identical parameters
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["training"]["test_size"],
    random_state=config["training"]["random_state"]
)

# 5. Generate predictions using the loaded model
pred = model.predict(X_test)

# 6. Print your performance metrics
print("\n=== Classification Report ===")
print(classification_report(y_test, pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, pred))

print("=== ROC AUC Score ===")
print(f"{roc_auc_score(y_test, pred):.4f}\n")

logger.info("Evaluation Completed Successfully")
