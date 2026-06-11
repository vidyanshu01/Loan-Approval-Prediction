import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import logger
from .config_loader import load_config

# 1. Load configuration file
config = load_config()
logger.info("Training Started")

# 2. Read processed dataset
df = pd.read_csv(config["data"]["processed_data_path"])

# 3. Separate features and target
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)

# 4. Map or Encode Target Variable (Y/N -> 1/0)
le = LabelEncoder()
y = le.fit_transform(df["Loan_Status"])

# Save label encoder for runtime inference later
os.makedirs(os.path.dirname(config["artifacts"]["encoder_path"]), exist_ok=True)
joblib.dump(le, config["artifacts"]["encoder_path"])
logger.info("Label encoder saved successfully")

# 5. One-Hot Encode all categorical text columns into 0 and 1 numerical flags
categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Save feature columns template so your API/Streamlit app knows the final training structure
joblib.dump(list(X.columns), config["artifacts"]["feature_columns_path"])

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["training"]["test_size"],
    random_state=config["training"]["random_state"]
)

# 7. Extract XGBoost nested configurations safely
xgb_config = config["models"]["xgboost"]

# 8. Initialize XGBClassifier with numeric-only fields
model = XGBClassifier(
    n_estimators=xgb_config["n_estimators"],
    learning_rate=xgb_config["learning_rate"],
    max_depth=xgb_config["max_depth"],
    subsample=xgb_config["subsample"],
    colsample_bytree=xgb_config["colsample_bytree"],
    random_state=config["training"]["random_state"]
)

# 9. Fit model on strictly numeric training vectors
model.fit(X_train, y_train)

# 10. Persist Model Binary Artifacts safely
os.makedirs(os.path.dirname(config["artifacts"]["model_path"]), exist_ok=True)
joblib.dump(model, config["artifacts"]["model_path"])

print("Model Saved Successfully")
logger.info("Model Saved Successfully")
