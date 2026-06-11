# 🏦 AI-Powered Loan Approval Prediction System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green.svg)]()

## 📌 Overview

The Loan Approval Prediction System is an end-to-end Machine Learning project designed to predict whether a loan application will be approved based on an applicant's financial and demographic information.

This project demonstrates the complete ML lifecycle:

- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Model Evaluation
- Batch Prediction
- Streamlit Dashboard
- Cloud Deployment

The system helps financial institutions automate loan approval decisions, reduce manual effort, and improve decision-making efficiency.

---

# 🎯 Business Problem

Banks receive thousands of loan applications every day.

Manual verification involves:

- Checking applicant income
- Verifying credit history
- Evaluating repayment capability
- Assessing risk

This process is:

❌ Time-consuming

❌ Expensive

❌ Prone to human errors

Machine Learning can automate this process by learning patterns from historical loan data.

---

# 🚀 Project Features

### Machine Learning Features

✅ Data Preprocessing

✅ Missing Value Treatment

✅ Feature Engineering

✅ Label Encoding

✅ XGBoost Classification

✅ Model Evaluation

✅ Feature Importance Analysis

---

### Streamlit Features

✅ Single Applicant Prediction

✅ Batch Prediction Using CSV Upload

✅ Data Insights Dashboard

✅ Model Performance Dashboard

✅ Download Prediction Results

---

### Software Engineering Features

✅ Config-Based Project

✅ Virtual Environment Support

✅ Logging

✅ Unit Testing

✅ Modular Code Structure

✅ GitHub Ready

---

# 📂 Dataset Information

This project uses the Loan Prediction Dataset.

### Dataset Download

Dataset Source:

📥 Kaggle Dataset

🔗 https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

After downloading, place files inside:

```text
data/raw/
│
├── train.csv
└── test.csv
```

---

# 📊 Dataset Files

## train.csv

Contains:

- Features
- Target Variable (Loan_Status)

Used for:

- Training
- Validation
- Model Evaluation

---

## test.csv

Contains:

- Features Only
- No Loan_Status Column

Used for:

- Batch Prediction
- Kaggle Submission
- Production Testing

---

# 📋 Dataset Features

| Feature | Description |
|----------|-------------|
| Loan_ID | Unique Loan Identifier |
| Gender | Applicant Gender |
| Married | Marital Status |
| Dependents | Number of Dependents |
| Education | Education Level |
| Self_Employed | Self Employment Status |
| ApplicantIncome | Applicant Monthly Income |
| CoapplicantIncome | Co-applicant Income |
| LoanAmount | Requested Loan Amount |
| Loan_Amount_Term | Loan Repayment Term |
| Credit_History | Credit Score Indicator |
| Property_Area | Urban / Semiurban / Rural |
| Loan_Status | Target Variable |

---

# 🎯 Target Variable

| Value | Meaning |
|---------|---------|
| Y | Loan Approved |
| N | Loan Rejected |

---

# 🏗 Project Architecture

```text
loan-approval-prediction/
│
├── venv/
│
├── app/
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── 1_Data_Insights.py
│   │   ├── 2_Model_Performance.py
│   │   └── 3_Batch_Prediction.py
│
├── artifacts/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb
│
├── src/
│   ├── config_loader.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── batch_predict.py
│
├── tests/
│   ├── test_prediction.py
│   ├── test_model_loading.py
│   └── test_preprocessing.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/loan-approval-prediction.git

cd loan-approval-prediction
```

---

# 🐍 Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

---

Linux/Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📜 Requirements

```text
streamlit
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
plotly
joblib
pyyaml
pytest
```

Generate automatically:

```bash
pip freeze > requirements.txt
```

---

# 🔄 Workflow

Step 1

Preprocess Data

```bash
python src/data_preprocessing.py
```

---

Step 2

Feature Engineering

```bash
python src/feature_engineering.py
```

---

Step 3

Train Model

```bash
python src/train_model.py
```

Output:

```text
artifacts/model.pkl
```

---

Step 4

Evaluate Model

```bash
python src/evaluate_model.py
```

---

Step 5

Run Batch Prediction

```bash
python src/batch_predict.py
```

Output:

```text
submission.csv
```

---

Step 6

Run Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

Application URL:

```text
http://localhost:8501
```

---

# 🤖 Machine Learning Models

Models Evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

Final Model:

### XGBoost Classifier

Reasons:

- High Accuracy
- Fast Training
- Better Generalization
- Handles Non-linear Relationships

---

# 📈 Expected Performance

| Metric | Score |
|----------|---------|
| Accuracy | 87–92% |
| Precision | 86–90% |
| Recall | 85–89% |
| F1 Score | 86–90% |
| ROC-AUC | 90%+ |

---

# 📊 Feature Engineering

Created New Features:

### TotalIncome

```python
ApplicantIncome + CoapplicantIncome
```

### IncomePerPerson

```python
TotalIncome / Dependents
```

### LoanIncomeRatio

```python
LoanAmount / TotalIncome
```

These engineered features significantly improve model performance.

---

# 📁 Batch Prediction

Upload:

```text
test.csv
```

Predict:

```text
Approved
Rejected
Approved
...
```

Download:

```text
loan_predictions.csv
```

---

# 🧪 Testing

Run All Tests

```bash
pytest
```

Run Specific Test

```bash
pytest tests/test_prediction.py
```

Expected:

```text
4 passed
```

---

# ☁ Streamlit Cloud Deployment

## Step 1

Push Project to GitHub

```bash
git init
git add .
git commit -m "Initial Commit"
git push
```

---

## Step 2

Open

https://share.streamlit.io

---

## Step 3

Connect GitHub Repository

---

## Step 4

Select

```text
Repository:
loan-approval-prediction

Branch:
main

Main File:
app/streamlit_app.py
```

---

## Step 5

Deploy

Application will be live in a few minutes.

---

# ✅ Advantages

### For Banks

- Faster Approval Process
- Reduced Manual Work
- Consistent Decision Making
- Lower Operational Costs

### For Data Science Portfolio

- End-to-End ML Lifecycle
- Production-Level Structure
- Streamlit Deployment
- Recruiter-Friendly Project

---

# ❌ Limitations

- Dataset is relatively small.
- Real-world banking data is more complex.
- No live credit bureau integration.
- No fraud detection module.
- Model performance depends on historical data quality.

---

# 🔮 Future Enhancements

- SHAP Explainability
- MLflow Tracking
- Docker Support
- CI/CD Pipeline
- FastAPI Backend
- AWS Deployment
- Real-Time Loan Scoring
- Fraud Detection System

---

# 👨‍💻 Author

Vidyanshu Kushawaha

Aspiring Data Scientist | Machine Learning Engineer

GitHub:
https://github.com/vidyanshu01

LinkedIn:
www.linkedin.com/in/vidyanshu-kushawaha-551a6a2a7

Email:
vidy0nshu@gmail.com

---

# ⭐ If you found this project useful, please consider giving it a star.
