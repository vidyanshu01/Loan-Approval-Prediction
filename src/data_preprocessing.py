import pandas as pd
from sklearn.impute import SimpleImputer
from src.utils.logger import logger

def load_data(path):
    logger.info("Raw Data load Successfully")
    return pd.read_csv(path)

def preprocess_data(df):
    logger.info("Preprocessing Started")
    cat_cols = [
        'Gender',
        'Married',
        'Dependents',
        'Self_Employed',
        'Education',
        'Property_Area'
    ]

    num_cols = [
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Credit_History'
    ]
    logger.info("Handling Null value")
    print("Total null value in each column: Before Simple imputer")
    print(df.isnull().sum())   
    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = SimpleImputer(strategy='median')

    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    print("Total null value in each column: After Simple imputer")
    print(df.isnull().sum())    
    logger.info("Preprocessing Completed")
    return df

if __name__ == "__main__":

    df = load_data("data/raw/train.csv")

    df = preprocess_data(df)

    df.to_csv(
        "data/processed/clean_loan_data.csv",
        index=False
    )

    print("Processed data saved.")