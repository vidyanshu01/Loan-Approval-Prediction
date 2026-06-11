import pandas as pd
from src.utils.logger import logger
def create_features(df):
    logger.info('Creating New Feature started')
    df["TotalIncome"] = (
        df["ApplicantIncome"] +
        df["CoapplicantIncome"]
    )

    df["EMI"] = (
        df["LoanAmount"] /
        df["Loan_Amount_Term"]
    )

    df["BalanceIncome"] = (
        df["TotalIncome"] -
        (df["EMI"] * 1000)
    )
    logger.info("Feature Engineering Done")
    return df

if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/clean_loan_data.csv"
    )

    df = create_features(df)

    df.to_csv(
        "data/processed/clean_loan_data.csv",
        index=False
    )

    print("Feature Engineering Completed")