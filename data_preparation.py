from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DATASET_PATH = DATASET_DIR / "customer_churn_dataset.csv"
CLEANED_DATASET_PATH = DATASET_DIR / "cleaned_customer_churn_dataset.csv"

FEATURE_ORDER = [
    "Age",
    "Gender",
    "City",
    "Subscription_Type",
    "Monthly_Spending",
    "Tenure",
    "Number_of_Purchases",
    "Customer_Support_Requests",
    "Login_Frequency",
    "Satisfaction_Score",
    "Payment_Delay",
    "Contract_Length",
    "Total_Spending",
    "Days_Since_Last_Activity",
    "Churn_Status",
]


def prepare_data():
    df = pd.read_csv(RAW_DATASET_PATH)

    print("Shape before cleaning:", df.shape)
    print("\nMissing values before cleaning:")
    print(df.isna().sum())

    df = df.drop_duplicates()
    print("\nShape after duplicate removal:", df.shape)

    for column in ["Age", "Monthly_Spending", "Satisfaction_Score"]:
        df[column] = df[column].fillna(df[column].median())

    print("\nMissing values after filling:")
    print(df.isna().sum())

    df["Last_Activity_Date"] = pd.to_datetime(df["Last_Activity_Date"], errors="coerce")
    df = df.drop(columns=["Last_Activity_Date", "Customer_ID"])

    categorical_columns = [
        "Gender",
        "City",
        "Subscription_Type",
        "Login_Frequency",
        "Contract_Length",
        "Churn_Status",
    ]

    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        mapping = {label: int(value) for label, value in zip(encoder.classes_, encoder.transform(encoder.classes_))}
        print(f"{column} encoding:", mapping)

    df = df[FEATURE_ORDER]

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DATASET_PATH, index=False)

    print("\nFinal shape:", df.shape)
    print(f"Cleaned dataset saved at: {CLEANED_DATASET_PATH.relative_to(BASE_DIR)}")
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    prepare_data()
