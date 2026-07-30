"""Data preparation and cleaning pipeline for the Customer Churn Prediction System.
Handles duplicate removal, missing value imputation, datetime parsing, categorical encoding,
and saving the final cleaned dataset.
"""

from typing import Dict, List
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import (
    BASE_DIR,
    CLEANED_DATASET_PATH,
    DATASET_DIR,
    FEATURE_ORDER,
    RAW_DATASET_PATH,
    CATEGORICAL_COLUMNS,
)


def load_raw_dataset(path: str = RAW_DATASET_PATH) -> pd.DataFrame:
    """Load the raw dataset from CSV."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {path}")
    return pd.read_csv(path)


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numerical values with column medians."""
    df_clean = df.copy()
    for column in ["Age", "Monthly_Spending", "Satisfaction_Score"]:
        if column in df_clean.columns and df_clean[column].isna().sum() > 0:
            median_val = df_clean[column].median()
            df_clean[column] = df_clean[column].fillna(median_val)
    return df_clean


def encode_categorical_features(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, int]]:
    """Encode categorical features using LabelEncoder and return encoding mappings."""
    mappings = {}
    for column in columns:
        if column in df.columns:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])
            mapping = {str(label): int(val) for label, val in zip(encoder.classes_, encoder.transform(encoder.classes_))}
            mappings[column] = mapping
            print(f"{column} encoding:", mapping)
    return mappings


def prepare_data() -> None:
    """Run the complete data preparation pipeline."""
    df = pd.read_csv(RAW_DATASET_PATH)

    print("Shape before cleaning:", df.shape)
    print("\nMissing values before cleaning:")
    print(df.isna().sum())

    df = df.drop_duplicates()
    print("\nShape after duplicate removal:", df.shape)

    df = clean_missing_values(df)

    print("\nMissing values after filling:")
    print(df.isna().sum())

    if "Last_Activity_Date" in df.columns:
        df["Last_Activity_Date"] = pd.to_datetime(df["Last_Activity_Date"], errors="coerce")

    cols_to_drop = [c for c in ["Last_Activity_Date", "Customer_ID"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    encode_categorical_features(df, CATEGORICAL_COLUMNS)

    df = df[FEATURE_ORDER]

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DATASET_PATH, index=False)

    print("\nFinal shape:", df.shape)
    print(f"Cleaned dataset saved at: {CLEANED_DATASET_PATH.relative_to(BASE_DIR)}")
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    from pathlib import Path
    prepare_data()
