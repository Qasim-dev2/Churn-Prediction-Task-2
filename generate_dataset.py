from datetime import datetime, timedelta
from pathlib import Path
import random

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DATASET_PATH = DATASET_DIR / "customer_churn_dataset.csv"

RANDOM_SEED = 42
NUM_RECORDS = 2000
DUPLICATE_ROWS = 10
MISSING_RATE = 0.02


def spending_range(subscription_type):
    ranges = {
        "Basic": (500, 1500),
        "Standard": (1500, 3000),
        "Premium": (3000, 6000),
    }
    return ranges[subscription_type]


def calculate_churn_status(
    satisfaction_score,
    support_requests,
    login_frequency,
    payment_delay,
    tenure,
    days_since_last_activity,
    contract_length,
    subscription_type,
):
    churn_score = 0

    if satisfaction_score <= 2:
        churn_score += 3
    if support_requests >= 6:
        churn_score += 2
    if login_frequency == "Low":
        churn_score += 2
    elif login_frequency == "Medium":
        churn_score += 1
    if payment_delay > 15:
        churn_score += 2
    if tenure <= 6:
        churn_score += 2
    if days_since_last_activity > 45:
        churn_score += 2
    if contract_length == "Monthly":
        churn_score += 1
    elif contract_length == "Yearly":
        churn_score -= 1
    if subscription_type == "Premium" and satisfaction_score >= 4:
        churn_score -= 1

    return "Yes" if churn_score >= 6 else "No"


def generate_dataset():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    genders = ["Male", "Female"]
    cities = [
        "Lahore",
        "Karachi",
        "Islamabad",
        "Rawalpindi",
        "Faisalabad",
        "Multan",
        "Peshawar",
        "Quetta",
    ]
    subscription_types = ["Basic", "Standard", "Premium"]
    login_frequencies = ["Low", "Medium", "High"]
    contract_lengths = ["Monthly", "Quarterly", "Yearly"]

    rows = []
    today = datetime.today()

    for index in range(1, NUM_RECORDS + 1):
        customer_id = f"CUST{index:04d}"
        age = np.random.randint(18, 66)
        gender = random.choice(genders)
        city = random.choice(cities)
        subscription_type = random.choice(subscription_types)
        tenure = np.random.randint(1, 61)

        low, high = spending_range(subscription_type)
        monthly_spending = np.random.randint(low, high + 1)

        number_of_purchases = np.random.randint(1, 81)
        support_requests = np.random.randint(0, 11)
        login_frequency = random.choice(login_frequencies)
        satisfaction_score = np.random.randint(1, 6)
        payment_delay = np.random.randint(0, 31)
        contract_length = random.choice(contract_lengths)
        days_since_last_activity = np.random.randint(1, 91)
        last_activity_date = today - timedelta(days=int(days_since_last_activity))
        total_spending = monthly_spending * tenure

        churn_status = calculate_churn_status(
            satisfaction_score=satisfaction_score,
            support_requests=support_requests,
            login_frequency=login_frequency,
            payment_delay=payment_delay,
            tenure=tenure,
            days_since_last_activity=days_since_last_activity,
            contract_length=contract_length,
            subscription_type=subscription_type,
        )

        rows.append(
            [
                customer_id,
                age,
                gender,
                city,
                subscription_type,
                monthly_spending,
                tenure,
                number_of_purchases,
                support_requests,
                login_frequency,
                last_activity_date.strftime("%Y-%m-%d"),
                satisfaction_score,
                payment_delay,
                contract_length,
                total_spending,
                days_since_last_activity,
                churn_status,
            ]
        )

    columns = [
        "Customer_ID",
        "Age",
        "Gender",
        "City",
        "Subscription_Type",
        "Monthly_Spending",
        "Tenure",
        "Number_of_Purchases",
        "Customer_Support_Requests",
        "Login_Frequency",
        "Last_Activity_Date",
        "Satisfaction_Score",
        "Payment_Delay",
        "Contract_Length",
        "Total_Spending",
        "Days_Since_Last_Activity",
        "Churn_Status",
    ]

    df = pd.DataFrame(rows, columns=columns)

    # Intentionally add missing values for cleaning practice.
    rng = np.random.default_rng(RANDOM_SEED)
    missing_count = int(NUM_RECORDS * MISSING_RATE)
    for column in ["Age", "Monthly_Spending", "Satisfaction_Score"]:
        missing_indices = rng.choice(df.index, size=missing_count, replace=False)
        df.loc[missing_indices, column] = np.nan

    duplicate_rows = df.sample(n=DUPLICATE_ROWS, random_state=RANDOM_SEED)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATASET_PATH, index=False)

    print("Dataset created successfully")
    print(f"File path: {RAW_DATASET_PATH.relative_to(BASE_DIR)}")
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    generate_dataset()
