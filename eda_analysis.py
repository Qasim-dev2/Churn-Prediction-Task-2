from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
GRAPHS_DIR = BASE_DIR / "graphs"
RAW_DATASET_PATH = DATASET_DIR / "customer_churn_dataset.csv"
CLEANED_DATASET_PATH = DATASET_DIR / "cleaned_customer_churn_dataset.csv"


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def run_eda():
    df = pd.read_csv(RAW_DATASET_PATH)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset Info:")
    df.info()

    print("\nSummary Statistics:")
    print(df.describe(include="all"))

    print("\nChurn Count:")
    print(df["Churn_Status"].value_counts())

    sns.set_theme(style="whitegrid", palette="Set2")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn_Status", order=["No", "Yes"])
    plt.title("Churn Distribution")
    plt.xlabel("Churn Status")
    plt.ylabel("Number of Customers")
    save_plot(GRAPHS_DIR / "churn_distribution.png")

    plt.figure(figsize=(7, 4))
    sns.histplot(df["Age"], bins=20, kde=True, color="#4C78A8")
    plt.title("Age Distribution of Customers")
    plt.xlabel("Age")
    plt.ylabel("Count")
    save_plot(GRAPHS_DIR / "age_distribution.png")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Gender", hue="Churn_Status")
    plt.title("Gender vs Churn")
    plt.xlabel("Gender")
    plt.ylabel("Number of Customers")
    save_plot(GRAPHS_DIR / "gender_vs_churn.png")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Subscription_Type", hue="Churn_Status")
    plt.title("Subscription Type vs Churn")
    plt.xlabel("Subscription Type")
    plt.ylabel("Number of Customers")
    save_plot(GRAPHS_DIR / "subscription_vs_churn.png")

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn_Status", y="Monthly_Spending", order=["No", "Yes"])
    plt.title("Monthly Spending vs Churn")
    plt.xlabel("Churn Status")
    plt.ylabel("Monthly Spending")
    save_plot(GRAPHS_DIR / "monthly_spending_vs_churn.png")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Satisfaction_Score", hue="Churn_Status")
    plt.title("Satisfaction Score vs Churn")
    plt.xlabel("Satisfaction Score")
    plt.ylabel("Number of Customers")
    save_plot(GRAPHS_DIR / "satisfaction_vs_churn.png")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Login_Frequency", hue="Churn_Status")
    plt.title("Login Frequency vs Churn")
    plt.xlabel("Login Frequency")
    plt.ylabel("Number of Customers")
    save_plot(GRAPHS_DIR / "login_frequency_vs_churn.png")

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn_Status", y="Customer_Support_Requests", order=["No", "Yes"])
    plt.title("Support Requests vs Churn")
    plt.xlabel("Churn Status")
    plt.ylabel("Customer Support Requests")
    save_plot(GRAPHS_DIR / "support_requests_vs_churn.png")

    cleaned_df = pd.read_csv(CLEANED_DATASET_PATH)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cleaned_df.corr(), annot=True, cmap="vlag", fmt=".2f", linewidths=0.4)
    plt.title("Feature Correlation Heatmap")
    save_plot(GRAPHS_DIR / "correlation_heatmap.png")

    print("\nEDA completed successfully")
    print(f"All graphs saved in: {GRAPHS_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    run_eda()
