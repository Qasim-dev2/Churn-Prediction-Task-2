"""Exploratory Data Analysis (EDA) module for Customer Churn dataset.
Generates distribution plots, feature comparisons against churn, and correlation heatmaps.
"""

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import (
    BASE_DIR,
    CLEANED_DATASET_PATH,
    GRAPHS_DIR,
    RAW_DATASET_PATH,
)


def save_plot(path: Union[str, Path], dpi: int = 150) -> None:
    """Save current Matplotlib figure cleanly with layout adjustments."""
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_countplot(
    df: pd.DataFrame,
    x: str,
    hue: str = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Number of Customers",
    save_path: Union[str, Path] = None,
) -> None:
    """Helper function to create and save Seaborn count plots."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=x, hue=hue)
    plt.title(title)
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel)
    if save_path:
        save_plot(save_path)


def run_eda() -> None:
    """Execute complete EDA pipeline and output visualization charts."""
    df = pd.read_csv(RAW_DATASET_PATH)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset Info:")
    df.info()

    print("\nSummary Statistics:")
    print(df.describe(include="all"))

    print("\nChurn Count:")
    print(df["Churn_Status"].value_counts())

    sns.set_theme(style="whitegrid", palette="Set2")

    # Churn Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn_Status", order=["No", "Yes"])
    plt.title("Churn Distribution")
    plt.xlabel("Churn Status")
    plt.ylabel("Number of Customers")
    save_plot(GRAPHS_DIR / "churn_distribution.png")

    # Age Distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(df["Age"], bins=20, kde=True, color="#4C78A8")
    plt.title("Age Distribution of Customers")
    plt.xlabel("Age")
    plt.ylabel("Count")
    save_plot(GRAPHS_DIR / "age_distribution.png")

    # Gender vs Churn
    plot_countplot(
        df,
        x="Gender",
        hue="Churn_Status",
        title="Gender vs Churn",
        save_path=GRAPHS_DIR / "gender_vs_churn.png",
    )

    # Subscription Type vs Churn
    plot_countplot(
        df,
        x="Subscription_Type",
        hue="Churn_Status",
        title="Subscription Type vs Churn",
        save_path=GRAPHS_DIR / "subscription_vs_churn.png",
    )

    # Monthly Spending vs Churn
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn_Status", y="Monthly_Spending", order=["No", "Yes"])
    plt.title("Monthly Spending vs Churn")
    plt.xlabel("Churn Status")
    plt.ylabel("Monthly Spending")
    save_plot(GRAPHS_DIR / "monthly_spending_vs_churn.png")

    # Satisfaction vs Churn
    plot_countplot(
        df,
        x="Satisfaction_Score",
        hue="Churn_Status",
        title="Satisfaction Score vs Churn",
        save_path=GRAPHS_DIR / "satisfaction_vs_churn.png",
    )

    # Login Frequency vs Churn
    plot_countplot(
        df,
        x="Login_Frequency",
        hue="Churn_Status",
        title="Login Frequency vs Churn",
        save_path=GRAPHS_DIR / "login_frequency_vs_churn.png",
    )

    # Support Requests vs Churn
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn_Status", y="Customer_Support_Requests", order=["No", "Yes"])
    plt.title("Support Requests vs Churn")
    plt.xlabel("Churn Status")
    plt.ylabel("Customer Support Requests")
    save_plot(GRAPHS_DIR / "support_requests_vs_churn.png")

    # Correlation Heatmap
    if CLEANED_DATASET_PATH.exists():
        cleaned_df = pd.read_csv(CLEANED_DATASET_PATH)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cleaned_df.corr(), annot=True, cmap="vlag", fmt=".2f", linewidths=0.4)
        plt.title("Feature Correlation Heatmap")
        save_plot(GRAPHS_DIR / "correlation_heatmap.png")

    print("\nEDA completed successfully")
    print(f"All graphs saved in: {GRAPHS_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    run_eda()
