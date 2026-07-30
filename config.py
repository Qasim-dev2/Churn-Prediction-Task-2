"""Centralized configuration module for the Customer Churn Prediction System.
Defines directory paths, dataset constants, random seeds, logging paths, and feature lists.
"""

from pathlib import Path

# Version Metadata
__version__ = "1.1.0"
__author__ = "Qasim"

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = BASE_DIR / "graphs"
EVALUATION_DIR = GRAPHS_DIR / "evaluation"
LOGS_DIR = BASE_DIR / "logs"

RAW_DATASET_PATH = DATASET_DIR / "customer_churn_dataset.csv"
CLEANED_DATASET_PATH = DATASET_DIR / "cleaned_customer_churn_dataset.csv"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"
PIPELINE_LOG_PATH = LOGS_DIR / "pipeline.log"

# Parameters
RANDOM_SEED = 42
NUM_RECORDS = 2000
DUPLICATE_ROWS = 10
MISSING_RATE = 0.02

# Feature Definitions
CATEGORICAL_COLUMNS = [
    "Gender",
    "City",
    "Subscription_Type",
    "Login_Frequency",
    "Contract_Length",
    "Churn_Status",
]

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
