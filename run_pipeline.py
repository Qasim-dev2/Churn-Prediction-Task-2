"""Master pipeline orchestration script for Customer Churn Prediction System.
Runs data generation, preparation, EDA analysis, model training, and evaluation in sequence.
"""

from config import PIPELINE_LOG_PATH
from logger import get_logger
from generate_dataset import generate_dataset
from data_preparation import prepare_data
from eda_analysis import run_eda
from train_model import train_models
from model_analysis import analyze_model

logger = get_logger("run_pipeline", log_file=PIPELINE_LOG_PATH)


def run_full_pipeline() -> None:
    """Execute all pipeline steps sequentially with file and console logging."""
    logger.info("Starting Customer Churn Prediction System pipeline...")

    logger.info("Step 1: Generating synthetic dataset...")
    generate_dataset()

    logger.info("Step 2: Cleaning and preparing dataset...")
    prepare_data()

    logger.info("Step 3: Running Exploratory Data Analysis (EDA)...")
    run_eda()

    logger.info("Step 4: Training machine learning models...")
    train_models()

    logger.info("Step 5: Analyzing best model performance...")
    analyze_model()

    logger.info("Pipeline execution completed successfully! Log saved to %s", PIPELINE_LOG_PATH)


if __name__ == "__main__":
    run_full_pipeline()
