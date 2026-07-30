"""Unit tests for the Customer Churn Prediction System pipeline.
Tests configuration paths, dataset generation heuristics, data cleaning routines,
logger instantiation, and ChurnPredictor risk factor evaluation.
"""

import unittest
from pathlib import Path
import pandas as pd

from config import (
    BASE_DIR,
    CATEGORICAL_COLUMNS,
    FEATURE_ORDER,
)
from generate_dataset import calculate_churn_status, spending_range
from data_preparation import clean_missing_values
from logger import get_logger
from predictor import ChurnPredictor


class TestConfig(unittest.TestCase):
    """Test suite for configuration constants."""

    def test_paths_exist_or_valid(self):
        self.assertIsInstance(BASE_DIR, Path)

    def test_feature_order_length(self):
        self.assertEqual(len(FEATURE_ORDER), 15)

    def test_categorical_columns(self):
        self.assertIn("Gender", CATEGORICAL_COLUMNS)
        self.assertIn("Churn_Status", CATEGORICAL_COLUMNS)


class TestDatasetGeneration(unittest.TestCase):
    """Test suite for dataset generation logic."""

    def test_spending_range(self):
        low, high = spending_range("Basic")
        self.assertEqual(low, 500)
        self.assertEqual(high, 1500)

        low_p, high_p = spending_range("Premium")
        self.assertEqual(low_p, 3000)
        self.assertEqual(high_p, 6000)

    def test_calculate_churn_status_high_risk(self):
        status = calculate_churn_status(
            satisfaction_score=1,
            support_requests=8,
            login_frequency="Low",
            payment_delay=20,
            tenure=2,
            days_since_last_activity=60,
            contract_length="Monthly",
            subscription_type="Basic",
        )
        self.assertEqual(status, "Yes")

    def test_calculate_churn_status_low_risk(self):
        status = calculate_churn_status(
            satisfaction_score=5,
            support_requests=0,
            login_frequency="High",
            payment_delay=0,
            tenure=40,
            days_since_last_activity=5,
            contract_length="Yearly",
            subscription_type="Premium",
        )
        self.assertEqual(status, "No")


class TestDataPreparation(unittest.TestCase):
    """Test suite for data cleaning functions."""

    def test_clean_missing_values(self):
        df = pd.DataFrame(
            {
                "Age": [20, 30, None, 40],
                "Monthly_Spending": [1000, None, 2000, 3000],
                "Satisfaction_Score": [1, 5, 3, None],
            }
        )
        cleaned_df = clean_missing_values(df)
        self.assertEqual(cleaned_df["Age"].isna().sum(), 0)
        self.assertEqual(cleaned_df["Monthly_Spending"].isna().sum(), 0)
        self.assertEqual(cleaned_df["Satisfaction_Score"].isna().sum(), 0)


class TestLoggerAndPredictor(unittest.TestCase):
    """Test suite for logger and predictor module utilities."""

    def test_logger_instantiation(self):
        log = get_logger("test_logger")
        self.assertEqual(log.name, "test_logger")

    def test_predictor_risk_signals(self):
        predictor = ChurnPredictor.__new__(ChurnPredictor)
        customer = {
            "Satisfaction_Score": 1,
            "Customer_Support_Requests": 7,
            "Login_Frequency": "Low",
            "Payment_Delay": 20,
        }
        signals = predictor.evaluate_risk_signals(customer)
        self.assertIn("Low satisfaction score", signals)
        self.assertIn("High number of customer support requests", signals)


if __name__ == "__main__":
    unittest.main()
