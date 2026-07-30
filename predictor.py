"""Inference helper module for the Customer Churn Prediction System.
Encapsulates model loading, input validation, feature scaling, risk scoring, and batch predictions.
"""

from typing import Dict, List, Any, Tuple
import joblib
import pandas as pd

from config import FEATURE_ORDER, MODEL_PATH, SCALER_PATH


class ChurnPredictor:
    """Predictor class for loading saved models and evaluating customer churn probability."""

    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained model and scaler artifacts from disk."""
        if not self.model_path.exists() or not self.scaler_path.exists():
            raise FileNotFoundError(f"Model or scaler artifact not found in {self.model_path.parent}")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def predict_single(self, feature_df: pd.DataFrame) -> Tuple[int, float]:
        """Predict churn status (0 or 1) and churn probability for a single customer DataFrame."""
        if list(feature_df.columns) != FEATURE_ORDER:
            feature_df = feature_df[FEATURE_ORDER]

        scaled_data = self.scaler.transform(feature_df)
        prediction = int(self.model.predict(scaled_data)[0])
        probability = float(self.model.predict_proba(scaled_data)[0][1])

        return prediction, probability

    def evaluate_risk_signals(self, customer: Dict[str, Any]) -> List[str]:
        """Analyze raw customer record dictionary and return list of risk factor descriptions."""
        risk_signals = []

        if customer.get("Satisfaction_Score", 5) <= 2:
            risk_signals.append("Low satisfaction score")
        if customer.get("Customer_Support_Requests", 0) >= 6:
            risk_signals.append("High number of customer support requests")
        if customer.get("Login_Frequency") == "Low":
            risk_signals.append("Low login frequency")
        if customer.get("Payment_Delay", 0) > 15:
            risk_signals.append("High payment delay")
        if customer.get("Tenure", 12) <= 6:
            risk_signals.append("Low customer tenure")
        if customer.get("Days_Since_Last_Activity", 0) > 45:
            risk_signals.append("Customer inactive for extended period")
        if customer.get("Contract_Length") == "Monthly":
            risk_signals.append("Monthly contract subscription")

        return risk_signals
