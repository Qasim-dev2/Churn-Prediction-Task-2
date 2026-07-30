"""Model training and evaluation module for the Customer Churn Prediction System.
Trains Logistic Regression, Decision Tree, and Random Forest classifiers, evaluates metrics,
selects the best model based on F1-score/ROC-AUC, and persists serialized model and scaler objects.
"""

from typing import Dict, Tuple, Any
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import (
    BASE_DIR,
    CLEANED_DATASET_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SCALER_PATH,
)


def evaluate_model(model: Any, X_test_scaled: Any, y_test: Any) -> Dict[str, Any]:
    """Evaluate trained classifier and return comprehensive metrics dictionary."""
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "Confusion_Matrix": confusion_matrix(y_test, y_pred),
        "Classification_Report": classification_report(y_test, y_pred, zero_division=0),
    }


def select_best_model(
    results_df: pd.DataFrame, trained_models: Dict[str, Any]
) -> Tuple[str, Any, float]:
    """Select optimal model prioritizing Random Forest when metric scores are close."""
    max_f1 = results_df["F1_Score"].max()
    rf_rows = results_df[results_df["Model"] == "Random Forest"]

    if not rf_rows.empty:
        random_forest_row = rf_rows.iloc[0]
        if random_forest_row["F1_Score"] >= max_f1 - 0.01:
            return "Random Forest", trained_models["Random Forest"], float(random_forest_row["F1_Score"])

    best_row = results_df.sort_values("F1_Score", ascending=False).iloc[0]
    best_name = str(best_row["Model"])
    return best_name, trained_models[best_name], float(best_row["F1_Score"])


def train_models() -> None:
    """Load cleaned dataset, split, scale features, train models, and persist best model."""
    if not CLEANED_DATASET_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATASET_PATH}. Run data_preparation.py first.")

    df = pd.read_csv(CLEANED_DATASET_PATH)
    print("Cleaned dataset loaded successfully. Shape:", df.shape)

    X = df.drop(columns=["Churn_Status"])
    y = df["Churn_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }

    results = []
    trained_models = {}

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(f"Training Model: {model_name}")
        print("=" * 60)

        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model
        metrics = evaluate_model(model, X_test_scaled, y_test)

        print("Accuracy:", round(metrics["Accuracy"], 4))
        print("Precision:", round(metrics["Precision"], 4))
        print("Recall:", round(metrics["Recall"], 4))
        print("F1 Score:", round(metrics["F1_Score"], 4))
        print("ROC-AUC:", round(metrics["ROC_AUC"], 4))
        print("\nConfusion Matrix:\n", metrics["Confusion_Matrix"])
        print("\nClassification Report:\n", metrics["Classification_Report"])

        results.append(
            {
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1_Score": metrics["F1_Score"],
                "ROC_AUC": metrics["ROC_AUC"],
            }
        )

    results_df = pd.DataFrame(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(MODEL_COMPARISON_PATH, index=False)

    best_model_name, best_model, best_f1_score = select_best_model(results_df, trained_models)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\nModel comparison saved at:", MODEL_COMPARISON_PATH.relative_to(BASE_DIR))
    print("\nModel Comparison Table:")
    print(results_df)
    print("\nBest model name:", best_model_name)
    print("Best F1 score:", round(best_f1_score, 4))
    print("Saved model path:", MODEL_PATH.relative_to(BASE_DIR))
    print("Saved scaler path:", SCALER_PATH.relative_to(BASE_DIR))


if __name__ == "__main__":
    train_models()
