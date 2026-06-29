from pathlib import Path

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


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "cleaned_customer_churn_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"


def evaluate_model(model, X_test_scaled, y_test):
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


def select_best_model(results_df, trained_models):
    max_f1 = results_df["F1_Score"].max()
    random_forest_row = results_df[results_df["Model"] == "Random Forest"].iloc[0]

    # Prefer Random Forest when it is practically tied because it gives native feature importances.
    if random_forest_row["F1_Score"] >= max_f1 - 0.01:
        return "Random Forest", trained_models["Random Forest"], random_forest_row["F1_Score"]

    best_row = results_df.sort_values("F1_Score", ascending=False).iloc[0]
    best_name = best_row["Model"]
    return best_name, trained_models[best_name], best_row["F1_Score"]


def train_models():
    df = pd.read_csv(DATASET_PATH)
    print("Cleaned dataset loaded successfully")
    print("Dataset shape:", df.shape)

    X = df.drop(columns=["Churn_Status"])
    y = df["Churn_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
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
        print("\nConfusion Matrix:")
        print(metrics["Confusion_Matrix"])
        print("\nClassification Report:")
        print(metrics["Classification_Report"])

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
    print("\nModel Comparison:")
    print(results_df)
    print("\nBest model name:", best_model_name)
    print("Best F1 score:", round(best_f1_score, 4))
    print("Saved model path:", MODEL_PATH.relative_to(BASE_DIR))
    print("Saved scaler path:", SCALER_PATH.relative_to(BASE_DIR))


if __name__ == "__main__":
    train_models()
