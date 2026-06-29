from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "cleaned_customer_churn_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
MODEL_COMPARISON_PATH = BASE_DIR / "results" / "model_comparison.csv"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "results" / "feature_importance.csv"
EVALUATION_DIR = BASE_DIR / "graphs" / "evaluation"


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_model():
    df = pd.read_csv(DATASET_PATH)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    results_df = pd.read_csv(MODEL_COMPARISON_PATH)

    print("Dataset, model, scaler, and comparison results loaded successfully")

    X = df.drop(columns=["Churn_Status"])
    y = df["Churn_Status"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot(EVALUATION_DIR / "confusion_matrix.png")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x="Model", y="F1_Score", hue="Model", palette="Set2", legend=False)
    plt.title("Model Comparison Based on F1 Score")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    save_plot(EVALUATION_DIR / "model_comparison_f1.png")

    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame(
            {
                "Feature": X.columns,
                "Importance": model.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)

        feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=feature_importance,
            x="Importance",
            y="Feature",
            hue="Feature",
            palette="viridis",
            legend=False,
        )
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        save_plot(EVALUATION_DIR / "feature_importance.png")

        print("\nFeature Importance:")
        print(feature_importance)
        print(f"\nFeature importance saved at: {FEATURE_IMPORTANCE_PATH.relative_to(BASE_DIR)}")
    else:
        print("\nSelected best model does not support feature_importances_.")
        print("Feature importance file and plot were not created for this model.")

    print("\nModel analysis completed successfully")
    print(f"Evaluation graphs saved in: {EVALUATION_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    analyze_model()
