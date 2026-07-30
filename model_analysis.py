"""Model analysis and evaluation plotting module.
Generates confusion matrix heatmaps, feature importance charts, model comparison plots,
and ROC curve visualizations for the trained churn prediction model.
"""

from pathlib import Path
from typing import Union

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

from config import (
    BASE_DIR,
    CLEANED_DATASET_PATH,
    EVALUATION_DIR,
    FEATURE_IMPORTANCE_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    SCALER_PATH,
)


def save_plot(path: Union[str, Path], dpi: int = 150) -> None:
    """Save plot cleanly with layout adjustments."""
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path: Union[str, Path]) -> None:
    """Generate and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    save_plot(save_path)


def analyze_model() -> None:
    """Load model, compute test set predictions, and export evaluation graphs."""
    if not (MODEL_PATH.exists() and SCALER_PATH.exists() and CLEANED_DATASET_PATH.exists()):
        raise FileNotFoundError("Required model, scaler, or cleaned dataset file missing.")

    df = pd.read_csv(CLEANED_DATASET_PATH)
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
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Confusion Matrix Plot
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

    # Model Comparison Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x="Model", y="F1_Score", hue="Model", palette="Set2", legend=False)
    plt.title("Model Comparison Based on F1 Score")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    save_plot(EVALUATION_DIR / "model_comparison_f1.png")

    # ROC Curve Plot
    if y_prob is not None:
        plot_roc_curve(y_test, y_prob, EVALUATION_DIR / "roc_curve.png")

    # Feature Importance
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

    print("\nModel analysis completed successfully")
    print(f"Evaluation graphs saved in: {EVALUATION_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    analyze_model()
