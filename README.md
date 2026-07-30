# Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/Library-scikit--learn-orange.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Objective

This project builds an end-to-end machine learning system that predicts whether a customer is likely to stop using a service based on demographics, subscription details, engagement behavior, support activity, satisfaction score, payment delay, and recent activity.

Prepared for Teyzix Core Internship Task ID **ML-2**.

## Problem Statement

Customer churn reduces recurring revenue and increases acquisition costs. The goal of this system is to identify high-risk customers early so the business can take proactive retention actions such as service follow-ups, targeted promotional offers, support improvements, and yearly contract incentives.

## System Features

- **Dataset Generation**: Synthetic customer churn dataset with 2,000 original records (`generate_dataset.py`).
- **Data Cleaning & Pipeline**: Handles duplicate removal, missing value imputation, datetime parsing, and categorical label encoding (`data_preparation.py`).
- **Exploratory Data Analysis**: Generates comprehensive plots covering customer demographics, churn distribution, spending patterns, behavior trends, and correlation heatmaps (`eda_analysis.py`).
- **Model Comparison**: Benchmarks Logistic Regression, Decision Tree, and Random Forest models using 80/20 stratified split (`train_model.py`).
- **Performance Metrics**: Calculates Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix, and Classification Report.
- **Advanced Visualizations**: Produces feature importance charts, model comparison bar graphs, and Receiver Operating Characteristic (ROC) curves (`model_analysis.py`).
- **Pipeline Orchestrator**: Single command automated execution script (`run_pipeline.py`).
- **Inference Wrapper**: Encapsulates model scaling and risk signal breakdown (`predictor.py`).
- **Structured Logging**: Preconfigured timestamped console/file logger (`logger.py`).
- **Unit Testing Suite**: Verified component functionality and risk heuristic logic via `test_pipeline.py`.
- **Interactive Web App**: Streamlit prediction dashboard with real-time risk indicators and contributing risk factor breakdowns (`app.py`).

## Folder Structure

```text
Customer-Churn-Prediction-System/
├── config.py
├── logger.py
├── predictor.py
├── generate_dataset.py
├── data_preparation.py
├── eda_analysis.py
├── train_model.py
├── model_analysis.py
├── run_pipeline.py
├── test_pipeline.py
├── app.py
├── requirements.txt
├── README.md
├── report.md
├── dataset/
│   ├── customer_churn_dataset.csv
│   └── cleaned_customer_churn_dataset.csv
├── graphs/
│   ├── churn_distribution.png
│   ├── age_distribution.png
│   ├── gender_vs_churn.png
│   ├── subscription_vs_churn.png
│   ├── monthly_spending_vs_churn.png
│   ├── satisfaction_vs_churn.png
│   ├── login_frequency_vs_churn.png
│   ├── support_requests_vs_churn.png
│   ├── correlation_heatmap.png
│   └── evaluation/
│       ├── confusion_matrix.png
│       ├── model_comparison_f1.png
│       ├── feature_importance.png
│       └── roc_curve.png
├── models/
│   ├── churn_model.pkl
│   └── scaler.pkl
└── results/
    ├── model_comparison.csv
    └── feature_importance.csv
```

## How to Install

```cmd
python -m pip install -r requirements.txt
```

## Running the Pipeline

To execute the entire pipeline with a single master script:

```cmd
python run_pipeline.py
```

Or execute scripts individually:

```cmd
python generate_dataset.py
python data_preparation.py
python eda_analysis.py
python train_model.py
python model_analysis.py
```

## Running Unit Tests

```cmd
python test_pipeline.py
```

## How to Run the Streamlit Dashboard

```cmd
streamlit run app.py
```

If the `streamlit` command is not bound to PATH:

```cmd
python -m streamlit run app.py
```

## Model Evaluation Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8350 | 0.8246 | 0.7966 | 0.8103 | 0.9180 |
| Decision Tree | 0.9000 | 0.8703 | 0.9096 | 0.8895 | 0.9010 |
| **Random Forest** | **0.9300** | **0.9515** | **0.8870** | **0.9181** | **0.9798** |

**Selected Model**: Random Forest (Highest F1 Score & ROC-AUC with native feature importances).
