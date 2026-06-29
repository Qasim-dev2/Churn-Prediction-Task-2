# Customer Churn Prediction System

## Objective

This project builds a machine learning system that predicts whether a customer is likely to stop using a service based on demographics, subscription details, engagement behavior, support activity, satisfaction, payment delay, and recent activity.

The project was prepared for Teyzix Core Internship Task ID ML-2.

## Problem Statement

Customer churn reduces recurring revenue and increases acquisition costs. The goal of this system is to identify high-risk customers early so the business can take retention actions such as service follow-up, targeted offers, support improvements, and yearly contract incentives.

## Features

- Synthetic customer churn dataset with 2,000 original records.
- 10 duplicate rows and around 2 percent missing values added intentionally for data cleaning.
- Data preparation pipeline for duplicates, missing values, categorical encoding, and feature ordering.
- Exploratory data analysis with customer demographics, churn distribution, spending patterns, behavior trends, and correlation analysis.
- Model comparison using Logistic Regression, Decision Tree, and Random Forest.
- Evaluation using Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix, and Classification Report.
- Streamlit prediction app with churn probability and important contributing factors.
- Generated trained model, scaler, graphs, result CSVs, README, and analysis report.

## Dataset Description

The dataset is self-created and saved at `dataset/customer_churn_dataset.csv`.

Columns:

- `Customer_ID`
- `Age`
- `Gender`
- `City`
- `Subscription_Type`
- `Monthly_Spending`
- `Tenure`
- `Number_of_Purchases`
- `Customer_Support_Requests`
- `Login_Frequency`
- `Last_Activity_Date`
- `Satisfaction_Score`
- `Payment_Delay`
- `Contract_Length`
- `Total_Spending`
- `Days_Since_Last_Activity`
- `Churn_Status`

Churn is generated using a score-based business logic. Churn risk increases for low satisfaction, many support requests, low login frequency, high payment delay, short tenure, long inactivity, and monthly contracts. Churn risk decreases for yearly contracts and satisfied premium customers.

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- Streamlit

## Folder Structure

```text
Customer-Churn-Prediction-System/
+-- dataset/
|   +-- customer_churn_dataset.csv
|   +-- cleaned_customer_churn_dataset.csv
+-- graphs/
|   +-- churn_distribution.png
|   +-- age_distribution.png
|   +-- gender_vs_churn.png
|   +-- subscription_vs_churn.png
|   +-- monthly_spending_vs_churn.png
|   +-- satisfaction_vs_churn.png
|   +-- login_frequency_vs_churn.png
|   +-- support_requests_vs_churn.png
|   +-- correlation_heatmap.png
|   +-- evaluation/
|       +-- confusion_matrix.png
|       +-- model_comparison_f1.png
|       +-- feature_importance.png
+-- models/
|   +-- churn_model.pkl
|   +-- scaler.pkl
+-- results/
|   +-- model_comparison.csv
|   +-- feature_importance.csv
+-- generate_dataset.py
+-- data_preparation.py
+-- eda_analysis.py
+-- train_model.py
+-- model_analysis.py
+-- app.py
+-- requirements.txt
+-- README.md
+-- report.md
```

## How to Install

Run this command from the main project folder:

```cmd
python -m pip install -r requirements.txt
```

## How to Run the Full Project

Run each script from the main project folder in this order:

```cmd
python generate_dataset.py
python data_preparation.py
python eda_analysis.py
python train_model.py
python model_analysis.py
```

## How to Run the Streamlit App

```cmd
streamlit run app.py
```

If the `streamlit` command does not work, use:

```cmd
python -m streamlit run app.py
```

The app loads:

- `models/churn_model.pkl`
- `models/scaler.pkl`

## Model Evaluation Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8350 | 0.8246 | 0.7966 | 0.8103 | 0.9180 |
| Decision Tree | 0.9000 | 0.8703 | 0.9096 | 0.8895 | 0.9010 |
| Random Forest | 0.9300 | 0.9515 | 0.8870 | 0.9181 | 0.9798 |

Best model: Random Forest

The Random Forest model was selected because it produced the strongest F1 Score and ROC-AUC while also supporting feature importance analysis.

## Screenshots

Add screenshots of the Streamlit interface and generated graphs here before final submission:

- Home screen of Streamlit app
- Prediction result for a high-risk customer
- Prediction result for a low-risk customer
- EDA graphs from the `graphs/` folder
- Evaluation graphs from `graphs/evaluation/`

## Common Errors

If `joblib` is missing:

```cmd
python -m pip install joblib
```

If multiple packages are missing:

```cmd
python -m pip install -r requirements.txt
```

If Streamlit is missing:

```cmd
python -m pip install streamlit
```

If `streamlit run app.py` is not recognized:

```cmd
python -m streamlit run app.py
```

If model files are missing, run:

```cmd
python train_model.py
```

## GitHub Submission Notes

Before uploading to GitHub:

- Confirm `app.py` is in the main project folder.
- Confirm model files are inside `models/`.
- Confirm generated graphs are inside `graphs/`.
- Confirm result CSV files are inside `results/`.
- Keep `README.md`, `report.md`, and `requirements.txt` in the main folder.

CMD commands to push:

```cmd
git init
git add .
git commit -m "Complete customer churn prediction system"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Customer-Churn-Prediction-System.git
git push -u origin main
```
