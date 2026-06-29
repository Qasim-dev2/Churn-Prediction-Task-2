# Customer Churn Prediction System - Analysis Report

## Dataset Description

The project uses a self-created synthetic customer churn dataset with 2,000 original customer records. The raw dataset also includes 10 intentionally duplicated rows, giving 2,010 rows before cleaning.

The dataset contains customer demographics, subscription details, spending behavior, purchase activity, support usage, engagement frequency, payment delay, contract length, last activity information, and churn status.

Important columns include:

- `Age`, `Gender`, and `City` for demographic analysis.
- `Subscription_Type`, `Monthly_Spending`, `Tenure`, and `Contract_Length` for subscription behavior.
- `Number_of_Purchases`, `Login_Frequency`, and `Days_Since_Last_Activity` for customer engagement.
- `Customer_Support_Requests`, `Satisfaction_Score`, and `Payment_Delay` for service experience and risk analysis.
- `Churn_Status` as the target variable.

The target column uses `Yes` for customers who churn and `No` for customers who do not churn.

## Data Preparation Process

The data preparation script loads `dataset/customer_churn_dataset.csv` and performs the required cleaning operations.

Steps completed:

- Checked dataset shape before cleaning: 2,010 rows and 17 columns.
- Checked missing values before cleaning.
- Removed duplicate records, reducing the dataset to 2,000 rows.
- Filled missing values in `Age`, `Monthly_Spending`, and `Satisfaction_Score` using median values.
- Converted `Last_Activity_Date` to datetime.
- Dropped `Last_Activity_Date` because `Days_Since_Last_Activity` already captures the same useful modeling signal.
- Encoded categorical columns using LabelEncoder.
- Dropped `Customer_ID` because it is an identifier and not a predictive feature.
- Saved the cleaned dataset as `dataset/cleaned_customer_churn_dataset.csv`.

The final cleaned dataset has 2,000 rows and 15 columns, including the target.

## Feature Engineering

The dataset includes engineered behavior and value features:

- `Total_Spending` is calculated as `Monthly_Spending * Tenure`.
- `Days_Since_Last_Activity` measures recent engagement.
- `Payment_Delay` captures billing risk.
- `Contract_Length` captures customer commitment level.

The feature order used for model training and Streamlit prediction is:

```text
Age
Gender
City
Subscription_Type
Monthly_Spending
Tenure
Number_of_Purchases
Customer_Support_Requests
Login_Frequency
Satisfaction_Score
Payment_Delay
Contract_Length
Total_Spending
Days_Since_Last_Activity
```

## Exploratory Data Analysis

EDA was performed on the original dataset and generated visual outputs in the `graphs/` folder.

Generated graphs:

- `graphs/churn_distribution.png`
- `graphs/age_distribution.png`
- `graphs/gender_vs_churn.png`
- `graphs/subscription_vs_churn.png`
- `graphs/monthly_spending_vs_churn.png`
- `graphs/satisfaction_vs_churn.png`
- `graphs/login_frequency_vs_churn.png`
- `graphs/support_requests_vs_churn.png`
- `graphs/correlation_heatmap.png`

The raw churn distribution was:

- No churn: 1,120 records
- Churn: 890 records

This distribution is reasonably balanced for training and evaluation.

## Model Selection

Three machine learning models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest

The dataset was split using:

- Test size: 0.2
- Random state: 42
- Stratified split based on `Churn_Status`

StandardScaler was fitted on the training set and used to transform both training and test features.

## Evaluation Results

The models were evaluated using Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix, and Classification Report.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8350 | 0.8246 | 0.7966 | 0.8103 | 0.9180 |
| Decision Tree | 0.9000 | 0.8703 | 0.9096 | 0.8895 | 0.9010 |
| Random Forest | 0.9300 | 0.9515 | 0.8870 | 0.9181 | 0.9798 |

Random Forest confusion matrix:

```text
[[215   8]
 [ 20 157]]
```

This means the model correctly identified 215 non-churn customers and 157 churn customers in the test set.

## Best Model

The best model is Random Forest.

Random Forest achieved:

- Accuracy: 0.9300
- Precision: 0.9515
- Recall: 0.8870
- F1 Score: 0.9181
- ROC-AUC: 0.9798

It was selected because it had the highest F1 Score and strongest ROC-AUC performance. It also supports feature importance, which helps explain the most important churn drivers.

## Feature Importance

The most important features from the final Random Forest model were:

| Feature | Importance |
|---|---:|
| Satisfaction_Score | 0.2617 |
| Days_Since_Last_Activity | 0.1188 |
| Payment_Delay | 0.1087 |
| Customer_Support_Requests | 0.0997 |
| Contract_Length | 0.0643 |

These results show that churn is most strongly affected by satisfaction, inactivity, payment delays, support problems, and contract commitment.

## Key Findings

- Customers with low satisfaction scores are much more likely to churn.
- Customers inactive for a long time show higher churn risk.
- High payment delays are a strong warning signal.
- Customers with many support requests are at greater risk.
- Monthly contract customers are more flexible and may churn more easily.
- Yearly contracts and satisfied premium customers are associated with lower churn risk.

## Business Recommendations

- Target customers with low satisfaction scores using follow-up surveys and service recovery calls.
- Improve support response for customers with many support requests.
- Send offers, reminders, or reactivation campaigns to inactive users.
- Encourage yearly contracts through discounts, loyalty rewards, or bundled benefits.
- Monitor customers with high payment delays and provide payment reminders or flexible options.
- Create retention campaigns for high-risk customers identified by the model.
- Track satisfaction and login behavior continuously because they are strong churn indicators.

## Conclusion

The Customer Churn Prediction System successfully creates a realistic dataset, prepares the data, performs exploratory analysis, trains multiple machine learning models, evaluates performance, and provides a working Streamlit prediction interface.

The final Random Forest model performs strongly, with an F1 Score of 0.9181 and ROC-AUC of 0.9798. The project is ready for internship submission and GitHub upload.
