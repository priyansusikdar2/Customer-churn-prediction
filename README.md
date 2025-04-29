# Customer Churn Prediction Project

## 🔍 Objective
This project aims to build a machine learning model that predicts whether a customer will discontinue a subscription-based service (i.e., churn). The goal is to assist the business in proactively identifying at-risk customers and taking retention measures.

---

## 📊 Dataset Overview

- Source: Provided by the organization (`Churn_Modelling.csv`)
- Size: 10,000 customer records
- Key Features:
  - **Demographics**: `Geography`, `Gender`, `Age`
  - **Account Information**: `Tenure`, `Balance`, `CreditScore`, `NumOfProducts`
  - **Customer Activity**: `IsActiveMember`, `HasCrCard`, `EstimatedSalary`
  - **Target Variable**: `Exited` (1 = churned, 0 = retained)

---

## 🧪 Project Workflow

### 1. **Data Preprocessing**
- Removed irrelevant columns: `RowNumber`, `CustomerId`, `Surname`
- Handled categorical variables:
  - Encoded `Gender` with Label Encoding
  - One-hot encoded `Geography` (drop-first strategy)
- Applied `StandardScaler` for feature normalization

### 2. **Model Development**
- Model Used: `RandomForestClassifier` (100 estimators)
- Train/test split: 80/20
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

### 3. **Insights & Interpretability**
- Feature importance extracted and visualized to identify key drivers of churn

---

## 📈 Results

The Random Forest model demonstrated strong performance and successfully classified churn cases. Important features affecting churn include:

- `Age`
- `Balance`
- `NumOfProducts`
- `IsActiveMember`
- `CreditScore`

---

## 📦 Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

Install dependencies using:
```bash
pip install pandas scikit-learn matplotlib seaborn

## Output Image
![image](https://github.com/user-attachments/assets/b19a661b-2430-4f90-a34b-91beaf886c04)
