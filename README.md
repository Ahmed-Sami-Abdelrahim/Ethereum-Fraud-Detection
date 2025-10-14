# Ethereum Fraud Detection

💳 **Ethereum Fraud Detection** is a project designed to detect suspicious Ethereum transactions using AI.

<img width="1917" height="849" alt="png1" src="https://github.com/user-attachments/assets/7e28b9cc-aaad-40dd-ac6f-ca415c205db6" />

---

## 📌 Project Overview

Fraud in cryptocurrency transactions is rare but critical. This project aims to:

- Detect suspicious Ethereum transactions.
- Provide a **fraud probability score** for each transaction.
- Offer an **interactive dashboard** to explore transaction risks.

The dataset is **imbalanced** (more legitimate transactions than frauds), so special care was taken to ensure the model captures fraudulent transactions effectively.  

**Dataset:** [Ethereum Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)

---

## 🛠️ Features & Steps

### 1. Data Cleaning & Preprocessing
- Handled negative balances and extreme outliers.
- Standardized numeric features for proper slider ranges in the dashboard.
- Processed categorical features using Label Encoding.
- Handled inconsistencies in categorical data.

### 2. Exploratory Data Analysis (EDA) & Visualization
- **Univariate Analysis:** inspected numeric distributions and categorical value counts.
- **Bivariate Analysis:** examined relationships between features and the fraud flag.
- Created **clear visualizations** for numeric and categorical features to gain insights.

### 3. Modeling & Interactive Dashboard
- **Models tried**:
  - Decision Tree Classifier
  - Random Forest Classifier
  - **XGBoost Classifier** (best performing)
- **Hyperparameter tuning** using **GridSearchCV**.
- **Evaluation** on test data with low threshold (0.01) to capture rare frauds:

### 4. **Interactive Dashboard (Streamlit)**:
- Users can adjust sliders for transaction features.
- Shows **fraud probability (%)** and risk levels:
  - Low Risk 🟢
  - Moderate Risk 🟡
  - High Risk 🟠
  - Very High Risk 🔴
- Blue-themed interface with readable fonts and clear layout.

## 🚀 Key Achievements
- Built an **Ethereum Fraud Detection** system using XGBoost with high accuracy on an imbalanced dataset.  
- Implemented **data preprocessing** including outlier handling, negative value filtering, and feature encoding.  
- Performed **EDA**: univariate & bivariate analysis, and used visualizations to understand feature behavior.  
- Used **GridSearchCV** to tune hyperparameters for optimal model performance.  
- Developed an **interactive Streamlit dashboard** to predict fraud probability with risk levels.

## 💻 Technologies & Libraries Used
- **Data Manipulation & Analysis**: `pandas`, `numpy`  
- **Visualization**: `matplotlib`, `seaborn`  
- **Machine Learning & Modeling**: `scikit-learn` (`RandomForestClassifier`, `DecisionTreeClassifier`, `GridSearchCV`), `XGBoost`  
- **Model Persistence**: `joblib`  
- **Web App / Dashboard**: `Streamlit`

## 🌐 Live Demo
- Interactive dashboard deployed on Streamlit Community Cloud: [Live App Link](https://ethereum-fraud-detection.streamlit.app/)



