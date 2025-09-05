# Customer Churn Prediction System

##Overview
This project predicts telecom customer churn using machine learning and provides actionable insights for retention strategies.  
Built on the **IBM Telco Churn Dataset (7k+ customers)**, the system is deployed as a real-time Streamlit app.  

## Key Features
- Predicts churn probability for a given customer profile  
- Identifies top factors driving churn (contract type, tenure, monthly charges)  
- Business insights via EDA visualizations  
- End-to-end pipeline: data preprocessing → model training → deployment  

## Tech Stack
- **Python**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **ML Models**: Random Forest, XGBoost  
- **Deployment**: Streamlit Cloud (free hosting)  

## Dataset
- Source: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- Size: ~7,000 customers, 21 features  
- Target: `Churn` (Yes/No)  

## Model Performance
- Accuracy: ~85%  
- Precision/Recall tuned to handle class imbalance  
- Top churn drivers: **Month-to-Month contracts, short tenure, high monthly charges** 
