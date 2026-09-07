# 🏠 House Price Prediction

A machine learning project that predicts house prices using three regression algorithms:

- Decision Tree Regressor
- Random Forest Regressor
- K-Nearest Neighbors (KNN)

## 📊 Dataset

The project uses a house-price dataset containing 4,600 records and 8 features used for prediction.

### Features

- Bedrooms
- Bathrooms
- Sqft Living
- Sqft Lot
- Floors
- Waterfront
- View
- Condition

## 🤖 Machine Learning Models

### Decision Tree
A Decision Tree learns rules from the training data to predict house prices.

### Random Forest
Random Forest combines multiple decision trees to improve prediction performance.

### KNN
KNN predicts the price based on nearby houses. Feature scaling is used because KNN relies on distances.

## 📈 Model Comparison

The models are evaluated using:

- MAE
- RMSE
- R² Score

The Streamlit application displays predictions from all three models.

## 🚀 Streamlit App

The application allows the user to enter house details and compare predictions from:

1. Decision Tree
2. Random Forest
3. KNN

## 📁 Project Structure

```text
House_Price_Prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
└── models/
    ├── decision_tree.pkl
    ├── random_forest.pkl
    ├── knn.pkl
    └── scaler.pkl
