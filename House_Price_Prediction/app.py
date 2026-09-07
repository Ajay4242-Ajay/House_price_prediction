import streamlit as st
import pandas as pd
import joblib
dt_model = joblib.load("models/decision_tree.pkl")
rf_model = joblib.load("models/random_forest.pkl")
knn_model = joblib.load("models/knn.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("🏠 House Price Prediction")

st.write("Enter the details of the house:")

bedrooms = st.number_input("Bedrooms", min_value=0.0, value=3.0)
bathrooms = st.number_input("Bathrooms", min_value=0.0, value=2.0)
sqft_living = st.number_input("Living Area (sqft)", min_value=0, value=1500)
sqft_lot = st.number_input("Lot Area (sqft)", min_value=0, value=5000)
floors = st.number_input("Floors", min_value=0.0, value=1.0)
waterfront = st.number_input("Waterfront (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
view = st.number_input("View", min_value=0, max_value=4, value=0)
condition = st.number_input("Condition", min_value=1, max_value=5, value=3)
input_data = pd.DataFrame({
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "sqft_living": [sqft_living],
    "sqft_lot": [sqft_lot],
    "floors": [floors],
    "waterfront": [waterfront],
    "view": [view],
    "condition": [condition]
})
if st.button("Predict Price"):

    dt_prediction = dt_model.predict(input_data)[0]

    rf_prediction = rf_model.predict(input_data)[0]

    input_scaled = scaler.transform(input_data)
    knn_prediction = knn_model.predict(input_scaled)[0]

    st.subheader("Predicted Prices")

    st.write("Decision Tree:", round(dt_prediction, 2))
    st.write("Random Forest:", round(rf_prediction, 2))
    st.write("KNN:", round(knn_prediction, 2))