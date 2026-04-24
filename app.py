import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

# ================================
# Load model & scaler
# ================================
model = keras.models.load_model("best_model.keras")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check if it is fraudulent.")

# ================================
# Input Fields
# ================================

time = st.number_input("Time", value=0.0)
amount = st.number_input("Amount", value=0.0)

st.subheader("PCA Features (V1 - V28)")
features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    features.append(val)

# ================================
# Prediction
# ================================
if st.button("Predict"):

    # Create input DataFrame
    input_data = [time, amount] + features
    columns = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    
    input_df = pd.DataFrame([input_data], columns=columns)

    # 🔥 Apply same scaling used in training
    input_df[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])

    # Predict probability
    prob = model.predict(input_df)[0][0]

    # Decision threshold
    if prob > 0.5:
        st.error(f"⚠️ Fraudulent Transaction (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Genuine Transaction (Probability: {prob:.2f})")

    st.write(f"Prediction Probability: {prob:.4f}")
