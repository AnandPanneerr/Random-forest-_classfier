import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered"
)

st.title("🫀 Heart Disease Prediction")
st.write("Predict the risk of heart disease based on health details")

# ---------------- Load Model & Encoder ----------------
try:
    model = joblib.load("heart_rf_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    sex_encoder = joblib.load("sex_encoder.pkl")
    st.success("✅ Model & Encoder Loaded Successfully")
except Exception as e:
    st.error(f"❌ Model loading error: {e}")
    st.stop()

# ---------------- User Inputs ----------------
st.subheader("Enter Your Health Details")

age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
blood_pressure = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
max_heart_rate = st.number_input("Max Heart Rate", min_value=70, max_value=250, value=150)

# ---------------- Encode Categorical Input ----------------
sex_encoded = sex_encoder.transform([sex])[0]

# ---------------- Prediction ----------------
if st.button("Predict Heart Disease"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex_encoded,
        "Blood_Pressure": blood_pressure,
        "Cholesterol": cholesterol,
        "Max_Heart_Rate": max_heart_rate
    }])

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Confidence: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Confidence: {probability*100:.2f}%)")

# ---------------- Sidebar Info ----------------
st.sidebar.title("About")
st.sidebar.info(
    """
    This app predicts heart disease risk using a trained Random Forest model.
    Features used:
    - Age
    - Sex
    - Blood Pressure
    - Cholesterol
    - Max Heart Rate
    """
)
