import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model (save your model first using joblib)
model = joblib.load("solar_model.pkl")

st.title("🌞 Solar Power Prediction App")

# Inputs
windspeed = st.number_input("Wind Speed")
sunshine = st.number_input("Sunshine")
airpressure = st.number_input("Air Pressure")
radiation = st.number_input("Radiation")
airtemperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")

# Dummy time features (or compute automatically)
daysin = 0.5
daycos = 0.8
yearsin = 0.2
yearcos = 0.9

# Predict button
if st.button("Predict"):
    input_data = np.array([[windspeed, sunshine, airpressure, radiation,
                            airtemperature, humidity,
                            daysin, daycos, yearsin, yearcos]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Solar Power: {prediction[0]}")