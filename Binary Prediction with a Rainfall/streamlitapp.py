import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open("../Binary Prediction with a Rainfall/rainfall_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("🌧️ Rainfall Prediction App")
st.markdown("Enter weather conditions to predict if it will rain.")

# Sidebar for user input
st.sidebar.header("Enter Weather Details")

# Input fields for numerical features
def user_input():
    pressure = st.sidebar.slider("Pressure", 900, 1100, 1013)
    maxtemp = st.sidebar.slider("Max Temperature (°C)", 10, 50, 25)
    temparature = st.sidebar.slider("Temperature (°C)", 5, 40, 20)
    mintemp = st.sidebar.slider("Min Temperature (°C)", 0, 30, 15)
    dewpoint = st.sidebar.slider("Dew Point", 0, 30, 10)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)
    cloud = st.sidebar.slider("Cloud Cover (%)", 0, 100, 50)
    sunshine = st.sidebar.slider("Sunshine Hours", 0, 15, 5)
    winddirection = st.sidebar.slider("Wind Direction (°)", 0, 360, 180)
    windspeed = st.sidebar.slider("Wind Speed (km/h)", 0, 100, 20)
    
    # Store inputs in a dataframe
    data = {
        'pressure': pressure, 'maxtemp': maxtemp, 'temparature': temparature,
        'mintemp': mintemp, 'dewpoint': dewpoint, 'humidity': humidity,
        'cloud': cloud, 'sunshine': sunshine, 'winddirection': winddirection,
        'windspeed': windspeed
    }
    
    return pd.DataFrame([data])

# Get user input
input_data = user_input()

# Show user inputs
st.write("### Weather Conditions Entered")
st.write(input_data)

# Predict button
if st.button("Predict Rainfall 🌦️"):
    prediction = model.predict(input_data)
    result = "Yes, it will rain! ☔" if prediction[0] == 1 else "No, it won't rain. ☀️"
    
    # Display prediction
    st.subheader("Prediction Result")
    st.success(result)
