import streamlit as st
import pickle
import numpy as np
import os

# Load the model
model_path = "D:/My Projects/Projects/Heart Dieases Project/heart_disease_model.pkl"

# Initialize model variable
model = None

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found. Please check that the file path is correct and the file exists.")
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")

# Streamlit app interface
st.title("Heart Disease Prediction App")

if model is not None:  # Check if model is loaded
    # Form for user inputs
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    sex = st.selectbox("Sex", ("Male", "Female"))
    cp = st.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"))
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, step=1)
    chol = st.number_input("Serum Cholesterol in mg/dl", min_value=100, max_value=400, step=1)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"))
    restecg = st.number_input("Resting Electrocardiographic Results", min_value=0, max_value=2, step=1)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, step=1)
    exang = st.selectbox("Exercise Induced Angina", ("Yes", "No"))
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, step=0.1)
    slope = st.number_input("Slope of Peak Exercise ST Segment", min_value=0, max_value=2, step=1)
    ca = st.number_input("Number of Major Vessels (0-3) Colored by Fluoroscopy", min_value=0, max_value=3, step=1)
    thal = st.number_input("Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)", min_value=0, max_value=3, step=1)

    # Convert categorical inputs to numerical values
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    # Map chest pain type to numerical values
    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-Anginal Pain": 2,
        "Asymptomatic": 3
    }
    cp = cp_map[cp]

    # Create feature array
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Predict button
    if st.button("Predict"):
        try:
            # Predict using the loaded model
            prediction = model.predict(features)
            output = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
            
            # Display result
            st.success(f'Result: {output}')
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
else:
    st.error("Model is not loaded. Please check the model file.")
