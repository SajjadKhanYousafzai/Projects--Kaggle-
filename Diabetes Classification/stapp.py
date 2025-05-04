import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Load model, scaler, and feature names
@st.cache_resource
def load_resources():
    try:
        model = pickle.load(open('./models/diabetes_model.pkl', 'rb'))
        scaler = pickle.load(open('./models/scaler.pkl', 'rb'))
        feature_names = pickle.load(open('./models/feature_names.pkl', 'rb'))
        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Ensure model, scaler, and feature_names files are in the same directory.")
        st.stop()

model, scaler, feature_names = load_resources()

# App layout
st.title("🩺 Diabetes Prediction System")
st.markdown("Enter patient details to predict diabetes risk. This tool uses a machine learning model trained on clinical data.")

# Input form
st.subheader("Patient Information")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, help="Number of times pregnant")
    glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=200.0, value=100.0, help="Plasma dextrose concentration")
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=150.0, value=70.0, help="Diastolic blood pressure")
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, help="Triceps skin fold thickness")

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=100.0, help="2-Hour serum insulin")
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, help="Body mass index")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, help="Diabetes pedigree function")
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, help="Patient's age")

# Prediction logic
def predict_diabetes(data):
    input_data = pd.DataFrame({
        'Pregnancies': [data['pregnancies']],
        'Glucose': [data['glucose']],
        'BloodPressure': [data['blood_pressure']],
        'SkinThickness': [data['skin_thickness']],
        'Insulin': [data['insulin']],
        'BMI': [data['bmi']],
        'DiabetesPedigreeFunction': [data['dpf']],
        'Age': [data['age']],
        'Insulin_Log': [np.log1p(data['insulin'])],
        'DPF_Log': [np.log1p(data['dpf'])],
        'BMI_Category_Normal': [1 if 18.5 <= data['bmi'] < 25 else 0],
        'BMI_Category_Overweight': [1 if 25 <= data['bmi'] < 30 else 0],
        'BMI_Category_Obese': [1 if data['bmi'] >= 30 else 0],
        'Glucose_Level_Prediabetes': [1 if 100 <= data['glucose'] < 126 else 0],
        'Glucose_Level_Diabetes': [1 if data['glucose'] >= 126 else 0],
        'Age_Group_Middle': [1 if 30 <= data['age'] < 45 else 0],
        'Age_Group_Senior': [1 if 45 <= data['age'] < 60 else 0],
        'Age_Group_Elderly': [1 if data['age'] >= 60 else 0]
    })
    
    # Ensure all required features are present
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0
    
    input_data = input_data[feature_names]
    
    # Scale and predict
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]
    
    return prediction, probability

# Predict button
if st.button("Predict Diabetes Risk"):
    input_data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age
    }
    
    prediction, probability = predict_diabetes(input_data)
    
    # Display results
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk: The patient is likely to have diabetes (Probability: {probability:.2%})")
        st.markdown("**Recommendation:** Consult a healthcare professional for further evaluation.")
    else:
        st.success(f"✅ Low Risk: The patient is unlikely to have diabetes (Probability: {probability:.2%})")
        st.markdown("**Recommendation:** Maintain a healthy lifestyle to prevent future risk.")
    
    # Display feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance)
        plt.title('Feature Importance in Diabetes Prediction')
        plt.tight_layout()
        st.pyplot(fig)

# Add footer
st.markdown("---")
st.markdown("Developed by Sajjad Ali Shah | Powered by Streamlit | Model trained on Kaggle Diabetes Dataset")