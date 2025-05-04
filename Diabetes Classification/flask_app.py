from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load model, scaler, and feature names
try:
    model = pickle.load(open('./models/diabetes_model.pkl', 'rb'))
    scaler = pickle.load(open('./models/scaler.pkl', 'rb'))
    feature_names = pickle.load(open('./models/feature_names.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure model, scaler, and feature_names files are in the same directory.")
    exit(1)

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

# Generate feature importance plot
def generate_feature_importance_plot():
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        sns.set_theme(style='whitegrid', palette='Set2')
        sns.barplot(x='Importance', y='Feature', data=importance)
        plt.title('Feature Importance in Diabetes Prediction', fontsize=16, weight='bold')
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()
        
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return img_str
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            input_data = {
                'pregnancies': int(request.form['pregnancies']),
                'glucose': float(request.form['glucose']),
                'blood_pressure': float(request.form['blood_pressure']),
                'skin_thickness': float(request.form['skin_thickness']),
                'insulin': float(request.form['insulin']),
                'bmi': float(request.form['bmi']),
                'dpf': float(request.form['dpf']),
                'age': int(request.form['age'])
            }
            
            # Validate inputs
            if (input_data['pregnancies'] < 0 or input_data['pregnancies'] > 20 or
                input_data['glucose'] < 0 or input_data['glucose'] > 200 or
                input_data['blood_pressure'] < 0 or input_data['blood_pressure'] > 150 or
                input_data['skin_thickness'] < 0 or input_data['skin_thickness'] > 100 or
                input_data['insulin'] < 0 or input_data['insulin'] > 900 or
                input_data['bmi'] < 0 or input_data['bmi'] > 70 or
                input_data['dpf'] < 0 or input_data['dpf'] > 3 or
                input_data['age'] < 0 or input_data['age'] > 120):
                return render_template('index.html', error="Input values out of valid range.")
            
            # Predict
            prediction, probability = predict_diabetes(input_data)
            
            # Prepare result
            result = {
                'status': 'High Risk' if prediction == 1 else 'Low Risk',
                'probability': f"{probability:.2%}",
                'recommendation': ("Consult a healthcare professional for further evaluation."
                                if prediction == 1 else
                                "Maintain a healthy lifestyle to prevent future risk.")
            }
            
            # Generate feature importance plot
            feature_plot = generate_feature_importance_plot()
            
            return render_template('index.html', result=result, feature_plot=feature_plot)
        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)