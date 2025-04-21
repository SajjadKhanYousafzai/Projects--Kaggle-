from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model and scaler
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define feature names (same as training)
feature_names = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
                 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed', 
                 'temp_diff', 'humidity_cloud', 'day_sin', 'day_cos']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'day': float(request.form['day']),
            'pressure': float(request.form['pressure']),
            'maxtemp': float(request.form['maxtemp']),
            'temparature': float(request.form['temparature']),
            'mintemp': float(request.form['mintemp']),
            'dewpoint': float(request.form['dewpoint']),
            'humidity': float(request.form['humidity']),
            'cloud': float(request.form['cloud']),
            'sunshine': float(request.form['sunshine']),
            'winddirection': float(request.form['winddirection']),
            'windspeed': float(request.form['windspeed'])
        }

        # Create DataFrame
        df = pd.DataFrame([data])

        # Feature engineering
        df['temp_diff'] = df['maxtemp'] - df['mintemp']
        df['humidity_cloud'] = df['humidity'] * df['cloud']
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)

        # Ensure correct feature order
        X = df[feature_names]

        # Scale features
        X_scaled = scaler.transform(X)

        # Predict probability
        prob = model.predict_proba(X_scaled)[0, 1]
        result = f"Probability of Rainfall: {prob:.2%}"

        return render_template('index.html', prediction=result)
    except Exception as e:
        error = f"Error: {str(e)}"
        return render_template('index.html', prediction=error)

if __name__ == '__main__':
    app.run(debug=True)