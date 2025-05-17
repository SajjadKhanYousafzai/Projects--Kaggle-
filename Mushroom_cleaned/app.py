from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load models and scaler
def load_models():
    try:
        best_model = joblib.load('./models/random_forest_model.joblib')
        lr_model = joblib.load('./models/logistic_regression_model.joblib')
        scaler = joblib.load('./models/standard_scaler1.joblib')
        feature_names = joblib.load('./models/feature_names.joblib')
        if 'class' in feature_names:
            feature_names.remove('class')
        return {
            'best_model': best_model,
            'lr_model': lr_model,
            'scaler': scaler,
            'feature_names': feature_names
        }
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

# Load dataset
def load_data():
    try:
        df = pd.read_csv("./dataset/mushroom_cleaned.csv")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Prediction function
def predict_mushroom(input_data, models_dict):
    try:
        feature_names = models_dict['feature_names']
        input_df = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Fill numerical values
        for col in ['cap-diameter', 'stem-height', 'stem-width']:
            if col in input_data:
                input_df[col] = float(input_data[col])
        
        # Handle categorical features
        categorical_cols = ['cap-shape', 'gill-attachment', 'gill-color', 'stem-color', 'season']
        for col in categorical_cols:
            if col in input_data:
                col_name = f"{col}_{input_data[col]}"
                if col_name in input_df.columns:
                    input_df[col_name] = 1
        
        # Scale input
        input_scaled = models_dict['scaler'].transform(input_df)
        
        # Predict with both models
        prediction = models_dict['best_model'].predict(input_scaled)
        probability = models_dict['best_model'].predict_proba(input_scaled)
        lr_prediction = models_dict['lr_model'].predict(input_scaled)
        lr_probability = models_dict['lr_model'].predict_proba(input_scaled)
        
        return {
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist(),
            'lr_prediction': int(lr_prediction[0]),
            'lr_probability': lr_probability[0].tolist()
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# Function to convert matplotlib figure to base64 string
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

# Load models and data at startup
models_dict = load_models()
df = load_data()

@app.route('/')
def index():
    if models_dict is None or df is None:
        return "Error: Failed to load models or data.", 500
    # Get unique values for form dropdowns
    form_data = {
        'cap_shapes': sorted(df['cap-shape'].unique().tolist()),
        'gill_attachments': sorted(df['gill-attachment'].unique().tolist()),
        'gill_colors': sorted(df['gill-color'].unique().tolist()),
        'stem_colors': sorted(df['stem-color'].unique().tolist()),
        'seasons': sorted(df['season'].unique().tolist()),
        'cap_diameter': {
            'min': float(df['cap-diameter'].min()),
            'max': float(df['cap-diameter'].max()),
            'value': float(df['cap-diameter'].median())
        },
        'stem_height': {
            'min': float(df['stem-height'].min()),
            'max': float(df['stem-height'].max()),
            'value': float(df['stem-height'].median())
        },
        'stem_width': {
            'min': float(df['stem-width'].min()),
            'max': float(df['stem-width'].max()),
            'value': float(df['stem-width'].median())
        }
    }
    return render_template('index.html', form_data=form_data)

@app.route('/predict', methods=['POST'])
def predict():
    if models_dict is None:
        return jsonify({'error': 'Models not loaded'}), 500
    input_data = request.form.to_dict()
    result = predict_mushroom(input_data, models_dict)
    if result is None:
        return jsonify({'error': 'Prediction failed'}), 500
    
    # Generate probability gauge charts
    fig_rf, ax_rf = plt.subplots(figsize=(4, 0.5))
    ax_rf.barh([''], [result['probability'][1]], color='green', alpha=0.6)
    ax_rf.barh([''], [1], color='lightgray', alpha=0.2)
    ax_rf.set_xlim(0, 1)
    ax_rf.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax_rf.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax_rf.set_yticks([])
    ax_rf.axvline(x=result['probability'][1], color='black', linestyle='-', linewidth=2)
    ax_rf.set_title('RF: Probability Edible')
    rf_gauge = fig_to_base64(fig_rf)
    
    fig_lr, ax_lr = plt.subplots(figsize=(4, 0.5))
    ax_lr.barh([''], [result['lr_probability'][1]], color='green', alpha=0.6)
    ax_lr.barh([''], [1], color='lightgray', alpha=0.2)
    ax_lr.set_xlim(0, 1)
    ax_lr.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax_lr.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax_lr.set_yticks([])
    ax_lr.axvline(x=result['lr_probability'][1], color='black', linestyle='-', linewidth=2)
    ax_lr.set_title('LR: Probability Edible')
    lr_gauge = fig_to_base64(fig_lr)
    
    return jsonify({
        'rf_prediction': 'Edible' if result['prediction'] == 1 else 'Poisonous',
        'rf_confidence': result['probability'][result['prediction']],
        'rf_gauge': rf_gauge,
        'lr_prediction': 'Edible' if result['lr_prediction'] == 1 else 'Poisonous',
        'lr_confidence': result['lr_probability'][result['lr_prediction']],
        'lr_gauge': lr_gauge
    })

@app.route('/visualizations')
def visualizations():
    visualizations = {}
    
    # Class Distribution
    try:
        with open('class_distribution.png', 'rb') as f:
            img = base64.b64encode(f.read()).decode('utf-8')
            visualizations['class_distribution'] = img
    except:
        fig, ax = plt.subplots(figsize=(10, 6))
        class_counts = df['class'].value_counts()
        sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
        ax.set_title('Distribution of Mushroom Classes')
        ax.set_xlabel('Class (0: Poisonous, 1: Edible)')
        ax.set_ylabel('Count')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Poisonous', 'Edible'])
        for i, v in enumerate(class_counts):
            ax.text(i, v + 50, f'{v} ({v/len(df):.1%})', ha='center')
        visualizations['class_distribution'] = fig_to_base64(fig)
    
    # Other visualizations (placeholders if not found)
    for viz in ['numerical_distributions', 'numerical_boxplots_by_class', 
                'correlation_heatmap', 'model_accuracy_comparison',
                'confusion_matrix_random_forest', 'confusion_matrix_logistic_regression']:
        try:
            with open(f'{viz}.png', 'rb') as f:
                img = base64.b64encode(f.read()).decode('utf-8')
                visualizations[viz] = img
        except:
            visualizations[viz] = None
    
    return render_template('visualizations.html', visualizations=visualizations)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)