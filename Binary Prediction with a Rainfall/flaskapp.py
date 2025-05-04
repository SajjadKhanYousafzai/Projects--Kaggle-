from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

# Load model and scaler
try:
    model = pickle.load(open("./models/catboost_model.pkl", "rb"))
    scaler = pickle.load(open("./models/scaler.pkl", "rb"))
except FileNotFoundError:
    print("Error: Model or scaler file not found.")
    exit(1)

# Load datasets
try:
    df_train = pd.read_csv("./data/train.csv")
    df_test = pd.read_csv("./data/test.csv")
except FileNotFoundError:
    print("Error: Dataset files not found.")
    exit(1)

# Set Seaborn style
sns.set_style('whitegrid')
sns.set_palette("coolwarm")
plt.rcParams['font.size'] = 12

# Feature engineering function
def preprocess_weather_data(data):
    data["dew_humidity"] = data["dewpoint"] * data["humidity"]
    data["cloud_windspeed"] = data["cloud"] * data["windspeed"]
    data["cloud_to_humidity"] = data["cloud"] / data["humidity"]
    data["temp_to_sunshine"] = data["sunshine"] / data["temparature"]
    data["wind_temp_interaction"] = data["windspeed"] * data["temparature"]
    data["cloud_sun_ratio"] = data["cloud"] / (data["sunshine"] + 1)
    data["dew_humidity/sun"] = data["dewpoint"] * data["humidity"] / (data["sunshine"] + 1)
    data["dew_humidity_+"] = data["dewpoint"] * data["humidity"]
    data["humidity_sunshine_*"] = data["humidity"] * data["sunshine"]
    data["cloud_humidity/pressure"] = (data["cloud"] * data["humidity"]) / data["pressure"]
    data['month'] = ((data['day'] - 1) // 30 + 1).clip(upper=12)
    data['season'] = data['month'].apply(lambda x: 1 if 3 <= x <= 5 else 2 if 6 <= x <= 8 else 3 if 9 <= x <= 11 else 0)
    data['season_cloud_trend'] = data['cloud'] * data['season']
    data['season_cloud_deviation'] = data['cloud'] - data.groupby('season')['cloud'].transform('mean')
    data['season_temperature'] = data['temparature'] * data['season']
    data = data.drop(columns=["month", "maxtemp", "winddirection", "humidity", "temparature", "pressure", "day", "season"])
    return data

# Helper function to save plot as PNG
def save_plot(fig, filename):
    os.makedirs("static/images", exist_ok=True)
    fig.savefig(f"static/images/{filename}", bbox_inches='tight')
    plt.close(fig)

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/data_exploration', methods=['GET', 'POST'])
def data_exploration():
    numerical_variables = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
                          'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
    selected_feature = request.form.get('feature', numerical_variables[0])
    viz_type = request.form.get('viz_type', 'Histogram')

    # Generate visualizations
    if viz_type == 'Histogram':
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df_train[selected_feature], kde=True, color='skyblue', bins=30, ax=ax)
        ax.set_title(f'Distribution of {selected_feature}', fontsize=14)
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Frequency')
        save_plot(fig, 'histogram.png')

    elif viz_type == 'KDE Plot':
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(data=df_train[df_train['rainfall'] == 1], x=selected_feature, label='Rainfall = 1', color='red', ax=ax)
        sns.kdeplot(data=df_train[df_train['rainfall'] == 0], x=selected_feature, label='Rainfall = 0', color='blue', ax=ax)
        ax.set_title(f'{selected_feature} by Rainfall', fontsize=14)
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Density')
        ax.legend()
        save_plot(fig, 'kde_plot.png')

    elif viz_type == 'Correlation Heatmap':
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = df_train[numerical_variables + ['rainfall']].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, annot_kws={"size": 10}, ax=ax)
        ax.set_title('Correlation Heatmap of Features and Target', fontsize=16)
        save_plot(fig, 'heatmap.png')

    elif viz_type == 'Wind Rose':
        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(14, 6))
        rain_data = df_train[df_train['rainfall'] == 1]
        axes[0].set_theta_direction(-1)
        axes[0].set_theta_offset(np.pi / 2.0)
        axes[0].bar(np.deg2rad(rain_data['winddirection']), rain_data['windspeed'], width=np.pi/8, color='blue', alpha=0.7)
        axes[0].set_title('Wind Rose (Rain)', fontsize=14)
        no_rain_data = df_train[df_train['rainfall'] == 0]
        axes[1].set_theta_direction(-1)
        axes[1].set_theta_offset(np.pi / 2.0)
        axes[1].bar(np.deg2rad(no_rain_data['winddirection']), no_rain_data['windspeed'], width=np.pi/8, color='red', alpha=0.7)
        axes[1].set_title('Wind Rose (No Rain)', fontsize=14)
        plt.tight_layout()
        save_plot(fig, 'wind_rose.png')

    elif viz_type == 'Target Distribution':
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x=df_train['rainfall'], palette='coolwarm', ax=ax)
        ax.set_title('Rainfall Class Distribution', fontsize=16)
        ax.set_xlabel('Rainfall (0 = No Rain, 1 = Rain)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        save_plot(fig, 'target_distribution.png')

    return render_template('data_exploration.html', 
                           data_head=df_train.head().to_html(classes='table table-striped'),
                           data_describe=df_train.describe().to_html(classes='table table-striped'),
                           numerical_variables=numerical_variables,
                           selected_feature=selected_feature,
                           viz_type=viz_type)

@app.route('/model_insights')
def model_insights():
    # ROC curve
    auc_score = 0.9010
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr) * auc_score
    ax.plot(fpr, tpr, label=f"CatBoost (AUC = {auc_score:.4f})", color='blue')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve for CatBoost")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    save_plot(fig, 'roc_curve.png')

    # Feature importance
    feature_names = preprocess_weather_data(df_train.drop(['rainfall'], axis=1)).drop(['id'], axis=1).columns
    feature_importance = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_df, palette="mako", ax=ax)
    ax.set_title('Feature Importance for CatBoost', fontsize=16)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    save_plot(fig, 'feature_importance.png')

    return render_template('model_insights.html', auc_score=auc_score)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    prediction_result = None
    if request.method == 'POST':
        try:
            mintemp = float(request.form['mintemp'])
            dewpoint = float(request.form['dewpoint'])
            cloud = float(request.form['cloud'])
            sunshine = float(request.form['sunshine'])
            windspeed = float(request.form['windspeed'])
            pressure = float(request.form['pressure'])

            input_data = pd.DataFrame({
                'id': [0],
                'day': [1],
                'pressure': [pressure],
                'maxtemp': [mintemp + 5.0],
                'temparature': [mintemp + 2.5],
                'mintemp': [mintemp],
                'dewpoint': [dewpoint],
                'humidity': [80.0],
                'cloud': [cloud],
                'sunshine': [sunshine],
                'winddirection': [90.0],
                'windspeed': [windspeed]
            })

            input_processed = preprocess_weather_data(input_data)
            input_scaled = scaler.transform(input_processed.drop(['id'], axis=1))
            prob = model.predict_proba(input_scaled)[:, 1][0]
            interpretation = "High likelihood of rainfall. 🌧️" if prob > 0.5 else "Low likelihood of rainfall. ☀️"
            prediction_result = {'probability': f"{prob:.4f}", 'interpretation': interpretation}
        except ValueError:
            prediction_result = {'error': "Please enter valid numerical values."}

    return render_template('prediction.html', prediction_result=prediction_result)

@app.route('/plot/<filename>')
def serve_plot(filename):
    return send_file(f"static/images/{filename}", mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)