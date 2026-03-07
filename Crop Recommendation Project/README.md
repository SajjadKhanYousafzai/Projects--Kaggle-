# 🌾 AI Smart Farming Advisor — Crop Recommendation System

> **Empowering Farmers with Data-Driven Precision Agriculture**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview

This end-to-end machine learning project helps farmers make informed decisions about:

- 🌱 **Crop Recommendation** — Which crop to grow based on soil & climate data
- 🧪 **Fertilizer Suggestion** — What nutrients the soil needs
- 💧 **Irrigation Planning** — How much water the crop requires

The system takes soil parameters as input and uses trained ML models to provide actionable recommendations.

---

## 🧬 Dataset Features

| Feature | Description |
|---|---|
| **N** | Ratio of Nitrogen content in soil |
| **P** | Ratio of Phosphorous content in soil |
| **K** | Ratio of Potassium content in soil |
| **Temperature** | Temperature in degree Celsius |
| **Humidity** | Relative humidity in % |
| **pH** | pH value of the soil |
| **Rainfall** | Rainfall in mm |
| **Label** | Target crop name |

**Source:** [Kaggle — Crop Recommendation Dataset by Atharva Ingle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

---

## 🛠️ Skills & Techniques Used

- **Classification** — Multi-class crop prediction (22 crop types)
- **Recommendation Systems** — Cosine similarity-based fertilizer/crop matching
- **Agricultural Analytics** — Feature importance, soil segmentation via clustering
- **Regression** — Irrigation water requirement estimation
- **EDA** — Distribution plots, correlation heatmaps, Plotly interactive charts

---

## 🤖 Models Trained

| Model | Task |
|---|---|
| Random Forest | Crop classification |
| XGBoost | Crop classification |
| LightGBM | Crop classification |
| CatBoost | Crop classification |
| SVM | Crop classification |
| Logistic Regression | Baseline |
| Gradient Boosting Regressor | Irrigation planning |
| KMeans Clustering | Soil segmentation |

---

## 📁 Project Structure

```
Crop Recommendation Project/
├── crop.ipynb              # Main end-to-end notebook
├── requirements.txt        # Package dependencies
├── README.md               # Project documentation
└── Dataset/
    └── Crop_recommendation.csv
```

---

## ⚙️ Setup & Installation

### 1. Create and activate the conda environment

```bash
conda create -n ml_env python=3.11
conda activate ml_env
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook crop.ipynb
```

---

## 📦 Dependencies

```
numpy==2.4.2
pandas==3.0.1
matplotlib==3.10.8
seaborn==0.13.2
plotly==6.6.0
scikit-learn==1.8.0
xgboost==3.2.0
lightgbm==4.6.0
catboost==1.2.10
joblib==1.5.3
scipy==1.17.1
```

---

## 📊 Key Results

- Achieved **99%+ accuracy** with Random Forest & XGBoost on crop classification
- Feature importance analysis reveals **Rainfall**, **Humidity**, and **K (Potassium)** as top predictors
- Soil segmentation identifies distinct soil profiles across the dataset

---

## 👤 Author

**Sajjad Ali Shah**
- LinkedIn: [sajjad-ali-shah47](https://www.linkedin.com/in/sajjad-ali-shah47/)
- Dataset: [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

---

## 📄 License

This project is licensed under the MIT License.

