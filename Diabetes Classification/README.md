# 📈 Diabetes Classification Project

## Project Overview

This project focuses on diabetes classification and customer segmentation using machine learning. It includes a comprehensive analysis notebook for customer segmentation and placeholders for diabetes classification, along with interactive Streamlit and Flask web applications for real-time insights.

## 📊 Dataset

The dataset (`marketing_campaign.csv`) contains customer data with features including:

* Year\_Birth
* Education
* Marital\_Status
* Income
* Kidhome & Teenhome
* Dt\_Customer (enrollment date)
* Recency
* Spending on various products (e.g., wines, fruits)
* Campaign acceptance and purchase behavior

## 📁 Project Structure

```bash
Diabetes_Classification/
├── dataset/
│   └── marketing_campaign.csv         # Raw dataset for customer segmentation
├── models/                            # Placeholder for trained models
├── plots/                             # Generated visualizations (e.g., PNGs, HTML)
├── templates/                         # HTML templates for Flask app
├── Diabetes.ipynb                     # Placeholder notebook for diabetes classification
├── mushroom.ipynb                     # Main Jupyter notebook for customer segmentation
├── flask_app.py                       # Flask app for serving predictions
├── stapp.py                           # Streamlit app for interactive visualization
├── requirements.txt                   # Dependencies for the project
└── README.md                          # Project overview and instructions
```

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* pip

### Installation

1. Clone the repository

```bash
git clone <repository-url>
cd Diabetes_Classification
```

2. Install the required packages

```bash
pip install -r requirements.txt
```

### Running the Streamlit App

```bash
streamlit run stapp.py
```

This will launch the interactive web application where you can:

* Explore customer segments
* Visualize data and clustering results
* (Future) Input data for diabetes predictions

## 🤖 Machine Learning Models

### Used for Customer Segmentation:

* PCA (dimensionality reduction)
* Agglomerative Clustering (customer segmentation)

### Planned for Diabetes Classification:

* Random Forest
* Logistic Regression
* Gradient Boosting
* K-Nearest Neighbors

## 📊 Visualizations

The project includes various visualizations:

* Distribution of customer demographics
* Feature distributions by segment
* Boxplots of numerical features
* Correlation heatmap
* Cluster scatter plots (2D/3D)
* Cluster profiles (income vs. expenses)

## ⚠️ Important Disclaimer

This application is for educational purposes only. Customer segmentation insights should be validated with domain experts before use in marketing strategies. Diabetes predictions should not be used for medical diagnosis.

## 🛠️ Technical Implementation

* **Feature Engineering**: Created features like Age, Days\_of\_client, Kids, and Expenses
* **Outlier Handling**: Z-score based removal with justification
* **Preprocessing**: One-hot encoding for categorical features, StandardScaler for numerical features
* **Clustering**: Elbow method for optimal clusters, silhouette score for evaluation
* **Visualization**: Interactive Plotly charts and Seaborn plots

## 📋 Requirements

```bash
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
streamlit>=1.10.0
plotly>=5.0.0
yellowbrick>=1.3
scipy>=1.7.0
flask>=2.0.0
```

## 📑 Future Improvements

* Implement diabetes classification models (e.g., XGBoost, neural networks)
* Add feature importance analysis for clustering
* Expand dataset with health-related features for diabetes prediction
* Enhance web apps with prediction capabilities

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgements

* Dataset sourced from Kaggle - Customer Personality Analysis
* Inspired by data science communities and tools like Jupyter, Flask, and Streamlit
