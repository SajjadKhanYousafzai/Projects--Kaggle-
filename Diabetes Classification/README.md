📈 Diabetes Classification Project
Project Overview
This project focuses on diabetes classification and customer segmentation using machine learning. It includes an analysis notebook for classifying diabetes cases and segmenting customers based on behavioral data, along with interactive web applications (Flask and Streamlit) for real-time insights and predictions.
📊 Dataset
The dataset (marketing_campaign.csv) contains customer data with features including:

Year_Birth
Education
Marital_Status
Income
Kidhome & Teenhome
Dt_Customer (enrollment date)
Recency
Spending on various products (e.g., wines, fruits)
Campaign acceptance and purchase behavior

📁 Project Structure
Diabetes_Classification/
├── dataset/
│   └── marketing_campaign.csv         # Raw dataset for customer segmentation
├── models/                            # Placeholder for trained models
├── plots/                             # Generated visualizations (e.g., PNGs, HTML)
├── templates/                         # HTML templates for Flask app
├── Diabetes.ipynb                     # Notebook for diabetes classification
├── flask_app.py                       # Flask app for serving predictions
├── stapp.py                           # Streamlit app for interactive visualization
├── requirements.txt                   # Dependencies for the project
└── README.md                          # Project overview and instructions

🚀 Getting Started
Prerequisites

Python 3.8+
pip

Installation

Clone the repository
git clone <repository-url>
cd Diabetes_Classification


Install the required packages
pip install -r requirements.txt



Running the Web Applications

Flask App:
python flask_app.py

 Access at http://localhost:5000.

Streamlit App:
streamlit run stapp.py

 Access at http://localhost:8501.


This will launch interactive applications where you can:

Explore customer segments
Visualize data and clustering results
(Future) Get real-time predictions for diabetes classification

🤖 Machine Learning Models
The customer segmentation notebook uses:

PCA for dimensionality reduction
Agglomerative Clustering for segmenting customers

Diabetes classification models (in Diabetes.ipynb) are placeholders for:

Logistic Regression
Random Forest
Gradient Boosting

📊 Visualizations
The project includes various visualizations:

Age and income distributions
Education and marital status breakdowns
Spending patterns across product categories
2D/3D PCA scatter plots for clusters
Cluster profiles (income vs. expenses)

⚠️ Important Disclaimer
This application is for educational purposes only. Customer segmentation insights should be validated with domain experts before use in marketing strategies.
🛠️ Technical Implementation

Feature Engineering: Created features like Age, Days_of_client, Kids, and Expenses.
Outlier Handling: Z-score based removal with justification.
Preprocessing: One-hot encoding for categorical features, StandardScaler for numerical features.
Clustering: Elbow method for optimal clusters, silhouette score for evaluation.
Visualization: Interactive Plotly charts and Seaborn plots.

📋 Requirements
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

📑 Future Improvements

Add diabetes classification models (e.g., XGBoost, neural networks).
Implement feature importance analysis for clustering.
Expand dataset with health-related features for better diabetes prediction.
Enhance web apps with prediction capabilities.

📜 License
This project is open source and available under the MIT License.
🙏 Acknowledgements

Dataset sourced from Kaggle - Customer Personality Analysis.
Inspired by data science communities and tools like Jupyter, Flask, and Streamlit.

