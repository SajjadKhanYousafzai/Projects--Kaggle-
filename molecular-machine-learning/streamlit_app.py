import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
import io
import base64
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# Streamlit app configuration
st.set_page_config(page_title="Molecular Photostability Prediction", layout="wide")
st.title("🔬 Molecular Photostability (T80) Prediction")
st.markdown("""
Welcome to the **Molecular Photostability Prediction App**!  
This professional tool, developed for the NSF's Molecule Maker Lab Institute competition, predicts the photostability lifetime (T80) of molecules using a data-driven machine learning pipeline. Upload your dataset or use the default data, explore features, train models, and generate predictions for submission.

**Key Features**:
- Upload `train.csv` and `test.csv` or use provided datasets.
- Visualize molecular feature distributions.
- Train a robust ML pipeline with SVR, Random Forest, XGBoost, and LASSO.
- View feature importance and model performance (RMSE).
- Download predictions as a submission file.
""")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload", "Data Exploration", "Model Training", "Predictions"])

# Initialize session state for data
if 'df_train' not in st.session_state:
    st.session_state.df_train = None
if 'df_test' not in st.session_state:
    st.session_state.df_test = None
if 'submission' not in st.session_state:
    st.session_state.submission = None

# Data Upload
if page == "Data Upload":
    st.header("Data Upload")
    st.markdown("Upload your `train.csv` and `test.csv` files or use the default datasets.")
    
    col1, col2 = st.columns(2)
    with col1:
        train_file = st.file_uploader("Upload train.csv", type=["csv"])
        if train_file:
            st.session_state.df_train = pd.read_csv(train_file)
            st.success("Train data uploaded successfully!")
    with col2:
        test_file = st.file_uploader("Upload test.csv", type=["csv"])
        if test_file:
            st.session_state.df_test = pd.read_csv(test_file)
            st.success("Test data uploaded successfully!")
    
    if st.button("Use Default Datasets"):
        try:
            st.session_state.df_train = pd.read_csv("../molecular-machine-learning/data/train.csv")
            st.session_state.df_test = pd.read_csv("../molecular-machine-learning/data/test.csv")
            st.success("Default datasets loaded successfully!")
        except FileNotFoundError:
            st.error("Default datasets not found. Please upload train.csv and test.csv.")

    if st.session_state.df_train is not None and st.session_state.df_test is not None:
        st.write("Train Data Preview:")
        st.dataframe(st.session_state.df_train.head())
        st.write("Test Data Preview:")
        st.dataframe(st.session_state.df_test.head())

# Data Exploration
if page == "Data Exploration" and st.session_state.df_train is not None:
    st.header("Data Exploration")
    df_train = st.session_state.df_train
    
    st.subheader("Dataset Summary")
    st.write(f"Rows: {df_train.shape[0]}, Columns: {df_train.shape[1]}")
    st.write(f"Numerical Columns: {df_train.select_dtypes(include=['int64', 'float64']).shape[1]}")
    st.write(f"Categorical Columns: {df_train.select_dtypes(include=['object', 'category']).shape[1]}")
    st.write(f"Missing Values: {df_train.isnull().sum().sum()}")
    
    st.subheader("Feature Distributions")
    key_features = ['TDOS4.0', 'NumHeteroatoms', 'Mass', 'T80']
    for feature in key_features:
        if feature in df_train.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df_train[feature], ax=ax1)
            ax1.set_title(f"Distribution of {feature}")
            sns.boxplot(x=df_train[feature], ax=ax2)
            ax2.set_title(f"Boxplot of {feature}")
            st.pyplot(fig)

# Model Training
if page == "Model Training" and st.session_state.df_train is not None and st.session_state.df_test is not None:
    st.header("Model Training")
    df_train = st.session_state.df_train.copy()
    df_test = st.session_state.df_test.copy()
    
    # Impute missing values
    numerical_columns = df_train.select_dtypes(include=[np.number]).columns.drop(['T80'], errors='ignore').tolist()
    for column in numerical_columns:
        if df_train[column].isnull().any():
            df_train[column].fillna(df_train[column].median(), inplace=True)
            df_test[column].fillna(df_train[column].median(), inplace=True)
    
    # Cap outliers
    def cap_outliers(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column] = data[column].clip(lower_bound, upper_bound)
        return data
    
    key_features = ['T80', 'TDOS4.0', 'NumHeteroatoms', 'Mass']
    for feature in key_features:
        if feature in df_train.columns:
            df_train = cap_outliers(df_train, feature)
            if feature != 'T80':
                df_test = cap_outliers(df_test, feature)
    
    # RDKit feature engineering
    def compute_rdkit_features(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return pd.Series({
                'MolWt': Descriptors.MolWt(mol),
                'MolLogP': Descriptors.MolLogP(mol),
                'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
                'TPSA': Descriptors.TPSA(mol)
            })
        except:
            return None
    
    df_train_rdkit = df_train['Smiles'].apply(compute_rdkit_features)
    df_test_rdkit = df_test['Smiles'].apply(compute_rdkit_features)
    df_train = df_train.loc[~df_train_rdkit.isna().any(axis=1)]
    df_test = df_test.loc[~df_test_rdkit.isna().any(axis=1)]
    df_train = pd.concat([df_train, df_train_rdkit], axis=1)
    df_test = pd.concat([df_test, df_test_rdkit], axis=1)
    
    # Feature selection
    numerical_columns = df_train.select_dtypes(include=[np.number]).columns.drop(['T80'], errors='ignore').tolist()
    variances = df_train[numerical_columns].var()
    numerical_columns = [col for col in numerical_columns if variances[col] > 1e-10]
    
    corr_matrix = df_train[numerical_columns].corr().abs()
    st.write("NaN in correlation matrix:", corr_matrix.isna().any().any())
    corr_matrix = corr_matrix.fillna(0)
    to_drop = []
    protected_features = ['TDOS4.0', 'NumHeteroatoms', 'Mass', 'MolLogP', 'TPSA']
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col_i, col_j = corr_matrix.columns[i], corr_matrix.columns[j]
            if col_i in protected_features or col_j in protected_features:
                continue
            corr_value = corr_matrix.iloc[i, j]
            if np.isscalar(corr_value) and corr_value > 0.85:
                var_i = df_train[col_i].var()
                var_j = df_train[col_j].var()
                to_drop.append(col_j if var_i > var_j else col_i)
    
    to_drop = list(set(to_drop))
    df_train = df_train.drop(columns=to_drop, errors='ignore')
    df_test = df_test.drop(columns=to_drop, errors='ignore')
    numerical_columns = [col for col in df_train.select_dtypes(include=[np.number]).columns if col != 'T80']
    
    X_train = df_train[numerical_columns]
    y_train = df_train['T80']
    if X_train.empty:
        st.error("Feature selection resulted in an empty dataset. Using protected features only.")
        numerical_columns = [f for f in protected_features if f in df_train.columns]
        X_train = df_train[numerical_columns]
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=10)
    rfe.fit(X_train, y_train)
    selected_features = X_train.columns[rfe.support_].tolist()
    selected_features = list(set(selected_features + protected_features).intersection(X_train.columns))
    
    X_train = df_train[selected_features]
    X_test = df_test[selected_features]
    
    # Data scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training
    models = {
        'SVR': SVR(),
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(random_state=42, n_jobs=-1),
        'LASSO': Lasso(random_state=42)
    }
    
    st.subheader("Model Performance")
    rmse_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores.mean())
        rmse_results[name] = rmse
        st.write(f"{name} RMSE: {rmse:.4f} ± {np.sqrt(scores.std()):.4f}")
    
    # Hyperparameter tuning for SVR
    param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 1],
        'kernel': ['linear', 'rbf']
    }
    grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    st.write("\nBest SVR Parameters:", grid_search.best_params_)
    st.write("Best SVR RMSE:", np.sqrt(-grid_search.best_score_))
    
    # Ensemble predictions
    svr = SVR(**grid_search.best_params_).fit(X_train_scaled, y_train)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1).fit(X_train_scaled, y_train)
    xgb = XGBRegressor(random_state=42, n_jobs=-1).fit(X_train_scaled, y_train)
    
    y_pred_svr = svr.predict(X_test_scaled)
    y_pred_rf = rf.predict(X_test_scaled)
    y_pred_xgb = xgb.predict(X_test_scaled)
    y_pred_ensemble = 0.5 * y_pred_svr + 0.3 * y_pred_rf + 0.2 * y_pred_xgb
    y_pred_ensemble = np.clip(y_pred_ensemble, 0, None)
    
    st.session_state.submission = pd.DataFrame({'Batch_ID': df_test['Batch_ID'], 'T80': y_pred_ensemble})
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index, ax=ax)
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)

# Predictions
if page == "Predictions" and st.session_state.submission is not None:
    st.header("Predictions")
    st.subheader("Submission Preview")
    st.dataframe(st.session_state.submission.head(10))
    
    # Download submission
    csv = st.session_state.submission.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="submission.csv">Download Submission File</a>'
    st.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
**Developed by Sajjad Ali Shah**  
[LinkedIn](https://www.linkedin.com/in/sajjad-ali-shah-120341305) | [Competition Dataset](https://www.kaggle.com/competitions/molecular-machine-learning/overview)  
Built with ❤️ using Streamlit for the NSF Molecule Maker Lab Institute Competition
""")