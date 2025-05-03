import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from HROCH import Xicor
import joblib
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Load and explore data
df_train = pd.read_csv("../molecular-machine-learning/data/train.csv")
df_test = pd.read_csv("../molecular-machine-learning/data/test.csv")
submission = pd.read_csv("../molecular-machine-learning/data/sample_submission.csv")

# Preprocess data
# Impute missing values with median for numerical columns
for column in df_train.select_dtypes(include=[np.number]).columns:
    if df_train[column].isnull().any():
        df_train[column].fillna(df_train[column].median(), inplace=True)
        df_test[column].fillna(df_train[column].median(), inplace=True)

# Cap outliers for key features
def cap_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = data[column].clip(lower_bound, upper_bound)
    return data

key_features = ['T80', 'Mass', 'NumHeteroatoms', 'TDOS4.0']
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

# Compute Morgan fingerprints
def smiles_to_morgan_bits(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array([int(x) for x in fp])

train_morgan = np.array([smiles_to_morgan_bits(smiles) for smiles in df_train['Smiles']])
test_morgan = np.array([smiles_to_morgan_bits(smiles) for smiles in df_test['Smiles']])
morgan_columns = [f'frag_{i}' for i in range(train_morgan.shape[1])]
df_train_morgan = pd.DataFrame(train_morgan, columns=morgan_columns, index=df_train.index)
df_test_morgan = pd.DataFrame(test_morgan, columns=morgan_columns, index=df_test.index)
df_train = pd.concat([df_train, df_train_morgan], axis=1)
df_test = pd.concat([df_test, df_test_morgan], axis=1)

# Feature selection
numerical_columns = df_train.select_dtypes(include=[np.number]).columns.drop('T80')
corr_matrix = df_train[numerical_columns].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
df_train = df_train.drop(columns=to_drop, errors='ignore')
df_test = df_test.drop(columns=to_drop, errors='ignore')

# Xicor feature selection
numerical_columns = [col for col in df_train.select_dtypes(include=[np.number]).columns if col != 'T80']
frag_columns = [col for col in df_train.columns if col.startswith('frag_')]
corel_num = sorted([(c, Xicor(df_train[c].values, df_train['T80'].values)) for c in numerical_columns], key=lambda x: x[1], reverse=True)
corel_frag = sorted([(c, Xicor(df_train[c].values, df_train['T80'].values)) for c in frag_columns], key=lambda x: x[1], reverse=True)
top_num_features = [c for c, _ in corel_num[:30]]
top_frag_features = [c for c, _ in corel_frag[:100]]
selected_features = top_num_features + top_frag_features
key_features = ['TDOS4.0', 'NumHeteroatoms', 'Mass']
selected_features = list(set(selected_features + key_features).intersection(df_train.columns))
X_train = df_train[selected_features]
X_test = df_test[selected_features]
y_train = df_train['T80']

# Data scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and selected features for Streamlit app
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_features, 'selected_features.pkl')

# Model training
svr = SVR(C=1, epsilon=0.1, kernel='rbf')
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
xgb = XGBRegressor(random_state=42, n_jobs=-1)

svr.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)
xgb.fit(X_train_scaled, y_train)

# Save models
joblib.dump(svr, 'svr_model.pkl')
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(xgb, 'xgb_model.pkl')

# Ensemble predictions
y_pred_svr = svr.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test_scaled)
y_pred_xgb = xgb.predict(X_test_scaled)
y_pred_ensemble = 0.5 * y_pred_svr + 0.3 * y_pred_rf + 0.2 * y_pred_xgb
y_pred_ensemble = np.clip(y_pred_ensemble, 0, 1000)

# Generate submission
submission = pd.DataFrame({'Batch_ID': df_test['Batch_ID'], 'T80': y_pred_ensemble})
submission.to_csv('submission2.csv', index=False)
print("Submission saved to submission2.csv")