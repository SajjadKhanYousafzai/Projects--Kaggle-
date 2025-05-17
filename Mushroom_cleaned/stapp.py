import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Mushroom Classifier",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .title-text {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
    }
    .subtitle-text {
        font-size: 1.5rem;
        font-weight: 500;
        color: #FFFDF6;
        text-align: center;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1B5E20;
        margin-top: 2rem;
    }
    .result-text {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 5px;
    }
    .edible {
        background-color: #C8E6C9;
        color: #1B5E20;
    }
    .poisonous {
        background-color: #FFCDD2;
        color: #B71C1C;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='title-text'>🍄 Mushroom Classification 🍄</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Predict whether a mushroom is edible or poisonous</div>", unsafe_allow_html=True)

# Load the saved files
@st.cache_resource
def load_models():
    try:
        # Load the best model (change to your best model's name)
        best_model = joblib.load('./models/random_forest_model.joblib')
        # Also load other models for comparison
        lr_model = joblib.load('./models/logistic_regression_model.joblib')
        scaler = joblib.load('./models/standard_scaler1.joblib')
        # Load the feature names to ensure consistency
        feature_names = joblib.load('./models/feature_names.joblib')
        
        # Remove 'class' from feature names if it exists
        if 'class' in feature_names:
            feature_names.remove('class')
            
        return {
            'best_model': best_model,
            'lr_model': lr_model,
            'scaler': scaler,
            'feature_names': feature_names
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Function to load the original dataset for reference values
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("./dataset/mushroom_cleaned.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Function to make prediction
def predict_mushroom(input_data, models_dict):
    try:
        # Get all feature names from the model
        feature_names = models_dict['feature_names']
        
        # Create a DataFrame with one row initially filled with zeros
        input_df = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Fill in the numerical values
        for col in ['cap-diameter', 'stem-height', 'stem-width']:
            if col in input_data and col in input_df.columns:
                input_df[col] = input_data[col]
        
        # Handle categorical features correctly
        categorical_cols = ['cap-shape', 'gill-attachment', 'gill-color', 'stem-color', 'season']
        
        for col in categorical_cols:
            if col in input_data:
                # Find the column that corresponds to this category value
                col_name = f"{col}_{input_data[col]}"
                # Set this column to 1 if it exists in our feature names
                if col_name in input_df.columns:
                    input_df[col_name] = 1
        
        # Scale the input
        input_scaled = models_dict['scaler'].transform(input_df)
        
        # Make prediction with best model
        prediction = models_dict['best_model'].predict(input_scaled)
        probability = models_dict['best_model'].predict_proba(input_scaled)
        
        # Also get logistic regression prediction for comparison
        lr_prediction = models_dict['lr_model'].predict(input_scaled)
        lr_probability = models_dict['lr_model'].predict_proba(input_scaled)
        
        return {
            'prediction': prediction[0],
            'probability': probability[0],
            'lr_prediction': lr_prediction[0],
            'lr_probability': lr_probability[0]
        }
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        st.error(f"Input Data: {input_data}")
        if models_dict:
            st.error(f"Feature Names: {models_dict['feature_names'][:10]}...")
        return None

# Main function to run the app
def main():
    # Load models and data
    models_dict = load_models()
    df = load_data()
    
    if models_dict is None or df is None:
        st.error("Failed to load essential components. Please check the file paths.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Classifier", "📊 Visualizations", "ℹ️ About"])
    
    # Tab 1: Classifier
    with tab1:
        st.markdown("<div class='section-header'>Enter Mushroom Characteristics</div>", unsafe_allow_html=True)
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        # Get unique values for categorical features
        cap_shapes = sorted(df['cap-shape'].unique().tolist())
        gill_attachments = sorted(df['gill-attachment'].unique().tolist())
        gill_colors = sorted(df['gill-color'].unique().tolist())
        stem_colors = sorted(df['stem-color'].unique().tolist())
        seasons = sorted(df['season'].unique().tolist())
        
        # Input form with more descriptive labels
        with col1:
            st.subheader("Physical Characteristics")
            cap_diameter = st.slider("Cap Diameter (cm)", min_value=float(df['cap-diameter'].min()), 
                                    max_value=float(df['cap-diameter'].max()), 
                                    value=float(df['cap-diameter'].median()))
            
            cap_shape = st.selectbox("Cap Shape", cap_shapes, 
                                    help="Shape of the mushroom cap")
            
            gill_attachment = st.selectbox("Gill Attachment", gill_attachments,
                                        help="How the gills are attached to the stem")
            
            gill_color = st.selectbox("Gill Color", gill_colors,
                                    help="Color of the gills under the cap")
        
        with col2:
            st.subheader("Additional Features")
            stem_height = st.slider("Stem Height (cm)", min_value=float(df['stem-height'].min()), 
                                    max_value=float(df['stem-height'].max()), 
                                    value=float(df['stem-height'].median()))
            
            stem_width = st.slider("Stem Width (cm)", min_value=float(df['stem-width'].min()), 
                                max_value=float(df['stem-width'].max()), 
                                value=float(df['stem-width'].median()))
            
            stem_color = st.selectbox("Stem Color", stem_colors,
                                    help="Color of the mushroom stem")
            
            season = st.selectbox("Growing Season", seasons,
                                help="Season when the mushroom was found")
        
        # Create input data dictionary
        input_data = {
            'cap-diameter': cap_diameter,
            'stem-height': stem_height,
            'stem-width': stem_width,
            'cap-shape': cap_shape,
            'gill-attachment': gill_attachment,
            'gill-color': gill_color,
            'stem-color': stem_color,
            'season': season
        }
        
        # Add a predict button with better styling
        predict_button = st.button("🔍 Predict Mushroom Type")
        
        if predict_button:
            with st.spinner('Analyzing mushroom characteristics...'):
                # Make prediction
                result = predict_mushroom(input_data, models_dict)
                
                if result:
                    # Display the result
                    st.markdown("<div class='section-header'>Prediction Results</div>", unsafe_allow_html=True)
                    
                    # Create columns for the prediction displays
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.subheader("Random Forest Prediction")
                        prediction = "Edible" if result['prediction'] == 1 else "Poisonous"
                        prediction_class = "edible" if result['prediction'] == 1 else "poisonous"
                        
                        st.markdown(f"<div class='result-text {prediction_class}'>{prediction}</div>", 
                                  unsafe_allow_html=True)
                        
                        # Show probabilities
                        st.write(f"Confidence: {result['probability'][result['prediction']]:.2%}")
                        
                        # Create a gauge chart for the probability
                        fig, ax = plt.subplots(figsize=(4, 0.5))
                        ax.barh([''], [result['probability'][1]], color='green', alpha=0.6)
                        ax.barh([''], [1], color='lightgray', alpha=0.2)
                        ax.set_xlim(0, 1)
                        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                        ax.set_yticks([])
                        ax.axvline(x=result['probability'][1], color='black', linestyle='-', linewidth=2)
                        plt.title('Probability of being Edible')
                        st.pyplot(fig)
                    
                    with res_col2:
                        st.subheader("Logistic Regression Prediction")
                        lr_prediction = "Edible" if result['lr_prediction'] == 1 else "Poisonous"
                        lr_prediction_class = "edible" if result['lr_prediction'] == 1 else "poisonous"
                        
                        st.markdown(f"<div class='result-text {lr_prediction_class}'>{lr_prediction}</div>", 
                                  unsafe_allow_html=True)
                        
                        # Show probabilities
                        st.write(f"Confidence: {result['lr_probability'][result['lr_prediction']]:.2%}")
                        
                        # Create a gauge chart for the probability
                        fig, ax = plt.subplots(figsize=(4, 0.5))
                        ax.barh([''], [result['lr_probability'][1]], color='green', alpha=0.6)
                        ax.barh([''], [1], color='lightgray', alpha=0.2)
                        ax.set_xlim(0, 1)
                        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                        ax.set_yticks([])
                        ax.axvline(x=result['lr_probability'][1], color='black', linestyle='-', linewidth=2)
                        plt.title('Probability of being Edible')
                        st.pyplot(fig)
                    
                    # Add disclaimer
                    st.warning("⚠️ **Important Disclaimer**: This tool is for educational purposes only. Never consume wild mushrooms based solely on this classification. Always consult with a mushroom expert or mycologist for proper identification.")
    
    # Tab 2: Visualizations
    with tab2:
        st.markdown("<div class='section-header'>Dataset Visualizations</div>", unsafe_allow_html=True)
        
        # Create tabs for different visualization categories
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Distribution", "Relationships", "Model Performance"])
        
        with viz_tab1:
            st.subheader("Class Distribution")
            # Display the distribution chart
            try:
                img = Image.open('class_distribution.png')
                st.image(img, caption='Distribution of Mushroom Classes')
            except:
                # If the image doesn't exist, create it
                plt.figure(figsize=(10, 6))
                class_counts = df['class'].value_counts()
                sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
                plt.title('Distribution of Mushroom Classes', fontsize=16)
                plt.xlabel('Class (0: Poisonous, 1: Edible)', fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.xticks([0, 1], ['Poisonous', 'Edible'])
                for i, v in enumerate(class_counts):
                    plt.text(i, v + 50, f'{v} ({v/len(df):.1%})', ha='center', fontsize=12)
                st.pyplot(plt)
            
            st.subheader("Numerical Features Distribution")
            try:
                img = Image.open('numerical_distributions.png')
                st.image(img, caption='Distribution of Numerical Features')
            except:
                st.info("Numerical distributions visualization image not found.")
                
        with viz_tab2:
            st.subheader("Feature Relationships")
            # Show boxplots of numerical features by class
            try:
                img = Image.open('numerical_boxplots_by_class.png')
                st.image(img, caption='Numerical Features by Class')
            except:
                st.info("Numerical boxplots visualization image not found.")
            
            # Show correlation heatmap
            st.subheader("Feature Correlation")
            try:
                img = Image.open('correlation_heatmap.png')
                st.image(img, caption='Correlation Heatmap of Features')
            except:
                st.info("Correlation heatmap visualization image not found.")
                
        with viz_tab3:
            st.subheader("Model Performance Comparison")
            try:
                img = Image.open('model_accuracy_comparison.png')
                st.image(img, caption='Model Accuracy Comparison')
            except:
                st.info("Model accuracy comparison visualization image not found.")
            
            # Display confusion matrices if available
            st.subheader("Confusion Matrices")
            col1, col2 = st.columns(2)
            with col1:
                try:
                    img = Image.open('confusion_matrix_random_forest.png')
                    st.image(img, caption='Random Forest Confusion Matrix')
                except:
                    st.info("Random Forest confusion matrix image not found.")
            
            with col2:
                try:
                    img = Image.open('confusion_matrix_logistic_regression.png')
                    st.image(img, caption='Logistic Regression Confusion Matrix')
                except:
                    st.info("Logistic Regression confusion matrix image not found.")
                    
    # Tab 3: About
    with tab3:
        st.markdown("<div class='section-header'>About This Project</div>", unsafe_allow_html=True)
        
        st.markdown("""
        ### 🍄 Mushroom Classification Project
        
        This application uses machine learning to classify mushrooms as edible or poisonous based on their physical characteristics. The models were trained on a dataset of mushroom samples with various features including cap diameter, stem dimensions, colors, and seasonal information.
        
        #### 📊 Dataset
        The dataset contains information about various mushroom species, labeled as either edible (1) or poisonous (0). The features include:
        
        - **Cap diameter**: Size of the mushroom cap in centimeters
        - **Cap shape**: Shape classification of the mushroom cap
        - **Gill attachment**: How the gills are attached to the stem
        - **Gill color**: Color of the gills under the cap
        - **Stem height**: Height of the mushroom stem in centimeters
        - **Stem width**: Width of the mushroom stem in centimeters
        - **Stem color**: Color of the mushroom stem
        - **Season**: Season when the mushroom typically grows
        
        #### 🤖 Machine Learning Models
        Several classification models were trained and evaluated:
        
        - **Random Forest**: Best performing model with excellent accuracy
        - **Logistic Regression**: Simple yet effective baseline model
        - **Gradient Boosting**: Advanced ensemble learning method
        - **K-Nearest Neighbors**: Instance-based learning algorithm
        
        #### ⚠️ Important Disclaimer
        This application is for educational purposes only. Never consume wild mushrooms based solely on this classification. Always consult with a mushroom expert or mycologist for proper identification before consuming any wild mushrooms. Misidentification can lead to serious illness or death.
        
        #### 🔍 How to Use
        1. Navigate to the **Classifier** tab
        2. Enter the mushroom's characteristics using the sliders and dropdown menus
        3. Click the "Predict Mushroom Type" button
        4. View the prediction results and confidence levels
        
        #### 📈 Visualizations
        The **Visualizations** tab provides various charts and graphs showing the distribution of features in the dataset, relationships between variables, and model performance metrics.
        
        #### 🧪 Project Methodology
        1. Data cleaning and exploratory analysis
        2. Feature engineering and preprocessing
        3. Model training and evaluation
        4. Selection of the best performing model
        5. Development of this interactive application
        
        #### 📚 Learn More
        To learn more about mushroom identification, please consult official mycology resources and field guides.
        """)
        
        # Add a footer
        st.markdown("""
        ---
        <p style='text-align: center; color: #666; font-size: 0.8rem;'>
        Mushroom Classification App | Created with Streamlit | Data Science Project
        </p>
        """, unsafe_allow_html=True)

# Run the main function
if __name__ == "__main__":
    main()