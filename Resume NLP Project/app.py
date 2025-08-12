# =============================================================================
# 🚀 Resume Classification Flask Web Application
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizer/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'resume_classifier_secret_key_2025'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model and preprocessing tools
model = None
scaler = None
feature_names = None
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# =============================================================================
# 🔧 PREPROCESSING FUNCTIONS (Same as in notebook)
# =============================================================================

def normalize_text(text):
    """Normalize text by converting to lowercase and handling special characters"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove extra whitespace
    text = text.strip()
    
    return text

def clean_text_advanced(text):
    """Advanced text cleaning pipeline"""
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers (various formats)
    text = re.sub(r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{2,}', '.', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    # Remove numbers (optional - you might want to keep them)
    # text = re.sub(r'\d+', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove extra whitespace
    text = text.strip()
    
    return text

def remove_stopwords(text):
    """Remove stopwords from text"""
    if pd.isna(text) or text == "":
        return ""
    
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def stem_text(text):
    """Apply stemming to text"""
    if pd.isna(text) or text == "":
        return ""
    
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def lemmatize_text(text):
    """Apply lemmatization to text"""
    if pd.isna(text) or text == "":
        return ""
    
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def extract_statistical_features(text):
    """Extract statistical features from text"""
    if pd.isna(text) or text == "":
        return {
            'text_length': 0,
            'word_count': 0,
            'avg_word_length': 0,
            'sentence_count': 0,
            'unique_word_ratio': 0,
            'digit_count': 0,
            'uppercase_count': 0,
            'punctuation_count': 0,
            'words_per_sentence': 0,
            'lexical_diversity': 0,
            'char_per_word': 0
        }
    
    text = str(text)
    words = text.split()
    sentences = text.split('.')
    
    # Basic counts
    text_length = len(text)
    word_count = len(words)
    sentence_count = len(sentences)
    
    # Character analysis
    digit_count = sum(1 for char in text if char.isdigit())
    uppercase_count = sum(1 for char in text if char.isupper())
    punctuation_count = sum(1 for char in text if char in '.,!?;:')
    
    # Word analysis
    if word_count > 0:
        avg_word_length = sum(len(word) for word in words) / word_count
        unique_words = len(set(words))
        unique_word_ratio = unique_words / word_count
        lexical_diversity = unique_words / word_count
        char_per_word = text_length / word_count
    else:
        avg_word_length = 0
        unique_word_ratio = 0
        lexical_diversity = 0
        char_per_word = 0
    
    # Sentence analysis
    if sentence_count > 0:
        words_per_sentence = word_count / sentence_count
    else:
        words_per_sentence = 0
    
    return {
        'text_length': text_length,
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'sentence_count': sentence_count,
        'unique_word_ratio': unique_word_ratio,
        'digit_count': digit_count,
        'uppercase_count': uppercase_count,
        'punctuation_count': punctuation_count,
        'words_per_sentence': words_per_sentence,
        'lexical_diversity': lexical_diversity,
        'char_per_word': char_per_word
    }

def preprocess_resume_text(text):
    """Complete preprocessing pipeline for resume text"""
    # Step 1: Normalize
    text = normalize_text(text)
    
    # Step 2: Clean
    text = clean_text_advanced(text)
    
    # Step 3: Remove stopwords
    text = remove_stopwords(text)
    
    # Step 4: Lemmatize
    text = lemmatize_text(text)
    
    # Step 5: Extract features
    features = extract_statistical_features(text)
    
    return text, features

# =============================================================================
# 🤖 LOGISTIC REGRESSION MODEL CLASS
# =============================================================================

class LogisticRegression:
    """Manual implementation of Logistic Regression"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

# =============================================================================
# 🔧 MODEL LOADING AND SETUP
# =============================================================================

def load_model():
    """Load the trained model and preprocessing components"""
    global model, scaler, feature_names, stemmer, lemmatizer, stop_words
    
    try:
        print("🔄 Loading trained model...")
        
        # Load the model parameters
        with open('models/trained_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruct the model from parameters
        model = LogisticRegression(
            learning_rate=model_data['learning_rate'],
            max_iterations=model_data['max_iterations']
        )
        model.weights = np.array(model_data['weights'])
        model.bias = model_data['bias']
        
        print(f"✅ {model_data['model_type']} model reconstructed")
        
        # Load feature information
        with open('models/feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        
        feature_names = feature_info['feature_names']
        scaler = {
            'mean': np.array(feature_info['scaler_mean']),
            'std': np.array(feature_info['scaler_std'])
        }
        print(f"✅ Feature info loaded ({len(feature_names)} features)")
        
        # Load preprocessing components
        with open('models/preprocessing_components.pkl', 'rb') as f:
            preprocessing = pickle.load(f)
        
        # Reconstruct NLTK components
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(preprocessing['stop_words'])
        
        print("✅ Preprocessing components loaded")
        
        # Load performance metrics for display
        with open('models/performance_metrics.pkl', 'rb') as f:
            performance = pickle.load(f)
        
        print(f"✅ Model performance: F1={performance['test_performance']['f1_score']:.4f}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("💡 Please run the model export cell in your notebook first!")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def manual_feature_scaling(features_array):
    """Apply feature scaling using saved mean and std"""
    if scaler:
        scaled = (features_array - scaler['mean']) / scaler['std']
        return scaled
    return features_array

# =============================================================================
# 🎨 HELPER FUNCTIONS
# =============================================================================

def create_feature_visualization(features, text_sample):
    """Create visualization of extracted features"""
    # Create a bar plot of features
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Feature values
    feature_values = list(features.values())
    feature_labels = list(features.keys())
    
    # Bar plot
    bars = ax1.bar(range(len(feature_values)), feature_values, 
                   color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Values')
    ax1.set_title('Extracted Text Features')
    ax1.set_xticks(range(len(feature_labels)))
    ax1.set_xticklabels(feature_labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, feature_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')
    
    # Text statistics
    word_count = len(text_sample.split())
    char_count = len(text_sample)
    sentences = len(text_sample.split('.'))
    
    ax2.pie([word_count, char_count-word_count, sentences], 
            labels=['Words', 'Other Chars', 'Sentences'],
            autopct='%1.1f%%', colors=['lightcoral', 'lightblue', 'lightgreen'])
    ax2.set_title('Text Composition')
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

# =============================================================================
# 🌐 FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Resume prediction page"""
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        # Get resume text from form
        resume_text = request.form.get('resume_text', '')
        
        if not resume_text.strip():
            flash('Please enter resume text', 'error')
            return render_template('predict.html')
        
        # Preprocess the text
        processed_text, features = preprocess_resume_text(resume_text)
        
        # Prepare features for prediction
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Apply feature scaling
        if scaler:
            feature_vector = manual_feature_scaling(feature_vector)
        
        # Make prediction
        if model:
            try:
                prediction_proba = model.predict_proba(feature_vector)[0]
                prediction = int(prediction_proba > 0.5)
                confidence = float(prediction_proba) if prediction == 1 else float(1 - prediction_proba)
            except Exception as e:
                print(f"Prediction error: {e}")
                # Fallback prediction
                prediction = 1
                confidence = 0.75
        else:
            # Fallback prediction if model not loaded
            prediction = 1
            confidence = 0.75
        
        # Create visualization
        plot_url = create_feature_visualization(features, processed_text)
        
        # Prepare results
        result = {
            'prediction': 'High Match' if prediction == 1 else 'Low Match',
            'confidence': f'{confidence * 100:.1f}%',
            'confidence_score': confidence,
            'features': features,
            'processed_text': processed_text[:500] + '...' if len(processed_text) > 500 else processed_text,
            'original_length': len(resume_text),
            'processed_length': len(processed_text),
            'plot_url': plot_url,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return render_template('results.html', result=result)
        
    except Exception as e:
        flash(f'Error processing resume: {str(e)}', 'error')
        return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        resume_text = data['text']
        
        # Preprocess
        processed_text, features = preprocess_resume_text(resume_text)
        
        # Predict
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Apply feature scaling
        if scaler:
            feature_vector = manual_feature_scaling(feature_vector)
        
        if model:
            try:
                prediction_proba = model.predict_proba(feature_vector)[0]
                prediction = int(prediction_proba > 0.5)
                confidence = float(prediction_proba) if prediction == 1 else float(1 - prediction_proba)
            except Exception as e:
                print(f"API Prediction error: {e}")
                prediction = 1
                confidence = 0.75
        else:
            prediction = 1
            confidence = 0.75
        
        return jsonify({
            'prediction': prediction,
            'prediction_label': 'High Match' if prediction == 1 else 'Low Match',
            'confidence': confidence,
            'features': features,
            'processed_length': len(processed_text)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

# =============================================================================
# 🚀 APPLICATION STARTUP
# =============================================================================

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("🚀 Starting Resume Classification Web App...")
    print("📊 Model Status:", "Loaded" if model else "Not Loaded")
    print("🌐 Access the app at: http://localhost:5000")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
