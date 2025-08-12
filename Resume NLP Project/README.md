# 📄 Resume NLP Project: AI-Powered Resume Classification

A complete end-to-end machine learning pipeline and web application for automatic resume analysis and job category prediction using advanced Natural Language Processing (NLP) and Machine Learning.

---

## 🚀 Project Overview

- **Goal:** Automatically classify resumes and predict job category/match using AI
- **Tech:** Python, Flask, NLTK, NumPy, pandas, Bootstrap, HTML/CSS/JS
- **ML:** Custom Logistic Regression, KNN, Naive Bayes, Decision Tree (manual implementations)
- **NLP:** Text cleaning, stemming, lemmatization, feature engineering
- **Web App:** Beautiful, responsive Flask frontend for real-time resume analysis

---

## 📦 Features

- **Text Preprocessing:** Advanced cleaning, normalization, stopword removal, stemming, lemmatization
- **Feature Engineering:** 11+ statistical and linguistic features extracted from resume text
- **Model Training:** Multiple ML algorithms, 5-fold cross-validation, robust evaluation
- **Performance:** Best F1-Score: 63.49% (Logistic Regression)
- **Web Interface:** Paste resume, get instant AI-powered analysis, feature visualization, and recommendations
- **API:** REST endpoint for programmatic predictions

---

## 🗂️ Project Structure

```
Resume NLP Project/
├── app.py                  # Flask web application
├── export_model.py         # Script to export model from notebook
├── requirements.txt        # Python dependencies
├── Resume_data.ipynb       # Main Jupyter notebook (ML pipeline)
├── models/                 # Exported model, features, and preprocessing files
├── static/                 # CSS, JS, images
├── templates/              # HTML templates (Jinja2)
└── README.md               # Project documentation
```

---

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd "Resume NLP Project"
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Train & Export Model:**
   - Open `Resume_data.ipynb` in Jupyter
   - Run all cells to preprocess data, train models, and export the best model (creates files in `models/`)
4. **Run the Web App:**
   ```bash
   python app.py
   ```
   - Visit [http://localhost:5000](http://localhost:5000) in your browser

---

## 🌐 Web Application

- **Homepage:** Project overview, stats, and features
- **Predict:** Paste your resume, get instant analysis and match score
- **Results:** Feature visualization, confidence score, recommendations
- **About:** Project details, ML pipeline, and performance

---

## 🧠 Machine Learning Pipeline

- **Data Preprocessing:**
  - Text normalization, cleaning, stopword removal, stemming, lemmatization
- **Feature Engineering:**
  - Extracts 11+ features (length, word count, lexical diversity, etc.)
- **Model Training:**
  - Logistic Regression, KNN, Naive Bayes, Decision Tree (manual)
  - 5-fold cross-validation, performance comparison
- **Model Export:**
  - Saves model weights, bias, feature info, and preprocessing for Flask app

---

## 📊 Results

- **Best Model:** Logistic Regression
- **Test F1-Score:** 0.6349
- **Test Accuracy:** 0.5383
- **Features Used:** 11
- **Cross-Validation:** 5-fold

---

## 🔗 API Usage

- **Endpoint:** `POST /api/predict`
- **Payload:**
  ```json
  { "text": "Paste resume text here..." }
  ```
- **Response:**
  ```json
  {
    "prediction": 1,
    "prediction_label": "High Match",
    "confidence": 0.85,
    "features": { ... },
    "processed_length": 1234
  }
  ```

---

## 👨‍💻 Author

- **Name:** Sajjad Ali Shah
- **LinkedIn:** [Profile](https://www.linkedin.com/in/sajjad-ali-shah47/)
- **GitHub:** [SajjadKhanYousafzai](https://github.com/SajjadKhanYousafzai)

---

## 📜 License

This project is for educational and demonstration purposes. For commercial use, please contact the author.
