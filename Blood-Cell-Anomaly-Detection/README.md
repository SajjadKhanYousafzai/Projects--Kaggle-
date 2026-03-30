## Blood Cell Anomaly Detection

End-to-end machine learning project for detecting blood cell anomalies using tabular and model-based workflows.

## What this project includes

- Data preprocessing and feature handling
- Model training notebook for experimentation
- Saved model artifacts in the `Models` directory
- Backend API scaffold in `Backend/main.py`
- Frontend folder for UI integration

## Project structure

Blood-Cell-Anomaly-Detection/
- Blood_Cell_Anomaly_Detection_Complete.ipynb
- Dataset/
- Models/
- Backend/
- frontend/
- requirements.txt

## Quick start

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open and run the notebook:

- `Blood_Cell_Anomaly_Detection_Complete.ipynb`

4. Run backend (from `Backend` folder):

```bash
pip install -r requirements.txt
python main.py
```

## Dataset

Main dataset files are available in `Dataset/`, including:

- `blood_cell_anomaly_detection.csv`
- `cell_type_reference.csv`
- `cytodiffusion_benchmark_scores.csv`

## Next improvements

- Add model evaluation metrics table in this README
- Add API endpoint documentation with sample requests
- Add screenshots or demo GIF of frontend output

## Author

Built as part of the Kaggle projects workspace.
