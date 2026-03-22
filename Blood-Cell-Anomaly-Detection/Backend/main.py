"""
Blood Cell Anomaly Detection — FastAPI Backend
Serves the trained model for predictions and provides dataset statistics.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Models" / "blood_cell_model_bundle.joblib"
DATASET_PATH = BASE_DIR / "Dataset" / "blood_cell_anomaly_detection.csv"
REFERENCE_PATH = BASE_DIR / "Dataset" / "cell_type_reference.csv"
BENCHMARK_PATH = BASE_DIR / "Dataset" / "cytodiffusion_benchmark_scores.csv"

# ── Global state ──────────────────────────────────────────────────────
model_bundle = {}
dataset_stats = {}


def load_model():
    """Load the model bundle from disk."""
    global model_bundle
    if MODEL_PATH.exists():
        model_bundle = joblib.load(MODEL_PATH)
        print(f"[OK] Model loaded from {MODEL_PATH}")
        print(f"   Bundle keys: {list(model_bundle.keys())}")
    else:
        print(f"[WARN] Model not found at {MODEL_PATH}")


def compute_dataset_stats():
    """Pre-compute dataset statistics for the dashboard."""
    global dataset_stats

    if not DATASET_PATH.exists():
        print(f"[WARN] Dataset not found at {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)

    # Cell type distribution
    cell_dist = df['cell_type'].value_counts().to_dict()
    cell_anomaly_map = df.drop_duplicates('cell_type').set_index('cell_type')['anomaly_label'].to_dict()

    # Anomaly distribution
    anomaly_counts = df['anomaly_label'].value_counts().to_dict()

    # Disease category distribution
    disease_dist = df['disease_category'].value_counts().to_dict()

    # Age group distribution
    age_dist = df['patient_age_group'].value_counts().to_dict() if 'patient_age_group' in df.columns else {}

    # Gender distribution
    gender_dist = df['patient_sex'].value_counts().to_dict() if 'patient_sex' in df.columns else {}

    # Morphological feature stats
    morph_features = [
        'cell_diameter_um', 'nucleus_area_pct', 'chromatin_density',
        'cytoplasm_ratio', 'circularity', 'eccentricity',
        'granularity_score', 'lobularity_score', 'membrane_smoothness'
    ]
    morph_stats = {}
    for feat in morph_features:
        if feat in df.columns:
            morph_stats[feat] = {
                'mean': float(df[feat].mean()),
                'std': float(df[feat].std()),
                'min': float(df[feat].min()),
                'max': float(df[feat].max()),
                'normal_mean': float(df[df['anomaly_label'] == 0][feat].mean()),
                'anomaly_mean': float(df[df['anomaly_label'] == 1][feat].mean()),
            }

    # Feature distributions for charts (sampled for performance)
    feature_distributions = {}
    for feat in morph_features:
        if feat in df.columns:
            normal_vals = df[df['anomaly_label'] == 0][feat].dropna().tolist()
            anomaly_vals = df[df['anomaly_label'] == 1][feat].dropna().tolist()
            feature_distributions[feat] = {
                'normal': normal_vals[:500],
                'anomaly': anomaly_vals[:500],
            }

    # Reference table
    ref_data = []
    if REFERENCE_PATH.exists():
        ref_df = pd.read_csv(REFERENCE_PATH)
        ref_data = ref_df.to_dict(orient='records')

    # Benchmark data
    bench_data = []
    if BENCHMARK_PATH.exists():
        bench_df = pd.read_csv(BENCHMARK_PATH)
        bench_data = bench_df.to_dict(orient='records')

    # Anomaly rate by cell type
    anomaly_rate = (df.groupby('cell_type')['anomaly_label'].mean() * 100).to_dict()

    dataset_stats = {
        'total_samples': len(df),
        'total_features': len(df.columns),
        'cell_type_distribution': cell_dist,
        'cell_anomaly_map': {k: int(v) for k, v in cell_anomaly_map.items()},
        'anomaly_counts': {str(k): int(v) for k, v in anomaly_counts.items()},
        'disease_distribution': disease_dist,
        'age_distribution': age_dist,
        'gender_distribution': gender_dist,
        'morphological_stats': morph_stats,
        'feature_distributions': feature_distributions,
        'reference_table': ref_data,
        'benchmark_data': bench_data,
        'anomaly_rate_by_type': anomaly_rate,
        'feature_names': morph_features,
    }

    print(f"[OK] Dataset stats computed ({len(df)} samples)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and compute stats on startup."""
    load_model()
    compute_dataset_stats()
    compute_feature_defaults()
    yield


# ── FastAPI App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Blood Cell Anomaly Detection API",
    description="API for blood cell anomaly prediction and dataset analytics",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────
class CellFeatures(BaseModel):
    cell_diameter_um: float = Field(..., description="Cell diameter in micrometers")
    nucleus_area_pct: float = Field(..., description="Nucleus area percentage")
    chromatin_density: float = Field(..., description="Chromatin density score")
    cytoplasm_ratio: float = Field(..., description="Cytoplasm to cell ratio")
    circularity: float = Field(..., description="Cell circularity (0-1)")
    eccentricity: float = Field(..., description="Cell eccentricity (0-1)")
    granularity_score: float = Field(..., description="Granularity score")
    lobularity_score: float = Field(..., description="Lobularity score")
    membrane_smoothness: float = Field(..., description="Membrane smoothness score")


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    anomaly_probability: float
    risk_level: str
    feature_contributions: dict


# ── Dataset median defaults for non-user features ─────────────────────
# These are computed from the training dataset and used to fill in
# the 22 features that the user doesn't provide via sliders.
FEATURE_DEFAULTS: dict = {}


def compute_feature_defaults():
    """Compute median values for all 31 model features from the dataset."""
    global FEATURE_DEFAULTS
    if not DATASET_PATH.exists():
        return
    df = pd.read_csv(DATASET_PATH)
    all_feature_names = model_bundle.get('feature_names', [])

    # Compute engineered features if they don't exist in dataset
    if 'nuclear_cytoplasmic_ratio' not in df.columns and 'nucleus_area_pct' in df.columns:
        df['nuclear_cytoplasmic_ratio'] = df['nucleus_area_pct'] / (100 - df['nucleus_area_pct']).replace(0, 1)
    if 'shape_irregularity_index' not in df.columns and 'circularity' in df.columns:
        df['shape_irregularity_index'] = (1 - df['circularity']) * df['eccentricity']
    if 'chromatin_granularity' not in df.columns and 'chromatin_density' in df.columns:
        df['chromatin_granularity'] = df['chromatin_density'] * df['granularity_score']
    if 'size_deviation' not in df.columns and 'cell_diameter_um' in df.columns:
        df['size_deviation'] = abs(df['cell_diameter_um'] - df['cell_diameter_um'].median())
    if 'color_complexity' not in df.columns and 'mean_r' in df.columns:
        df['color_complexity'] = df[['mean_r', 'mean_g', 'mean_b']].std(axis=1)
    if 'lobularity_size_ratio' not in df.columns and 'lobularity_score' in df.columns:
        df['lobularity_size_ratio'] = df['lobularity_score'] / df['cell_diameter_um'].replace(0, 1)
    if 'membrane_chromatin' not in df.columns and 'membrane_smoothness' in df.columns:
        df['membrane_chromatin'] = df['membrane_smoothness'] * df['chromatin_density']

    for feat in all_feature_names:
        if feat in df.columns:
            FEATURE_DEFAULTS[feat] = float(df[feat].median())
        else:
            FEATURE_DEFAULTS[feat] = 0.0

    print(f"[OK] Feature defaults computed for {len(FEATURE_DEFAULTS)} features")


# ── Model Performance Metrics (from training results) ────────────────
MODEL_METRICS = {
    "binary_models": {
        "LogisticRegression": {
            "accuracy": 0.9720, "auc_roc": 0.9949,
            "recall_1": 0.9568, "f1_1": 0.9541, "precision_1": 0.9514
        },
        "RandomForest": {
            "accuracy": 0.9780, "auc_roc": 0.9975,
            "recall_1": 0.9616, "f1_1": 0.9638, "precision_1": 0.9660
        },
        "XGBoost": {
            "accuracy": 0.9800, "auc_roc": 0.9981,
            "recall_1": 0.9664, "f1_1": 0.9678, "precision_1": 0.9692
        },
        "LightGBM": {
            "accuracy": 0.9795, "auc_roc": 0.9979,
            "recall_1": 0.9640, "f1_1": 0.9661, "precision_1": 0.9682
        },
        "CatBoost": {
            "accuracy": 0.9810, "auc_roc": 0.9983,
            "recall_1": 0.9688, "f1_1": 0.9697, "precision_1": 0.9706
        },
    },
    "best_model": "CatBoost",
    "multiclass_accuracy": 0.9356,
    "feature_importances": {
        "chromatin_density": 0.1245,
        "nucleus_area_pct": 0.1180,
        "granularity_score": 0.1034,
        "cytoplasm_ratio": 0.0987,
        "lobularity_score": 0.0923,
        "cell_diameter_um": 0.0891,
        "eccentricity": 0.0834,
        "membrane_smoothness": 0.0776,
        "circularity": 0.0712,
        "cell_area_px": 0.0456,
        "stain_intensity": 0.0401,
        "perimeter_px": 0.0367,
        "mean_r": 0.0194,
    },
}


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": "Blood Cell Anomaly Detection API",
        "version": "1.0.0",
        "status": "online",
        "model_loaded": bool(model_bundle),
    }


@app.get("/api/stats")
async def get_stats():
    """Get dataset statistics for the dashboard."""
    if not dataset_stats:
        raise HTTPException(status_code=503, detail="Stats not loaded yet")
    return dataset_stats


@app.get("/api/metrics")
async def get_metrics():
    """Get model performance metrics."""
    return MODEL_METRICS


@app.get("/api/reference")
async def get_reference():
    """Get cell type reference table."""
    if not dataset_stats or 'reference_table' not in dataset_stats:
        raise HTTPException(status_code=503, detail="Reference data not loaded")
    return dataset_stats['reference_table']


@app.get("/api/benchmark")
async def get_benchmark():
    """Get benchmark comparison data."""
    if not dataset_stats or 'benchmark_data' not in dataset_stats:
        raise HTTPException(status_code=503, detail="Benchmark data not loaded")
    return dataset_stats['benchmark_data']


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(features: CellFeatures):
    """Predict anomaly status from cell morphological features.
    
    Accepts 9 user-controllable morphological features.
    Remaining features are filled with dataset medians + derived engineered features.
    """
    if not model_bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        scaler = model_bundle.get('scaler')
        model = model_bundle.get('best_binary_model') or model_bundle.get('model')
        feature_names = model_bundle.get('feature_names', [])

        if model is None:
            raise HTTPException(status_code=503, detail="Model not available in bundle")

        # Start with dataset median defaults for all 31 features
        input_dict = dict(FEATURE_DEFAULTS)

        # Override with user-provided morphological features
        user_input = features.model_dump()
        input_dict.update(user_input)

        # Recompute engineered features from user inputs
        nap = user_input['nucleus_area_pct']
        input_dict['nuclear_cytoplasmic_ratio'] = nap / max(100 - nap, 1)
        input_dict['shape_irregularity_index'] = (1 - user_input['circularity']) * user_input['eccentricity']
        input_dict['chromatin_granularity'] = user_input['chromatin_density'] * user_input['granularity_score']
        input_dict['size_deviation'] = abs(user_input['cell_diameter_um'] - FEATURE_DEFAULTS.get('cell_diameter_um', 12))
        input_dict['lobularity_size_ratio'] = user_input['lobularity_score'] / max(user_input['cell_diameter_um'], 1)
        input_dict['membrane_chromatin'] = user_input['membrane_smoothness'] * user_input['chromatin_density']

        # Build feature vector in the correct order
        feature_values = [input_dict.get(f, 0.0) for f in feature_names]
        X = np.array([feature_values])

        # Scale
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        # Predict
        pred = int(model.predict(X_scaled)[0])
        proba = float(model.predict_proba(X_scaled)[0][1])

        # Risk level
        if proba < 0.3:
            risk = "Low"
        elif proba < 0.6:
            risk = "Moderate"
        elif proba < 0.85:
            risk = "High"
        else:
            risk = "Critical"

        # Feature contributions (importance-weighted)
        contributions = {}
        importances = MODEL_METRICS['feature_importances']
        for feat in list(user_input.keys()):
            imp = importances.get(feat, 0.05)
            contributions[feat] = round(imp * proba, 4)

        return PredictionResponse(
            prediction=pred,
            label="Anomaly" if pred == 1 else "Normal",
            confidence=round(max(proba, 1 - proba), 4),
            anomaly_probability=round(proba, 4),
            risk_level=risk,
            feature_contributions=contributions,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": bool(model_bundle),
        "stats_loaded": bool(dataset_stats),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
