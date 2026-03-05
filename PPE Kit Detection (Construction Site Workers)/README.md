# 🦺 PPE Kit Detection — Construction Site Safety

An end-to-end **YOLO11s** object detection pipeline that monitors PPE compliance on construction sites, paired with a **React + TypeScript** frontend and a **FastAPI** backend deployed to the cloud.

🚀 **Live Demo:** [ppe-detection-api-orpin.vercel.app](https://ppe-detection-api-orpin.vercel.app)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Frontend](#frontend)
- [Deployment](#deployment)
- [Pipeline Summary](#pipeline-summary)
- [Results](#results)

---

## Overview

The system detects 11 classes across construction-site images and video streams — identifying both **correctly worn PPE** and **safety violations** (missing PPE) in real time.

| Type                    | Classes                                                |
| ----------------------- | ------------------------------------------------------ |
| ✅ Positive (PPE worn)  | Helmet · Gloves · Vest · Boots · Goggles · Person |
| ❌ Negative (violation) | no_helmet · no_goggle · no_gloves · no_boots        |
| ⬜ Neutral              | none                                                   |

---

## Dataset

**Source:** [Kaggle — PPE Kit Detection (Construction Site Workers)](https://www.kaggle.com/datasets/ketakichalke/ppe-kit-detection-construction-site-workers)

Place the raw dataset under `data/` before running the notebook:

```
PPE Kit Detection (Construction Site Workers)/
└── data/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

The notebook performs a clean **stratified 75 / 15 / 10 re-split**, pHash deduplication, and rare-class augmentation automatically.

---

## Model Performance

| Split          | mAP50            | mAP50-95         |
| -------------- | ---------------- | ---------------- |
| Validation     | 0.4719           | 0.2216           |
| **Test** | **0.5385** | **0.2780** |

- **Architecture:** YOLO11s (~9.4 M params)
- **Epochs:** 120 · **Batch:** 32 · **Image size:** 640 × 640
- **Optimizer:** AdamW · **LR schedule:** Cosine
- **Augmentation:** HFlip · Brightness/Contrast · HSV · GaussNoise · Mixup · Copy-Paste · Rotation ±10°

---

## Project Structure

```
PPE Kit Detection (Construction Site Workers)/
├── construction_kit.ipynb          # Main training notebook (local)
├── construction_kit_kaggle.ipynb   # Kaggle-ready training notebook
├── data/                           # Raw dataset (not committed)
│   ├── images/
│   └── labels/
├── Artifacts/                      # Saved model outputs
│   ├── best_ppe_model.pt           # PyTorch weights (18 MB)
│   ├── dataset.yaml
│   ├── model_metadata.json
│   └── ...
├── backend/                        # FastAPI inference server
│   ├── main.py                     # FastAPI app + YOLO11s endpoint
│   ├── requirements.txt
│   └── Dockerfile                  # For HuggingFace Spaces deployment
└── frontend/                       # React + TypeScript web app
    ├── src/
    │   ├── components/
    │   └── utils/
    ├── .env                        # Local: VITE_API_URL=http://localhost:7860
    ├── .env.example
    ├── vercel.json                 # Vercel deployment config
    └── package.json
```

---

## Quick Start

### 0 — Clone the repository

```bash
git clone https://github.com/SajjadKhanYousafzai/Kaggle-ML-Portfolio.git
cd "Kaggle-ML-Portfolio/PPE Kit Detection (Construction Site Workers)"
```

### 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2 — Run the training notebook

Open `construction_kit.ipynb` in VS Code or Jupyter and run all cells top-to-bottom. Trained weights are saved to `Artifacts/`.

> **GPU recommended.** The notebook runs on CPU but is ~10× faster on an NVIDIA GPU (tested on Colab T4).

### 3 — Run the backend

```bash
cd backend
pip install -r requirements.txt
python main.py
# → http://localhost:7860
```

### 4 — Run the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## Frontend

A React web app that sends images/video frames to the FastAPI backend for YOLO11s inference.

### Features

| Mode                | Description                                            |
| ------------------- | ------------------------------------------------------ |
| 📷 Live Webcam      | Real-time detection via `requestAnimationFrame` loop |
| 🖼️ Image Upload   | Drag & drop or file picker inference                   |
| 🎬 Video File       | Frame-by-frame detection on local video files          |
| 📊 Compliance Panel | Per-class confidence bars, worn / missing summary      |
| 📝 Violation Log    | Timestamped log with snapshots + CSV export            |

**Stack:** Vite 5 · React 18 · TypeScript 5 · Tailwind CSS 3 · FastAPI · YOLO11s

---

## Deployment

| Component | Platform | URL |
| --------- | -------- | --- |
| **Frontend** | Vercel | [ppe-detection-api-orpin.vercel.app](https://ppe-detection-api-orpin.vercel.app) |
| **Backend** | HuggingFace Spaces (Docker) | [sajjad-ali-shah-ppe-detection-api.hf.space](https://sajjad-ali-shah-ppe-detection-api.hf.space) |
| **Model weights** | HuggingFace Hub | [Sajjad-Ali-Shah/ppe-yolo11s](https://huggingface.co/Sajjad-Ali-Shah/ppe-yolo11s) |

### Deploy your own

1. Push `backend/` to a HuggingFace Docker Space (set `HF_TOKEN` as a repository secret)
2. Import the repo on [vercel.com](https://vercel.com), set root directory to `frontend/`
3. Add environment variable: `VITE_API_URL=https://<your-hf-space>.hf.space`
4. Deploy

---

## Pipeline Summary

| Step          | Details                                                        |
| ------------- | -------------------------------------------------------------- |
| Data download | Raw dataset placed in `data/`                                |
| EDA           | Class counts, GT box visualisation                             |
| Re-split      | Stratified 75 / 15 / 10 + pHash dedup                          |
| Augmentation  | Albumentations pipeline for rare negative classes (4× copies) |
| Training      | YOLO11s · AdamW · cosine LR · 120 epochs                    |
| Evaluation    | Per-class P / R / mAP50 / mAP50-95 on val & test               |
| Export        | `best.pt` + `best.onnx` saved to `Artifacts/`            |
| Frontend      | In-browser ONNX inference — no backend required               |
| Demo          | Live at [ppe-detection-api-orpin.vercel.app](https://ppe-detection-api-orpin.vercel.app) |

---

## Results

| Artifact         | Path                                                      |
| ---------------- | --------------------------------------------------------- |
| PyTorch weights  | `Artifacts/best_ppe_model.pt`                           |
| ONNX model       | `Artifacts/ppe_yolo11s/weights/best.onnx`               |
| Training curves  | `Artifacts/training_curves.png`                         |
| Confusion matrix | `Artifacts/ppe_yolo11s/confusion_matrix_normalized.png` |
| Test predictions | `Artifacts/test_predictions.png`                        |
| Metadata         | `Artifacts/model_metadata.json`                         |

---

*Part of a Kaggle Computer Vision portfolio — [SajjadKhanYousafzai/Kaggle-ML-Portfolio](https://github.com/SajjadKhanYousafzai/Kaggle-ML-Portfolio)*
