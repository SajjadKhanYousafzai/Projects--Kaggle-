# 🦺 PPE Kit Detection — Construction Site Safety

An end-to-end **YOLO11s** object detection pipeline that monitors PPE compliance on construction sites, paired with a fully in-browser **React + TypeScript** frontend powered by ONNX Runtime Web.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Frontend](#frontend)
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
├── requirements.txt                # Python dependencies
├── data/                           # Raw dataset (not committed)
│   ├── images/
│   └── labels/
├── Artifacts/                      # Saved model outputs
│   ├── best_ppe_model.pt           # PyTorch weights (18 MB)
│   ├── ppe_yolo11s/
│   │   └── weights/
│   │       └── best.onnx           # ONNX export (36 MB)
│   ├── model_metadata.json
│   ├── dataset.yaml
│   ├── training_log.txt
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── ...
└── frontend/                       # React + TypeScript in-browser app
    ├── src/
    │   ├── components/
    │   ├── utils/
    │   │   ├── detector.ts         # ONNX inference + NMS
    │   │   └── drawing.ts          # Canvas annotation
    │   └── types/
    ├── public/
    │   ├── models/best.onnx
    │   └── ort-wasm-*.wasm
    └── package.json
```

---

## Quick Start

### 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2 — Run the training notebook

Open `construction_kit_colabb.ipynb` in VS Code or Jupyter and run all cells top-to-bottom.Trained weights are saved to `Artifacts/`.

> **GPU recommended.** The notebook runs on CPU but is ~10× faster on an NVIDIA GPU (tested on Colab T4).

---

## Frontend

A zero-server-needed web app that runs ONNX inference entirely in the browser.

### Features

| Mode                | Description                                            |
| ------------------- | ------------------------------------------------------ |
| 📷 Live Webcam      | Real-time detection via `requestAnimationFrame` loop |
| 🖼️ Image Upload   | Drag & drop or file picker inference                   |
| 🎬 Video File       | Frame-by-frame detection on local video files          |
| 📊 Compliance Panel | Per-class confidence bars, worn / missing summary      |
| 📝 Violation Log    | Timestamped log with snapshots + CSV export            |

### Run locally

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Build for production

```bash
npm run build
# Output: frontend/dist/
```

**Stack:** Vite 5 · React 18 · TypeScript 5 · Tailwind CSS 3 · ONNX Runtime Web 1.17

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
| Demo          | Gradio app (Step 14 in notebook)                               |

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

*Part of a Kaggle Computer Vision portfolio — [SajjadKhanYousafzai/Projects--Kaggle-](https://github.com/SajjadKhanYousafzai/Projects--Kaggle-)*
