# Garbage Classification Project

This project trains an image classification model for garbage sorting using transfer learning in PyTorch.

## Overview

The notebook `garbage-data.ipynb` builds an end-to-end pipeline to:
- Load and inspect the Garbage Classification V2 dataset
- Train a transfer learning model for multi-class waste classification
- Evaluate model performance using common classification metrics
- Export model artifacts for deployment

Target classes include:
- battery
- biological
- cardboard
- clothes
- glass
- metal
- paper
- plastic
- shoes
- trash

## Dataset

Source: Garbage Classification V2 on Kaggle
https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2

## Project Files

- `garbage-data.ipynb`: Full training and evaluation workflow
- `requirements.txt`: Python dependencies

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## How To Run

1. Open `garbage-data.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
2. Update dataset paths if needed.
3. Run all cells in order.

## Outputs

The notebook can generate model and metadata artifacts, including:
- Trained PyTorch checkpoint
- TorchScript model
- Optional ONNX model export
- Class-name mapping JSON

Note: ONNX export may require both `onnx` and `onnxscript`, which are included in `requirements.txt`.

## Notes

- The notebook is designed for both Kaggle and local environments.
- If running locally, make sure your dataset folder structure matches the expected class-folder format.
