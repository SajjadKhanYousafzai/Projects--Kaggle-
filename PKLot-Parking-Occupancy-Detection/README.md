# PKLot Parking Occupancy Detection

This project performs parking-space occupancy detection on the PKLot dataset
using object detection models. The main workflow compares YOLOv11, YOLOv12,
and RT-DETR for both accuracy and deployment efficiency.

## Overview

The notebook `PKLot-Parking-Occupancy-Detection.ipynb` includes:

- COCO annotation inspection and visualization
- COCO-to-YOLO label conversion
- `data.yaml` generation for Ultralytics training
- RT-DETR training and evaluation
- Comparative analysis of YOLOv11 vs YOLOv12 vs RT-DETR

Target classes:

- `space-empty`
- `space-occupied`

## Benchmark Summary

The following results are reported in the notebook comparison section:

| Model | IoU (overall) | FPS | FLOPs (G) | Model Size (MB) |
|---|---:|---:|---:|---:|
| YOLOv11 | 0.890 | 29.78 | 6.31 | 5.19 |
| YOLOv12 | 0.897 | 38.52 | 6.32 | 5.23 |
| RT-DETR | 0.920 | 19.10 | 103.44 | 63.11 |

Key takeaway: RT-DETR gives stronger localization quality, while YOLOv12 is
more practical for real-time deployment due to higher FPS and smaller size.

## Dataset

Source:
https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset/data

Expected local splits:

- `Dataset/train`
- `Dataset/valid`
- `Dataset/test`

Each split uses COCO annotations (`_annotations.coco.json`), which are
converted to YOLO labels under `custom_yolo_data`.

## Project Structure

```text
PKLot-Parking-Occupancy-Detection/
|-- PKLot-Parking-Occupancy-Detection.ipynb
|-- data.yaml
|-- requirements.txt
|-- rtdetr-l.pt
|-- Dataset/
|   |-- train/
|   |-- valid/
|   `-- test/
|-- custom_yolo_data/
|   |-- train/
|   |-- valid/
|   `-- test/
`-- runs/
	`-- detect/
```

## Setup

1. Create and activate a virtual environment.
2. Install project requirements.
3. Install notebook runtime dependencies used by the detection pipeline.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install ultralytics opencv-python pyyaml
```

## How To Run

1. Open `PKLot-Parking-Occupancy-Detection.ipynb` in VS Code, JupyterLab, or Kaggle.
2. Verify path variables in the notebook match your local folder layout.
3. Run cells in order:
   - dataset loading and visualization
   - COCO-to-YOLO conversion
   - `data.yaml` creation
   - model training/evaluation and comparison

## Optional Training Snippet

```python
from ultralytics import YOLO

model = YOLO("rtdetr-l.pt")
results = model.train(data="data.yaml", epochs=50, imgsz=640)
metrics = model.val(data="data.yaml")
```

## Outputs

Running the notebook generates:

- YOLO-format labels in `custom_yolo_data/*/labels`
- Training artifacts in `runs/detect/train*`
- Comparison tables and qualitative plots in notebook cells

Note: trained best/last checkpoints are typically produced during training,
but may not be committed to the repository.

## Notes

- The committed `rtdetr-l.pt` is used as the starting checkpoint for RT-DETR.
- Large raw datasets may be excluded from version control due file-size limits.
