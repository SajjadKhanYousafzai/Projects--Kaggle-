"""
PPE Kit Detection — FastAPI Backend
Model hosted on HuggingFace Hub: Sajjad-Ali-Shah/ppe-yolo11s
"""

import io
import os
import time

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PPE Kit Detection API",
    description="YOLO11s object detection for construction-site PPE compliance",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO_ID      = "Sajjad-Ali-Shah/ppe-yolo11s"
MODEL_FILENAME  = "best_ppe_model.pt"
HF_TOKEN        = os.getenv("HF_TOKEN")          # set as a Secret in HF Spaces
IMGSZ           = 640
DEFAULT_CONF    = 0.35
DEFAULT_IOU     = 0.45

CLASS_NAMES = {
    0: "Helmet",    1: "Gloves",    2: "Vest",
    3: "Boots",     4: "Goggles",   5: "none",
    6: "Person",    7: "no_helmet", 8: "no_goggle",
    9: "no_gloves", 10: "no_boots",
}
POSITIVE_CLASSES = {"Helmet", "Gloves", "Vest", "Boots", "Goggles", "Person"}
NEGATIVE_CLASSES = {"no_helmet", "no_goggle", "no_gloves", "no_boots"}

# ── Model (loaded once at startup) ────────────────────────────────────────────
model: YOLO | None = None


@app.on_event("startup")
async def load_model() -> None:
    global model
    print(f"📥 Downloading model from HuggingFace: {HF_REPO_ID}/{MODEL_FILENAME}")
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        token=HF_TOKEN,
    )
    model = YOLO(model_path)
    print(f"✅ Model loaded from {model_path}")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "model": HF_REPO_ID, "version": "1.0.0"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", tags=["inference"])
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG / PNG)"),
    conf: float = Query(DEFAULT_CONF, ge=0.05, le=0.95, description="Confidence threshold"),
    iou:  float = Query(DEFAULT_IOU,  ge=0.1,  le=0.9,  description="NMS IoU threshold"),
):
    """
    Run PPE detection on an uploaded image.

    Returns bounding boxes, class names, confidence scores and a
    compliance summary (worn PPE vs missing PPE).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet — please retry.")

    # ── Read image ────────────────────────────────────────────────────────────
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    # ── Inference ─────────────────────────────────────────────────────────────
    t0      = time.perf_counter()
    results = model(image, conf=conf, iou=iou, imgsz=IMGSZ, verbose=False)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    # ── Parse detections ──────────────────────────────────────────────────────
    detections = []
    wearing:  list[str] = []
    missing:  list[str] = []

    result = results[0]
    if result.boxes is not None:
        for box in result.boxes:
            cls_id     = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            cls_name   = CLASS_NAMES.get(cls_id, "unknown")

            detections.append({
                "class_id":   cls_id,
                "class_name": cls_name,
                "confidence": round(confidence, 4),
                "bbox":       [x1, y1, x2, y2],
            })

            if cls_name in NEGATIVE_CLASSES:
                missing.append(cls_name)
            elif cls_name in POSITIVE_CLASSES:
                wearing.append(cls_name)

    return {
        "detections": detections,
        "compliance": {
            "wearing":      list(set(wearing)),
            "missing":      list(set(missing)),
            "is_compliant": len(missing) == 0,
        },
        "inference_time_ms": elapsed,
        "image_size": {"width": image.width, "height": image.height},
    }


# ── Entry point (local dev) ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
