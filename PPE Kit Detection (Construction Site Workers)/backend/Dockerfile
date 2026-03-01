# ── HuggingFace Spaces — Dockerfile Space ────────────────────────────────────
# Port 7860 is required by HF Spaces.
# Set HF_TOKEN as a Repository Secret in the Space settings (not in this file).

FROM python:3.10-slim

WORKDIR /app

# System libs needed by OpenCV / ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

EXPOSE 7860

CMD ["python", "main.py"]
