# Backend image: FastAPI + CPU-only PyTorch, with CLIP and EasyOCR weights baked in so the
# container runs fully offline (spec sections 9, 10.1, 12.3).
# Layers are ordered from least to most frequently changed for build-cache efficiency.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    EASYOCR_MODULE_PATH=/models/easyocr

# Shared libraries used by opencv-python-headless / EasyOCR at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0t64 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) CPU-only torch/torchvision from the PyTorch index (avoids ~3 GB of CUDA wheels).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0

# 2) Everything else. torch==2.10.0 is already satisfied by the +cpu build, so pip keeps it.
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -c "import cv2, easyocr, torch, transformers; print('torch', torch.__version__, 'cv2', cv2.__version__)"

# 3) Pre-download CLIP + EasyOCR weights (only re-runs if the script or config.json changes).
COPY scripts/download_models.py scripts/download_models.py
COPY model_outputs/config.json /tmp/model_cfg/config.json
RUN python scripts/download_models.py /tmp/model_cfg \
    && rm -rf /tmp/model_cfg \
    && chmod -R a+rX /models

# From here on nothing may be downloaded at runtime.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODEL_DIR=/app/model_outputs \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

# 4) Application code and model artifacts (change most often, so they come last).
COPY pytest.ini .
COPY model_outputs/ model_outputs/
COPY app/ app/
COPY tests/ tests/

RUN useradd --uid 10001 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
