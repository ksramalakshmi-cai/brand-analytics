FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL maintainer="brand-analytics"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    SAGEMAKER_BIND_TO_PORT=8080 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

# System deps for OpenCV, ffmpeg, OCR, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY config.py db.py pipeline.py api.py serve ./
COPY src/ src/
COPY labels.yaml ./
COPY logos/ logos/

RUN chmod +x /app/serve

EXPOSE 8080

ENTRYPOINT ["/app/serve"]