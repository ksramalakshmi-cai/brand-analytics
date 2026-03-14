# ── Brand Analytics – SageMaker-compatible GPU container ──────────────
#
# Build:
#   docker build -t brand-analytics .
#
# Run locally:
#   docker run --gpus all -p 8000:8000 \
#     -e BA_MONGO_URI="mongodb+srv://..." \
#     -e GOOGLE_API_KEY="..." \
#     -e AWS_ACCESS_KEY_ID="..." \
#     -e AWS_SECRET_ACCESS_KEY="..." \
#     brand-analytics
#
# SageMaker deploys this container automatically; it calls /ping for
# health checks and /invocations for inference.  The same image works
# standalone (port 8000) and behind a SageMaker endpoint (port 8080).
# ─────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="brand-analytics"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    # SageMaker sets this env var; default to 8000 for standalone use
    SAGEMAKER_BIND_TO_PORT=8080 \
    # Reduce MKL/OpenMP thread contention inside the container
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

# System deps for OpenCV, ffmpeg, PaddleOCR, etc.
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

# Python deps (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY config.py db.py pipeline.py api.py serve ./
COPY src/ src/
COPY labels.yaml ./

# Default logos dir (can be overridden by S3 at runtime)
COPY logos/ logos/

# The 'serve' script is the SageMaker entry point
RUN chmod +x /app/serve

EXPOSE 8000
EXPOSE 8080

# SageMaker invokes 'serve'; standalone uses CMD directly
# When SAGEMAKER_BIND_TO_PORT is set, 'serve' binds to that port.
# For standalone/ECS/docker-compose, the CMD below runs on port 8000.
ENTRYPOINT ["/app/serve"]
