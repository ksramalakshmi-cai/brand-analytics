#!/usr/bin/env bash
# Run pipeline with env vars that reduce PaddleOCR segfaults on CPU-only machines.
# Usage: ./run_ocr.sh --input videoplayback.mp4
# Or:    ./run_ocr.sh --input videoplayback.mp4 --mode both

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

exec python3 pipeline.py "$@"
