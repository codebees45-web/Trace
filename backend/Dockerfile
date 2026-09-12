# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# TRACE backend — production image
#
# Notes on size/runtime:
#   - torch + diffusers pull in a large dependency tree; this image is not
#     small (expect 4-6GB). That's expected for a Stable Diffusion pipeline.
#   - If you don't need face reconstruction, set ENABLE_GENERATION=false at
#     runtime (see .env.example) — the model still won't be downloaded until
#     first load, but you can also strip diffusers/transformers/accelerate
#     from requirements.txt entirely for a much smaller image.
#   - CPU inference works but is slow (SD_INFERENCE_STEPS controls the
#     quality/speed tradeoff). For real throughput, run on a GPU host and
#     switch the base image to a CUDA-enabled one (nvidia/cuda:...-runtime)
#     plus the matching torch wheel.
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS base

# OpenCV needs these system libs even in headless mode; libgomp1 is required
# by torch/scikit-learn's OpenMP-based kernels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Don't ship the training notebook, a dev sqlite DB, or __pycache__ into the image.
RUN rm -rf __pycache__ *.ipynb trace.db .env

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

# SQLite DB and any generated files live here — mount a volume at this path
# in production so data survives container restarts/redeploys.
ENV DATABASE_URL=sqlite:////app/data/trace.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 1 worker: the SD pipeline is already lock-guarded and loaded once per
# process; each additional worker duplicates the model in memory. Scale
# horizontally (more containers behind a load balancer) instead of with
# --workers if you need more predict throughput.
CMD ["gunicorn", "main:app", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]