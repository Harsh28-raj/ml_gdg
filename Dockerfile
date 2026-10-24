# PS-23 ML service — CPU-only, models baked into the image (no downloads at startup).
# Needs >= 2 GB RAM on the host (~620 MB steady, ~1 GB peak while loading models).
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf_cache \
    ML_DEVICE=cpu \
    ML_CACHE_DB=/app/data/embedding_cache.sqlite3 \
    PORT=8001

WORKDIR /app

# 1. Dependencies (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2. Download models into the image (cached unless the loader/config changes)
COPY app/__init__.py app/config.py app/models_loader.py app/
COPY scripts/download_models.py scripts/
RUN python scripts/download_models.py

# 3. App code + sample images
COPY app/ app/
COPY scripts/ scripts/
COPY sample_data/ sample_data/

# Run offline at runtime (models are already in the image), as a non-root user
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN useradd --create-home --uid 1000 appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health', timeout=4)" || exit 1

# One worker: each worker would load its own copy of the models
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
