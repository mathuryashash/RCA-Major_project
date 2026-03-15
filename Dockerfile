# ---- Build stage (no spaCy model download at build time) ----
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for psycopg2, scipy/numpy, spaCy, and h5py
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Docker-optimized: CPU-only torch, no unused packages)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt \
    && pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY templates/ ./templates/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Expose FastAPI port
EXPOSE 8000

# Health check (start_period=60s for model loading: transformers, spaCy, etc.)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run FastAPI via Uvicorn
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
