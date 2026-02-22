# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first — Docker layer cache: pip only re-runs when this changes
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# WORKDIR is /app — this is where uvicorn runs from
# The module path "app.main" means /app/app/main.py must exist
WORKDIR /app

# Copy installed packages from builder (no build tools in final image)
COPY --from=builder /install /usr/local

# Copy the ENTIRE project root into /app
# This means:
#   /app/app/main.py        ← uvicorn finds "app.main:app"
#   /app/app/config.py
#   /app/app/ingestion/...
#   /app/app/retrieval/...
#   /app/app/llm/...
COPY app/ ./app/
COPY requirements.txt .

# Create data directories (ChromaDB + uploads)
RUN mkdir -p /app/data/chroma_db /app/data/uploads && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Health check — used by docker-compose depends_on condition
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# uvicorn runs from WORKDIR=/app, finds module at app/main.py → "app.main:app"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
