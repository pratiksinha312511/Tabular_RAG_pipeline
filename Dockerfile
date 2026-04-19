# ── Stage 1: Build dependencies ──
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies for kaleido/plotly chart rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Production image ──
FROM python:3.12-slim

WORKDIR /app

# Kaleido needs chromium libs for server-side chart rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libxshmfence1 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY config.py .
COPY server.py .
COPY pipeline.py .
COPY guardrails.py .
COPY llm_client.py .
COPY cache.py .
COPY categories.py .
COPY visualizations.py .
COPY main.py .

# Copy static assets and data
COPY static/ ./static/
COPY data/ ./data/

# Create directories for runtime output
RUN mkdir -p output logs

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Railway sets PORT automatically; default to 8000 for local Docker
ENV PORT=8000
ENV ENV=production

EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/api/users')" || exit 1

# Start server — Railway injects PORT env var
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT} --workers 2
