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

# Kaleido needs Chromium for server-side chart rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Tell Kaleido / Plotly where Chromium lives
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROME_PATH=/usr/bin/chromium
ENV CHROMIUM_PATH=/usr/bin/chromium
ENV CHROMIUM_FLAGS="--no-sandbox --disable-gpu --disable-dev-shm-usage"

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
