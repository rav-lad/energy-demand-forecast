FROM python:3.10-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY dashboard/ ./dashboard/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ── data-collector ────────────────────────────────────────────────────────────
FROM base AS data-collector
CMD ["python", "scripts/update_data.py"]

# ── trainer ───────────────────────────────────────────────────────────────────
FROM base AS trainer
CMD ["python", "scripts/train.py"]

# ── trainer-quantile ──────────────────────────────────────────────────────────
FROM base AS trainer-quantile
CMD ["python", "scripts/train_quantile.py"]

# ── inference ─────────────────────────────────────────────────────────────────
FROM base AS inference
CMD ["python", "scripts/infer.py"]

# ── backtest ──────────────────────────────────────────────────────────────────
FROM base AS backtest
CMD ["python", "scripts/run_backtest.py"]

# ── dashboard ─────────────────────────────────────────────────────────────────
FROM base AS dashboard
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
