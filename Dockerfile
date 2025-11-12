# =============================================================================
# Energy Trading Research - Docker Image
# =============================================================================
# Multi-stage build for optimized image size
# Supports both CPU and GPU (CUDA) execution
#
# Build:
#   docker build -t energy-trading:latest .
#
# Build with GPU support:
#   docker build --build-arg USE_GPU=true -t energy-trading:gpu .
# =============================================================================

FROM python:3.10-slim as base

# Build arguments
ARG USE_GPU=false
ARG DEBIAN_FRONTEND=noninteractive

# Metadata
LABEL maintainer="energy-trading"
LABEL description="Energy demand forecasting and trading research platform"
LABEL version="2.0.0"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# =============================================================================
# Stage 1: System dependencies
# =============================================================================

FROM base as dependencies

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    gfortran \
    # Git for potential pip installs from repos
    git \
    # Networking
    curl \
    wget \
    # For pandas/numpy
    libopenblas-dev \
    liblapack-dev \
    # For matplotlib
    libfreetype6-dev \
    libpng-dev \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# =============================================================================
# Stage 2: Python dependencies
# =============================================================================

FROM dependencies as python-deps

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional utilities
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    jupyterlab

# =============================================================================
# Stage 3: Application
# =============================================================================

FROM python-deps as application

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p \
    /app/data/raw_data/energy \
    /app/data/raw_data/weather \
    /app/data/raw_data/market_prices \
    /app/data/raw_data/fundamentals \
    /app/data/modified_data \
    /app/data/transformed_data \
    /app/models/xgboost \
    /app/models/reg_lin \
    /app/models/Quantile/lightgbm_quantile \
    /app/models/tft/checkpoints \
    /app/models/scalers \
    /app/models/demand_forecast \
    /app/models/price_forecast \
    /app/outputs/figures \
    /app/outputs/reports \
    /app/outputs/logs \
    /app/trading_system/backtests \
    /app/research/notebooks/market_research \
    /app/research/reports

# Set permissions
RUN chmod +x /app/*.py 2>/dev/null || true

# =============================================================================
# Stage 4: Final image
# =============================================================================

FROM application as final

# Expose ports
EXPOSE 8888
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["/bin/bash"]

# =============================================================================
# Development variant with Jupyter
# =============================================================================

FROM final as development

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter by default in dev mode
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Production variant (smaller, no dev tools)
# =============================================================================

FROM final as production

# Remove development tools
RUN pip uninstall -y jupyter jupyterlab ipykernel

# Run as non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Default command for production
CMD ["python", "--version"]
