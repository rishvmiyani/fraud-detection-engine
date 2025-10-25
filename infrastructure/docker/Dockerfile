# =================================================================
# FRAUD DETECTION ENGINE - MULTI-STAGE DOCKERFILE
# =================================================================

# Build Arguments
ARG PYTHON_VERSION=3.12
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# =================================================================
# BASE STAGE - Common dependencies
# =================================================================
FROM python:UTF8{PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# =================================================================
# DEPENDENCIES STAGE - Install Python packages
# =================================================================
FROM base as dependencies

# Copy requirements files
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prod.txt

# =================================================================
# DEVELOPMENT STAGE
# =================================================================
FROM dependencies as development

# Install development dependencies
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =================================================================
# PRODUCTION STAGE
# =================================================================
FROM dependencies as production

# Copy only necessary files
COPY src/ src/
COPY config/ config/
COPY models/ models/
COPY alembic.ini ./
COPY requirements*.txt ./

# Create necessary directories
RUN mkdir -p logs data/processed data/models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Labels for metadata
LABEL maintainer="Fraud Detection Team <team@frauddetection.com>" \
      version="" \
      description="Fraud Detection Engine - Production Ready" \
      build-date="" \
      vcs-ref="" \
      vcs-url="https://github.com/fraud-detection/fraud-detection-engine"

# Production command
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# =================================================================
# TESTING STAGE
# =================================================================
FROM development as testing

# Copy test files
COPY tests/ tests/

# Install testing dependencies
RUN pip install --no-cache-dir pytest pytest-cov pytest-asyncio

# Run tests
RUN python -m pytest tests/ -v --cov=src --cov-report=html

# =================================================================
# FINAL STAGE - Choose target based on build arg
# =================================================================
FROM production as final
