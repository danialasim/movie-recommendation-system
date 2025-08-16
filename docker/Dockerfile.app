# Optimized multi-stage build for Movie Recommendation System
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set pip timeout and retries for better network handling
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_RETRIES=5

# Copy and install Python dependencies (use Docker-optimized requirements)
COPY docker/requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir --user --timeout=300 -r requirements.txt

# Production stage
FROM python:3.10-slim as production

# Set working directory
WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy the application code (optimized layer order)
COPY src/ ./src/
COPY config/ ./config/
COPY static/ ./static/
COPY templates/ ./templates/
COPY models/ ./models/
COPY *.py ./
COPY config.yaml ./

# Copy and make entrypoint executable
COPY docker/entrypoint-app.sh ./entrypoint-app.sh
RUN chmod +x ./entrypoint-app.sh

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONPATH=/app:/app/src
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8001 8002

# Health check with timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Use entrypoint script
ENTRYPOINT ["./entrypoint-app.sh"]

# Default command
CMD ["python", "-m", "uvicorn", "src.api.enhanced_fastapi_app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
