# Multi-stage build for production readiness
FROM python:3.11-slim as builder

WORKDIR /app

# Install system compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies to local prefix
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runner stage
FROM python:3.11-slim as runner

WORKDIR /app

# Install runtime dependencies (like curl for healthcheck validation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application source code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8501
ENV CHROMA_DB_DIR=/app/chroma_db
ENV PDFS_DIR=/app/pdfs

# Expose port
EXPOSE 8501

# Healthcheck checking Streamlit status
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run command
ENTRYPOINT ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
