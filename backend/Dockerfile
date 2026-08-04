# --- Stage 1: Build Frontend React App ---
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend
# Copy package files first
COPY frontend/package*.json ./
# Install dependencies
RUN npm install
# Copy frontend source
COPY frontend/ ./
# Build production assets to /frontend/dist
RUN npm run build

# --- Stage 2: Build FastAPI Python App ---
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/cache/huggingface \
    EASYOCR_CACHE=/app/cache/easyocr \
    TORCH_HOME=/app/cache/torch \
    ENABLE_MOCK_FALLBACK=True

WORKDIR /app

# Install system dependencies needed by OpenCV and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to utilize Docker layer caching
COPY backend/requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend app source code and pre-download script
COPY backend/app/ ./app/
COPY backend/download_models.py .

# Copy the built frontend static assets from Stage 1 into the FastAPI app's static/ directory
COPY --from=frontend-builder /frontend/dist ./app/static

# Pre-download ML models so they are baked into the container image
RUN python download_models.py

# Expose port (Railway injects PORT env variable)
EXPOSE 8000

# Start Uvicorn server, binding to the injected PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
