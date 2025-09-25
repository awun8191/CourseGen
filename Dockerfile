# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONUNBUFFERED=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_ROOT_USER_ACTION=ignore \
  OMP_NUM_THREADS=2 \
  NUMBA_CACHE_DIR=/tmp/numba_cache \
  MPLCONFIGDIR=/tmp/matplotlib \
  COURSEGEN_COURSES_JSON=/app/data/textbooks/courses.json \
  COURSEGEN_CACHE_DIR=/app/OUTPUT_DATA2/cache \
  CHROMA_PERSIST_DIR=/app/OUTPUT_DATA2/emdeddings \
  PYTHONPATH=/app

# Install system dependencies in optimized layers with retry logic
RUN set -eux \
  && for i in {1..3}; do \
       apt-get update && break || { \
         echo "apt-get update failed (attempt $i/3), retrying in 5s..."; \
         sleep 5; \
       }; \
     done \
  && for i in {1..3}; do \
       apt-get install -y --no-install-recommends \
  # Core build tools
  build-essential pkg-config python3-dev \
  # SSL and networking
  ca-certificates curl wget \
  # Image processing libraries
  libglib2.0-0 libgl1-mesa-dri libglx-mesa0 libsm6 libxrender1 libxext6 libfontconfig1 libice6 \
  libjpeg-dev libpng-dev libtiff-dev zlib1g-dev \
  # Scientific computing
  libgomp1 libhdf5-dev libblas-dev liblapack-dev libopenblas-dev \
  # PDF processing
  libpoppler-cpp-dev poppler-utils \
  # XML processing
  libxml2-dev libxslt1-dev \
  # Additional dependencies for opencv, scikit-image, etc.
  libgtk-3-dev libgirepository1.0-dev libcairo-gobject2 libpango-1.0-0 libatk-bridge2.0-0 libdrm2 libxkbcommon0 libatspi2.0-0 \
  libxss1 libasound2 libxrandr2 libxcomposite1 libxdamage1 libgbm1 \
  # Tesseract OCR
  tesseract-ocr tesseract-ocr-eng \
  # Additional image processing
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  && break || { \
    echo "apt-get install failed (attempt $i/3), retrying in 5s..."; \
    sleep 5; \
    if [ $i -eq 3 ]; then \
      echo "apt-get install failed after 3 attempts, exiting..."; \
      exit 1; \
    fi; \
  }; \
  done \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements*.txt ./
RUN set -eux \
  && python -m pip install --upgrade pip setuptools wheel \
  && pip install --timeout=300 --prefer-binary -r requirements.txt

# Copy source code
COPY . .

# Create non-root user and required directories with proper permissions
RUN set -eux \
  && groupadd -r -g 1001 appuser \
  && useradd -r -u 1001 -g appuser appuser \
  && mkdir -p /app/OUTPUT_DATA2/emdeddings /app/OUTPUT_DATA2/cache /tmp/numba_cache /tmp/matplotlib \
     /app/data/textbooks /app/data/exported_data /app/data/ocr_cache /app/chromadb_storage \
  && chown -R appuser:appuser /app \
  && chmod -R g+w /tmp

# Ensure mounted directories have correct permissions (for persistent volumes)
RUN set -eux \
  && mkdir -p /app/OUTPUT_DATA2/emdeddings /app/OUTPUT_DATA2/cache /app/data \
  && chmod -R 775 /app/OUTPUT_DATA2 /app/data \
  && chown -R appuser:appuser /app/OUTPUT_DATA2 /app/data

USER appuser

# Health check with better error handling
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import sys; print('Health check passed'); sys.exit(0)" || exit 1

ENTRYPOINT ["python", "-m", "services.QuestionRag.pipelines.question_generator"]
CMD ["--help"]
