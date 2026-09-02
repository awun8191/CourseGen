# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

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
    COURSEGEN_CACHE_DIR=/app/output_data/cache \
    CHROMA_PERSIST_DIR=/app/output_data/vector_database \
    PYTHONPATH=/app

RUN set -eux \
  && for i in 1 2 3; do apt-get update && break || { echo "retry $i"; sleep 5; }; done \
  && apt-get install -y --no-install-recommends \
    build-essential pkg-config python3-dev \
    ca-certificates curl \
    libglib2.0-0 libgl1-mesa-dri libglx-mesa0 libsm6 libxrender1 libxext6 libfontconfig1 libice6 \
    libjpeg-dev libpng-dev libtiff-dev zlib1g-dev \
    libgomp1 libhdf5-dev libblas-dev liblapack-dev libopenblas-dev \
    libpoppler-cpp-dev poppler-utils \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
  && pip install --timeout=300 --prefer-binary -r requirements.txt \
  && rm -rf /root/.cache

COPY . ./

RUN mkdir -p /app/output_data/vector_database /app/output_data/cache \
  && useradd -m appuser \
  && chown -R appuser:appuser /app

USER appuser

VOLUME ["/app/output_data/vector_database", "/app/output_data/cache"]

ENTRYPOINT ["python", "-m", "services.QuestionRag.pipelines.question_generator"]
CMD ["--help"]
