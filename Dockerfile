FROM python:3.10-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    COURSEGEN_COURSES_JSON=/app/data/textbooks/courses.json \
    COURSEGEN_CACHE_DIR=/app/.cache/coursegen \
    CHROMA_PERSIST_DIR=/app/OUTPUT_DATA2/emdeddings \
    PYTHONPATH=/app

RUN set -eux \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        ca-certificates \
        pkg-config \
        python3-dev \
        python3-venv \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        libjpeg-dev \
        zlib1g-dev \
        libpng-dev \
        libtiff-dev \
        libxml2-dev \
        libxslt1-dev \
        libgomp1 \
        libomp-dev \
        libhdf5-dev \
        libpoppler-cpp-dev \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        libtesseract-dev \
        unzip \
        ghostscript \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN set -eux \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/chromadb_storage /app/.cache/coursegen

COPY . .

ENTRYPOINT ["python", "-m", "services.QuestionRag.pipelines.question_generator"]
CMD ["--help"]
