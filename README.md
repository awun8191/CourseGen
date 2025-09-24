# CourseGen: Recursive PDF Extraction, RAG, and Automated Course Content Generation

CourseGen is a comprehensive, modular pipeline for processing educational materials (PDFs, images, textbooks) through OCR, embedding, retrieval-augmented generation (RAG), and automated content creation. It supports generating course outlines, interactive questions, and more, with features like resumable processing, multi-provider API integration (Gemini, Cloudflare, Ollama), robust caching, cost tracking, and scalable architecture.

## Key Features
- **End-to-End Pipeline**: From raw PDFs to searchable embeddings and AI-generated educational content.
- **Robust OCR**: Handles scanned documents with preprocessing, rotation detection, and tunable Tesseract parameters.
- **Intelligent RAG**: Semantic search over ChromaDB with metadata filtering (e.g., by course code, department).
- **Automated Generation**: Course outlines and questions using structured Gemini prompts.
- **Load Balancing**: Rotates API keys to handle high-volume requests without rate limits.
- **Observability**: Detailed logging, progress tracking, billing ledgers, and resumability.
- **Modularity**: Separate services for RAG, Question Generation, providers (Gemini/Cloudflare/Ollama/Firestore), and utilities.
- **Scalability**: Parallel processing, batching, and caching for large datasets (e.g., university course libraries).

The project is organized into `services/` (core pipelines), `data_models/` (Pydantic schemas), `utils/` (helpers), `data/` (inputs/outputs, gitignored), and `tests/` (PyTest suite).

## Project Structure Overview
- `services/RAG/`: PDF ingestion, OCR, chunking, embedding, and ChromaDB storage. See [detailed README](README_convert_to_embeddings.md).
- `services/QuestionRag/`: RAG-based course outline and question generation. See [Course Outline README](README_course_outline_generation.md) and [Question Generation README](README_question_generation.md).
- `services/Gemini/`: API client with key load balancing. See [API Key Load Balancer README](README_api_key_load_balancer.md).
- `services/{Cloudflare, Ollama, Firestore}/`: Provider-specific clients.
- `utils/`: Caching, data cleaning, progress tracking, and more.
- `data_models/`: Typed models for courses, questions, documents, etc.
- `data/`: Sample textbooks, courses.json, caches (gitignored).
- `specs/` and `steering/`: Design documents and high-level architecture.
- `tests/`: Unit/integration tests.

## Docker Setup
Use `docker-compose.yml` for containerized runs with **persistent volumes** that survive container rebuilds. The setup mounts embeddings, caches, and data directories to preserve your work across updates.

### 🚀 Quick Start with Persistent Volumes
```bash
# Build the optimized Docker image with persistent volume support
./build.sh

# Start with persistent volumes (recommended)
docker-compose up

# Or run specific service with volumes
docker-compose up coursegen-questions

# Generate questions for all courses (20 per subtopic: 10 theory + 10 calculation)
docker-compose run --rm coursegen --theory-per-request 10 --calc-per-request 10

# Generate questions for specific course
docker-compose run --rm coursegen --course-code "AAE 101" --theory-per-request 10 --calc-per-request 10
```

### 📁 Persistent Data Directories
Your data persists across container rebuilds in these locations:
- `./OUTPUT_DATA2/emdeddings/` - ChromaDB vector embeddings
- `./.cache/coursegen/` - Question generation caches
- `./data/` - Course data and configurations

### 🔧 Advanced Usage
```bash
# Interactive shell with persistent volumes
docker-compose run --rm coursegen bash

# Custom environment file with volumes
docker-compose --env-file .env.production up

# Run with specific settings (volumes automatically mounted)
docker-compose run --rm coursegen \
  --course-code "AAE 101" \
  --theory-per-request 10 \
  --calc-per-request 10 \
  --request-delay 2 \
  --temperature 0.7

# Debug mode with verbose output
docker-compose run --rm coursegen \
  --course-code "AAE 101" \
  --theory-per-request 5 \
  --calc-per-request 5 \
  --no-resume \
  --request-delay 2
```

### Available Courses
Check available courses in `data/textbooks/courses.json`:
```bash
# List all available course codes
grep '"code"' data/textbooks/courses.json | head -10

# Example courses: "AAE 101", "AAE 331", "AAE 335", etc.
```

### Question Generation Troubleshooting
- **"Course code not found"**: Check available courses in `data/textbooks/courses.json`
- **"No RAG context found"**: Ensure ChromaDB embeddings exist in `OUTPUT_DATA2/emdeddings/`
- **API errors**: Verify API keys in `.env` file are valid and have sufficient quota
- **0 questions generated**: Course may not have sufficient RAG context or outlines
- **Memory issues**: Reduce `--theory-per-request` and `--calc-per-request` values
- **Volume permission errors**: Ensure host directories have proper permissions (775 recommended)
- **Firestore errors**: Check Firebase credentials and network connectivity

### 🚀 Recent Improvements
- ✅ **Persistent Volumes**: Embeddings and caches now survive container rebuilds
- ✅ **Enhanced Reliability**: Added retry logic for network failures during build
- ✅ **Fixed Dependencies**: Resolved numpy/albumentations version conflicts
- ✅ **Better Error Handling**: Improved build script with debugging capabilities
- ✅ **Path Consistency**: Fixed typos and ensured consistent directory paths
- ✅ **Improved Health Checks**: Container now verifies ChromaDB embeddings directory exists
- ✅ **Optimized Docker Compose**: Cleaner configuration with better defaults
- ✅ **Comprehensive Documentation**: See [Docker README](DOCKER_README.md) for detailed troubleshooting

### Build Script Features
The `./build.sh` script now includes:
- **System Resource Checks**: Validates disk space and Docker daemon status
- **Retry Logic**: Automatically retries failed builds with exponential backoff
- **Debug Mode**: Provides detailed system information for troubleshooting
- **Cleanup Options**: Removes old images and containers to free space
- **Verbose Logging**: Shows detailed build progress and error information

### Dockerfile Optimizations
- **Multi-layer Caching**: Optimized layer structure for faster rebuilds
- **Network Resilience**: Automatic retry logic for apt-get operations
- **Security**: Non-root user with proper permissions
- **Health Checks**: Built-in monitoring and health verification
- **Resource Optimization**: Configured for optimal memory and CPU usage
- **Persistent Volume Support**: Proper permissions and ownership for mounted directories
- **Directory Structure**: Ensures all required directories exist with correct permissions

### 📋 Prerequisites
- **API Keys**: Ensure `.env` has valid API keys for Gemini, Cloudflare, and Firestore
- **Persistent Data**: Your embeddings and caches are preserved in:
  - `OUTPUT_DATA2/emdeddings/` (ChromaDB embeddings)
  - `.cache/coursegen/` (Generation caches)
  - `data/` (Course data and configurations)
- **Course Data**: Verify `data/textbooks/courses.json` contains your course outlines

## Quick Start

### 🐳 Docker Setup (Recommended - with Persistent Volumes)
1. **Build and Start**:
    ```bash
    # Build the optimized Docker image
    ./build.sh

    # Start with persistent volumes (data survives rebuilds)
    docker-compose up
    ```

2. **Configure Secrets**:
    ```bash
    # Copy and edit environment file
    cp .env.example .env
    # Edit .env with your API keys:
    # - GOOGLE_API_KEY (Gemini)
    # - CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN
    # - TESSDATA_PREFIX (Tesseract OCR path)
    ```

3. **Generate Embeddings** (one-time setup):
    ```bash
    # Process PDFs with persistent storage
    docker-compose run --rm coursegen \
      python -m services.RAG.convert_to_embeddings \
      -i data/textbooks/COMPILATION/EEE \
      --with-chroma \
      -c pdfs_bge_m3_cloudflare \
      --workers 4 \
      --resume
    ```

4. **Generate Questions** (20 per subtopic: 10 theory + 10 calculation):
    ```bash
    # Generate for all courses
    docker-compose run --rm coursegen \
      --theory-per-request 10 \
      --calc-per-request 10 \
      --request-delay 2

    # Or for specific course
    docker-compose run --rm coursegen \
      --course-code "EEE 315" \
      --theory-per-request 10 \
      --calc-per-request 10
    ```

5. **Run Tests**:
    ```bash
    docker-compose run --rm coursegen pytest tests/ -v
    ```

### 💻 Local Development Setup
1. **Setup Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # or .venv\Scripts\activate  # Windows
    pip install -r requirements.txt
    ```
    Install Tesseract OCR and set `TESSDATA_PREFIX` env var.

2. **Configure Secrets**: Same as Docker setup above.

3. **Process PDFs** (Embeddings):
    ```bash
    python -m services.RAG.convert_to_embeddings \
      -i data/textbooks/COMPILATION/EEE \
      --export-dir data/exported_data \
      --cache-dir data/ocr_cache \
      --with-chroma \
      -c pdfs_bge_m3_cloudflare \
      -p chromadb_storage \
      --workers 4 \
      --resume \
      --ocr-dpi 450
    ```

4. **Generate Questions**:
    ```bash
    python -m services.QuestionRag.pipelines.question_generator \
      --theory-per-request 10 \
      --calc-per-request 10 \
      --request-delay 2
    ```

5. **Run Tests**:
    ```bash
    pytest tests/ -v
    ```

## Detailed Documentation
For in-depth guides:
- [Course Outline Generation](README_course_outline_generation.md): Automating syllabi from RAG-retrieved content.
- [Question Generation](README_question_generation.md): Creating MCQs, essays, and more aligned with outlines.
- [API Key Load Balancer](README_api_key_load_balancer.md): Scaling Gemini requests across multiple keys.
- [Convert to Embeddings Pipeline](README_convert_to_embeddings.md): Core PDF processing and vectorization.
- [Other Components](README_other_components.md): Data models, utils, testing, and deployment.

## Environment Variables
- `TESSDATA_PREFIX`: Path to Tesseract data.
- `OCR_DPI`: Rendering DPI (default 300).
- `OMP_NUM_THREADS`: OCR threads (default 4).
- `CF_EMBED_MAX_BATCH`: Embedding batch size (≤100).
- `BILLING_ENABLED`: Track costs (1/0).
- `CF_PRICE_PER_M_TOKENS`: Cloudflare pricing (default 0.02 USD/M tokens).

## Contributing
- Follow PEP 8; add tests for new features.
- Use imperative commit messages (e.g., "Add rotation detection to OCR").
- Report issues at https://github.com/sst/opencode/issues (for tool feedback).
- For help: Run `/help` in opencode.

See `AGENTS.md` for agent-specific instructions. This project is licensed under MIT.