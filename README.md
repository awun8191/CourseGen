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
- **Docker Integration**: Full containerization with persistent volumes and AWS ECR deployment.
- **Production Ready**: Optimized for deployment to AWS ECS, EKS, or other container platforms.

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

### ☁️ AWS ECR Deployment
Deploy your CourseGen application to AWS Elastic Container Registry for production use:

```bash
# Fix Docker credential issues (if needed)
./build.sh --fix-credentials

# Build and deploy to AWS ECR
./build.sh --deploy

# Run from ECR with persistent volumes
./run.sh --course-code "EEE 315"

# Manual ECR authentication (if needed)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 888429341445.dkr.ecr.us-east-1.amazonaws.com
```

**ECR Repository**: `888429341445.dkr.ecr.us-east-1.amazonaws.com/rag:latest`

### 📁 Persistent Data Directories
Your data persists across container rebuilds in these locations:
- `./OUTPUT_DATA2/emdeddings/` - ChromaDB vector embeddings
- `./.cache/coursegen/` - Question generation caches
- `./data/` - Course data and configurations

### 🔄 Updating Embeddings Only
To update just the embeddings without rebuilding the entire container:

#### Method 1: Update Both Local Volume AND AWS Image (Complete Workflow)
```bash
# Step 1: Update local persistent volume embeddings
docker-compose run --rm coursegen \
  python -m services.RAG.convert_to_embeddings \
  -i data/textbooks/COMPILATION/EEE \
  --with-chroma \
  -c pdfs_bge_m3_cloudflare \
  --workers 4 \
  --resume

# Step 2: Rebuild image with updated embeddings
./build.sh --cleanup

# Step 3: Deploy updated image to AWS ECR
./build.sh --deploy
```

#### Method 2: Using Docker with Volume Mounts (Local Volume Only)
```bash
# Run embeddings generation with persistent volumes
docker run --rm -it \
  -v $(pwd)/OUTPUT_DATA2:/app/OUTPUT_DATA2 \
  -v $(pwd)/data:/app/data \
  888429341445.dkr.ecr.us-east-1.amazonaws.com/rag:latest \
  python -m services.RAG.convert_to_embeddings \
  -i data/textbooks/COMPILATION/EEE \
  --with-chroma \
  -c pdfs_bge_m3_cloudflare \
  --workers 4 \
  --resume
```

#### Method 3: Using Docker Compose (Local Volume Only)
```bash
# Update embeddings using docker-compose
docker-compose run --rm coursegen \
  python -m services.RAG.convert_to_embeddings \
  -i data/textbooks/COMPILATION/EEE \
  --with-chroma \
  -c pdfs_bge_m3_cloudflare \
  --workers 4 \
  --resume
```

#### Method 4: Manual Directory Replacement (Local Volume Only)
```bash
# Backup current embeddings
cp -r OUTPUT_DATA2/emdeddings OUTPUT_DATA2/emdeddings.backup.$(date +%s)

# Replace with new embeddings directory
rm -rf OUTPUT_DATA2/emdeddings
cp -r /path/to/new/emdeddings OUTPUT_DATA2/
```

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

### Docker & AWS ECR Troubleshooting
- **"Error saving credentials"**: Run `./build.sh --fix-credentials` to resolve credential helper issues
- **ECR authentication failed**: Check AWS CLI configuration and permissions
- **Image not found locally**: The run script will automatically pull from ECR if available
- **Permission denied on volumes**: Ensure host directories have proper permissions (775 recommended)
- **Build fails with "invalid tag"**: Use the fixed build script with correct ECR URI format
- **AWS CLI not found**: Install AWS CLI or authenticate manually with `aws ecr get-login-password`

### Embeddings-Specific Troubleshooting
- **"No RAG context found"**: Embeddings directory may be empty or corrupted
- **"ChromaDB connection failed"**: Check if `OUTPUT_DATA2/emdeddings/` exists and has content
- **"Permission denied on embeddings"**: Ensure proper ownership: `sudo chown -R $USER:$USER OUTPUT_DATA2/`
- **"Embeddings outdated"**: Regenerate using the embeddings update commands above
- **"Disk space full"**: Embeddings can be large; check available space with `df -h`
- **"ChromaDB locked"**: Stop all containers and try again: `docker-compose down`

### 🚀 Recent Improvements
- ✅ **AWS ECR Deployment**: Full integration with AWS Elastic Container Registry
- ✅ **Persistent Volumes**: Embeddings and caches now survive container rebuilds
- ✅ **Enhanced Reliability**: Added retry logic for network failures during build
- ✅ **Fixed Dependencies**: Resolved numpy/albumentations version conflicts
- ✅ **Better Error Handling**: Improved build script with debugging capabilities
- ✅ **Path Consistency**: Fixed typos and ensured consistent directory paths
- ✅ **Improved Health Checks**: Container now verifies ChromaDB embeddings directory exists
- ✅ **Optimized Docker Compose**: Cleaner configuration with better defaults
- ✅ **Docker Credential Helper Fix**: Automatic resolution of credential helper issues
- ✅ **Question Generation Fix**: Resolved "missing solution steps" error for calculation questions
- ✅ **Enhanced Scripts**: Added deployment, credential fixing, and debugging options
- ✅ **Automated Embeddings Update**: One-command workflow for updating embeddings in both local volume and AWS image
- ✅ **Comprehensive Documentation**: Complete guide for all features and troubleshooting

### Build Script Features
The `./build.sh` script now includes:
- **System Resource Checks**: Validates disk space and Docker daemon status
- **Retry Logic**: Automatically retries failed builds with exponential backoff
- **Debug Mode**: Provides detailed system information for troubleshooting
- **Cleanup Options**: Removes old images and containers to free space
- **Verbose Logging**: Shows detailed build progress and error information
- **AWS ECR Deployment**: Automated push to AWS Elastic Container Registry
- **Credential Helper Fix**: Resolves Docker credential helper configuration issues
- **Multiple Build Targets**: Support for full and minimal Dockerfiles
- **Health Verification**: Validates built images can run successfully

### Dockerfile Optimizations
- **Multi-layer Caching**: Optimized layer structure for faster rebuilds
- **Network Resilience**: Automatic retry logic for apt-get operations
- **Security**: Non-root user with proper permissions
- **Health Checks**: Built-in monitoring and health verification
- **Resource Optimization**: Configured for optimal memory and CPU usage
- **Persistent Volume Support**: Proper permissions and ownership for mounted directories
- **Directory Structure**: Ensures all required directories exist with correct permissions

### Run Script Features
The `./run.sh` script provides enhanced container execution with:
- **AWS ECR Integration**: Automatic authentication and image pulling from ECR
- **Persistent Volume Management**: Automatic mounting of data directories
- **Flexible Configuration**: Support for custom environment files and parameters
- **Interactive/Background Modes**: Choose between interactive and detached execution
- **Smart Prerequisites**: Validates Docker image availability and pulls from ECR if needed
- **Error Recovery**: Graceful handling of authentication and network issues

### 📋 Prerequisites
- **API Keys**: Ensure `.env` has valid API keys for Gemini, Cloudflare, and Firestore
- **Persistent Data**: Your embeddings and caches are preserved in:
  - `OUTPUT_DATA2/emdeddings/` (ChromaDB embeddings)
  - `.cache/coursegen/` (Generation caches)
  - `data/` (Course data and configurations)
- **Course Data**: Verify `data/textbooks/courses.json` contains your course outlines
- **AWS ECR (Optional)**: For deployment, ensure AWS CLI is configured with proper permissions

### Enhanced Script Usage
The enhanced build and run scripts provide powerful deployment and management features:

#### Build Script (`./build.sh`)
```bash
# ONE COMMAND: Update embeddings in volume, rebuild image, and deploy to AWS
./build.sh --update-embeddings

# Basic build
./build.sh

# Build with cleanup
./build.sh --cleanup

# Build and deploy to AWS ECR
./build.sh --deploy

# Fix Docker credential issues
./build.sh --fix-credentials

# Debug build issues
./build.sh --debug

# Build minimal version
./build.sh --minimal

# Show all options
./build.sh --help
```

#### Run Script (`./run.sh`)
```bash
# Basic usage
./run.sh

# Run specific course
./run.sh --course-code "EEE 315"

# Custom question counts
./run.sh --theory-per-request 5 --calc-per-request 3

# Interactive mode
./run.sh -i --course-code "AAE 101"

# Background mode
./run.sh -b --course-code "EEE 315"

# Debug mode
./run.sh --debug --course-code "AAE 101"

# Custom environment file
./run.sh --env-file .env.production --course-code "EEE 471"

# Show all options
./run.sh --help
```

#### Docker Compose Commands
```bash
# Start all services
docker-compose up

# Run question generation
docker-compose run --rm coursegen --course-code "EEE 315"

# Update embeddings (with proper command override)
docker-compose run --rm \
  -e PYTHONPATH=/app \
  coursegen \
  python -m services.RAG.convert_to_embeddings \
  -i data/textbooks/COMPILATION/EEE \
  --with-chroma \
  -c pdfs_bge_m3_cloudflare \
  --workers 4 \
  --resume

# Run tests
docker-compose run --rm coursegen pytest tests/ -v
```

### Embeddings Management
Manage ChromaDB embeddings independently of the main application:

#### Complete Workflow: Update Both Local Volume AND AWS Image
```bash
# ONE COMMAND: Update embeddings, rebuild image, and deploy to AWS
./build.sh --update-embeddings

# OR manually (3-step process):
# Step 1: Update local persistent volume embeddings
docker run --rm \
  -v $(pwd)/OUTPUT_DATA2:/app/OUTPUT_DATA2 \
  -v $(pwd)/data:/app/data \
  888429341445.dkr.ecr.us-east-1.amazonaws.com/rag:latest \
  python -m services.RAG.convert_to_embeddings \
  -i data/textbooks/COMPILATION/EEE \
  --with-chroma \
  -c pdfs_bge_m3_cloudflare \
  --workers 4 \
  --resume

# Step 2: Rebuild image with updated embeddings
./build.sh --cleanup

# Step 3: Deploy updated image to AWS ECR
./build.sh --deploy
```

#### Generate/Update Embeddings (Local Volume Only)
```bash
# Using docker run with persistent volumes (recommended)
docker run --rm \
  -v $(pwd)/OUTPUT_DATA2:/app/OUTPUT_DATA2 \
  -v $(pwd)/data:/app/data \
  888429341445.dkr.ecr.us-east-1.amazonaws.com/rag:latest \
  python -m services.RAG.convert_to_embeddings \
  -i data/textbooks/COMPILATION/EEE \
  --with-chroma \
  -c pdfs_bge_m3_cloudflare \
  --workers 4 \
  --resume

# Using docker-compose (alternative)
docker-compose run --rm \
  -e PYTHONPATH=/app \
  coursegen \
  python -m services.RAG.convert_to_embeddings \
  -i data/textbooks/COMPILATION/EEE \
  --with-chroma \
  -c pdfs_bge_m3_cloudflare \
  --workers 4 \
  --resume
```

#### Verify Embeddings
```bash
# Check embeddings directory exists and has content
ls -la OUTPUT_DATA2/emdeddings/

# Verify ChromaDB is accessible
docker-compose run --rm coursegen \
  python -c "from services.RAG.chroma_store import ChromaStore; print('ChromaDB accessible')"
```

#### Backup/Restore Embeddings
```bash
# Create backup
tar -czf embeddings_backup_$(date +%Y%m%d_%H%M%S).tar.gz OUTPUT_DATA2/emdeddings/

# Restore from backup
tar -xzf embeddings_backup_20250101_120000.tar.gz
```

#### Understanding Persistent vs Image Embeddings
- **Persistent Volume** (`OUTPUT_DATA2/emdeddings/`): Survives container rebuilds, used for development
- **Image Embeddings**: Baked into the Docker image, used for production deployments
- **AWS ECR Image**: Contains embeddings that are deployed to AWS services

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

    # Or use the enhanced run script
    ./run.sh --course-code "EEE 315"
    ```

5. **Deploy to AWS ECR**:
    ```bash
    # ONE COMMAND: Update embeddings and deploy to AWS ECR
    ./build.sh --update-embeddings

    # OR manually:
    # Fix credential issues (if needed)
    ./build.sh --fix-credentials

    # Build and deploy to AWS ECR
    ./build.sh --deploy

    # Run from ECR
    ./run.sh --course-code "EEE 315"
    ```

6. **Run Tests**:
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

## Project Status
This CourseGen project is **production-ready** with comprehensive Docker integration and AWS ECR deployment capabilities. All major components have been implemented and tested:

### ✅ Completed Features
- **Full Docker Integration**: Containerized application with persistent volumes
- **AWS ECR Deployment**: Automated deployment to AWS Elastic Container Registry
- **Embeddings Management**: Complete workflow for updating ChromaDB embeddings
- **Question Generation**: Fixed all known issues including solution steps validation
- **Error Handling**: Comprehensive error recovery and troubleshooting
- **Documentation**: Complete guides for all features and use cases

### 🚀 Ready for Production
- Deploy to AWS ECS, EKS, or other container platforms
- Scale horizontally with multiple container instances
- Use in CI/CD pipelines for automated updates
- Monitor and manage through Docker and AWS tools

## Contributing
- Follow PEP 8; add tests for new features.
- Use imperative commit messages (e.g., "Add rotation detection to OCR").
- Report issues at https://github.com/sst/opencode/issues (for tool feedback).
- For help: Run `/help` in opencode.

See `AGENTS.md` for agent-specific instructions. This project is licensed under MIT.