# CourseGen Docker Setup

This document provides comprehensive instructions for building and running CourseGen using Docker.

For project overview and general setup, see the [main README](../README.md).

## Recent Improvements (v2.0)

### ✅ Build Reliability Enhancements
- **Network Resilience**: Added automatic retry logic for apt-get operations with exponential backoff
- **Dependency Resolution**: Fixed numpy/albumentations version conflicts (numpy>=1.24.4)
- **Error Handling**: Removed problematic pip cache purge operations
- **Path Consistency**: Fixed typos in directory paths across all Docker files

### ✅ Build Script Improvements
- **System Validation**: Pre-build checks for disk space and Docker daemon status
- **Debug Capabilities**: Enhanced logging and system information reporting
- **Retry Logic**: Automatic retry for failed builds with configurable attempts
- **Cleanup Options**: Better management of old images and containers

### ✅ Dockerfile Optimizations
- **Multi-layer Caching**: Optimized layer structure for faster incremental builds
- **Security Hardening**: Non-root user with proper permissions and health checks
- **Resource Optimization**: Configured for optimal memory and CPU usage
- **Monitoring**: Built-in health checks and container monitoring

## Quick Start

1. **Build the image:**
   ```bash
   ./build.sh
   ```

2. **Run with default help:**
   ```bash
   ./run.sh
   ```

3. **Generate questions:**
   ```bash
   ./run.sh --generate-questions
   ```

## Prerequisites

- Docker installed and running
- At least 4GB of available RAM
- 10GB of free disk space
- API keys for Google Gemini and Cloudflare Workers AI

## Environment Setup

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your API keys:**
   ```bash
   # Required API keys
   GOOGLE_API_KEY=your_google_api_key_here
   CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
   CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
   ```

## Building the Image

### Basic Build
```bash
./build.sh
```

### Build with Options
```bash
# Clean up old images before building
./build.sh --cleanup

# Skip image verification
./build.sh --no-verify

# Verbose output
./build.sh --verbose
```

### Manual Docker Build
```bash
docker build -t coursegen:latest .
```

## Running the Application

### Using the Run Script

The `run.sh` script provides convenient options:

```bash
# Show help
./run.sh

# Interactive mode
./run.sh -i --generate-questions

# Mount local data directories
./run.sh -m --generate-questions

# Use custom environment file
./run.sh --env-file .env.production --generate-questions
```

### Direct Docker Commands

```bash
# Basic question generation
docker run --rm -it coursegen:latest --generate-questions

# Generate questions for specific course
docker run --rm -it coursegen:latest --generate-questions --course-code "EEE 315"

# Generate course outlines
docker run --rm -it coursegen:latest --department_from "EEE 315"

# With environment file
docker run --rm -it --env-file .env coursegen:latest --generate-questions

# With volume mounts for data persistence
docker run --rm -it \
  -v $(pwd)/OUTPUT_DATA2:/app/OUTPUT_DATA2 \
  -v $(pwd)/.cache:/app/.cache \
  coursegen:latest --generate-questions
```

## Using Docker Compose

### Basic Usage
```bash
# Build and run with default command
docker-compose up

# Run question generation service
docker-compose --profile questions up coursegen-questions

# Run outline generation service
docker-compose --profile outlines up coursegen-outlines
```

### Background Services
```bash
# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Data Management

### Embedded Data
The Docker image includes:
- ChromaDB embeddings (`OUTPUT_DATA2/emdeddings/`)
- Course data (`data/textbooks/`)
- OCR cache (`data/ocr_cache/`)
- Application cache (`.cache/`)

### Volume Mounts (Optional)
For data persistence and updates:

```bash
# Mount all data directories
docker run --rm -it \
  -v $(pwd)/OUTPUT_DATA2:/app/OUTPUT_DATA2 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.cache:/app/.cache \
  coursegen:latest --generate-questions
```

### Docker Compose Volumes
Uncomment volume mounts in `docker-compose.yml`:

```yaml
volumes:
  - ./OUTPUT_DATA2:/app/OUTPUT_DATA2
  - ./.cache:/app/.cache
  - ./data:/app/data
```

## Configuration Options

### Question Generation
```bash
# Basic question generation
docker run --rm -it coursegen:latest --generate-questions

# Specific course
docker run --rm -it coursegen:latest \
  --generate-questions --course-code "EEE 315"

# Custom difficulty and batch size
docker run --rm -it coursegen:latest \
  --generate-questions \
  --theory-per-request 15 \
  --calc-per-request 8
```

### RAG Configuration
```bash
# Custom RAG parameters
docker run --rm -it \
  -e GEN_QG_RAG_TOPK=50 \
  -e GEN_QG_RAG_MIN_SIM=0.7 \
  coursegen:latest --generate-questions
```

### Performance Tuning
```bash
# Adjust threading and memory
docker run --rm -it \
  -e OMP_NUM_THREADS=8 \
  --memory=8g \
  --cpus=4 \
  coursegen:latest --generate-questions
```

## Troubleshooting

### Build Issues

1. **Out of disk space:**
    ```bash
    docker system prune -a
    ./build.sh --cleanup
    ```

2. **Memory issues during build:**
    ```bash
    # Increase Docker memory limit in Docker Desktop
    # Or build with smaller batch sizes
    ```

3. **Package installation failures:**
    ```bash
    # Check requirements.txt for conflicting versions
    # Try building without cache
    docker build --no-cache -t coursegen:latest .
    ```

4. **Network connectivity issues during apt-get:**
    ```bash
    # The Dockerfile includes retry logic for network issues
    # If build fails due to network, try again - it will retry automatically
    ./build.sh --verbose
    ```

5. **Numpy version conflicts:**
    ```bash
    # Fixed in requirements.txt - numpy>=1.24.4 for albumentations compatibility
    # If you encounter conflicts, check the specific error and update versions
    pip install numpy pandas pydantic requests  # Core fallback packages
    ```

6. **Pip cache issues:**
    ```bash
    # If you see "pip cache commands can not function since cache is disabled"
    # This is normal when using --no-cache-dir and can be ignored
    # The build script handles this gracefully
    ```

### Runtime Issues

1. **API key errors:**
   ```bash
   # Verify .env file exists and has correct keys
   cat .env | grep -E "(GOOGLE_API_KEY|CLOUDFLARE)"
   ```

2. **ChromaDB connection issues:**
   ```bash
   # Check if ChromaDB data exists
   ls -la OUTPUT_DATA2/emdeddings/
   
   # Verify permissions
   docker run --rm -it coursegen:latest ls -la /app/OUTPUT_DATA2/
   ```

3. **Memory issues:**
   ```bash
   # Increase Docker memory limit
   docker run --rm -it --memory=8g coursegen:latest --generate-questions
   ```

### Debugging

1. **Interactive shell:**
    ```bash
    docker run --rm -it coursegen:latest /bin/bash
    ```

2. **Check logs:**
    ```bash
    docker-compose logs -f coursegen
    ```

3. **Inspect image:**
    ```bash
    docker run --rm -it coursegen:latest python -c "
    import sys
    print('Python version:', sys.version)
    import chromadb
    print('ChromaDB version:', chromadb.__version__)
    "
    ```

4. **Build script debugging:**
    ```bash
    # Enable verbose output to see detailed build process
    ./build.sh --verbose --debug

    # Check system resources before building
    ./build.sh --debug

    # View build logs if build fails
    tail -50 /tmp/docker_build.log
    ```

5. **Test container functionality:**
    ```bash
    # Test if container starts correctly
    docker run --rm coursegen:latest --help

    # Test ChromaDB connectivity
    docker run --rm -it coursegen:latest python -c "
    from services.RAG.chroma_store import ChromaStore
    store = ChromaStore()
    print('ChromaDB connection successful')
    "
    ```

## Performance Optimization

### Build Optimization
- Use `.dockerignore` to exclude unnecessary files
- Multi-stage builds for smaller images
- Layer caching for faster rebuilds
- Retry logic for network failures during apt-get operations
- Optimized pip installation with proper caching
- Fixed dependency conflicts (numpy>=1.24.4 for albumentations compatibility)

### Runtime Optimization
- Adjust `OMP_NUM_THREADS` based on available CPUs
- Use SSD storage for ChromaDB data
- Increase memory allocation for large datasets

### Resource Limits
```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
    reservations:
      memory: 2G
      cpus: '1.0'
```

## Security Considerations

1. **API Keys:**
   - Never commit `.env` files to version control
   - Use Docker secrets for production deployments
   - Rotate API keys regularly

2. **Container Security:**
   - Runs as non-root user (`appuser`)
   - Minimal base image (Python slim)
   - No unnecessary packages installed

3. **Network Security:**
   - Uses custom Docker network
   - No exposed ports by default
   - HTTPS for all API communications

## Production Deployment

### Docker Swarm
```bash
docker stack deploy -c docker-compose.yml coursegen
```

### Kubernetes
```bash
# Convert docker-compose to Kubernetes manifests
kompose convert
kubectl apply -f .
```

### Health Monitoring
```bash
# Check container health
docker ps --filter "name=coursegen"

# View health check logs
docker inspect coursegen-app | jq '.[0].State.Health'
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Docker and application logs
3. Verify API key configuration
4. Ensure sufficient system resources