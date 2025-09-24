#!/bin/bash

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="coursegen"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi

    # Check if required files exist
    if [[ ! -f "Dockerfile" ]]; then
        print_error "Dockerfile not found in current directory"
        exit 1
    fi

    if [[ ! -f "requirements.txt" ]]; then
        print_error "requirements.txt not found in current directory"
        exit 1
    fi

    # Check for embeddings data (IMPORTANT: embeddings must be preserved)
    if [[ -d "OUTPUT_DATA2/embeddings" ]]; then
        print_status "Found existing embeddings data - will be preserved in container"
    else
        print_warning "No embeddings data found. Container will start with empty embeddings."
    fi

    print_success "Prerequisites check passed"
}

# Function to clean up old images (optional)
cleanup_old_images() {
    print_status "Cleaning up old images..."
    
    # Remove dangling images
    if docker images -f "dangling=true" -q | grep -q .; then
        docker rmi $(docker images -f "dangling=true" -q) 2>/dev/null || true
        print_success "Removed dangling images"
    fi
    
    # Remove old versions of our image (keep latest)
    OLD_IMAGES=$(docker images "${IMAGE_NAME}" --format "{{.ID}} {{.Tag}}" | grep -v "${IMAGE_TAG}" | awk '{print $1}' || true)
    if [[ -n "$OLD_IMAGES" ]]; then
        echo "$OLD_IMAGES" | xargs -r docker rmi 2>/dev/null || true
        print_success "Removed old image versions"
    fi
}

# Function to build the Docker image
build_image() {
    print_status "Building CourseGen Docker image..."
    print_status "Image: ${FULL_IMAGE_NAME}"
    print_status "Build context: $(pwd)"

    # Check available disk space
    DISK_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ $DISK_SPACE -lt 5 ]]; then
        print_warning "Low disk space: ${DISK_SPACE}GB available. Build may fail."
    fi

    # Build arguments for optimization
    BUILD_ARGS=(
        --tag "${FULL_IMAGE_NAME}"
        --build-arg BUILDKIT_INLINE_CACHE=1
        --progress=plain
    )

    # Add cache from previous builds if available
    if docker image inspect "${FULL_IMAGE_NAME}" &> /dev/null; then
        BUILD_ARGS+=(--cache-from "${FULL_IMAGE_NAME}")
        print_status "Using cache from previous build"
    fi

    # Perform the build with better error handling
    if [[ "$USE_MINIMAL" == "true" ]]; then
        print_status "Building with minimal Dockerfile (--minimal flag used)..."
        if [[ -f "Dockerfile.minimal" ]]; then
            BUILD_ARGS_MINIMAL=(
                --tag "${FULL_IMAGE_NAME}"
                --file "Dockerfile.minimal"
                --progress=plain
            )

            if docker build "${BUILD_ARGS_MINIMAL[@]}" . 2>&1 | tee /tmp/docker_build.log; then
                print_success "Minimal Docker build completed successfully!"
                print_warning "Note: Some optional features may not be available in minimal build"
                return 0
            else
                print_error "Minimal build failed! Check /tmp/docker_build.log for details"
                return 1
            fi
        else
            print_error "Dockerfile.minimal not found!"
            return 1
        fi
    else
        print_status "Attempting build with full Dockerfile..."

        # Try the build with timeout and better error capture
        if timeout 1800 docker build "${BUILD_ARGS[@]}" . 2>&1 | tee /tmp/docker_build.log; then
            print_success "Docker build completed successfully!"
            return 0
        else
            BUILD_EXIT_CODE=${PIPESTATUS[0]}
            print_error "Docker build failed with exit code: $BUILD_EXIT_CODE"

            # Show relevant error lines from the log
            print_status "Last 20 lines of build log:"
            tail -20 /tmp/docker_build.log | while read line; do
                echo "  $line"
            done

            print_warning "Full build failed, trying minimal Dockerfile as fallback..."

            # Try with minimal Dockerfile as fallback
            if [[ -f "Dockerfile.minimal" ]]; then
                print_status "Building with minimal dependencies..."
                BUILD_ARGS_MINIMAL=(
                    --tag "${FULL_IMAGE_NAME}"
                    --file "Dockerfile.minimal"
                    --progress=plain
                )

                if docker build "${BUILD_ARGS_MINIMAL[@]}" . 2>&1 | tee /tmp/docker_build_minimal.log; then
                    print_success "Minimal Docker build completed successfully!"
                    print_warning "Note: Some optional features may not be available in minimal build"
                    return 0
                else
                    print_error "Both full and minimal builds failed!"
                    print_status "Check /tmp/docker_build_minimal.log for details"
                    return 1
                fi
            else
                print_error "Docker build failed and no minimal Dockerfile found!"
                print_status "Troubleshooting tips:"
                echo "  1. Check available disk space: df -h"
                echo "  2. Try building without cache: docker build --no-cache -t ${FULL_IMAGE_NAME} ."
                echo "  3. Check Docker daemon memory limits in Docker Desktop settings"
                echo "  4. Review /tmp/docker_build.log for specific error details"
                return 1
            fi
        fi
    fi
}

# Function to display build results
show_build_results() {
    print_status "Build Results:"
    
    # Get image size
    IMAGE_SIZE=$(docker images "${FULL_IMAGE_NAME}" --format "{{.Size}}" 2>/dev/null || echo "Unknown")
    echo "  Image: ${FULL_IMAGE_NAME}"
    echo "  Size: ${IMAGE_SIZE}"
    
    # Get image ID and creation date
    IMAGE_INFO=$(docker images "${FULL_IMAGE_NAME}" --format "{{.ID}} {{.CreatedAt}}" 2>/dev/null || echo "Unknown Unknown")
    IMAGE_ID=$(echo "$IMAGE_INFO" | awk '{print $1}')
    CREATED_AT=$(echo "$IMAGE_INFO" | awk '{print $2, $3}')
    echo "  Image ID: ${IMAGE_ID}"
    echo "  Created: ${CREATED_AT}"
    
    print_success "Image built successfully!"
}

# Function to show usage examples
show_usage_examples() {
    print_status "Usage Examples:"
    echo ""
    echo "  # Run with default help command:"
    echo "  docker run --rm ${FULL_IMAGE_NAME}"
    echo ""
    echo "  # Generate questions (uses embedded ChromaDB data):"
    echo "  docker run --rm -it ${FULL_IMAGE_NAME} --generate-questions"
    echo ""
    echo "  # Generate questions for specific course:"
    echo "  docker run --rm -it ${FULL_IMAGE_NAME} --generate-questions --course-code 'EEE 315'"
    echo ""
    echo "  # IMPORTANT: Preserve embeddings data with volume mounts:"
    echo "  docker run --rm -it \\"
    echo "    -v \$(pwd)/OUTPUT_DATA2:/app/OUTPUT_DATA2 \\"
    echo "    -v \$(pwd)/.cache:/app/.cache \\"
    echo "    ${FULL_IMAGE_NAME} --generate-questions"
    echo ""
    echo "  # Run outline generation:"
    echo "  docker run --rm -it ${FULL_IMAGE_NAME} --department_from 'EEE 315'"
    echo ""
    echo "  # Note: ChromaDB embeddings and course data are included in the image"
    echo "  # Volume mounts are CRITICAL for preserving embeddings data between runs"
    echo ""
}

# Function to verify the built image
verify_image() {
    print_status "Verifying built image..."

    # Test if the image can run
    if docker run --rm "${FULL_IMAGE_NAME}" --help &> /dev/null; then
        print_success "Image verification passed - container can run successfully"
        return 0
    else
        print_warning "Image verification failed - container may have issues"
        return 1
    fi
}

# Function to debug build issues
debug_build_issues() {
    print_status "Debugging build issues..."

    echo "=== System Information ==="
    echo "Docker version: $(docker --version)"
    echo "Docker Compose version: $(docker-compose --version 2>/dev/null || echo 'Not available')"
    echo "Available disk space: $(df -h . | tail -1)"
    echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $7}')"

    echo ""
    echo "=== Docker Status ==="
    echo "Docker daemon running: $(docker info &>/dev/null && echo 'Yes' || echo 'No')"
    echo "BuildKit enabled: $(docker buildx version &>/dev/null && echo 'Yes' || echo 'No')"

    echo ""
    echo "=== File Checks ==="
    echo "Dockerfile exists: $([[ -f "Dockerfile" ]] && echo 'Yes' || echo 'No')"
    echo "requirements.txt exists: $([[ -f "requirements.txt" ]] && echo 'Yes' || echo 'No')"
    echo "Dockerfile.minimal exists: $([[ -f "Dockerfile.minimal" ]] && echo 'Yes' || echo 'No')"

    echo ""
    echo "=== Build Logs ==="
    if [[ -f "/tmp/docker_build.log" ]]; then
        echo "Last 10 lines of build log:"
        tail -10 /tmp/docker_build.log
    else
        echo "No build log found. Run a build first."
    fi

    echo ""
    echo "=== Recommendations ==="
    echo "1. Ensure Docker daemon is running"
    echo "2. Check available disk space (need at least 5GB)"
    echo "3. Try building without cache: docker build --no-cache -t ${FULL_IMAGE_NAME} ."
    echo "4. Check Docker daemon memory limits in Docker Desktop settings"
    echo "5. Verify all required files are present"
}

# Main execution
main() {
    print_status "Starting CourseGen Docker build process..."
    echo "=================================================="
    
    # Parse command line arguments
    CLEANUP=false
    VERIFY=true
    VERBOSE=false
    USE_MINIMAL=false
    DEBUG=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --cleanup)
                CLEANUP=true
                shift
                ;;
            --no-verify)
                VERIFY=false
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --minimal)
                USE_MINIMAL=true
                shift
                ;;
            --ultra-minimal)
                USE_MINIMAL="ultra"
                shift
                ;;
            --debug)
                DEBUG=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --cleanup        Clean up old Docker images before building"
                echo "  --no-verify      Skip image verification after build"
                echo "  --verbose        Enable verbose output"
                echo "  --minimal        Use minimal Dockerfile (fewer dependencies)"
                echo "  --ultra-minimal  Use ultra-minimal Dockerfile (core packages only)"
                echo "  --debug          Show system information and debug build issues"
                echo "  --help, -h       Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Enable verbose mode if requested
    if [[ "$VERBOSE" == "true" ]]; then
        set -x
    fi
    
    # Execute build steps
    if [[ "$DEBUG" == "true" ]]; then
        debug_build_issues
        exit 0
    fi

    check_prerequisites

    if [[ "$CLEANUP" == "true" ]]; then
        cleanup_old_images
    fi

    if build_image; then
        show_build_results

        if [[ "$VERIFY" == "true" ]]; then
            verify_image
        fi

        show_usage_examples

        print_success "Build process completed successfully!"
        exit 0
    else
        print_error "Build process failed!"
        print_status "Troubleshooting tips:"
        echo "  1. Check Docker daemon is running: docker info"
        echo "  2. Verify Dockerfile syntax and dependencies"
        echo "  3. Check requirements.txt for conflicting packages"
        echo "  4. Ensure sufficient disk space: df -h"
        echo "  5. Try building with --cleanup flag to remove old images"
        echo "  6. Check Docker logs: docker system events"
        echo "  7. Run with --debug flag for detailed system information"
        echo "  8. Try building without cache: docker build --no-cache -t ${FULL_IMAGE_NAME} ."
        exit 1
    fi
}

# Run main function with all arguments
main "$@"