#!/bin/bash

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="888429341445.dkr.ecr.us-east-1.amazonaws.com/rag"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
HAS_BUILDX=false

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

    # Check if Docker Buildx is available and configure BuildKit accordingly
    if docker buildx version &> /dev/null; then
        HAS_BUILDX=true
        export DOCKER_BUILDKIT=1
        print_status "Docker Buildx detected - BuildKit enabled"
    else
        HAS_BUILDX=false
        export DOCKER_BUILDKIT=0
        print_warning "Docker Buildx plugin not found. Falling back to legacy builder"
    fi

    # Check if AWS CLI is available for ECR authentication
    if ! command -v aws &> /dev/null; then
        print_warning "AWS CLI not found. You'll need to authenticate with ECR manually:"
        echo "  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 888429341445.dkr.ecr.us-east-1.amazonaws.com"
    else
        print_status "AWS CLI found - will authenticate with ECR automatically"
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

    # Check for embeddings data (baked into the image during build)
    if [[ -d "OUTPUT_DATA2/emdeddings" ]]; then
        print_status "Found embeddings bundle - will be baked into the image"
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

    # Remove current tagged image to avoid cache reuse when requested
    if docker image inspect "${FULL_IMAGE_NAME}" &> /dev/null; then
        docker image rm -f "${FULL_IMAGE_NAME}" 2>/dev/null || true
        print_status "Removed existing image ${FULL_IMAGE_NAME}"
    fi

    # Prune builder cache (safe to ignore failures)
    if docker builder prune -f >/dev/null 2>&1; then
        print_status "Cleared Docker build cache"
    else
        print_warning "Could not prune builder cache (BuildKit may be disabled)"
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
    )

    if [[ "$HAS_BUILDX" == "true" ]]; then
        BUILD_ARGS+=(--progress=plain)
    fi

    if [[ "${CLEANUP:-false}" == "true" ]]; then
        print_status "Cleanup flag detected - building without cache"
        BUILD_ARGS+=(--no-cache)
    else
        if docker image inspect "${FULL_IMAGE_NAME}" &> /dev/null; then
            BUILD_ARGS+=(--cache-from "${FULL_IMAGE_NAME}")
            print_status "Using cache from previous build"
        fi
    fi

    # Perform the build with better error handling
    if [[ "$USE_MINIMAL" == "true" ]]; then
        print_status "Building with minimal Dockerfile (--minimal flag used)..."
        if [[ -f "Dockerfile.minimal" ]]; then
            BUILD_ARGS_MINIMAL=(
                --tag "${FULL_IMAGE_NAME}"
                --file "Dockerfile.minimal"
            )

            if [[ "$HAS_BUILDX" == "true" ]]; then
                BUILD_ARGS_MINIMAL+=(--progress=plain)
            fi

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
                )

                if [[ "$HAS_BUILDX" == "true" ]]; then
                    BUILD_ARGS_MINIMAL+=(--progress=plain)
                fi

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
    echo "  # ONE COMMAND: Update embeddings in volume, rebuild image, and deploy to AWS:"
    echo "  ./build.sh --update-embeddings"
    echo ""
    echo "  # Fix Docker credential issues:"
    echo "  ./build.sh --fix-credentials"
    echo ""
    echo "  # Build and run locally:"
    echo "  ./build.sh && ./run.sh"
    echo ""
    echo "  # Build and deploy to AWS ECR:"
    echo "  ./build.sh --deploy"
    echo ""
    echo "  # Manual ECR authentication:"
    echo "  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 888429341445.dkr.ecr.us-east-1.amazonaws.com"
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
    echo "    ${FULL_IMAGE_NAME} --generate-questions"
    echo ""
    echo "  # Run outline generation:"
    echo "  docker run --rm -it ${FULL_IMAGE_NAME} --department_from 'EEE 315'"
    echo ""
    echo "  # Run with docker-compose:"
    echo "  docker-compose up"
    echo ""
    echo "  # Note: ChromaDB embeddings and course data are included in the image"
    echo "  # Volume mounts are CRITICAL for preserving embeddings data between runs"
    echo ""
}

# Function to push image to ECR
push_to_ecr() {
    print_status "Pushing image to AWS ECR..."

    # Check if AWS CLI is available
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI not found. Cannot push to ECR."
        print_status "Install AWS CLI or push manually:"
        echo "  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 888429341445.dkr.ecr.us-east-1.amazonaws.com"
        echo "  docker push ${FULL_IMAGE_NAME}"
        return 1
    fi

    # Method 1: Try direct docker login without credential helper
    print_status "Authenticating with AWS ECR (Method 1: Direct login)..."
    if echo "Logging into AWS ECR..." && aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 888429341445.dkr.ecr.us-east-1.amazonaws.com 2>/dev/null; then
        print_success "Successfully authenticated with ECR"
    else
        print_warning "Direct login failed, trying alternative method..."

        # Method 2: Try with explicit credential helper bypass
        print_status "Authenticating with AWS ECR (Method 2: Environment variables)..."
        AWS_PASSWORD=$(aws ecr get-login-password --region us-east-1)
        if [ $? -eq 0 ] && [ -n "$AWS_PASSWORD" ]; then
            if echo "$AWS_PASSWORD" | docker login --username AWS --password-stdin 888429341445.dkr.ecr.us-east-1.amazonaws.com; then
                print_success "Successfully authenticated with ECR"
            else
                print_error "Failed to authenticate with ECR using environment variables"
                print_status "Manual authentication required:"
                echo "  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 888429341445.dkr.ecr.us-east-1.amazonaws.com"
                echo "  docker push ${FULL_IMAGE_NAME}"
                return 1
            fi
        else
            print_error "Failed to get ECR login password"
            print_status "Manual authentication required:"
            echo "  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 888429341445.dkr.ecr.us-east-1.amazonaws.com"
            echo "  docker push ${FULL_IMAGE_NAME}"
            return 1
        fi
    fi

    # Push the image
    print_status "Pushing ${FULL_IMAGE_NAME} to ECR..."
    if docker push "${FULL_IMAGE_NAME}"; then
        print_success "Successfully pushed image to ECR!"
        print_status "Image available at: ${FULL_IMAGE_NAME}"
        return 0
    else
        print_error "Failed to push image to ECR"
        print_status "Try pushing manually:"
        echo "  docker push ${FULL_IMAGE_NAME}"
        return 1
    fi
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

# Function to fix Docker credential helper issues
fix_docker_credentials() {
    print_status "Fixing Docker credential helper configuration..."

    # Check current Docker config
    DOCKER_CONFIG="${HOME}/.docker/config.json"
    if [[ -f "$DOCKER_CONFIG" ]]; then
        print_status "Found existing Docker config at $DOCKER_CONFIG"

        # Backup current config
        cp "$DOCKER_CONFIG" "${DOCKER_CONFIG}.backup.$(date +%s)"

        # Remove problematic credential helpers
        if command -v jq &> /dev/null; then
            # Use jq if available
            jq 'del(.credsStore, .credHelpers)' "$DOCKER_CONFIG" > "${DOCKER_CONFIG}.tmp" && mv "${DOCKER_CONFIG}.tmp" "$DOCKER_CONFIG"
            print_success "Removed credential helpers from Docker config"
        else
            # Manual JSON editing
            sed -i '/credsStore/d; /credHelpers/d' "$DOCKER_CONFIG" 2>/dev/null || true
            print_warning "Could not automatically remove credential helpers. Manual edit may be needed."
        fi
    else
        print_status "No existing Docker config found - will create new one"
    fi

    print_success "Docker credential helper configuration fixed"
    print_status "You can now try ECR authentication again"
}

# Function to refresh host embeddings and rebuild image
update_embeddings_and_rebuild() {
    print_status "Starting complete embeddings update workflow..."
    echo "=================================================="

    # Check if image exists locally first
    if ! docker image inspect "${FULL_IMAGE_NAME}" &> /dev/null; then
        print_error "Docker image '${FULL_IMAGE_NAME}' not found locally"
        print_status "Building image first..."
        if ! build_image; then
            print_error "Failed to build Docker image"
            return 1
        fi
    fi

    # Step 1: Regenerate embeddings in the host directory
    print_status "Step 1: Regenerating embeddings in OUTPUT_DATA2/emdeddings..."
    if docker run --rm \
        --entrypoint python \
        -v "$(pwd)/OUTPUT_DATA2:/app/OUTPUT_DATA2" \
        -v "$(pwd)/data:/app/data" \
        -e PYTHONPATH=/app \
        "${FULL_IMAGE_NAME}" \
        -m services.RAG.convert_to_embeddings \
        -i data/textbooks/COMPILATION/EEE \
        --with-chroma \
        -c pdfs_bge_m3_cloudflare \
        --workers 4 \
        --resume 2>&1; then
        print_success "Embeddings regenerated in OUTPUT_DATA2/emdeddings"
    else
        print_error "Failed to regenerate embeddings in OUTPUT_DATA2/emdeddings"
        return 1
    fi

    # Step 2: Rebuild the image with updated embeddings
    print_status "Step 2: Rebuilding Docker image with updated embeddings..."
    CLEANUP=true  # Force cleanup for fresh build
    if build_image; then
        print_success "Docker image rebuilt with updated embeddings"
    else
        print_error "Failed to rebuild Docker image"
        return 1
    fi

    # Step 3: Deploy to AWS ECR
    print_status "Step 3: Deploying updated image to AWS ECR..."
    if push_to_ecr; then
        print_success "Updated image deployed to AWS ECR successfully!"
        print_status "Complete workflow finished!"
        return 0
    else
        print_error "Failed to deploy to AWS ECR"
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
    echo "=== Docker Configuration ==="
    DOCKER_CONFIG="${HOME}/.docker/config.json"
    if [[ -f "$DOCKER_CONFIG" ]]; then
        echo "Docker config exists: Yes"
        echo "Credential helpers configured: $(grep -c 'credsStore\|credHelpers' "$DOCKER_CONFIG" 2>/dev/null || echo '0')"
    else
        echo "Docker config exists: No"
    fi

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
    echo "=== ECR Troubleshooting ==="
    echo "1. Fix credential helpers: $0 --fix-credentials"
    echo "2. Manual ECR login: aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 888429341445.dkr.ecr.us-east-1.amazonaws.com"
    echo "3. Check AWS CLI: aws sts get-caller-identity"
    echo "4. Verify ECR permissions: aws ecr describe-repositories --repository-names rag"

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
    DEPLOY=false
    FIX_CREDENTIALS=false
    UPDATE_EMBEDDINGS=false

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
            --deploy)
                DEPLOY=true
                shift
                ;;
            --fix-credentials)
                FIX_CREDENTIALS=true
                shift
                ;;
            --update-embeddings)
                UPDATE_EMBEDDINGS=true
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
                echo "  --deploy         Build and push image to AWS ECR"
                echo "  --fix-credentials Fix Docker credential helper configuration"
                echo "  --update-embeddings Regenerate embeddings, rebuild image, and (optionally) deploy"
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

    if [[ "$FIX_CREDENTIALS" == "true" ]]; then
        fix_docker_credentials
        exit 0
    fi

    if [[ "$UPDATE_EMBEDDINGS" == "true" ]]; then
        check_prerequisites
        if update_embeddings_and_rebuild; then
            show_usage_examples
            print_success "Complete embeddings update workflow completed successfully!"
            exit 0
        else
            print_error "Embeddings update workflow failed!"
            exit 1
        fi
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

        # Deploy to ECR if requested
        if [[ "$DEPLOY" == "true" ]]; then
            if push_to_ecr; then
                print_success "Deployment to AWS ECR completed successfully!"
            else
                print_error "Deployment to AWS ECR failed!"
                exit 1
            fi
        fi

        show_usage_examples

        if [[ "$DEPLOY" == "true" ]]; then
            print_success "Build and deployment process completed successfully!"
        else
            print_success "Build process completed successfully!"
        fi
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
