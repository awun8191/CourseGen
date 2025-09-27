#!/usr/bin/env bash

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
DEFAULT_THEORY_COUNT=10
DEFAULT_CALC_COUNT=5
DEFAULT_REQUEST_DELAY=2

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

    # Check if AWS CLI is available for ECR authentication
    if ! command -v aws &> /dev/null; then
        print_warning "AWS CLI not found. You'll need to authenticate with ECR manually:"
        echo "  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 888429341445.dkr.ecr.us-east-1.amazonaws.com"
    else
        print_status "AWS CLI found - will authenticate with ECR automatically"
    fi

    # Check if image exists locally, if not try to pull from ECR
    if ! docker image inspect "${FULL_IMAGE_NAME}" &> /dev/null; then
        print_warning "Docker image '${FULL_IMAGE_NAME}' not found locally"
        print_status "Attempting to pull from ECR..."

        if command -v aws &> /dev/null; then
            # Authenticate with ECR using alternative methods
            print_status "Authenticating with AWS ECR..."
            if echo "Logging into AWS ECR..." && aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 888429341445.dkr.ecr.us-east-1.amazonaws.com 2>/dev/null; then
                print_success "Successfully authenticated with ECR"
            else
                print_warning "Direct login failed, trying alternative method..."
                AWS_PASSWORD=$(aws ecr get-login-password --region us-east-1)
                if [ $? -eq 0 ] && [ -n "$AWS_PASSWORD" ]; then
                    if echo "$AWS_PASSWORD" | docker login --username AWS --password-stdin 888429341445.dkr.ecr.us-east-1.amazonaws.com; then
                        print_success "Successfully authenticated with ECR"
                    else
                        print_error "Failed to authenticate with ECR"
                        print_status "Please authenticate manually:"
                        echo "  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 888429341445.dkr.ecr.us-east-1.amazonaws.com"
                        exit 1
                    fi
                else
                    print_error "Failed to get ECR login password"
                    print_status "Please authenticate manually:"
                    echo "  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 888429341445.dkr.ecr.us-east-1.amazonaws.com"
                    exit 1
                fi
            fi
        fi

        # Try to pull the image
        if docker pull "${FULL_IMAGE_NAME}"; then
            print_success "Successfully pulled image from ECR"
        else
            print_error "Failed to pull image from ECR"
            print_status "Build the image first:"
            echo "  ./build.sh"
            exit 1
        fi
    fi

    # Check for persistent data directories
    print_success "Prerequisites check passed"
}

ensure_volume_permissions() {
    local paths=("OUTPUT_DATA2" "OUTPUT_DATA2/cache" "OUTPUT_DATA2/data/gemini_cache" "data")
    for dir in "${paths[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
        fi
        print_status "Ensuring write access on $dir for container user"
        if ! chmod -R a+rwX "$dir" 2>/dev/null; then
            print_warning "Could not update permissions on $dir (non-writable entries may remain)."
            print_warning "If container writes still fail, run 'sudo chown -R 1001:1001 $dir' or adjust manually."
        fi
    done
}

# Function to build docker run command
build_docker_command() {
    local docker_args=(
        "docker" "run"
        "--rm"
        "-v" "$(pwd)/OUTPUT_DATA2/cache:/app/OUTPUT_DATA2/cache"
        "-v" "$(pwd)/OUTPUT_DATA2/data/gemini_cache:/app/OUTPUT_DATA2/data/gemini_cache"
        "-v" "$(pwd)/data:/app/data"
    )

    # Add environment file if it exists
    if [[ -f ".env" ]]; then
        docker_args+=("--env-file" ".env")
    fi

    # Add custom environment file if specified
    if [[ -n "$ENV_FILE" ]]; then
        docker_args+=("--env-file" "$ENV_FILE")
    fi

    # Add interactive flags if requested
    if [[ "$INTERACTIVE" == "true" ]]; then
        docker_args+=("-it")
    fi

    # Add image name
    docker_args+=("${FULL_IMAGE_NAME}")

    echo "${docker_args[@]}"
}

# Function to show usage examples
show_usage_examples() {
    print_status "Usage Examples:"
    echo ""
    echo "  # Generate questions for all courses (default):"
    echo "  ./run.sh"
    echo ""
    echo "  # Generate questions for specific course:"
    echo "  ./run.sh --course-code 'EEE 315'"
    echo ""
    echo "  # Custom question counts:"
    echo "  ./run.sh --theory-per-request 5 --calc-per-request 3"
    echo ""
    echo "  # Interactive mode:"
    echo "  ./run.sh -i --course-code 'AAE 101'"
    echo ""
    echo "  # With custom environment file:"
    echo "  ./run.sh --env-file .env.production --course-code 'EEE 471'"
    echo ""
    echo "  # Debug mode (no resume, verbose):"
    echo "  ./run.sh --no-resume --request-delay 1 --course-code 'AAE 101'"
    echo ""
    echo "  # Background mode (no interactive):"
    echo "  ./run.sh -b --course-code 'EEE 315'"
    echo ""
    echo "  # Help:"
    echo "  ./run.sh --help"
    echo ""
}

# Function to run container
run_container() {
    local docker_cmd
    docker_cmd=$(build_docker_command)

    local run_args=()

    # Add course code if specified
    if [[ -n "$COURSE_CODE" ]]; then
        run_args+=("--course-code" "$COURSE_CODE")
    fi

    # Add question generation parameters
    run_args+=("--theory-per-request" "$THEORY_COUNT")
    run_args+=("--calc-per-request" "$CALC_COUNT")
    run_args+=("--request-delay" "$REQUEST_DELAY")

    # Add optional flags
    if [[ "$NO_RESUME" == "true" ]]; then
        run_args+=("--no-resume")
    fi

    if [[ "$SKIP_FIRESTORE" == "true" ]]; then
        run_args+=("--skip-firestore")
    fi

    if [[ "$DEBUG" == "true" ]]; then
        run_args+=("--temperature" "0.1")
        run_args+=("--request-delay" "1")
    fi

    if [[ -n "$STRUCTURED_FLAG" ]]; then
        run_args+=("$STRUCTURED_FLAG")
    fi

    print_status "Starting CourseGen container..."
    print_status "Command: ${docker_cmd} ${run_args[*]}"

    ensure_volume_permissions

    # Execute the docker run command
    $docker_cmd "${run_args[@]}"
}

# Main execution
main() {
    # Default values
    COURSE_CODE=""
    THEORY_COUNT="$DEFAULT_THEORY_COUNT"
    CALC_COUNT="$DEFAULT_CALC_COUNT"
    REQUEST_DELAY="$DEFAULT_REQUEST_DELAY"
    INTERACTIVE=true
    NO_RESUME=false
    SKIP_FIRESTORE=false
    DEBUG=false
    ENV_FILE=""
    STRUCTURED_FLAG=""

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--interactive)
                INTERACTIVE=true
                shift
                ;;
            -b|--background)
                INTERACTIVE=false
                shift
                ;;
            --course-code)
                COURSE_CODE="$2"
                shift 2
                ;;
            --theory-per-request)
                THEORY_COUNT="$2"
                shift 2
                ;;
            --calc-per-request)
                CALC_COUNT="$2"
                shift 2
                ;;
            --request-delay)
                REQUEST_DELAY="$2"
                shift 2
                ;;
            --no-resume)
                NO_RESUME=true
                shift
                ;;
            --skip-firestore)
                SKIP_FIRESTORE=true
                shift
                ;;
            --debug)
                DEBUG=true
                shift
                ;;
            --env-file)
                ENV_FILE="$2"
                shift 2
                ;;
            --structured-output)
                STRUCTURED_FLAG="--structured-output"
                shift
                ;;
            --no-structured-output)
                STRUCTURED_FLAG="--no-structured-output"
                shift
                ;;
            -h|--help)
                echo "CourseGen Run Script with Persistent Volumes"
                echo ""
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  -i, --interactive     Run in interactive mode (default)"
                echo "  -b, --background      Run in background mode (no TTY)"
                echo "  --course-code CODE    Generate questions for specific course"
                echo "  --theory-per-request N    Number of theory questions per request (default: $DEFAULT_THEORY_COUNT)"
                echo "  --calc-per-request N     Number of calculation questions per request (default: $DEFAULT_CALC_COUNT)"
                echo "  --request-delay SECS     Delay between API calls in seconds (default: $DEFAULT_REQUEST_DELAY)"
                echo "  --no-resume           Do not reuse cached generations"
                echo "  --skip-firestore      Disable persistence to Firestore"
                echo "  --debug               Enable debug mode (lower temp, faster requests)"
                echo "  --structured-output   Enable Gemini structured output schema"
                echo "  --no-structured-output Disable Gemini structured output schema"
                echo "  --env-file FILE       Use custom environment file"
                echo "  -h, --help            Show this help message"
                echo ""
                show_usage_examples
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    print_status "CourseGen Container Runner with Persistent Volumes"
    echo "=================================================="

    check_prerequisites

    print_status "Configuration:"
    echo "  Course Code: ${COURSE_CODE:-All courses}"
    echo "  Theory Questions per Request: $THEORY_COUNT"
    echo "  Calculation Questions per Request: $CALC_COUNT"
    echo "  Request Delay: ${REQUEST_DELAY}s"
    echo "  Interactive Mode: $INTERACTIVE"
    echo "  Resume Enabled: $([ "$NO_RESUME" == "false" ] && echo "Yes" || echo "No")"
    echo "  Firestore Enabled: $([ "$SKIP_FIRESTORE" == "false" ] && echo "Yes" || echo "No")"
    echo "  Debug Mode: $DEBUG"
    echo ""

    if [[ "$INTERACTIVE" == "false" ]]; then
        print_warning "Running in background mode. Use Ctrl+C to stop."
    fi

    run_container

    print_success "Container execution completed!"
}

# Run main function with all arguments
main "$@"
