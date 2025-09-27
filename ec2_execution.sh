#!/usr/bin/env bash

set -euo pipefail

# ------------------------------------------------------------
# CourseGen EC2 Execution Helper
# ------------------------------------------------------------
# Syncs data baked into the container (courses.json + embeddings)
# onto the EC2 host before running the Docker image. This allows
# the host to pick up the latest assets every time a new image is
# pulled while still persisting runtime cache/output locally.
# ------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

HOST_SHORT=$(hostname -s 2>/dev/null || hostname)
HOST_FULL=$(hostname -f 2>/dev/null || echo "$HOST_SHORT")

IMAGE_NAME="888429341445.dkr.ecr.us-east-1.amazonaws.com/rag"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

DEFAULT_THEORY_COUNT=10
DEFAULT_CALC_COUNT=5
DEFAULT_REQUEST_DELAY=2.0

DATA_DIR="${DATA_DIR:-$HOME/data}"
TEXTBOOKS_DIR="${TEXTBOOKS_DIR:-${DATA_DIR}/textbooks}"
COURSES_FILE="${COURSES_FILE:-${TEXTBOOKS_DIR}/courses.json}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/OUTPUT_DATA2}"
CACHE_DIR="${CACHE_DIR:-${CACHE_ROOT}/cache}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-${CACHE_ROOT}/emdeddings}"
CONTAINER_NAME="${CONTAINER_NAME:-coursegen-rag}"
GEMINI_CACHE_DIR="${GEMINI_CACHE_DIR:-${CACHE_ROOT}/data/gemini_cache}"

ENV_FILE=""
COURSE_CODE=""
THEORY_COUNT="$DEFAULT_THEORY_COUNT"
CALC_COUNT="$DEFAULT_CALC_COUNT"
REQUEST_DELAY="$DEFAULT_REQUEST_DELAY"
NO_RESUME=false
SKIP_FIRESTORE=false
STRUCTURED_FLAG=""
MODE="interactive"  # interactive | background
SKIP_SYNC=false
PULL_IMAGE=true
PORTS=()
USER_FLAGS=()

send_email_notification() {
    local status="$1"
    local exit_code="$2"
    local detail="$3"

    local timestamp
    timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    local subject="CourseGen run ${status} (${HOST_SHORT})"
    local body="Status: ${status}
Exit code: ${exit_code}
Host: ${HOST_FULL}
Time: ${timestamp}

${detail}"
    local body_env
    body_env=${body//$'\n'/\\n}

    local tmp_script
    tmp_script=$(mktemp)
    cat <<'PY' > "$tmp_script"
import os
import sys

from services.Email.email_service import EmailService

subject = os.environ.get("COURSEGEN_EMAIL_SUBJECT", "CourseGen run update")
message = os.environ.get("COURSEGEN_EMAIL_MESSAGE", "").replace("\\n", "\n")

service = EmailService()
if not service.enabled:
    sys.exit(0)

if not service.send_email(subject, message):
    sys.exit(1)
PY

    local cmd=(docker run --rm)
    if [[ -n "$ENV_FILE" ]]; then
        cmd+=("--env-file" "$ENV_FILE")
    fi
    cmd+=(
        -e "COURSEGEN_EMAIL_SUBJECT=${subject}"
        -e "COURSEGEN_EMAIL_MESSAGE=${body_env}"
        -w /app
        -v "$tmp_script:/tmp/email_notify.py:ro"
        "$FULL_IMAGE_NAME"
        python
        /tmp/email_notify.py
    )

    if ! "${cmd[@]}"; then
        print_warning "Email notification failed (status: $status)"
    else
        print_success "Email notification sent ($status)"
    fi

    rm -f "$tmp_script"
}

ensure_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        print_error "Required command '$1' not found in PATH"
        exit 1
    fi
}

check_prerequisites() {
    print_status "Checking prerequisites..."
    ensure_command docker

    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Start Docker and retry."
        exit 1
    fi

    if command -v aws >/dev/null 2>&1; then
        print_status "AWS CLI detected"
    else
        print_warning "AWS CLI not found. Ensure ECR auth is handled manually."
    fi
}

authenticate_ecr() {
    if ! command -v aws >/dev/null 2>&1; then
        return
    fi

    print_status "Authenticating with AWS ECR..."
    if aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$IMAGE_NAME" >/dev/null 2>&1; then
        print_success "Authenticated with AWS ECR"
    else
        print_warning "Automatic ECR login failed. Run manually:"
        echo "  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $IMAGE_NAME"
    fi
}

pull_image_if_needed() {
    if [[ "$PULL_IMAGE" == "false" ]] && docker image inspect "$FULL_IMAGE_NAME" >/dev/null 2>&1; then
        print_status "Skipping image pull (requested)"
        return
    fi

    if docker image inspect "$FULL_IMAGE_NAME" >/dev/null 2>&1; then
        print_status "Pulling latest image for ${FULL_IMAGE_NAME}"
    else
        print_status "Image not found locally; pulling ${FULL_IMAGE_NAME}"
    fi

    docker pull "$FULL_IMAGE_NAME"
}

resolve_env_file() {
    if [[ -n "$ENV_FILE" ]]; then
        if [[ ! -f "$ENV_FILE" ]]; then
            print_error "Env file '$ENV_FILE' not found"
            exit 1
        fi
        return
    fi

    if [[ -f "$HOME/.env" ]]; then
        ENV_FILE="$HOME/.env"
        print_status "Using env file: $ENV_FILE"
    else
        print_warning "No env file supplied. Container will rely on baked-in defaults/env vars."
    fi
}

safe_reset_dir() {
    local target="$1"
    if [[ -z "$target" ]]; then
        print_error "safe_reset_dir called with empty path"
        exit 1
    fi

    local abs_path
    abs_path=$(realpath -m "$target")

    if [[ "$abs_path" != "$HOME"/* ]]; then
        print_warning "Refusing to reset directory outside \$HOME: $abs_path"
        mkdir -p "$target"
        return
    fi

    rm -rf "$abs_path"
    mkdir -p "$abs_path"
}

sync_data_from_image() {
    if [[ "$SKIP_SYNC" == "true" ]]; then
        print_status "Skipping data sync (per --skip-sync)"
        return
    fi

    print_status "Syncing courses.json and embeddings from ${FULL_IMAGE_NAME}"
    local sync_container="${CONTAINER_NAME}-sync-$$"
    docker rm -f "$sync_container" >/dev/null 2>&1 || true

    docker create --name "$sync_container" "$FULL_IMAGE_NAME" >/dev/null

    mkdir -p "$TEXTBOOKS_DIR"
    docker cp "$sync_container:/app/data/textbooks/courses.json" "$COURSES_FILE"
    print_success "Updated: $COURSES_FILE"

    safe_reset_dir "$EMBEDDINGS_DIR"
    docker cp "$sync_container:/app/OUTPUT_DATA2/emdeddings/." "$EMBEDDINGS_DIR/"
    print_success "Updated embeddings in: $EMBEDDINGS_DIR"

    docker rm "$sync_container" >/dev/null
}

ensure_volume_permissions() {
    local dirs=("$CACHE_ROOT" "$CACHE_DIR" "$DATA_DIR" "$TEXTBOOKS_DIR" "$EMBEDDINGS_DIR" "$GEMINI_CACHE_DIR")
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        if ! chmod -R a+rwX "$dir" >/dev/null 2>&1; then
            print_warning "Could not adjust permissions on $dir."
        fi
    done
}

remove_existing_container() {
    if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
        print_warning "Removing existing container named $CONTAINER_NAME"
        docker rm -f "$CONTAINER_NAME" >/dev/null
    fi
}

build_docker_command() {
    DOCKER_CMD=("docker" "run" "--name" "$CONTAINER_NAME")

    if [[ "$MODE" == "background" ]]; then
        DOCKER_CMD+=("-d")
    else
        DOCKER_CMD+=("--rm" "-it")
    fi

    DOCKER_CMD+=("-v" "$CACHE_DIR:/app/OUTPUT_DATA2/cache")

    mkdir -p "$GEMINI_CACHE_DIR"
    DOCKER_CMD+=("-v" "$GEMINI_CACHE_DIR:/app/OUTPUT_DATA2/data/gemini_cache")

    if [[ -f "$COURSES_FILE" ]]; then
        DOCKER_CMD+=("-v" "$COURSES_FILE:/app/data/textbooks/courses.json:ro")
    else
        print_warning "courses.json not found at $COURSES_FILE. Container will fall back to baked-in copy (if any)."
    fi

    if [[ -d "$EMBEDDINGS_DIR" ]]; then
        DOCKER_CMD+=("-v" "$EMBEDDINGS_DIR:/app/OUTPUT_DATA2/emdeddings")
    fi

    for port in "${PORTS[@]}"; do
        DOCKER_CMD+=("-p" "$port")
    done

    if [[ -n "$ENV_FILE" ]]; then
        DOCKER_CMD+=("--env-file" "$ENV_FILE")
    fi

    DOCKER_CMD+=("$FULL_IMAGE_NAME")
}

build_run_arguments() {
    RUN_ARGS=()
    if [[ -n "$COURSE_CODE" ]]; then
        RUN_ARGS+=("--course-code" "$COURSE_CODE")
    fi
    RUN_ARGS+=("--theory-per-request" "$THEORY_COUNT")
    RUN_ARGS+=("--calc-per-request" "$CALC_COUNT")
    RUN_ARGS+=("--request-delay" "$REQUEST_DELAY")

    if [[ "$NO_RESUME" == "true" ]]; then
        RUN_ARGS+=("--no-resume")
    fi
    if [[ "$SKIP_FIRESTORE" == "true" ]]; then
        RUN_ARGS+=("--skip-firestore")
    fi
    if [[ -n "$STRUCTURED_FLAG" ]]; then
        RUN_ARGS+=("$STRUCTURED_FLAG")
    fi

    if [[ ${#USER_FLAGS[@]} -gt 0 ]]; then
        RUN_ARGS+=("${USER_FLAGS[@]}")
    fi
}

format_command() {
    local -n ref=$1
    local formatted=""
    local piece
    for piece in "${ref[@]}"; do
        printf -v piece '%q' "$piece"
        formatted+=" $piece"
    done
    echo "${formatted# }"
}

show_help() {
    cat <<'USAGE'
CourseGen EC2 Execution Script

Usage: ./ec2_execution.sh [options]

Options:
  --env-file PATH          Use a specific env file (default: ~/.env if present)
  --course-code CODE       Restrict generation to a specific course
  --theory-per-request N   Theory questions per request (default: 10)
  --calc-per-request N     Calculation questions per request (default: 5)
  --request-delay SECS     Delay between Gemini calls (default: 2.0)
  --no-resume              Disable resume functionality
  --skip-firestore         Do not write to Firestore
  --structured-output      Enable Gemini structured output
  --no-structured-output   Disable Gemini structured output
  --background             Run container detached (keep running)
  --interactive            Run container in interactive mode (default)
  --port HOST:CONTAINER    Publish container port (can be repeated)
  --skip-sync              Do not copy courses.json/embeddings from the image
  --no-pull                Skip pulling the image (use local copy)
  --extra "ARGS"           Additional flags appended to the container command
  -h, --help               Show this help message

Environment overrides:
  IMAGE_TAG, DATA_DIR, TEXTBOOKS_DIR, CACHE_ROOT, CACHE_DIR,
  EMBEDDINGS_DIR, CONTAINER_NAME can be set before invoking.
USAGE
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --env-file)
                ENV_FILE="$2"; shift 2 ;;
            --course-code)
                COURSE_CODE="$2"; shift 2 ;;
            --theory-per-request)
                THEORY_COUNT="$2"; shift 2 ;;
            --calc-per-request)
                CALC_COUNT="$2"; shift 2 ;;
            --request-delay)
                REQUEST_DELAY="$2"; shift 2 ;;
            --no-resume)
                NO_RESUME=true; shift ;;
            --skip-firestore)
                SKIP_FIRESTORE=true; shift ;;
            --structured-output)
                STRUCTURED_FLAG="--structured-output"; shift ;;
            --no-structured-output)
                STRUCTURED_FLAG="--no-structured-output"; shift ;;
            --background)
                MODE="background"; shift ;;
            --interactive)
                MODE="interactive"; shift ;;
            --port)
                PORTS+=("$2"); shift 2 ;;
            --skip-sync)
                SKIP_SYNC=true; shift ;;
            --no-pull)
                PULL_IMAGE=false; shift ;;
            --extra)
                USER_FLAGS+=("$2"); shift 2 ;;
            -h|--help)
                show_help; exit 0 ;;
            --)
                shift
                USER_FLAGS+=("$@")
                break ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1 ;;
        esac
    done
}

print_configuration() {
    print_status "Configuration"
    echo "  Image: ${FULL_IMAGE_NAME}"
    echo "  Env File: ${ENV_FILE:-<none>}"
    echo "  Mode: $MODE"
    echo "  Course Code: ${COURSE_CODE:-all}"
    echo "  Theory per Request: $THEORY_COUNT"
    echo "  Calculation per Request: $CALC_COUNT"
    echo "  Request Delay: $REQUEST_DELAY"
    echo "  Data Dir: $DATA_DIR"
    echo "  Courses File: $COURSES_FILE"
    echo "  Cache Dir: $CACHE_DIR"
    echo "  Gemini Cache Dir: $GEMINI_CACHE_DIR"
    echo "  Embeddings Dir: $EMBEDDINGS_DIR"
    if [[ ${#PORTS[@]} -gt 0 ]]; then
        echo "  Ports: ${PORTS[*]}"
    fi
    if [[ ${#USER_FLAGS[@]} -gt 0 ]]; then
        echo "  Extra Flags: ${USER_FLAGS[*]}"
    fi
    echo ""
}

main() {
    parse_args "$@"

    resolve_env_file
    check_prerequisites
    authenticate_ecr
    pull_image_if_needed
    sync_data_from_image

    ensure_volume_permissions
    remove_existing_container

    print_configuration

    build_docker_command
    build_run_arguments

    local docker_cmd_pretty
    docker_cmd_pretty=$(format_command DOCKER_CMD)
    local run_args_pretty
    run_args_pretty=$(format_command RUN_ARGS)

    print_status "Launching container"
    print_status "Command: ${docker_cmd_pretty} ${run_args_pretty}"

    local run_exit=0
    local container_id=""

    if [[ "$MODE" == "background" ]]; then
        if [[ ${#RUN_ARGS[@]} -gt 0 ]]; then
            container_id=$("${DOCKER_CMD[@]}" "${RUN_ARGS[@]}")
        else
            container_id=$("${DOCKER_CMD[@]}")
        fi
        run_exit=$?

        if [[ $run_exit -eq 0 ]]; then
            container_id=$(echo "$container_id" | tr -d '\r\n')
            print_success "Container running in background as '$CONTAINER_NAME' (id: $container_id)"

            (
                local wait_output
                local exit_code
                local tail_logs
                local detail
                if wait_output=$(docker wait "$container_id" 2>/dev/null); then
                    exit_code="$wait_output"
                else
                    exit_code=$?
                fi
                tail_logs=$(docker logs --tail 60 "$container_id" 2>/dev/null || echo "(logs unavailable; container may have been removed)")
                detail="Container ${container_id} exited with status ${exit_code}.\n\nLast 60 log lines:\n${tail_logs}"
                send_email_notification "stopped" "$exit_code" "$detail"
                docker rm "$container_id" >/dev/null 2>&1 || true
            ) &
        else
            detail="docker run failed to start container in background mode.\nCommand: ${docker_cmd_pretty} ${run_args_pretty}"
            send_email_notification "failed_to_start" "$run_exit" "$detail"
            print_error "Failed to start container (exit $run_exit)"
        fi
    else
        if [[ ${#RUN_ARGS[@]} -gt 0 ]]; then
            "${DOCKER_CMD[@]}" "${RUN_ARGS[@]}"
        else
            "${DOCKER_CMD[@]}"
        fi
        run_exit=$?

        if [[ $run_exit -eq 0 ]]; then
            print_success "Container execution completed"
            send_email_notification "completed" "$run_exit" "Interactive run exited normally."
        else
            print_error "Container exited with status $run_exit"
            send_email_notification "failed" "$run_exit" "Interactive run terminated with exit code ${run_exit}. See console output for details."
        fi
    fi

    return $run_exit
}

main "$@"
