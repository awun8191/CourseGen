#!/usr/bin/env bash
set -euo pipefail

# --------------------------
# Config (can be overridden)
# --------------------------
IMAGE_NAME="${IMAGE_NAME:-coursegen:latest}"   # e.g. export IMAGE_NAME=888429341445.dkr.ecr.us-east-1.amazonaws.com/rag:latest
CONTAINER_NAME="${CONTAINER_NAME:-coursegen}"
DEFAULT_CMD=(--help)

# Colors
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BLUE=$'\033[0;34m'; NC=$'\033[0m'

log(){ printf "%s[INFO]%s %s\n"  "$BLUE" "$NC" "$*"; }
ok(){  printf "%s[SUCCESS]%s %s\n" "$GREEN" "$NC" "$*"; }
err(){ printf "%s[ERROR]%s %s\n" "$RED" "$NC" "$*" >&2; }

# --------------------------
# Usage
# --------------------------
usage() {
  cat <<EOF
Usage: $0 [options] [-- <command args...>]

Options:
  -i, --interactive        Run with TTY
  -m, --mount-data         Bind-mount ./OUTPUT_DATA2, ./.cache, ./data
  -e, --env-file FILE      Load environment variables from FILE
  -p, --pull               docker pull IMAGE_NAME before run
  -d, --detach             Run container in background
  -r, --resources C,M      Limit resources (e.g. -r "2,4g" = 2 CPUs, 4GB RAM)
  -n, --name NAME          Container name (default: ${CONTAINER_NAME})
  -u, --user               Run as current user (fixes file perms on mounts)
  -I, --image NAME:TAG     Override image (e.g., ECR URI)
  -h, --help               Show this help

Examples:
  $0 -m -- --generate-questions
  $0 -m -e .env -- --generate-questions --course-code "EEE 315"
  IMAGE_NAME=8884...amazonaws.com/rag:latest $0 -p -m -- --help
EOF
}

# --------------------------
# Parse args
# --------------------------
INTERACTIVE=false
MOUNT_DATA=false
DETACH=false
PULL=false
ENV_FILE=""
RES_CPUS=""
RES_MEM=""
RUN_AS_USER=false

declare -a CMD_ARGS=()
declare -a DOCKER_ARGS=("run" "--rm")

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--interactive) INTERACTIVE=true; shift ;;
    -m|--mount-data)  MOUNT_DATA=true; shift ;;
    -e|--env-file)    ENV_FILE="${2:-}"; shift 2 ;;
    -p|--pull)        PULL=true; shift ;;
    -d|--detach)      DETACH=true; shift ;;
    -r|--resources)   IFS=',' read -r RES_CPUS RES_MEM <<< "${2:-}"; shift 2 ;;
    -n|--name)        CONTAINER_NAME="${2:-}"; shift 2 ;;
    -u|--user)        RUN_AS_USER=true; shift ;;
    -I|--image)       IMAGE_NAME="${2:-}"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    --)               shift; CMD_ARGS+=("$@"); break ;;
    *)                CMD_ARGS+=("$1"); shift ;;
  esac
done

# --------------------------
# Pre-checks
# --------------------------
command -v docker >/dev/null || { err "Docker not found"; exit 1; }

if [[ "$PULL" == true ]]; then
  log "Pulling image: ${IMAGE_NAME}"
  docker pull "$IMAGE_NAME"
fi

# If image missing locally and not pulling, warn
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  log "Image '$IMAGE_NAME' not found locally."
  log "Build it (e.g.: docker build -t coursegen:latest .) or use -p/--pull with an ECR/Hub image."
fi

# --------------------------
# Compose docker args safely
# --------------------------
# TTY / detach
[[ "$INTERACTIVE" == true ]] && DOCKER_ARGS+=("-it")
[[ "$DETACH" == true ]] && DOCKER_ARGS+=("-d")
DOCKER_ARGS+=("--name" "$CONTAINER_NAME")

# Env file (auto-detect .env if not specified)
if [[ -n "$ENV_FILE" ]]; then
  [[ -f "$ENV_FILE" ]] || { err "Env file not found: $ENV_FILE"; exit 1; }
  log "Loading environment from: $ENV_FILE"
  DOCKER_ARGS+=("--env-file" "$ENV_FILE")
elif [[ -f ".env" ]]; then
  log "Auto-loading environment from: .env"
  DOCKER_ARGS+=("--env-file" ".env")
else
  log "No .env file found; ensure API keys are set in environment"
fi

# Resource limits
[[ -n "$RES_CPUS" ]] && DOCKER_ARGS+=("--cpus" "$RES_CPUS")
[[ -n "$RES_MEM"  ]] && DOCKER_ARGS+=("--memory" "$RES_MEM")

# Run as current user (optional; helps file ownership on Linux binds)
if [[ "$RUN_AS_USER" == true ]]; then
  DOCKER_ARGS+=("--user" "$(id -u):$(id -g)")
fi

# Volume mounts
if [[ "$MOUNT_DATA" == true ]]; then
  log "Mounting local data directories"
  mkdir -p "./OUTPUT_DATA2" "./.cache" "./data"
  DOCKER_ARGS+=("-v" "$(pwd)/OUTPUT_DATA2:/app/OUTPUT_DATA2")
  DOCKER_ARGS+=("-v" "$(pwd)/.cache:/app/.cache")
  DOCKER_ARGS+=("-v" "$(pwd)/data:/app/data")
  # Ensure the app uses a persistent embeddings dir inside OUTPUT_DATA2
  DOCKER_ARGS+=("-e" "CHROMA_PERSIST_DIR=/app/OUTPUT_DATA2/emdeddings")
fi

# Prevent giant logs on long runs
DOCKER_ARGS+=("--log-driver" "local" "--log-opt" "max-size=10m" "--log-opt" "max-file=3")

# Image
DOCKER_ARGS+=("$IMAGE_NAME")

# Command
if [[ ${#CMD_ARGS[@]} -eq 0 ]]; then
  CMD_ARGS=("${DEFAULT_CMD[@]}")
fi

# --------------------------
# Run
# --------------------------
log "Running container '$CONTAINER_NAME' from '$IMAGE_NAME'"
printf "%s[INFO]%s docker %q\n" "$BLUE" "$NC" "${DOCKER_ARGS[*]} ${CMD_ARGS[*]}"
docker "${DOCKER_ARGS[@]}" "${CMD_ARGS[@]}"
ok "Container exited"
