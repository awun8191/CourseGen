#!/bin/bash

set -e  # Exit on error

# Check for .env file
if [ ! -f .env ]; then
  echo "Error: .env file not found. Copy .env.example to .env and fill in your API keys."
  exit 1
fi

# Load .env (source it for vars)
source .env

# Default command (bash if none provided)
CMD="${1:-bash}"

# Path to Firestore service account (mount if exists)
FIRESTORE_JSON="enginneringhub_firebase_service.json"  # Your filename (note typo in 'enginneringhub')
if [ ! -f "$FIRESTORE_JSON" ]; then
  echo "Warning: Firestore JSON ($FIRESTORE_JSON) not found. Some features may fail."
  FIRESTORE_MOUNT=""
else
  FIRESTORE_MOUNT="-v $(pwd)/$FIRESTORE_JSON:/app/$FIRESTORE_SERVICE_ACCOUNT:ro"
fi

echo "Running CourseGen container with env from .env..."
echo "Command: $CMD"
echo "Firestore mount: $FIRESTORE_MOUNT"

# Run docker with bash -c to execute the command properly
docker run -it --rm -v "$(pwd)":/app $FIRESTORE_MOUNT --env-file .env coursegen:latest bash -c "$CMD"