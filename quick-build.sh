#!/bin/bash

# Quick build script for CourseGen - CPU-only, no NVIDIA packages

set -euo pipefail

echo "🚀 Quick CourseGen CPU-only Build"
echo "================================="

# Stop any running builds
echo "⏹️  Stopping any running builds..."
docker buildx stop || true

# Clean up build cache if requested
if [[ "${1:-}" == "--clean" ]]; then
    echo "🧹 Cleaning Docker build cache..."
    docker builder prune -f
    docker system prune -f
fi

# Build with minimal Dockerfile (CPU-only)
echo "🔨 Building CourseGen (CPU-only, minimal)..."
docker build \
    --file Dockerfile.minimal \
    --tag coursegen:latest \
    --progress=plain \
    --no-cache \
    .

if [[ $? -eq 0 ]]; then
    echo "✅ Build completed successfully!"
    echo "📊 Image size: $(docker images coursegen:latest --format '{{.Size}}')"
    echo ""
    echo "🎯 Quick test:"
    echo "  docker run --rm coursegen:latest --help"
    echo ""
    echo "🚀 Generate questions:"
    echo "  ./run.sh --generate-questions"
else
    echo "❌ Build failed!"
    exit 1
fi