#!/bin/bash

echo "Building CourseGen Docker image (CPU-only, no GPU/Tesseract)..."
docker build -t coursegen:latest .
echo "Build complete! Image: coursegen:latest"
echo "Size: $(docker images coursegen:latest --format '{{.Size}}')"
echo "To run: ./run.sh [optional-command]"
echo "Example: ./run.sh 'python -m services.QuestionRag.gemini_question_gen --generate-questions'"