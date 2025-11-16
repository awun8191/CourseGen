#!/bin/bash
# Deploy CourseGen Cloud Functions to Firebase

set -euo pipefail

echo "🚀 Deploying CourseGen Cloud Functions..."

# Check if Firebase CLI is installed
if ! command -v firebase &> /dev/null; then
    echo "❌ Firebase CLI not found. Install with: npm install -g firebase-tools"
    exit 1
fi

# Check if logged in
if ! firebase projects:list &> /dev/null; then
    echo "❌ Not logged in to Firebase. Run: firebase login"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Deploy functions
echo "🔥 Deploying functions..."
firebase deploy --only functions

echo "✅ Deployment complete!"
echo ""
echo "📊 Verify deployment:"
echo "  firebase functions:list"
echo ""
echo "📝 View logs:"
echo "  firebase functions:log --only updateQuestionStats"
echo "  firebase functions:log --only decrementQuestionStats"
