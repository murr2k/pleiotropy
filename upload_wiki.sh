#!/bin/bash

# Script to upload wiki pages to GitHub wiki repository

set -e

echo "📚 Uploading Wiki to GitHub..."

# Check if wiki directory exists
if [ ! -d "wiki" ]; then
    echo "❌ Error: wiki directory not found"
    exit 1
fi

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "📁 Created temporary directory: $TEMP_DIR"

# Clone the wiki repository
echo "📥 Cloning wiki repository..."
git clone https://github.com/murr2k/pleiotropy.wiki.git "$TEMP_DIR/wiki" || {
    echo "❌ Failed to clone wiki repository"
    echo "Make sure:"
    echo "1. The wiki is enabled in repository settings"
    echo "2. You have push access to the repository"
    exit 1
}

# Copy wiki files
echo "📋 Copying wiki files..."
cp wiki/*.md "$TEMP_DIR/wiki/"

# Navigate to wiki repo
cd "$TEMP_DIR/wiki"

# Check if there are changes
if git diff --quiet && git diff --staged --quiet; then
    echo "✅ Wiki is already up to date"
else
    # Commit and push
    echo "💾 Committing changes..."
    git add .
    git commit -m "Update wiki documentation

Updated with comprehensive documentation including:
- Project overview and status
- Installation and setup guides
- API reference and architecture
- Development roadmap
- Contributing guidelines"

    echo "🚀 Pushing to GitHub..."
    git push origin master || git push origin main

    echo "✅ Wiki successfully uploaded!"
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo "🎉 Done! View your wiki at: https://github.com/murr2k/pleiotropy/wiki"