#!/bin/bash

# Documentation deployment script for Entanglement-Enhanced NLP
# This script builds and deploys the documentation to GitHub Pages

set -e  # Exit on any error

echo "🚀 Starting documentation deployment..."

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo "❌ MkDocs is not installed. Installing..."
    pip install mkdocs mkdocs-material pymdown-extensions
fi

# Check if we're in the right directory
if [ ! -f "mkdocs.yml" ]; then
    echo "❌ mkdocs.yml not found. Please run this script from the project root."
    exit 1
fi

# Verify all documentation files exist
echo "📋 Checking documentation files..."
required_files=(
    "docs/index.md"
    "docs/installation.md"
    "docs/usage.md"
    "docs/api.md"
    "docs/theory.md"
    "docs/architecture.md"
    "docs/cli.md"
    "docs/examples.md"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing required file: $file"
        exit 1
    fi
done

echo "✅ All required documentation files found"

# Test the documentation build locally
echo "🔨 Testing documentation build..."
mkdocs build --strict

if [ $? -eq 0 ]; then
    echo "✅ Documentation build successful"
else
    echo "❌ Documentation build failed"
    exit 1
fi

# Deploy to GitHub Pages
echo "🌐 Deploying to GitHub Pages..."
mkdocs gh-deploy --force

if [ $? -eq 0 ]; then
    echo "✅ Documentation successfully deployed to GitHub Pages!"
    echo "📖 Your documentation will be available at: https://krish567366.github.io/entanglement-enhanced-nlp/"
else
    echo "❌ Deployment failed"
    exit 1
fi

echo "🎉 Documentation deployment complete!"
