#!/bin/bash
# Quick setup script for Modal training
# Run: bash src/post-training/scripts/setup_modal.sh

set -e  # Exit on error

echo "🚀 Modal GRPO Training Setup"
echo "=============================="
echo ""

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "📦 Installing Modal SDK..."
    pip install modal
else
    echo "✓ Modal SDK already installed"
fi

# Check if authenticated
if ! modal profile list &> /dev/null; then
    echo ""
    echo "🔐 Authenticating with Modal..."
    echo "   (This will open your browser)"
    modal setup
else
    echo "✓ Already authenticated with Modal"
fi

echo ""
echo "🔑 Setting up secrets..."
echo ""

# Check for .env file
if [ -f "src/post-training/.env" ]; then
    source src/post-training/.env
    echo "✓ Loaded .env file"
else
    echo "⚠️  No .env file found at src/post-training/.env"
fi

# Setup Gemini secret
if [ -z "$GEMINI_API_KEY" ]; then
    echo ""
    read -p "Enter your Gemini API key: " GEMINI_API_KEY
fi

if [ ! -z "$GEMINI_API_KEY" ]; then
    echo "Creating gemini-secret..."
    echo "$GEMINI_API_KEY" | modal secret create gemini-secret GEMINI_API_KEY=- 2>/dev/null || \
        echo "⚠️  gemini-secret may already exist (this is fine)"
else
    echo "⚠️  Skipping Gemini secret (no key provided)"
fi

# Setup HuggingFace secret
if [ -z "$HF_TOKEN" ]; then
    echo ""
    read -p "Enter your HuggingFace token (optional, press Enter to skip): " HF_TOKEN
fi

if [ ! -z "$HF_TOKEN" ]; then
    echo "Creating huggingface-secret..."
    echo "$HF_TOKEN" | modal secret create huggingface-secret HF_TOKEN=- 2>/dev/null || \
        echo "⚠️  huggingface-secret may already exist (this is fine)"
else
    echo "ℹ️  Skipping HuggingFace secret"
fi

echo ""
echo "📤 Uploading training data..."
cd "$(dirname "$0")/../../.."  # Go to project root

if [ -f "data/processed/train.jsonl" ]; then
    modal run src/post-training/scripts/train_grpo_modal.py::upload_data
    echo "✓ Training data uploaded"
else
    echo "⚠️  Training data not found at data/processed/train.jsonl"
    echo "   You'll need to generate it first, then run:"
    echo "   modal run src/post-training/scripts/train_grpo_modal.py::upload_data"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run training:"
echo "     modal run src/post-training/scripts/train_grpo_modal.py"
echo ""
echo "  2. View logs in real-time (automatic)"
echo ""
echo "  3. Download results after training:"
echo "     modal volume ls grpo-checkpoints"
echo "     modal volume get grpo-checkpoints runs/grpo_TIMESTAMP ./local_output"
echo ""
echo "See README_MODAL.md for more details"
