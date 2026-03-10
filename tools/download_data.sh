#!/bin/bash
# Download TinyStories training data pre-tokenized for different models
# Place the data files one directory above training/ (they're referenced as ../filename.bin)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")"  # repo root

echo "AutoANE Data Download"
echo "Data will be stored in: $DATA_DIR"
echo ""

# SmolLM2 tokenized TinyStories (default, ~40MB)
SMOLLM2_URL="https://huggingface.co/datasets/karpathy/tinystories_smollm2/resolve/main/tinystories_smollm2_data00.bin"
SMOLLM2_FILE="$DATA_DIR/tinystories_smollm2_data00.bin"

# Stories110M tokenized TinyStories (Llama2 tokenizer, ~41MB)
STORIES_URL="https://huggingface.co/datasets/karpathy/tinystories/resolve/main/tinystories_data00.bin"
STORIES_FILE="$DATA_DIR/tinystories_data00.bin"

download_if_needed() {
    local url="$1"
    local file="$2"
    local name="$3"

    if [ -f "$file" ]; then
        echo "  $name: already exists ($(du -h "$file" | cut -f1))"
    else
        echo "  Downloading $name..."
        curl -L -o "$file" "$url"
        echo "  $name: downloaded ($(du -h "$file" | cut -f1))"
    fi
}

echo "Downloading training data..."
download_if_needed "$SMOLLM2_URL" "$SMOLLM2_FILE" "SmolLM2 TinyStories"
download_if_needed "$STORIES_URL" "$STORIES_FILE" "Stories110M TinyStories"

echo ""
echo "Done! Data files:"
ls -lh "$DATA_DIR"/*.bin 2>/dev/null || echo "  No .bin files found"
