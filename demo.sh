#!/bin/bash
# AutoANE Demo — train a transformer from scratch and generate text in one command.
#
# Usage:
#   bash demo.sh                    # Full demo: download, train 2min, generate
#   bash demo.sh --quick            # Quick: train 30s
#   bash demo.sh --generate-only    # Skip training, generate from existing checkpoint
#
# Requirements: macOS 15+, Apple Silicon, Xcode Command Line Tools
#   For text decoding: pip install transformers

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_DIR="$SCRIPT_DIR/training"
DATA_FILE="$SCRIPT_DIR/tinystories_smollm2_data00.bin"
CKPT="$TRAIN_DIR/ane_autoresearch_ckpt.bin"

TIME_BUDGET=120
GENERATE_ONLY=false

for arg in "$@"; do
    case $arg in
        --quick) TIME_BUDGET=30 ;;
        --generate-only) GENERATE_ONLY=true ;;
        --help|-h)
            echo "Usage: bash demo.sh [--quick] [--generate-only]"
            echo "  --quick          Train for 30s instead of 120s"
            echo "  --generate-only  Skip training, use existing checkpoint"
            exit 0
            ;;
    esac
done

echo "============================================"
echo "  AutoANE Demo"
echo "  Train a transformer, then generate text"
echo "============================================"
echo ""

# Step 1: Download data
if [ ! -f "$DATA_FILE" ]; then
    echo "[1/4] Downloading training data (~40MB)..."
    bash "$SCRIPT_DIR/tools/download_data.sh"
else
    echo "[1/4] Training data found ($(du -h "$DATA_FILE" | cut -f1))"
fi

if [ "$GENERATE_ONLY" = true ]; then
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: No checkpoint found at $CKPT"
        echo "Run without --generate-only first to train a model."
        exit 1
    fi
    echo "[2/4] Skipping build (--generate-only)"
    echo "[3/4] Skipping training (--generate-only)"
else
    # Step 2: Build
    echo ""
    echo "[2/4] Building training binary..."
    cd "$TRAIN_DIR"
    make clean > /dev/null 2>&1
    make MODEL=autoresearch 2>&1 | tail -1
    echo "  Built successfully."

    # Step 3: Train
    echo ""
    echo "[3/4] Training 36M-param transformer for ${TIME_BUDGET}s..."
    echo "  Architecture: 512d/4L, SEQ=128, LR=4e-4, CPU-only"
    echo "  Data: TinyStories (children's stories, 20M tokens)"
    echo ""

    ./train --scratch --time $TIME_BUDGET --cpu-only \
        --lr 4e-4 --accum 10 --warmup 100 --clip 1.0 \
        --data ../tinystories_smollm2_data00.bin 2>&1 | \
        grep -E "^(Config|Params|Step|final_loss|val_loss|num_steps|training_seconds|num_params)" || true

    echo ""
    if [ -f "$CKPT" ]; then
        echo "  Checkpoint saved: $CKPT ($(du -h "$CKPT" | cut -f1))"
    else
        echo "  WARNING: No checkpoint found. Training may have been too short."
    fi
fi

# Step 4: Generate
echo ""
echo "[4/4] Generating text..."
echo ""

cd "$SCRIPT_DIR"
if command -v python3 &> /dev/null; then
    python3 generate.py "$CKPT" \
        --prompt "Once upon a time" \
        --tokens 200 \
        --temperature 0.8 \
        --top-k 40 \
        --seed 42
else
    echo "ERROR: python3 not found"
    exit 1
fi

echo ""
echo "============================================"
echo "  Demo complete!"
echo ""
echo "  Try different prompts:"
echo "    python3 generate.py $CKPT --prompt 'The little dog'"
echo "    python3 generate.py $CKPT --prompt 'Once there was a' --temperature 1.0"
echo ""
echo "  Export to GGUF:"
echo "    python3 tools/export_to_gguf.py $CKPT model.gguf"
echo ""
echo "  Run autonomous search (100 experiments):"
echo "    cd training && python3 run_autosearch.py --experiments 100"
echo "============================================"
