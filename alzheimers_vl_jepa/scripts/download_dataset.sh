#!/usr/bin/env bash
# ================================================================
# download_dataset.sh
# Downloads the "Augmented Alzheimer MRI Dataset" from Kaggle
# Requirements:
#   - kaggle CLI installed: pip install kaggle
#   - Kaggle API key at ~/.kaggle/kaggle.json
# ================================================================

set -e   # Exit on any error

DATASET_SLUG="uraninjo/augmented-alzheimer-mri-dataset"
TARGET_DIR="data/raw/augmented_alzheimer_mri_dataset"

echo "=== Alzheimer's MRI Dataset Downloader ==="

# Check kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: kaggle CLI not found."
    echo "Install with: pip install kaggle"
    echo "Then add your API key to ~/.kaggle/kaggle.json"
    exit 1
fi

mkdir -p "$TARGET_DIR"

echo "Downloading dataset from Kaggle..."
kaggle datasets download -d "$DATASET_SLUG" --unzip -p "$TARGET_DIR"

echo ""
echo "Dataset downloaded to: $TARGET_DIR"
echo "Expected structure:"
echo "  $TARGET_DIR/train/{MildDemented,ModerateDemented,NonDemented,VeryMildDemented}/"
echo "  $TARGET_DIR/test/{MildDemented,ModerateDemented,NonDemented,VeryMildDemented}/"
echo ""
echo "Done! Run training with: python main.py"
