#!/bin/bash
# Download nuScenes mini dataset
# Usage: bash scripts/download_data.sh

set -e

echo "=================================="
echo "nuScenes Mini Dataset Download"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Data directory
DATA_DIR="data"
NUSCENES_DIR="${DATA_DIR}/nuscenes"

# Create data directory
mkdir -p ${DATA_DIR}
cd ${DATA_DIR}

# Check if already downloaded
if [ -d "nuscenes/v1.0-mini" ]; then
    echo -e "${YELLOW}nuScenes mini dataset already exists at ${NUSCENES_DIR}${NC}"
    read -p "Do you want to re-download? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
    rm -rf nuscenes
fi

# Download mini split (4GB)
echo "Downloading nuScenes mini split (v1.0-mini, ~4GB)..."
echo "This may take a few minutes depending on your connection..."
echo ""

MINI_URL="https://www.nuscenes.org/data/v1.0-mini.tgz"

if command -v wget &> /dev/null; then
    wget ${MINI_URL}
elif command -v curl &> /dev/null; then
    curl -O ${MINI_URL}
else
    echo "Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Extract
echo ""
echo "Extracting archive..."
mkdir -p nuscenes
tar -xzf v1.0-mini.tgz -C nuscenes

# Cleanup
echo "Cleaning up..."
rm v1.0-mini.tgz

# Verify structure
echo ""
echo "Verifying dataset structure..."
cd ..

if [ -f "${NUSCENES_DIR}/v1.0-mini/scene.json" ]; then
    echo -e "${GREEN}✓ Dataset downloaded successfully!${NC}"

    # Show structure
    echo ""
    echo "Dataset structure:"
    ls -lh ${NUSCENES_DIR}/

    # Count scenes
    NUM_SCENES=$(ls ${NUSCENES_DIR}/samples/CAM_FRONT/ | wc -l)
    echo ""
    echo "Number of sample images: ${NUM_SCENES}"

    echo ""
    echo "=================================="
    echo -e "${GREEN}Download Complete!${NC}"
    echo "=================================="
    echo ""
    echo "Dataset location: ${NUSCENES_DIR}"
    echo ""
    echo "Next step: Test the dataset"
    echo "  bash scripts/test_dataloader.sh"
else
    echo "Error: Dataset verification failed!"
    exit 1
fi
