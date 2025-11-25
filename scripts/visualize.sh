#!/bin/bash
# Visualize predictions
# Usage: bash scripts/visualize.sh [sample_idx]

set -e

echo "=================================="
echo "Visualizing Predictions"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Config
WORK_DIR="work_dirs/quick_test"
CHECKPOINT="${WORK_DIR}/latest.pth"
OUT_DIR="${WORK_DIR}/visualizations"
SAMPLE_IDX=${1:-0}  # Default to sample 0

# Check checkpoint
if [ ! -f "${CHECKPOINT}" ]; then
    echo -e "${RED}Error: Checkpoint not found: ${CHECKPOINT}${NC}"
    echo "Please run training first: bash scripts/train_quick.sh"
    exit 1
fi

echo "Visualizing sample ${SAMPLE_IDX}..."
echo ""

# Create output directory
mkdir -p ${OUT_DIR}

# Run visualization
python tools/visualize.py \
    --data-root data/nuscenes \
    --checkpoint ${CHECKPOINT} \
    --sample-idx ${SAMPLE_IDX} \
    --out-dir ${OUT_DIR} \
    --use-mini \
    --gpu 0

# Check output
OUTPUT_FILE="${OUT_DIR}/sample_${SAMPLE_IDX}.png"

if [ -f "${OUTPUT_FILE}" ]; then
    echo ""
    echo "=================================="
    echo -e "${GREEN}Visualization complete! ✓${NC}"
    echo "=================================="
    echo ""
    echo "Output saved: ${OUTPUT_FILE}"
    echo ""
    echo "To visualize more samples:"
    echo "  bash scripts/visualize.sh 1"
    echo "  bash scripts/visualize.sh 2"
    echo "  ..."
else
    echo ""
    echo -e "${RED}Visualization failed!${NC}"
    exit 1
fi
