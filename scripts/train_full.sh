#!/bin/bash
# Full training (24 epochs)
# Usage: bash scripts/train_full.sh

set -e

echo "=================================="
echo "Full Training (24 epochs)"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check data
if [ ! -d "data/nuscenes/v1.0-mini" ]; then
    echo "Error: nuScenes data not found!"
    echo "Run: bash scripts/download_data.sh"
    exit 1
fi

# Training config
WORK_DIR="work_dirs/nvautonet_mini"
BATCH_SIZE=2
EPOCHS=24
GPU=0

# Confirm with user
echo "Training configuration:"
echo "  Work dir: ${WORK_DIR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "  GPU: ${GPU}"
echo ""
echo -e "${YELLOW}This will take approximately 2-3 hours.${NC}"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create work directory
mkdir -p ${WORK_DIR}

# Run training
echo ""
echo "Starting training..."
echo ""

python tools/train.py \
    --data-root data/nuscenes \
    --work-dir ${WORK_DIR} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 2e-4 \
    --use-mini \
    --gpu ${GPU} 2>&1 | tee ${WORK_DIR}/train.log

# Check completion
if [ -f "${WORK_DIR}/best.pth" ]; then
    echo ""
    echo "=================================="
    echo -e "${GREEN}Training completed successfully! ✓${NC}"
    echo "=================================="
    echo ""
    echo "Best checkpoint: ${WORK_DIR}/best.pth"
    echo "Latest checkpoint: ${WORK_DIR}/latest.pth"
    echo ""
    echo "View logs:"
    echo "  tensorboard --logdir ${WORK_DIR}/tensorboard"
    echo ""
    echo "Run evaluation:"
    echo "  python tools/test.py --checkpoint ${WORK_DIR}/best.pth --data-root data/nuscenes --use-mini"
else
    echo ""
    echo "Training may have issues. Check logs: ${WORK_DIR}/train.log"
fi
